"""
LLMParser: few-shot AMR parsing with HuggingFace instruction-tuned models.

Generates k diverse candidates per sentence via temperature sampling.
Supports any HF model; tested with Ministral-3-Instruct-2512 and Qwen3 families.
Prompt variants are defined in prompt_registry.py and selected by prompt_id.
"""

import os
import re
import torch
import penman
from penman.models.noop import NoOpModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List

from llm_parsers.prompt_registry import get_prompt, DEFAULT_PROMPT_ID

# Shared HF cache on the cluster — fall back to default if not available
_SHARED_CACHE = "/cl/home2/share/huggingface/hub"
HF_CACHE_DIR  = _SHARED_CACHE if os.path.isdir(_SHARED_CACHE) else None


def _patch_fp8_dequantize():
    """
    Monkey-patch transformers' FP8 dequantizer to handle per-tensor (scalar) scales.

    Ministral-3-2512 uses per-tensor FP8: each weight has a single scalar
    weight_scale_inv (shape []).  The stock _dequantize_one assumes a 2-D
    block-scale grid and crashes with "not enough values to unpack" on scalars.

    The patch adds a fast-path: if the scale has fewer than 2 dimensions, treat
    it as a per-tensor (or per-row) scale and multiply directly.
    """
    import transformers.integrations.finegrained_fp8 as _fp8_mod

    original = _fp8_mod.Fp8Dequantize._dequantize_one

    def _dequantize_one_patched(self, quantized, scales):
        fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
        if quantized.dtype == torch.int8 or (fp4_dtype is not None and quantized.dtype == fp4_dtype):
            quantized_fp32 = self._unpack_fp4(quantized)
        else:
            quantized_fp32 = quantized.to(torch.float32)

        # Per-tensor scale (scalar, ndim==0) or per-row scale (1-D): multiply directly.
        if scales.ndim < 2:
            out = quantized_fp32 * scales.to(torch.float32)
            return out.to(torch.bfloat16)

        # Original block-wise logic for 2-D+ scale grids.
        return original(self, quantized, scales)

    _fp8_mod.Fp8Dequantize._dequantize_one = _dequantize_one_patched


def _load_causal_lm(model_name: str, dtype, cache_dir):
    """
    Load a causal LM.

    Ministral-3 (2512 series) checkpoints are FP8-quantized multimodal models.
    - Uses Mistral3ForConditionalGeneration (matches the language_model.* weight layout).
    - FineGrainedFP8Config(dequantize=True) triggers FP8→bf16 conversion at load time,
      which works on any GPU (A100 sm_80 has no native FP8 compute support).
    - _patch_fp8_dequantize() fixes a transformers bug: scalar per-tensor scales
      (shape []) crash the stock block-scale dequantizer.
    - Vision tower weights are loaded but ignored during text-only generation.
    """
    kwargs = dict(device_map="auto", trust_remote_code=True, cache_dir=cache_dir)
    if "ministral" in model_name.lower():
        from transformers import Mistral3ForConditionalGeneration, FineGrainedFP8Config
        _patch_fp8_dequantize()
        kwargs["quantization_config"] = FineGrainedFP8Config(dequantize=True)
        return Mistral3ForConditionalGeneration.from_pretrained(model_name, **kwargs)
    # All other models: use AutoModelForCausalLM with explicit dtype
    kwargs["dtype"] = dtype
    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


# Matches opening markdown code fences: ```amr, ```penman, ``` etc.
_CODE_FENCE_RE = re.compile(r"```(?:amr|penman|[a-z]*)?\s*\n?", re.IGNORECASE)


def _clean_output(text: str, after_marker: str = "") -> str:
    """
    Preprocess raw model output before AMR extraction.

    Steps (in order):
      1. If after_marker is given (e.g. "AMR:" for p3 chain-of-thought), discard
         everything before the LAST occurrence — the model may emit reasoning first.
      2. Strip markdown code fences (```amr, ```penman, plain ```).
      3. Strip a leading "AMR:" label the model sometimes repeats in its response.
    """
    if after_marker:
        idx = text.rfind(after_marker)
        if idx != -1:
            text = text[idx + len(after_marker):]

    text = _CODE_FENCE_RE.sub("", text)
    text = text.replace("```", "")

    # Remove a repeated "AMR:" label the model echoes at the start of its response
    text = re.sub(r"^\s*AMR\s*:\s*", "", text, flags=re.IGNORECASE)

    return text


def _extract_amr(text: str) -> str:
    """
    Extract the first complete parenthesised AMR expression from text.

    Handles two failure modes:
    - Garbage before the graph (leading text, explanations): skip to first '('.
    - Truncated output (max_new_tokens hit mid-graph): depth > 0 at end of string.
      Recovery: append the right number of closing parens and let _validate_amr
      decide if the result is parseable.
    """
    start = text.find("(")
    if start == -1:
        return ""
    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]   # complete, balanced graph

    # Truncated: close open parens to attempt partial recovery
    if depth > 0:
        return text[start:] + ")" * depth
    return text[start:]


def _validate_amr(amr_str: str) -> str:
    """Return amr_str if parseable by penman, else return fallback."""
    if not amr_str.strip():
        return "(x / amr-empty)"
    try:
        penman.decode(amr_str.strip(), model=NoOpModel())
        return amr_str.strip()
    except Exception:
        return "(x / amr-empty)"


class LLMParser:
    def __init__(self, model_name: str, dtype=torch.float16,
                 prompt_id: str = DEFAULT_PROMPT_ID):
        """
        Args:
            model_name: HuggingFace model ID or local path.
            dtype:      torch dtype for model weights.
            prompt_id:  Prompt variant from prompt_registry.py (default: p1).
        """
        self.model_name = model_name
        self._prompt = get_prompt(prompt_id)
        self.prompt_id = prompt_id
        print(f"Prompt variant : {prompt_id} — {self._prompt['description']}")

        print(f"Loading tokenizer : {model_name}  (cache: {HF_CACHE_DIR or 'default'})")
        if "ministral" in model_name.lower():
            from transformers import MistralCommonBackend
            self.tokenizer = MistralCommonBackend.from_pretrained(
                model_name, cache_dir=HF_CACHE_DIR
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True, cache_dir=HF_CACHE_DIR
            )

        dtype_str = "native(FP8→auto)" if "ministral" in model_name.lower() else str(dtype)
        print(f"Loading model     : {model_name}  (device_map=auto, dtype={dtype_str})")
        self.model = _load_causal_lm(model_name, dtype, HF_CACHE_DIR)
        self.model.eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _build_inputs(self, sentence: str, device):
        """
        Build tokenized model inputs for one sentence using the selected prompt variant.

        Each prompt variant is a self-contained template with {sentence} as the only
        placeholder.  The full text is passed as a single user message (the role
        instruction is embedded in the template itself).

        MistralCommonBackend does not support the tokenize=False → re-encode
        round-trip: apply_chat_template(tokenize=False) returns an unsafe string
        that encodes back to wrong token IDs.  For Ministral we call
        apply_chat_template(tokenize=True) directly to get input_ids.
        All other tokenizers use the standard tokenize=False → string → encode path.
        """
        user_msg = self._prompt["template"].format(sentence=sentence)
        messages = [{"role": "user", "content": user_msg}]

        if "ministral" in self.model_name.lower():
            # apply_chat_template returns a BatchEncoding (not a plain Tensor) for
            # MistralCommonBackend; .to(device) works on BatchEncoding directly.
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            encoded = encoded.to(device)
            input_ids = encoded["input_ids"]
            return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

        # Standard path for all other models
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        # Qwen3 defaults to thinking (chain-of-thought) mode; disable it so the
        # model outputs AMR directly without <think>...</think> preamble tokens.
        if "qwen3" in self.model_name.lower():
            kwargs["enable_thinking"] = False
        prompt = self.tokenizer.apply_chat_template(messages, **kwargs)
        return self.tokenizer(prompt, return_tensors="pt").to(device)

    def generate_candidates(
        self,
        sentence: str,
        k: int,
        temperature: float = 0.8,
        max_new_tokens: int = 1024,
    ) -> List[str]:
        """
        Generate k AMR candidates for one sentence via temperature sampling.
        Returns a list of k strings (valid Penman AMR or '(x / amr-empty)').

        max_new_tokens=1024 is chosen to safely cover complex AMR graphs (complex
        sentences with named entities, multiple predicates, and reentrance can
        produce graphs with 300-700 tokens). 512 was observed to truncate mid-graph
        on longer sentences, producing unrecoverable partial outputs.
        """
        # device_map="auto" models may return 'meta' for .device; use first param's device instead
        device = next(self.model.parameters()).device
        inputs = self._build_inputs(sentence, device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                num_return_sequences=k,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        prefix        = self._prompt["forced_prefix"]
        extract_after = self._prompt["extract_after"]
        candidates = []
        for out in outputs:
            raw     = self.tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
            cleaned = _clean_output(prefix + raw, extract_after)
            candidates.append(_validate_amr(_extract_amr(cleaned)))

        return candidates
