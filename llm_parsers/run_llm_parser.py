"""
Generate k-best AMR candidates for all test sentences using a single LLM.

Reads gold data from the existing SPRING kbest_beam3 directory and writes
output in the same sent_XXXX.json format expected by the UE pipeline.

Output directory: data/k_top_AMR_parsers/{label}_{prompt_id}/kbest_beam{k}/sentences/

Settings (fixed for all runs):
  - temperature = 0.7
  - 3 few-shot examples embedded in each prompt template (prompt_registry.py)

Usage:
    python -m llm_parsers.run_llm_parser \\
        --model   Qwen/Qwen3-8B \\
        --label   Qwen3-8B \\
        --k       3 5 10 \\
        --prompt  p1
"""

import argparse
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from llm_parsers.prompt_registry import get_prompt, DEFAULT_PROMPT_ID

_TEMPERATURE  = 0.7   # fixed for all models and all k values
_GOLD_SOURCE_PARSER = "SPRING"
_GOLD_SOURCE_BEAM   = 3


def _load_gold_sentences(data_dir: Path, n_samples: int) -> list:
    source = data_dir / _GOLD_SOURCE_PARSER / f"kbest_beam{_GOLD_SOURCE_BEAM}"
    nested = source / f"kbest_beam{_GOLD_SOURCE_BEAM}" / "sentences"
    flat   = source / "sentences"
    sent_dir = nested if nested.exists() else flat
    if not sent_dir.exists():
        raise FileNotFoundError(f"Gold sentence dir not found under {source}")

    files   = sorted(sent_dir.glob("sent_*.json"))
    records = [json.loads(f.read_text()) for f in files]
    valid   = [r for r in records if r.get("gold_graph", "").strip()]
    return valid[:n_samples] if n_samples > 0 else valid


def run(model_name: str, label: str, k_values: list,
        n_samples: int, data_dir: Path, prompt_id: str = DEFAULT_PROMPT_ID):

    prompt     = get_prompt(prompt_id)
    parser_tag = f"{label}_{prompt_id}"   # e.g. Ministral-3B_p1, Qwen3-8B_p2

    print(f"\n{'='*60}")
    print(f"  Model     : {model_name}")
    print(f"  Label     : {label}")
    print(f"  Temp      : {_TEMPERATURE} (fixed)")
    print(f"  Prompt    : {prompt_id} — {prompt['description']}")
    print(f"  k values  : {k_values}")
    print(f"  Output tag: {parser_tag}")
    print(f"{'='*60}")

    gold_sentences = _load_gold_sentences(data_dir, n_samples)
    print(f"Loaded {len(gold_sentences)} gold sentences")

    # Import here so CUDA_VISIBLE_DEVICES is already set before torch is loaded
    from llm_parsers.llm_parser import LLMParser
    parser = LLMParser(model_name, prompt_id=prompt_id)

    for k in k_values:
        out_dir = data_dir / parser_tag / f"kbest_beam{k}" / "sentences"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Resume: count already-written files so a restarted job picks up where it left off
        already_done = {f.stem for f in out_dir.glob("sent_*.json")}
        n_resume     = len(already_done)
        if n_resume:
            print(f"\n--- k={k}  RESUMING from sentence {n_resume}/{len(gold_sentences)} ---")
        else:
            print(f"\n--- k={k}  temperature={_TEMPERATURE} ---")

        empty_count = 0
        todo = [r for r in gold_sentences
                if f"sent_{r['sent_idx']:04d}" not in already_done]

        for rec in tqdm(todo, desc=f"{parser_tag} k={k}",
                        initial=n_resume, total=len(gold_sentences)):
            idx      = rec["sent_idx"]
            sentence = rec["sentence"]
            gold     = rec["gold_graph"]

            candidates = parser.generate_candidates(sentence, k=k, temperature=_TEMPERATURE)
            if all(c == "(x / amr-empty)" for c in candidates):
                empty_count += 1

            sent_data = {
                "sent_idx":       idx,
                "sent_id":        rec.get("sent_id", ""),
                "sentence":       sentence,
                "gold_graph":     gold,
                "beam_size":      k,
                "num_candidates": len(candidates),
                "temperature":    _TEMPERATURE,
                "prompt_id":      prompt_id,
                "candidates": [
                    {"rank": r, "graph": g}
                    for r, g in enumerate(candidates, start=1)
                ],
            }
            (out_dir / f"sent_{idx:04d}.json").write_text(json.dumps(sent_data, indent=2))

        total_done = n_resume + len(todo)
        print(f"  Saved {total_done} sentences  |  resumed: {n_resume}  |  fully-empty: {empty_count}")

    meta = {
        "model": model_name, "label": label,
        "temperature": _TEMPERATURE,
        "prompt_id": prompt_id, "prompt_description": prompt["description"],
        "k_values": k_values, "n_sentences": len(gold_sentences),
    }
    (data_dir / parser_tag / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nDone → {data_dir / parser_tag}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",    required=True)
    ap.add_argument("--label",    required=True)
    ap.add_argument("--k",        nargs="+", type=int, default=[3, 5, 10])
    ap.add_argument("--prompt",   default=DEFAULT_PROMPT_ID,
                    help=f"Prompt variant ID (default: {DEFAULT_PROMPT_ID})")
    ap.add_argument("--n_samples", type=int, default=0)
    ap.add_argument("--gpu",      type=int, default=None)
    ap.add_argument("--data_dir", type=Path,
                    default=_ROOT / "data" / "k_top_AMR_parsers")
    args = ap.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    run(
        model_name=args.model,
        label=args.label,
        k_values=args.k,
        n_samples=args.n_samples,
        data_dir=args.data_dir,
        prompt_id=args.prompt,
    )
