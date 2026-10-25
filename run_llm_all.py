#!/usr/bin/env python3
"""
run_llm_all.py — Generate candidates + run UE pipeline for LLM parsers.

Two families, three sizes each:
  Ministral-3-Instruct-2512  : 3B -> 8B -> 14B
  Qwen3                      : 4B -> 8B -> 32B

All runs use 3 fixed few-shot examples and temperature=0.7.
Prompt variants (p1–p5) are defined in llm_parsers/prompt_registry.py.

───────────────────────────────────────────────────
USAGE
───────────────────────────────────────────────────

  Quick test (50 sentences, one model):
    python3 run_llm_all.py --model Ministral-3B --n_samples 50 --prompt p1

  Run one model directly (no SLURM):
    python3 run_llm_all.py --model Qwen3-4B --prompt p5

  Submit all models as parallel SLURM jobs (1 GPU each):
    python3 run_llm_all.py --submit --prompt p1

  Submit a specific model with a given prompt:
    python3 run_llm_all.py --submit --model Ministral-3B --prompt p5

  Evaluate only (candidates already generated):
    python3 run_llm_all.py --no-generate --model Qwen3-4B --prompt p1

───────────────────────────────────────────────────
SLURM job layout (--submit)
───────────────────────────────────────────────────
  One sbatch job per model, each requesting 1 GPU.
  Job scripts are saved to logs/sbatch_<label>_<prompt_id>.sh for inspection.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from estimators.pipeline import run_pipeline
from estimators.metrics import AMR_SIMILARITY_METRICS

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    # Ministral-3-Instruct-2512 family (largest available is 14B, not 32B)
    "Ministral-3B":  "mistralai/Ministral-3-3B-Instruct-2512",
    "Ministral-8B":  "mistralai/Ministral-3-8B-Instruct-2512",
    "Ministral-14B": "mistralai/Ministral-3-14B-Instruct-2512",
    # Qwen3 family (thinking mode disabled in llm_parser.py)
    "Qwen3-4B":  "Qwen/Qwen3-4B",
    "Qwen3-8B":  "Qwen/Qwen3-8B",
    "Qwen3-32B": "Qwen/Qwen3-32B",
}

# GPU resource name per model (see cluster GPU table).
# bf16 memory estimates: 3B≈6G, 4B≈8G, 8B≈16G, 14B≈28G, 32B≈64G.
MODEL_GPU = {
    "Ministral-3B":  "3090",      # 24 GB — plenty for 3B bf16
    "Ministral-8B":  "3090",      # 24 GB — fits 8B bf16
    "Ministral-14B": "a6000",     # 48 GB — fits 14B bf16
    "Qwen3-4B":      "3090",      # 24 GB
    "Qwen3-8B":      "3090",      # 24 GB
    "Qwen3-32B":     "a100-80",   # 80 GB — required for 32B bf16
}

DEFAULT_K_VALUES = [3, 5, 10]
DATA_DIR         = _ROOT / "data" / "k_top_AMR_parsers"
RESULTS_FILE     = _ROOT / "estimators" / "results" / "results.json"
CACHE_DIR        = _ROOT / "estimators" / "results" / "cache"
LOGS_DIR         = _ROOT / "logs"

UE_ROW_ORDER = (
    ["LexicalSimilarity", "NumSemSets"]
    + [f"EigValLaplacian_{m}" for m in AMR_SIMILARITY_METRICS]
    + [f"DegMat_{m}"          for m in AMR_SIMILARITY_METRICS]
    + [f"Eccentricity_{m}"    for m in AMR_SIMILARITY_METRICS]
)
QUALITY_COL_ORDER = ["S++", "S2", "SB"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parser_tag(label: str, prompt_id: str) -> str:
    """Unique key used as directory name and results.json key."""
    return f"{label}_{prompt_id}"


def _fmt(val, width=8):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return f"{'N/A':>{width}}"
    return f"{val:>{width}.4f}"


def print_table(prr: dict, label: str, k: int, prompt_id: str):
    col_w = 8
    print(f"\n{'─'*60}")
    print(f"  {label}  |  prompt={prompt_id}  |  k={k}  |  PRR")
    print(f"{'─'*60}")
    header = f"{'UE Method':<36}" + "".join(f"{q:>{col_w}}" for q in QUALITY_COL_ORDER)
    print(header)
    print("─" * len(header))
    for row in UE_ROW_ORDER:
        if row not in prr:
            continue
        vals = "".join(_fmt(prr[row].get(q)) for q in QUALITY_COL_ORDER)
        print(f"{row:<36}{vals}")


def load_results() -> dict:
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text())
    return {}


def save_results(all_results: dict):
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(all_results, indent=2))
    print(f"Results saved -> {RESULTS_FILE}")


def save_csv(prr: dict, tag: str, k: int):
    """Save one CSV per (parser_tag, beam) — same format as SPRING/IBM CSVs."""
    fname = RESULTS_FILE.parent / f"results_{tag}_beam{k}.csv"
    lines = ["UE_Method," + ",".join(QUALITY_COL_ORDER)]
    for row in UE_ROW_ORDER:
        if row not in prr:
            continue
        vals = ",".join(_fmt(prr[row].get(q)).strip() for q in QUALITY_COL_ORDER)
        lines.append(f"{row},{vals}")
    fname.write_text("\n".join(lines) + "\n")
    print(f"CSV  saved  -> {fname}")


# ---------------------------------------------------------------------------
# Step 1: Generate candidates (single model, called directly or via sbatch)
# ---------------------------------------------------------------------------

def generate(label: str, k_values: list, n_samples: int, prompt_id: str = "p1"):
    from llm_parsers.run_llm_parser import run as llm_run
    model_id = MODEL_REGISTRY[label]
    print(f"\n{'#'*60}")
    print(f"  Generating: {label}  ({model_id})  prompt={prompt_id}")
    print(f"{'#'*60}")
    t0 = time.time()
    llm_run(
        model_name=model_id,
        label=label,
        k_values=k_values,
        n_samples=n_samples,
        data_dir=DATA_DIR,
        prompt_id=prompt_id,
    )
    print(f"  [{label}] generation done in {(time.time()-t0)/60:.1f} min")


# ---------------------------------------------------------------------------
# Step 2: Run UE pipeline (single model)
# ---------------------------------------------------------------------------

def evaluate(label: str, k_values: list, n_samples: int, prompt_id: str = "p1"):
    tag         = f"{label}_{prompt_id}"
    all_results = load_results()
    if tag not in all_results:
        all_results[tag] = {}

    for k in k_values:
        print(f"\n{'='*60}")
        print(f"  Evaluating: {tag}  k={k}")
        print(f"{'='*60}")
        t0 = time.time()
        try:
            prr = run_pipeline(
                parser=tag,
                beam_size=k,
                n_samples=n_samples if n_samples > 0 else None,
                data_dir=DATA_DIR,
                cache_dir=CACHE_DIR,
                verbose=True,
            )
            all_results[tag][str(k)] = prr
            print_table(prr, label, k, prompt_id)
            save_csv(prr, tag, k)
            print(f"  [{tag} k={k}] done in {(time.time()-t0)/60:.1f} min")
        except FileNotFoundError as e:
            print(f"  SKIP {tag} k={k}: {e}")
        except Exception as e:
            print(f"  ERROR {tag} k={k}: {e}")
            import traceback; traceback.print_exc()

    save_results(all_results)


# ---------------------------------------------------------------------------
# Step 3: SLURM submission — one job per model
# ---------------------------------------------------------------------------

def submit_jobs(models: list, k_values: list, prompt_id: str, n_samples: int,
                conda_env: str, partition: str, mem: str, cpus: int, time_limit: str):
    """Generate one sbatch script per model and submit it."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    k_str    = " ".join(str(k) for k in k_values)
    submitted = []

    for label in models:
        gpu_type    = MODEL_GPU.get(label, "3090")
        script_name = LOGS_DIR / f"sbatch_{label}_{prompt_id}.sh"
        log_out     = LOGS_DIR / f"{label}_{prompt_id}_%j.out"
        log_err     = LOGS_DIR / f"{label}_{prompt_id}_%j.err"

        partition_line = f"#SBATCH --partition={partition}" if partition else ""

        script = f"""#!/bin/bash
#SBATCH --job-name=amr_{label}
#SBATCH --gres=gpu:{gpu_type}:1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time_limit}
#SBATCH --output={log_out}
#SBATCH --error={log_err}
{partition_line}

echo "=============================="
echo " Job  : $SLURM_JOB_ID"
echo " Node : $SLURMD_NODENAME"
echo " GPU  : $CUDA_VISIBLE_DEVICES"
echo " Model: {label}  gpu={gpu_type}  prompt={prompt_id}"
echo "=============================="

export HF_HOME="/cl/home2/share/huggingface"
export HF_HUB_CACHE="${{HF_HOME}}/hub"
export HF_ASSETS_CACHE="${{HF_HOME}}/assets"
export HF_TOKEN_PATH="${{HOME}}/.cache/huggingface/token"

eval "$(conda shell.bash hook)"
conda activate {conda_env}
cd {_ROOT}

python3 run_llm_all.py \\
    --model   {label} \\
    --k       {k_str} \\
    --prompt  {prompt_id} \\
    --n_samples {n_samples}
"""
        script_name.write_text(script)
        script_name.chmod(0o755)

        ret = os.system(f"sbatch {script_name}")
        if ret == 0:
            print(f"  Submitted: {label}  gpu={gpu_type}  (script: {script_name})")
            submitted.append(label)
        else:
            print(f"  FAILED to submit: {label}")

    print(f"\nSubmitted {len(submitted)}/{len(models)} jobs.")
    print(f"Monitor with: squeue -u $USER")
    print(f"Logs in     : {LOGS_DIR}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="LLM few-shot AMR parser — generate candidates + UE evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # What to run
    run_grp = ap.add_argument_group("What to run")
    run_grp.add_argument("--model",      choices=list(MODEL_REGISTRY.keys()),
                         help="Single model to run (required when not using --submit)")
    run_grp.add_argument("--models",     nargs="+", choices=list(MODEL_REGISTRY.keys()),
                         default=list(MODEL_REGISTRY.keys()),
                         help="Models to submit (used with --submit, default: all)")
    run_grp.add_argument("--k",          nargs="+", type=int, default=DEFAULT_K_VALUES,
                         metavar="K", help="k values (default: 3 5 10)")
    run_grp.add_argument("--prompt",     default="p1",
                         help="Prompt variant ID from prompt_registry.py (default: p1)")
    run_grp.add_argument("--n_samples",  type=int, default=0,
                         help="0=all sentences; positive=first N (for quick tests)")

    # Pipeline steps
    step_grp = ap.add_argument_group("Pipeline steps")
    step_grp.add_argument("--generate",    action="store_true",  default=True)
    step_grp.add_argument("--no-generate", dest="generate", action="store_false")
    step_grp.add_argument("--evaluate",    action="store_true",  default=True)
    step_grp.add_argument("--no-evaluate", dest="evaluate", action="store_false")

    # GPU (manual parallel runs)
    gpu_grp = ap.add_argument_group("GPU (manual parallel runs)")
    gpu_grp.add_argument("--gpu", type=int, default=None,
                         help="Pin to this GPU index (sets CUDA_VISIBLE_DEVICES)")

    # SLURM submission
    slurm_grp = ap.add_argument_group("SLURM submission (--submit)")
    slurm_grp.add_argument("--submit",     action="store_true", default=False,
                           help="Submit one sbatch job per model instead of running directly")
    slurm_grp.add_argument("--conda-env",  default="polygraph",
                           help="Conda environment name (default: polygraph)")
    slurm_grp.add_argument("--partition",  default="",
                           help="SLURM partition name (leave blank for default)")
    slurm_grp.add_argument("--mem",        default="64G",
                           help="Memory per job (default: 64G)")
    slurm_grp.add_argument("--cpus",       type=int, default=4,
                           help="CPUs per job (default: 4)")
    slurm_grp.add_argument("--time-limit", default="24:00:00",
                           help="Max wall time per job (default: 24:00:00)")

    args = ap.parse_args()

    # ── SLURM submission mode ────────────────────────────────────────────────
    if args.submit:
        models = args.models if not args.model else [args.model]
        print(f"Submitting {len(models)} SLURM jobs  (prompt={args.prompt})")
        submit_jobs(
            models=models,
            k_values=args.k,
            prompt_id=args.prompt,
            n_samples=args.n_samples,
            conda_env=args.conda_env,
            partition=args.partition,
            mem=args.mem,
            cpus=args.cpus,
            time_limit=args.time_limit,
        )
        return

    # ── Direct run mode ──────────────────────────────────────────────────────
    if args.model is None:
        ap.error("--model is required when not using --submit")

    # Set GPU before any torch import happens inside generate/evaluate
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"CUDA_VISIBLE_DEVICES={args.gpu}")

    print(f"\nModel     : {args.model}")
    print(f"Prompt    : {args.prompt}")
    print(f"k values  : {args.k}")
    print(f"Samples   : {'all' if args.n_samples == 0 else args.n_samples}")
    print(f"Generate  : {args.generate}   Evaluate: {args.evaluate}")

    if args.generate:
        generate(args.model, args.k, args.n_samples, args.prompt)

    if args.evaluate:
        evaluate(args.model, args.k, args.n_samples, args.prompt)


if __name__ == "__main__":
    main()
