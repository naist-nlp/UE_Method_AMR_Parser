#!/usr/bin/env python3
"""
run_all.py — UE pipeline for 'How Reliable are AMR Parser Outputs?
Uncertainty Estimation for AMR Parsing' (EMNLP 2026).

Reproduce the paper's PRR tables from the released score files
(no AMR data needed, all 6 parsers x 4 beams):
    python3 run_all.py --n_samples 0

Recompute everything from your own k-best outputs (needs AMR 3.0):
    python3 run_all.py --n_samples 0 --data_dir data/k_top_AMR_parsers \
        --cache_dir my_cache --tie_order numpy

Quick test on your own data (10 sentences):
    python3 run_all.py --n_samples 10 --parsers SPRING --beams 3 --cache_dir my_cache
"""

import argparse
import json
import sys
import time
from tqdm import tqdm
from pathlib import Path

import numpy as np

# Make sure the project root is on sys.path
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from estimators.pipeline import (
    PARSERS,
    BEAM_SIZES,
    QUALITY_METRIC_FNS,
    run_pipeline,
)
from estimators.metrics import AMR_SIMILARITY_METRICS

RESULTS_DIR = _ROOT / "estimators" / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Ordered row labels matching the results table
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


def _fmt(val, width: int = 8) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return f"{'N/A':>{width}}"
    return f"{val:>{width}.4f}"


def print_table(prr: dict, parser: str, beam: int):
    """Print normalized PRR results table (values = (AUCPRunc - random) / (AUCPRoracle - random))."""
    col_w = 8
    print(f"\n{'─'*60}")
    print(f"  Parser: {parser}  |  Beam: {beam}  |  PRR = (AUCPRunc-random)/(AUCPRoracle-random)")
    print(f"{'─'*60}")
    header = f"{'UE Method':<36}" + "".join(f"{q:>{col_w}}" for q in QUALITY_COL_ORDER)
    print(header)
    print("─" * len(header))
    for row in UE_ROW_ORDER:
        if row not in prr:
            continue
        row_vals = "".join(_fmt(prr[row].get(q)) for q in QUALITY_COL_ORDER)
        print(f"{row:<36}{row_vals}")


def save_results(all_results: dict):
    """Save full results as JSON."""
    out = RESULTS_DIR / "results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out}")


def save_csv(prr: dict, parser: str, beam: int):
    """Save one CSV per (parser, beam); values are normalized PRR = (AUCPRunc - random) / (AUCPRoracle - random)."""
    fname = RESULTS_DIR / f"results_{parser.replace('/', '_')}_beam{beam}.csv"
    lines = ["UE_Method," + ",".join(QUALITY_COL_ORDER)]
    for row in UE_ROW_ORDER:
        if row not in prr:
            continue
        vals = ",".join(
            _fmt(prr[row].get(q)).strip() for q in QUALITY_COL_ORDER
        )
        lines.append(f"{row},{vals}")
    fname.write_text("\n".join(lines) + "\n")
    print(f"CSV saved: {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    ap = argparse.ArgumentParser(description="AMR Parser UE pipeline")
    ap.add_argument(
        "--n_samples", type=int, default=10,
        help="Number of sentences to process (0 = all, default: 10 for testing)",
    )
    ap.add_argument(
        "--parsers", nargs="+", default=PARSERS,
        help="Parsers to evaluate (default: all six)",
    )
    ap.add_argument(
        "--beams", nargs="+", type=int, default=BEAM_SIZES,
        help="Beam sizes to evaluate (default: 3 5 8 10)",
    )
    ap.add_argument(
        "--metrics", nargs="+", default=list(AMR_SIMILARITY_METRICS.keys()),
        help="AMR similarity metrics (default: all five)",
    )
    ap.add_argument(
        "--quality_metrics", nargs="+", default=list(QUALITY_METRIC_FNS.keys()),
        help="Quality metrics for PRR (default: S++ S2 SB)",
    )
    ap.add_argument(
        "--data_dir", type=Path, default=None,
        help="Path to k_top_AMR_parsers/ (default: auto-detect)",
    )
    ap.add_argument(
        "--cache_dir", type=Path, default=_ROOT / "estimators" / "results" / "cache",
        help="Directory to cache similarity matrices (avoids recomputing on resume)",
    )
    ap.add_argument(
        "--no_cache", action="store_true",
        help="Disable caching (always recompute similarity matrices)",
    )
    ap.add_argument(
        "--tie_order", choices=["original", "numpy"], default="original",
        help="'original': sort tied uncertainty scores in the stored order of the "
             "paper's runs (exact paper numbers on any CPU); 'numpy': plain np.argsort",
    )
    ap.add_argument(
        "--recompute_ue", action="store_true",
        help="recompute the UE scores from the similarity matrices instead of using the "
             "stored ones (last bits of eigen-decompositions differ between CPUs)",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    n_samples = args.n_samples if args.n_samples > 0 else None
    cache_dir = None if args.no_cache else args.cache_dir

    print("=" * 60)
    print("  AMR Parser Uncertainty Estimation")
    print("  'Can we believe our AMR parsers?'")
    print("=" * 60)
    print(f"  parsers      : {args.parsers}")
    print(f"  beams        : {args.beams}")
    print(f"  sim metrics  : {args.metrics}")
    print(f"  qual metrics : {args.quality_metrics}")
    print(f"  n_samples    : {n_samples or 'ALL'}")
    print(f"  tie order    : {args.tie_order}")
    print(f"  UE scores    : {'recomputed' if args.recompute_ue else 'stored (recomputed if absent)'}")
    print("=" * 60)

    all_results: dict = {}
    t0 = time.time()

    for parser in tqdm(args.parsers, desc="parsers"):
        all_results[parser] = {}
        for beam in tqdm(args.beams, desc=f"  {parser} beams", leave=False):
            try:
                prr = run_pipeline(
                    parser=parser,
                    beam_size=beam,
                    n_samples=n_samples,
                    similarity_metrics=args.metrics,
                    quality_metrics=args.quality_metrics,
                    data_dir=args.data_dir,
                    cache_dir=cache_dir,
                    verbose=True,
                    tie_order=args.tie_order,
                    recompute_ue=args.recompute_ue,
                )
                all_results[parser][str(beam)] = prr
                print_table(prr, parser, beam)
                save_csv(prr, parser, beam)
            except FileNotFoundError as e:
                print(f"  SKIP {parser} beam={beam}: {e}")
                all_results[parser][str(beam)] = {}
            except Exception as e:
                print(f"  ERROR {parser} beam={beam}: {e}")
                import traceback; traceback.print_exc()
                all_results[parser][str(beam)] = {}

    save_results(all_results)
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
