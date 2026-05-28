#!/usr/bin/env python3
"""
Main entry point for the AMR Parser Uncertainty Estimation pipeline.

This script runs the full UE evaluation for all parsers and beam sizes,
computing PRR (Prediction Rejection Ratio) scores for each combination.

Quick test with 10 sentences:
    conda activate polygraph
    python run_all.py --n_samples 10

Full run over all 1898 sentences:
    python run_all.py --n_samples 0

Run specific parsers and beams:
    python run_all.py --parsers SPRING IBM_parser --beams 3 10
"""

import argparse
import json
import sys
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np

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

UE_ROW_ORDER = (
    ["LexicalSimilarity"]
    + [f"EigValLaplacian_{m}" for m in AMR_SIMILARITY_METRICS]
    + [f"DegMat_{m}"          for m in AMR_SIMILARITY_METRICS]
    + [f"Eccentricity_{m}"    for m in AMR_SIMILARITY_METRICS]
)

QUALITY_COL_ORDER = ["S++", "S2", "SB"]


def _fmt(val, width=8):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return f"{'N/A':>{width}}"
    return f"{val:>{width}.4f}"


def print_table(prr, parser, beam):
    col_w = 8
    print(f"\nParser: {parser}  |  Beam: {beam}")
    header = f"{'UE Method':<36}" + "".join(f"{q:>{col_w}}" for q in QUALITY_COL_ORDER)
    print(header)
    print("-" * len(header))
    for row in UE_ROW_ORDER:
        if row not in prr:
            continue
        row_vals = "".join(_fmt(prr[row].get(q)) for q in QUALITY_COL_ORDER)
        print(f"{row:<36}{row_vals}")


def save_results(all_results):
    out = RESULTS_DIR / "results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out}")


def save_csv(prr, parser, beam):
    fname = RESULTS_DIR / f"results_{parser.replace('/', '_')}_beam{beam}.csv"
    lines = ["UE_Method," + ",".join(QUALITY_COL_ORDER)]
    for row in UE_ROW_ORDER:
        if row not in prr:
            continue
        vals = ",".join(_fmt(prr[row].get(q)).strip() for q in QUALITY_COL_ORDER)
        lines.append(f"{row},{vals}")
    fname.write_text("\n".join(lines) + "\n")
    print(f"CSV saved: {fname}")


def parse_args():
    ap = argparse.ArgumentParser(description="AMR Parser Uncertainty Estimation pipeline")
    ap.add_argument(
        "--n_samples", type=int, default=10,
        help="Number of sentences to process (0 = all, default: 10 for quick testing)",
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
        help="AMR similarity metrics for UE (default: all five)",
    )
    ap.add_argument(
        "--quality_metrics", nargs="+", default=list(QUALITY_METRIC_FNS.keys()),
        help="Quality metrics used for PRR (default: S++ S2 SB)",
    )
    ap.add_argument(
        "--data_dir", type=Path, default=None,
        help="Path to k_top_AMR_parsers/ data directory (auto-detected by default)",
    )
    ap.add_argument(
        "--cache_dir", type=Path, default=_ROOT / "estimators" / "results" / "cache",
        help="Directory to cache similarity matrices so interrupted runs can resume",
    )
    ap.add_argument(
        "--no_cache", action="store_true",
        help="Disable caching and always recompute similarity matrices",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    n_samples = args.n_samples if args.n_samples > 0 else None
    cache_dir = None if args.no_cache else args.cache_dir

    print("AMR Parser Uncertainty Estimation")
    print(f"  parsers      : {args.parsers}")
    print(f"  beams        : {args.beams}")
    print(f"  sim metrics  : {args.metrics}")
    print(f"  qual metrics : {args.quality_metrics}")
    print(f"  n_samples    : {n_samples or 'ALL'}")

    all_results = {}
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
