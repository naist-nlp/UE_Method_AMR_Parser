#!/usr/bin/env python3
"""
export_scores.py — authors only: write the score files that let anyone
reproduce the paper's numbers without the AMR 3.0 data.

Needs the k-best data (AMR 3.0 licence) and the cached matrices in
estimators/results/cache/. Everything it writes is numbers only:

    <parser>_beam<k>_ue_LexicalSimilarity_n<N>.npz   LexSim scores (the only UE method that reads graph text)
    <parser>_beam<k>_ue_n<N>.npz                     all 17 UE scores, as computed on this machine
    <parser>_beam<k>_order_n<N>.npz                  sort order of every vector compute_prr sorts
    <parser>_beam3_top1_smatch_n<N>.npz              per-sentence top-1 Smatch (Tables 1 and 3)

np.argsort is not stable, and the order it gives tied uncertainty scores
depends on the CPU's SIMD sort routine, and the last bits of the
eigen-decompositions behind EigV and Ecc differ between CPUs. UE scores and
orders are therefore only written for a parser/beam whose 17 x 3 PRR values
all equal the paper's CSVs (expected/) on the machine running this script.

    # 1. fast check, no data needed: does np.argsort on this machine reproduce
    #    the paper for the graph-based methods?
    python3 scripts/export_scores.py --check_only

    # 2. full export on a machine where step 1 reports 24/24 cells
    python3 scripts/export_scores.py --data_dir /path/to/k_top_AMR_parsers
"""

import argparse
import csv
import platform
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from estimators.pipeline import (
    PARSERS,
    BEAM_SIZES,
    _build_stats,
    _make_ue_estimators,
    build_sim_matrices,
    compute_prr,
    compute_quality_vectors,
    load_data,
    order_path,
    run_ue_estimators,
    sort_orders,
    ue_path,
)
from estimators.metrics import AMR_SIMILARITY_METRICS
from estimators.metrics.smatch import compute_single as smatch_single

CACHE_DIR = _ROOT / "estimators" / "results" / "cache"
EXPECTED_DIR = _ROOT / "expected"
N = 1898
QUALITY = ["S++", "S2", "SB"]

# Tables 1 and 3: top-1 Smatch (mean of per-sentence Smatch, beam 3)
TOP1_PAPER = {
    "SPRING": 0.8449,
    "amr3joint_ontowiki2_seed44": 0.8374,
    "HMG-AMR-BART": 0.8358,
    "IBM_parser": 0.8128,
    "ibm_amr3.0_neur_al": 0.8115,
    "HMG-AMR-T5": 0.7693,
    "Qwen3-32B_p1": 0.1214,
    "Ministral-14B_p1": 0.0897,
}


def cpu_summary() -> str:
    model = "unknown CPU"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    simd = ""
    try:
        from numpy._core._multiarray_umath import __cpu_features__ as f
        simd = " ".join(k for k in ("AVX2", "AVX512F", "AVX512_SKX") if f.get(k))
    except ImportError:
        pass
    return f"{platform.node()} | {model} | numpy {np.__version__} | SIMD: {simd or 'none of AVX2/AVX512'}"


def read_expected(parser: str, beam: int) -> dict:
    with open(EXPECTED_DIR / f"results_{parser}_beam{beam}.csv") as f:
        return {r["UE_Method"]: {q: r[q] for q in QUALITY} for r in csv.DictReader(f)}


def mismatches(prr: dict, expected: dict, methods) -> list:
    """(method, metric, ours, paper) for every value that differs at the CSV's 4 decimals."""
    out = []
    for m in methods:
        for q in QUALITY:
            ours = f"{prr[m][q]:.4f}"
            if ours != expected[m][q]:
                out.append((m, q, ours, expected[m][q]))
    return out


def top1_smatch(parser: str, data_dir: Path) -> np.ndarray:
    """Per-sentence Smatch of the top-1 candidate vs gold, as for Tables 1 and 3."""
    sentences = load_data(parser, 3, None, data_dir)
    scores = []
    for s in sentences:
        try:
            scores.append(float(smatch_single(s["candidates"][0]["graph"], s["gold_graph"].strip())))
        except Exception:
            scores.append(0.0)
    return np.array(scores)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check_only", action="store_true",
                    help="only test np.argsort on this machine (graph-based methods, no data needed)")
    ap.add_argument("--data_dir", type=Path, default=_ROOT / "data" / "k_top_AMR_parsers")
    ap.add_argument("--parsers", nargs="+", default=PARSERS)
    ap.add_argument("--beams", nargs="+", type=int, default=BEAM_SIZES)
    ap.add_argument("--skip_top1", action="store_true", help="do not recompute top-1 Smatch")
    args = ap.parse_args()

    print(f"machine: {cpu_summary()}\n")
    metrics = list(AMR_SIMILARITY_METRICS)
    ok_cells, n_cells = 0, 0

    for parser in args.parsers:
        for beam in args.beams:
            n_cells += 1
            sentences = None if args.check_only else load_data(parser, beam, None, args.data_dir)
            sim = build_sim_matrices(sentences, metrics, CACHE_DIR, parser, beam, n=N)
            qual = compute_quality_vectors(sentences, QUALITY, CACHE_DIR, parser, beam, n=N)
            if args.check_only:
                est = _make_ue_estimators(metrics)
                ue = {"NumSemSets": est["NumSemSets"](_build_stats(None, sim[metrics[0]]))}
                for mn in metrics:
                    stats = _build_stats(None, sim[mn])
                    for fam in ("EigValLaplacian", "DegMat", "Eccentricity"):
                        ue[f"{fam}_{mn}"] = est[f"{fam}_{mn}"](stats)
            else:
                ue = run_ue_estimators(sentences, sim, metrics, CACHE_DIR, parser, beam, use_stored=False)

            prr = compute_prr(ue, qual)   # plain np.argsort, as in the paper's runs
            bad = mismatches(prr, read_expected(parser, beam), list(ue))
            status = "MATCH" if not bad else f"{len(bad)} of {3 * len(ue)} values differ"
            print(f"{parser:28s} k={beam:<2d} {status}")
            for m, q, ours, paper in bad[:3]:
                print(f"      {m} {q}: this machine {ours}, paper {paper}")
            if bad:
                continue
            ok_cells += 1
            if not args.check_only:
                np.savez_compressed(
                    order_path(CACHE_DIR, parser, beam, N),
                    **{k: v.astype(np.int32) for k, v in sort_orders(ue, qual).items()},
                )
                np.savez_compressed(ue_path(CACHE_DIR, parser, beam, N), **ue)

    what = "graph-based + NumSemSets" if args.check_only else "all 17 UE methods"
    print(f"\n{ok_cells}/{n_cells} parser/beam cells reproduce the paper exactly ({what}).")
    if args.check_only:
        print(f"{n_cells}/{n_cells}: run the full export on this node. Otherwise try another "
              "node type (sbatch -w <node>); see README, 'How PRR is computed'.")
        return
    print(f"UE scores and sort orders written for {ok_cells} cells to {CACHE_DIR}")

    if not args.skip_top1:
        print("\nTop-1 Smatch (Tables 1 and 3), per-sentence Smatch averaged:")
        for parser, paper in TOP1_PAPER.items():
            try:
                scores = top1_smatch(parser, args.data_dir)
            except FileNotFoundError as e:
                print(f"  {parser:28s} SKIP ({e})")
                continue
            np.savez_compressed(CACHE_DIR / f"{parser}_beam3_top1_smatch_n{len(scores)}.npz", scores=scores)
            mark = "MATCH" if f"{scores.mean():.4f}" == f"{paper:.4f}" else "DIFFERS"
            print(f"  {parser:28s} n={len(scores)}  mean {scores.mean():.4f}  paper {paper:.4f}  {mark}")


if __name__ == "__main__":
    main()
