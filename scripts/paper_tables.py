#!/usr/bin/env python3
"""
paper_tables.py — Tables 1, 3 and 4 of the paper, plus the PRR averages and
Spearman correlations quoted in the text (Figures 1 and 4).

Reads estimators/results/results_<parser>_beam<k>.csv (written by run_all.py)
and the cached top-1 scores. Prints the tables and writes them to
estimators/results/paper_tables.md.

    python3 scripts/paper_tables.py
"""

import csv
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = _ROOT / "estimators" / "results"
CACHE_DIR = RESULTS_DIR / "cache"
BEAMS = [3, 5, 8, 10]
QUALITY = ["S++", "S2", "SB"]

DISPLAY = {
    "SPRING": "SPRING",
    "IBM_parser": "IBM-Parser",
    "HMG-AMR-T5": "HMG-AMR-T5",
    "HMG-AMR-BART": "HMG-AMR-BART",
    "amr3joint_ontowiki2_seed44": "MBSE",
    "ibm_amr3.0_neur_al": "IBM-Neural",
    "Qwen3-32B_p1": "Qwen3-32B",
    "Ministral-14B_p1": "Ministral-14B",
}
TABLE1_ORDER = ["SPRING", "amr3joint_ontowiki2_seed44", "HMG-AMR-BART",
                "IBM_parser", "ibm_amr3.0_neur_al", "HMG-AMR-T5"]
TABLE3_ORDER = TABLE1_ORDER + ["Qwen3-32B_p1", "Ministral-14B_p1"]
TABLE4_ORDER = ["SPRING", "IBM_parser", "HMG-AMR-T5", "HMG-AMR-BART",
                "amr3joint_ontowiki2_seed44", "ibm_amr3.0_neur_al"]

# The 13 UE methods of the paper (Smatch++ is a quality metric only)
UE_METHODS = {
    "LexicalSimilarity": "LexSim",
    "EigValLaplacian_smatch": "EigV-Smatch",
    "EigValLaplacian_s2match": "EigV-S2match",
    "EigValLaplacian_sema": "EigV-SEMA",
    "EigValLaplacian_wlk": "EigV-WLK",
    "DegMat_smatch": "DegMat-Smatch",
    "DegMat_s2match": "DegMat-S2match",
    "DegMat_sema": "DegMat-SEMA",
    "DegMat_wlk": "DegMat-WLK",
    "Eccentricity_smatch": "Ecc-Smatch",
    "Eccentricity_s2match": "Ecc-S2match",
    "Eccentricity_sema": "Ecc-SEMA",
    "Eccentricity_wlk": "Ecc-WLK",
}


def load_prr(beam: int) -> dict:
    """{parser: {ue_method: {quality: prr}}} for one beam size."""
    out = {}
    for p in TABLE4_ORDER:
        with open(RESULTS_DIR / f"results_{p}_beam{beam}.csv") as f:
            out[p] = {r["UE_Method"]: {q: float(r[q]) for q in QUALITY}
                      for r in csv.DictReader(f) if r["UE_Method"] in UE_METHODS}
    return out


PAPER_RUN_LOG = _ROOT / "expected" / "top1_smatch_paper_run.log"
_LOG_NAMES = {"SPRING": "SPRING", "IBM": "IBM_parser", "HMG-T5": "HMG-AMR-T5",
              "HMG-BART": "HMG-AMR-BART", "amr3joint-seed44": "amr3joint_ontowiki2_seed44",
              "ibm-neur-al": "ibm_amr3.0_neur_al", "Qwen3-32B": "Qwen3-32B_p1",
              "Ministral-14B": "Ministral-14B_p1"}


def top1_smatch_paper_run() -> dict:
    """
    Top-1 Smatch of the paper's runs, read from their log. Smatch's hill
    climbing uses random restarts seeded from the clock, so a re-run changes
    these means by up to a few 1e-4; the per-sentence values were not kept.
    """
    out = {}
    for line in PAPER_RUN_LOG.read_text().splitlines():
        parts = line.split()
        if len(parts) > 2 and parts[0] in _LOG_NAMES and parts[2].startswith("mean="):
            out[_LOG_NAMES[parts[0]]] = float(parts[2][len("mean="):])
    return out


def top1_smatch(parser: str):
    """Mean per-sentence top-1 Smatch (beam 3) of the re-run, or None if the score file is missing."""
    files = sorted(CACHE_DIR.glob(f"{parser}_beam3_top1_smatch_n*.npz"))
    return float(np.load(files[0])["scores"].mean()) if files else None


def top1_smatchpp(parser: str) -> float:
    """Mean per-sentence top-1 Smatch++ (beam 3), the S++ quality scores of the PRR runs."""
    return float(np.load(CACHE_DIR / f"{parser}_beam3_qual_S++_n1898.npz")["scores"].mean())


def fmt(v) -> str:
    return "n/a" if v is None else f"{v:.4f}"


def table1() -> dict:
    """{parser: (Smatch of the paper's run, Smatch++, Smatch of the re-run)}"""
    paper = top1_smatch_paper_run()
    return {p: (paper.get(p), top1_smatchpp(p), top1_smatch(p)) for p in TABLE1_ORDER}


def table3() -> dict:
    """{parser: (Smatch of the paper's run, Smatch of the re-run)}"""
    paper = top1_smatch_paper_run()
    return {p: (paper.get(p), top1_smatch(p)) for p in TABLE3_ORDER}


def figure1(prr_by_beam: dict) -> dict:
    """Mean PRR per parser and quality metric over all beams and the 13 UE methods."""
    return {(p, q): np.mean([prr_by_beam[b][p][m][q] for b in BEAMS for m in UE_METHODS])
            for p in TABLE4_ORDER for q in QUALITY}


def figure4_rho(prr_by_beam: dict, smatch: dict) -> dict:
    """Spearman rho between top-1 Smatch and mean PRR over the 24 parser/beam points."""
    rho = {}
    for q in QUALITY:
        xs = [smatch[p] for b in BEAMS for p in TABLE4_ORDER]
        ys = [np.mean([prr_by_beam[b][p][m][q] for m in UE_METHODS]) for b in BEAMS for p in TABLE4_ORDER]
        rho[q] = spearmanr(xs, ys)[0]
    return rho


def main():
    prr = {b: load_prr(b) for b in BEAMS}
    lines = []

    note = ("Smatch: the paper's run (`expected/top1_smatch_paper_run.log`); "
            "re-run: `scripts/export_scores.py`, which differs by random restarts.")
    lines += ["## Table 1: top-1 parsing performance", "",
              "| Parser | Smatch | Smatch++ | Smatch (re-run) |", "|---|---|---|---|"]
    t1 = table1()
    lines += [f"| {DISPLAY[p]} | {fmt(s)} | {fmt(spp)} | {fmt(rerun)} |" for p, (s, spp, rerun) in t1.items()]
    lines += ["", note]

    lines += ["", "## Table 3: top-1 Smatch incl. LLM baselines", "",
              "| Parser | Smatch | Smatch (re-run) |", "|---|---|---|"]
    lines += [f"| {DISPLAY[p]} | {fmt(s)} | {fmt(rerun)} |" for p, (s, rerun) in table3().items()]

    lines += ["", "## Table 4: PRR at k = 10 (S++ = Smatch++, SB = SemBLEU)", "",
              "| UE method | " + " | ".join(f"{DISPLAY[p]} S++ | {DISPLAY[p]} SB" for p in TABLE4_ORDER) + " |",
              "|---|" + "---|" * (2 * len(TABLE4_ORDER))]
    for m, name in UE_METHODS.items():
        cells = [f"{prr[10][p][m][q]:.4f}" for p in TABLE4_ORDER for q in ("S++", "SB")]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    f1 = figure1(prr)
    lines += ["", "## Figure 1: mean PRR over all beams and UE methods", "",
              "| Parser | Smatch++ | S2match | SemBLEU | all three |", "|---|---|---|---|---|"]
    for p in TABLE4_ORDER:
        lines.append(f"| {DISPLAY[p]} | " + " | ".join(f"{f1[(p, q)]:.3f}" for q in QUALITY)
                     + f" | {np.mean([f1[(p, q)] for q in QUALITY]):.3f} |")

    smatch = {p: t1[p][0] for p in TABLE4_ORDER}
    if all(v is not None for v in smatch.values()):
        rho = figure4_rho(prr, smatch)
        lines += ["", "## Figure 4: Spearman rho (top-1 Smatch vs mean PRR, 24 points)", "",
                  "| Smatch++ | S2match | SemBLEU |", "|---|---|---|",
                  "| " + " | ".join(f"{rho[q]:.3f}" for q in QUALITY) + " |"]

    text = "\n".join(lines) + "\n"
    print(text)
    (RESULTS_DIR / "paper_tables.md").write_text(text)
    print(f"Written to {RESULTS_DIR / 'paper_tables.md'}")


if __name__ == "__main__":
    main()
