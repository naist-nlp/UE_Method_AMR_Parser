"""Shared constants and helpers for all paper figure scripts."""
import csv
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("estimators/results")
OUT_DIR     = Path("paper_figures")
BEAMS       = [3, 5, 8, 10]

BASELINE_PARSERS = [
    "SPRING",
    "IBM_parser",
    "HMG-AMR-T5",
    "HMG-AMR-BART",
    "amr3joint_ontowiki2_seed44",
    "ibm_amr3.0_neur_al",
]

PARSER_DISPLAY = {
    "SPRING":                     "SPRING",
    "IBM_parser":                 "IBM-Parser",
    "HMG-AMR-T5":                 "HMG-AMR-T5",
    "HMG-AMR-BART":               "HMG-AMR-BART",
    "amr3joint_ontowiki2_seed44": "MBSE",
    "ibm_amr3.0_neur_al":         "IBM-Neural",
}

PARSER_COLORS = {
    "SPRING":                     "#1565C0",
    "IBM_parser":                 "#E65100",
    "HMG-AMR-T5":                 "#2E7D32",
    "HMG-AMR-BART":               "#6A1B9A",
    "amr3joint_ontowiki2_seed44": "#C62828",
    "ibm_amr3.0_neur_al":         "#00838F",
}

PARSER_MARKERS = {
    "SPRING":                     "o",
    "IBM_parser":                 "s",
    "HMG-AMR-T5":                 "^",
    "HMG-AMR-BART":               "D",
    "amr3joint_ontowiki2_seed44": "P",
    "ibm_amr3.0_neur_al":         "X",
}

# Top-1 Smatch scores (nowiki, beam3)
SMATCH_SCORES = {
    "SPRING":                     0.8449,
    "IBM_parser":                 0.8128,
    "HMG-AMR-T5":                 0.7693,
    "HMG-AMR-BART":               0.8358,
    "amr3joint_ontowiki2_seed44": 0.8374,
    "ibm_amr3.0_neur_al":         0.8115,
}

UE_METHODS = [
    "LexicalSimilarity",
    "EigValLaplacian_smatch",
    "EigValLaplacian_s2match",
    "EigValLaplacian_sema",
    "EigValLaplacian_wlk",
    "DegMat_smatch",
    "DegMat_s2match",
    "DegMat_sema",
    "DegMat_wlk",
    "Eccentricity_smatch",
    "Eccentricity_s2match",
    "Eccentricity_sema",
    "Eccentricity_wlk",
]

UE_DISPLAY = {
    "LexicalSimilarity":      "LexSim",
    "EigValLaplacian_smatch": "EigV-Smatch",
    "EigValLaplacian_s2match":"EigV-S²match",
    "EigValLaplacian_sema":   "EigV-SEMA",
    "EigValLaplacian_wlk":    "EigV-WLK",
    "DegMat_smatch":          "DegMat-Smatch",
    "DegMat_s2match":         "DegMat-S²match",
    "DegMat_sema":            "DegMat-SEMA",
    "DegMat_wlk":             "DegMat-WLK",
    "Eccentricity_smatch":    "Ecc-Smatch",
    "Eccentricity_s2match":   "Ecc-S²match",
    "Eccentricity_sema":      "Ecc-SEMA",
    "Eccentricity_wlk":       "Ecc-WLK",
}

# UE families: each family line = mean of its members
UE_FAMILIES = {
    "LexSim": ["LexicalSimilarity"],
    "EigVal": ["EigValLaplacian_smatch","EigValLaplacian_s2match",
               "EigValLaplacian_sema","EigValLaplacian_wlk"],
    "DegMat": ["DegMat_smatch","DegMat_s2match",
               "DegMat_sema","DegMat_wlk"],
    "Ecc":    ["Eccentricity_smatch","Eccentricity_s2match",
               "Eccentricity_sema","Eccentricity_wlk"],
}

FAMILY_STYLE = {
    "LexSim": dict(color="#1565C0", marker="o"),
    "EigVal": dict(color="#2E7D32", marker="s"),
    "DegMat": dict(color="#E65100", marker="^"),
    "Ecc":    dict(color="#6A1B9A", marker="D"),
}

QUALITIES      = ["S++", "S2", "SB"]
QUALITY_LABELS = {"S++": "Smatch++", "S2": "S²match", "SB": "SemBLEU"}
QUALITY_COLORS = {"S++": "#1565C0",  "S2": "#2E7D32", "SB": "#E65100"}


def load_beam(beam: int) -> dict:
    """Return {parser: {ue_method: {S++, S2, SB}}}."""
    data = {}
    for parser in BASELINE_PARSERS:
        csv_path = RESULTS_DIR / f"results_{parser.replace('/', '_')}_beam{beam}.csv"
        if not csv_path.exists():
            continue
        rows = {}
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                if row["UE_Method"] in UE_METHODS:
                    rows[row["UE_Method"]] = {
                        "S++": float(row["S++"]),
                        "S2":  float(row["S2"]),
                        "SB":  float(row["SB"]),
                    }
        data[parser] = rows
    return data


def family_mean(data, parser, family_methods, quality):
    vals = [data.get(parser, {}).get(m, {}).get(quality, float("nan"))
            for m in family_methods]
    vals = [v for v in vals if not np.isnan(v)]
    return np.mean(vals) if vals else float("nan")


def bold_ticks(ax, which="y"):
    """Make tick labels bold on an axes. which='y', 'x', or 'both'."""
    if which in ("y", "both"):
        for lbl in ax.get_yticklabels():
            lbl.set_fontweight("bold")
    if which in ("x", "both"):
        for lbl in ax.get_xticklabels():
            lbl.set_fontweight("bold")


def save(fig, name: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  → paper_figures/{name}.pdf")


# Global rcParams for paper quality
RCPARAMS = {
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "legend.framealpha": 0.9,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
}
