#!/usr/bin/env python3
"""13x13 Spearman correlation heatmap between UE methods (Figure 7)."""
import sys; sys.path.insert(0, "paper_plots")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import spearmanr
from shared import *

FIG_SIZE = (9, 8)
CELL_FS  = 7
TICK_FS  = 8.5
TITLE_FS = 10

plt.rcParams.update(RCPARAMS)
n = len(UE_METHODS)

for quality in QUALITIES:
    method_vals = {m: [] for m in UE_METHODS}
    for beam in BEAMS:
        data = load_beam(beam)
        for parser in BASELINE_PARSERS:
            if parser not in data: continue
            for m in UE_METHODS:
                method_vals[m].append(
                    data[parser].get(m, {}).get(quality, float("nan")))

    rho_mat = np.ones((n, n))
    for i, m1 in enumerate(UE_METHODS):
        for j, m2 in enumerate(UE_METHODS):
            if i == j: continue
            a = np.array(method_vals[m1])
            b = np.array(method_vals[m2])
            mask = ~(np.isnan(a) | np.isnan(b))
            if mask.sum() > 2:
                rho_mat[i, j], _ = spearmanr(a[mask], b[mask])
            else:
                rho_mat[i, j] = float("nan")

    labels = [UE_DISPLAY[m] for m in UE_METHODS]
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im   = ax.imshow(rho_mat, cmap="RdYlGn", norm=norm, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=TICK_FS, fontweight="bold")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=TICK_FS, fontweight="bold")

    ax.set_xticks(np.arange(n) - 0.5, minor=True)
    ax.set_yticks(np.arange(n) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(n):
        for j in range(n):
            v = rho_mat[i, j]
            if np.isnan(v): continue
            tc = "white" if abs(v) > 0.6 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=CELL_FS, color=tc, fontweight="bold")

    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Spearman rho", fontsize=9, fontweight="bold")
    cb.ax.tick_params(labelsize=8)
    for lbl in cb.ax.get_yticklabels():
        lbl.set_fontweight("bold")

    ax.set_title(
        f"UE Method Pairwise Correlation (Spearman rho)  --  {QUALITY_LABELS[quality]}\n"
        f"across {len(BASELINE_PARSERS)} parsers x {len(BEAMS)} beams",
        fontsize=TITLE_FS, fontweight="bold", pad=12)

    plt.tight_layout()
    save(fig, f"ue_corr_heatmap_{quality}")
