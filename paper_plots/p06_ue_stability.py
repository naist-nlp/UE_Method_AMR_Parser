#!/usr/bin/env python3
"""ue_stability.pdf — Heatmap: UE methods × beam sizes, mean PRR over parsers."""
import sys; sys.path.insert(0, "paper_plots")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from shared import *

# ── TUNING ──────────────────────────────────────────────────────────────────
FIG_SIZE   = (13, 5.5)
CELL_FS    = 7.5    # font size inside cells
YLABEL_FS  = 8.5    # y-axis (UE method) label size
XLABEL_FS  = 8.5    # x-axis (beam) label size
TITLE_FS   = 10
# ────────────────────────────────────────────────────────────────────────────

plt.rcParams.update(RCPARAMS)
fig, axes = plt.subplots(1, 3, figsize=FIG_SIZE)

for ax, quality in zip(axes, QUALITIES):
    mat = np.full((len(UE_METHODS), len(BEAMS)), float("nan"))
    for bi, beam in enumerate(BEAMS):
        data = load_beam(beam)
        for mi, m in enumerate(UE_METHODS):
            vals = [data[p].get(m, {}).get(quality, float("nan"))
                    for p in BASELINE_PARSERS if p in data]
            vals = [v for v in vals if not np.isnan(v)]
            if vals:
                mat[mi, bi] = np.mean(vals)

    vabs = np.nanmax(np.abs(mat))
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
    im   = ax.imshow(mat, cmap="RdYlGn", norm=norm, aspect="auto")

    ax.set_xticks(range(len(BEAMS)))
    ax.set_xticklabels([f"k={b}" for b in BEAMS], fontsize=XLABEL_FS, fontweight="bold")
    ax.set_yticks(range(len(UE_METHODS)))
    ax.set_yticklabels([UE_DISPLAY[m] for m in UE_METHODS], fontsize=YLABEL_FS, fontweight="bold")

    # white cell borders
    ax.set_xticks(np.arange(len(BEAMS)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(UE_METHODS)) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(len(UE_METHODS)):
        for j in range(len(BEAMS)):
            v = mat[i, j]
            if np.isnan(v): continue
            tc = "white" if abs(norm(v) - 0.5) > 0.28 else "black"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=CELL_FS, color=tc, fontweight="bold")

    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Mean PRR", fontsize=9, fontweight="bold")
    cb.ax.tick_params(labelsize=8)
    for lbl in cb.ax.get_yticklabels():
        lbl.set_fontweight("bold")
    ax.set_title(QUALITY_LABELS[quality], fontsize=TITLE_FS, fontweight="bold")

fig.suptitle("UE Method Stability Across Beam Sizes  (mean PRR over 6 parsers)",
             fontsize=10, fontweight="bold", y=1.02)
plt.tight_layout()
save(fig, "ue_stability")
