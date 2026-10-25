#!/usr/bin/env python3
"""prr_vs_smatch.pdf — Scatter: mean PRR vs parser Smatch accuracy, 3 panels."""
import sys; sys.path.insert(0, "paper_plots")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from shared import *

# ── TUNING ──────────────────────────────────────────────────────────────────
FIG_SIZE    = (11, 3.8)
MARKER_BASE = 50    # base scatter dot size
MARKER_STEP = 18    # extra size per beam index (shows beam via size)
LINE_WIDTH  = 1.5
# ────────────────────────────────────────────────────────────────────────────

plt.rcParams.update(RCPARAMS)
fig, axes = plt.subplots(1, 3, figsize=FIG_SIZE)

for ax, quality in zip(axes, QUALITIES):
    all_x, all_y = [], []

    for bi, beam in enumerate(BEAMS):
        data = load_beam(beam)
        for parser in BASELINE_PARSERS:
            if parser not in data or parser not in SMATCH_SCORES: continue
            vals = [data[parser].get(m, {}).get(quality, float("nan")) for m in UE_METHODS]
            vals = [v for v in vals if not np.isnan(v)]
            if not vals: continue
            x = SMATCH_SCORES[parser]
            y = np.mean(vals)
            all_x.append(x); all_y.append(y)
            ax.scatter(x, y,
                       color=PARSER_COLORS[parser],
                       marker=PARSER_MARKERS[parser],
                       s=MARKER_BASE + bi * MARKER_STEP,
                       alpha=0.80, zorder=3,
                       label=PARSER_DISPLAY[parser] if bi == 0 else "")

    # regression + Spearman 
    if len(all_x) > 2:
        m_coef, b = np.polyfit(all_x, all_y, 1)
        xr = np.linspace(min(all_x)-0.005, max(all_x)+0.005, 100)
        ax.plot(xr, m_coef*xr + b, "k--", linewidth=LINE_WIDTH, alpha=0.6)
        rho, pval = spearmanr(all_x, all_y)
        sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
        ax.text(0.05, 0.95, f"ρ = {rho:.3f}{sig}",
                transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85))

    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_xlabel("Top-1 Smatch (parser accuracy)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean PRR (across UE methods)", fontsize=10, fontweight="bold")
    ax.set_title(QUALITY_LABELS[quality], fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=10)

# Parser legend on left panel
handles, lbls = axes[0].get_legend_handles_labels()
# Beam size size legend
from matplotlib.lines import Line2D
beam_handles = [Line2D([0],[0], marker="o", color="gray",
                        markersize=np.sqrt(MARKER_BASE + i*MARKER_STEP),
                        linestyle="None", label=f"k={b}")
                for i, b in enumerate(BEAMS)]
fig.legend(handles + beam_handles,
           lbls + [f"k={b}" for b in BEAMS],
           loc="lower center", ncol=5,
           bbox_to_anchor=(0.5, -0.16), fontsize=8, framealpha=0.9)
fig.suptitle("PRR vs Parser Accuracy (Smatch)  —  Does better accuracy mean better UE?",
             fontsize=10, fontweight="bold", y=1.02)

plt.draw()
for ax in axes:
    bold_ticks(ax, "both")
plt.tight_layout()
save(fig, "prr_vs_smatch")
