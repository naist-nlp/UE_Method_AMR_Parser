#!/usr/bin/env python3
"""1x6 row: PRR vs beam size per parser, averaged UE families (Figure 9)."""
import sys; sys.path.insert(0, "paper_plots")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shared import *

FIG_SIZE       = (14, 3.8)
PANEL_TITLE_FS = 9
AXIS_LABEL_FS  = 8.5
TICK_FS        = 8
LEGEND_FS      = 8
LINE_WIDTH     = 1.8
MARKER_SIZE    = 5

plt.rcParams.update(RCPARAMS)

for quality in QUALITIES:
    fig, axes = plt.subplots(1, 6, figsize=FIG_SIZE, sharey=True, sharex=True)
    for ax, parser in zip(axes, BASELINE_PARSERS):
        for fam, methods in UE_FAMILIES.items():
            beam_vals = []
            for beam in BEAMS:
                data = load_beam(beam)
                vals = [data.get(parser, {}).get(m, {}).get(quality, float("nan"))
                        for m in methods]
                vals = [v for v in vals if not np.isnan(v)]
                beam_vals.append(np.mean(vals) if vals else float("nan"))
            ax.plot(BEAMS, beam_vals,
                    color=FAMILY_STYLE[fam]["color"],
                    marker=FAMILY_STYLE[fam]["marker"],
                    linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label=fam)

        ax.axhline(0, color="gray", linewidth=0.7, linestyle="--", alpha=0.6, label="Random")
        ax.set_xticks(BEAMS)
        ax.set_title(PARSER_DISPLAY[parser], fontsize=PANEL_TITLE_FS, fontweight="bold")
        ax.tick_params(labelsize=TICK_FS)
        ax.set_xlabel("Beam $k$", fontsize=AXIS_LABEL_FS, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(f"PRR ({QUALITY_LABELS[quality]})", fontsize=AXIS_LABEL_FS, fontweight="bold")

    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center",
               ncol=len(UE_FAMILIES)+1,
               bbox_to_anchor=(0.5, -0.08), fontsize=LEGEND_FS, framealpha=0.9)
    fig.suptitle(f"PRR vs Beam Size  --  {QUALITY_LABELS[quality]}",
                 fontsize=10, fontweight="bold", y=1.02)
    plt.draw()
    for ax in axes:
        bold_ticks(ax, "both")
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    save(fig, f"prr_vs_beam_{quality}")
