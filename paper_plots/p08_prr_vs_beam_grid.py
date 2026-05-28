#!/usr/bin/env python3
"""2x3 grid: PRR vs beam size per parser, averaged UE families (Figure 8)."""
import sys; sys.path.insert(0, "paper_plots")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shared import *

FIG_SIZE       = (11, 6.0)
PANEL_TITLE_FS = 10
AXIS_LABEL_FS  = 9
TICK_FS        = 8
LEGEND_FS      = 9
LINE_WIDTH     = 2.0
MARKER_SIZE    = 6

PARSERS_GRID = [
    ["SPRING",     "HMG-AMR-T5",                "HMG-AMR-BART"],
    ["IBM_parser", "amr3joint_ontowiki2_seed44", "ibm_amr3.0_neur_al"],
]

plt.rcParams.update(RCPARAMS)

for quality in QUALITIES:
    fig, axes = plt.subplots(2, 3, figsize=FIG_SIZE, sharey=True, sharex=True)
    for ri, row_parsers in enumerate(PARSERS_GRID):
        for ci, parser in enumerate(row_parsers):
            ax = axes[ri][ci]
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

            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6, label="Random")
            ax.set_xticks(BEAMS)
            ax.set_title(PARSER_DISPLAY[parser], fontsize=PANEL_TITLE_FS, fontweight="bold")
            ax.tick_params(labelsize=TICK_FS)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if ri == 1:
                ax.set_xlabel("Beam size $k$", fontsize=AXIS_LABEL_FS, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"PRR ({QUALITY_LABELS[quality]})", fontsize=AXIS_LABEL_FS, fontweight="bold")

    handles, lbls = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center",
               ncol=len(UE_FAMILIES)+1,
               bbox_to_anchor=(0.5, -0.04), fontsize=LEGEND_FS, framealpha=0.9)
    fig.suptitle(f"PRR vs Beam Size  --  {QUALITY_LABELS[quality]}",
                 fontsize=10, fontweight="bold", y=1.01)
    plt.draw()
    for row in axes:
        for ax in row:
            bold_ticks(ax, "both")
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    safe = quality.replace("+", "p")
    save(fig, f"prr_vs_beam_grid_{safe}")
