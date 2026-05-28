#!/usr/bin/env python3
"""PRR vs beam size, single panel, averaged over all UE methods and quality metrics (Figure 2)."""
import sys; sys.path.insert(0, "paper_plots")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shared import *

FIG_SIZE    = (5.5, 3.5)
LINE_WIDTH  = 2.0
MARKER_SIZE = 6

plt.rcParams.update(RCPARAMS)
fig, ax = plt.subplots(figsize=FIG_SIZE)

for parser in BASELINE_PARSERS:
    beam_means = []
    for beam in BEAMS:
        data = load_beam(beam)
        if parser not in data:
            beam_means.append(float("nan")); continue
        vals = [data[parser].get(m, {}).get(q, float("nan"))
                for m in UE_METHODS for q in QUALITIES]
        vals = [v for v in vals if not np.isnan(v)]
        beam_means.append(np.mean(vals) if vals else float("nan"))
    ax.plot(BEAMS, beam_means,
            marker=PARSER_MARKERS[parser], color=PARSER_COLORS[parser],
            linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
            label=PARSER_DISPLAY[parser])

ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
ax.set_xticks(BEAMS)
ax.set_xlabel("Beam size $k$", fontsize=10, fontweight="bold")
ax.set_ylabel("Mean PRR (UE methods x quality metrics)", fontsize=9, fontweight="bold")
ax.tick_params(labelsize=10)
ax.legend(fontsize=7, loc="upper right", framealpha=0.7)

plt.draw()
bold_ticks(ax, "both")
plt.tight_layout()
save(fig, "prr_vs_beam_avg_quality")
