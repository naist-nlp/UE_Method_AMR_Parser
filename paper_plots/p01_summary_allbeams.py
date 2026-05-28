#!/usr/bin/env python3
"""Mean PRR per parser averaged over all beams and all UE methods (Figure 1)."""
import sys; sys.path.insert(0, "paper_plots")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from shared import *

FIG_SIZE     = (5.5, 3.2)
BAR_WIDTH    = 0.22
BAR_ANNOT_FS = 10
XTICK_ROT    = 15

plt.rcParams.update(RCPARAMS)

all_vals = {p: {q: [] for q in QUALITIES} for p in BASELINE_PARSERS}
for beam in BEAMS:
    data = load_beam(beam)
    for p in BASELINE_PARSERS:
        if p not in data: continue
        for q in QUALITIES:
            for m in UE_METHODS:
                v = data[p].get(m, {}).get(q, float("nan"))
                if not np.isnan(v):
                    all_vals[p][q].append(v)

parsers = [p for p in BASELINE_PARSERS if any(all_vals[p][q] for q in QUALITIES)]
x = np.arange(len(parsers))

fig, ax = plt.subplots(figsize=FIG_SIZE)

for qi, q in enumerate(QUALITIES):
    means = [np.mean(all_vals[p][q]) if all_vals[p][q] else float("nan") for p in parsers]
    offset = (qi - 1) * BAR_WIDTH
    bars = ax.bar(x + offset, means, BAR_WIDTH,
                  label=QUALITY_LABELS[q], color=QUALITY_COLORS[q],
                  alpha=0.88, edgecolor="white", linewidth=0.5)
    for bar, mv in zip(bars, means):
        if np.isnan(mv): continue
        ypos = mv + 0.004 if mv >= 0 else mv - 0.014
        ax.text(bar.get_x() + bar.get_width()/2, ypos, f"{mv:.3f}",
                ha="center", va="bottom" if mv >= 0 else "top",
                fontsize=BAR_ANNOT_FS, fontweight="bold", color="#333")

ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels([PARSER_DISPLAY[p] for p in parsers],
                   fontsize=10.5, fontweight="bold", rotation=XTICK_ROT, ha="right")
ax.tick_params(axis="y", labelsize=10)
ax.set_ylabel("Mean PRR (all UE methods, all beams)", fontsize=9, fontweight="bold")
ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.42, 1.1),
          handlelength=1.2, ncol=1)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

plt.draw()
for label in ax.get_yticklabels():
    label.set_fontweight("bold")

plt.tight_layout(pad=0.8)
save(fig, "summary_allbeams")
