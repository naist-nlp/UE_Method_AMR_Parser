#!/usr/bin/env python3
"""parser_radar_beam10_lexSim.pdf — Radar chart, beam 10, all 4 families incl. LexSim."""
import sys; sys.path.insert(0, "paper_plots")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from shared import *

# ── TUNING ──────────────────────────────────────────────────────────────────
BEAM       = 10
FIG_SIZE   = (8, 8)
SPOKE_FS   = 13
LEGEND_FS  = 11
TITLE_FS   = 10
LINE_WIDTH = 2.0
FILL_ALPHA = 0.07
RLABEL_POS = 160
# ────────────────────────────────────────────────────────────────────────────

plt.rcParams.update(RCPARAMS)
beam_data = load_beam(BEAM)

spokes       = [(fam, q) for q in QUALITIES for fam in UE_FAMILIES]
spoke_labels = [f"{fam}\n({QUALITY_LABELS[q]})" for fam, q in spokes]
N            = len(spokes)
angles       = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]
spoke_angles = angles[:-1]

fig, ax = plt.subplots(figsize=FIG_SIZE, subplot_kw=dict(polar=True))

for parser in BASELINE_PARSERS:
    if parser not in beam_data: continue
    values = []
    for fam, q in spokes:
        vals = [beam_data[parser].get(m, {}).get(q, float("nan"))
                for m in UE_FAMILIES[fam]]
        vals = [v for v in vals if not np.isnan(v)]
        values.append(np.mean(vals) if vals else 0.0)
    values += values[:1]
    ax.plot(angles, values, linewidth=LINE_WIDTH,
            color=PARSER_COLORS[parser], label=PARSER_DISPLAY[parser])
    ax.fill(angles, values, alpha=FILL_ALPHA, color=PARSER_COLORS[parser])

ax.set_xticks(spoke_angles)
ax.set_xticklabels(spoke_labels, fontsize=SPOKE_FS, fontweight="bold")
ax.set_rlabel_position(RLABEL_POS)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
ax.tick_params(labelsize=SPOKE_FS-1)
ax.grid(True, alpha=0.3)
ax.set_title(f"Parser Reliability Profile  |  Beam {BEAM}\n"
             "(mean PRR: LexSim, EigVal, DegMat, Ecc × Smatch++, S²match, SemBLEU)",
             fontsize=TITLE_FS, fontweight="bold", pad=22)
ax.legend(loc="upper right", bbox_to_anchor=(1.45, 1.15), fontsize=LEGEND_FS)

plt.draw()
bold_ticks(ax, "y")
plt.tight_layout()
save(fig, "parser_radar_beam10_lexSim")
