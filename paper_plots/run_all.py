#!/usr/bin/env python3
"""Run all paper figure scripts. Output → paper_figures/"""
import subprocess, sys
from pathlib import Path

scripts = [
    "paper_plots/p01_summary_allbeams.py",
    "paper_plots/p02_prr_vs_beam_avg.py",
    "paper_plots/p03_radar.py",
    "paper_plots/p04_radar_lexsim.py",
    "paper_plots/p05_prr_vs_smatch.py",
    "paper_plots/p06_ue_stability.py",
    "paper_plots/p07_ue_corr_heatmap.py",
    "paper_plots/p08_prr_vs_beam_grid.py",
    "paper_plots/p09_prr_vs_beam_quality.py",
]

for s in scripts:
    print(f"\n{'='*50}\n  {Path(s).name}\n{'='*50}")
    r = subprocess.run([sys.executable, s])
    if r.returncode != 0:
        print(f"  [FAILED] {s}")
