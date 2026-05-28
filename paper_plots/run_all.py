#!/usr/bin/env python3
"""Run all paper figure scripts in order. Must be run from the project root."""
import subprocess
import sys
from pathlib import Path

SCRIPTS = [
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

for script in SCRIPTS:
    print(f"\nRunning {script} ...")
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"  ERROR: {script} failed (exit code {result.returncode})")
    else:
        print(f"  OK")

print("\nAll figures saved to paper_figures/")
