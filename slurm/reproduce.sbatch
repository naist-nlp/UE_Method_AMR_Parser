#!/bin/bash
# Reproduce every number and figure of the paper from the released score
# files (no AMR data needed, any CPU node, a few minutes). Submit from the
# repository root with the environment activated; Slurm passes it on:
#   conda activate ue_amr && sbatch slurm/reproduce.sbatch
# Partition and account are those of our cluster; adjust them to yours.
#SBATCH --job-name=ue_reproduce
#SBATCH -p cpu_short
#SBATCH --account=is-nlp
#SBATCH -c 2
#SBATCH -t 4:00:00
#SBATCH --output=logs/reproduce_%j_%N.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME, start $(date)"
python3 run_all.py --n_samples 0
python3 scripts/paper_tables.py
python3 build_results_json.py
MPLBACKEND=Agg python3 paper_plots/run_all.py
python3 scripts/check_against_paper.py
echo "End $(date)"
