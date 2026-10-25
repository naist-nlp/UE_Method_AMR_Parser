#!/bin/bash
# Authors only, step 1 of the score export (no AMR data needed, ~10 min).
# Does np.argsort on this node reproduce the paper's PRR values exactly?
# Submit from the repository root with the environment activated, to a few
# node types, and keep a node whose last line says 24/24, e.g.
#   for n in <node1> <node2> <node3>; do sbatch -w $n slurm/check_tie_order.sbatch; done
# Partition and account are those of our cluster; adjust them to yours.
#SBATCH --job-name=ue_tiecheck
#SBATCH -p cpu_short
#SBATCH --account=is-nlp
#SBATCH -c 2
#SBATCH -t 4:00:00
#SBATCH --output=logs/tiecheck_%j_%N.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME, start $(date)"
python3 scripts/export_scores.py --check_only
echo "End $(date)"
