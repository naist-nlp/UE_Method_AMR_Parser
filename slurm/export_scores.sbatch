#!/bin/bash
# Authors only, step 2 of the score export: needs the AMR 3.0 k-best data.
# Run on a node where check_tie_order.sbatch reported 24/24, from the
# repository root with the environment activated:
#   sbatch -w <node> slurm/export_scores.sbatch [--skip_top1]
#   DATA_DIR=/path/to/k_top_AMR_parsers sbatch -w <node> slurm/export_scores.sbatch
# Writes numbers only into estimators/results/cache/ (UE scores, sort orders,
# top-1 Smatch). The k-best data is only read, never copied.
# Partition and account are those of our cluster; adjust them to yours.
#SBATCH --job-name=ue_export
#SBATCH -p cpu_long
#SBATCH --account=is-nlp
#SBATCH -c 2
#SBATCH -t 4-04:00:00
#SBATCH --output=logs/export_%j_%N.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME, start $(date)"
python3 scripts/export_scores.py --data_dir "${DATA_DIR:-data/k_top_AMR_parsers}" "$@"
echo "End $(date)"
