#!/bin/bash
#SBATCH --job-name=storm-p2-era5
#SBATCH --array=0-71%2
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/phase2_env.sh
year=$((2018 + SLURM_ARRAY_TASK_ID / 12))
month=$((1 + SLURM_ARRAY_TASK_ID % 12))
"${PHASE2_PYTHON}" src/phase2_download.py --only era5 \
  --years "${year}" --months "${month}"
