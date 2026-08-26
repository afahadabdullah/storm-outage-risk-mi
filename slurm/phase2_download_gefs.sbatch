#!/bin/bash -l
#SBATCH --job-name=storm-p2-gefs
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONNOUSERSITE=1
python src/phase2_download.py --only gefs
