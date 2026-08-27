#!/bin/bash
#SBATCH --job-name=storm-p2-train
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=18:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/phase2_env.sh
"${PHASE2_PYTHON}" src/phase2_train.py
