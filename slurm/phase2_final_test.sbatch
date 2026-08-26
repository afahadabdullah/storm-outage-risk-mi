#!/bin/bash -l
#SBATCH --job-name=storm-p2-final-test
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONNOUSERSITE=1
python src/phase2_build.py --through test --acknowledge-test
python src/phase2_train.py --evaluate-test
