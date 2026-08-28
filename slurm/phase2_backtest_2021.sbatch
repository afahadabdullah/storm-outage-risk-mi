#!/bin/bash
#SBATCH --job-name=storm-p2-backtest-2021
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

# Historical entry point retained to fail loudly. Under the current frozen
# split, 2021 is training data and cannot be presented as an out-of-sample test.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/phase2_env.sh
printf 'Refusing: 2021 is inside the frozen training window.\n' >&2
printf 'Use forward-year CV in phase2_cv_metrics.csv instead.\n' >&2
exit 2
