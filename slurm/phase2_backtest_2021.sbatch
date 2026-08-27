#!/bin/bash
#SBATCH --job-name=storm-p2-backtest-2021
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

# Exploratory retrospective score only.  This intentionally does not invoke
# phase2_train --evaluate-test, so the configured final-test year stays sealed.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/phase2_env.sh
"${PHASE2_PYTHON}" src/phase2_build.py --through backtest --backtest-year 2021
"${PHASE2_PYTHON}" src/phase2_backtest.py --year 2021
