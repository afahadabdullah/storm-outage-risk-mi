#!/bin/bash
#SBATCH --job-name=storm-p2-apply
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

# Spec sections 6.4, 8 and 9: composition, forecast application, decision value.
# None of these read the test year, so this runs against the frozen bundle while
# the held-out year is still sealed. Memory is higher than the training job
# because the Monte Carlo holds (county-days x n_mc_draws) floats.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/phase2_env.sh
"${PHASE2_PYTHON}" src/phase2_compose.py
"${PHASE2_PYTHON}" src/phase2_forecast.py
"${PHASE2_PYTHON}" src/phase2_value.py
