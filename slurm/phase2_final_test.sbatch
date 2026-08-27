#!/bin/bash -l
#SBATCH --job-name=storm-p2-final-test
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

# The one-time test-year score. This builds phase2_merged_test.parquet -- a
# SEPARATE file from the validation-scope table, so a refit remains possible
# afterwards without repeating the 72-file ERA5 aggregation.
#
# phase2_train --evaluate-test refuses to run if the reporting-county cohort in
# the test build differs from the cohort frozen into the model bundle. That is
# deliberate: scoring a frozen model against a different study population is not
# a held-out test. If it stops here, read the message before reaching for
# --force-test.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONNOUSERSITE=1
python src/phase2_build.py --through test --acknowledge-test
python src/phase2_train.py --evaluate-test

# With the test year open, redraw the section 8.4 money plot -- it can now carry
# the observed overlay -- and recompute the per-lead cost-loss curves.
python src/phase2_forecast.py
python src/phase2_value.py
