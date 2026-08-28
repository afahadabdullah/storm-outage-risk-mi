#!/bin/bash
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
source slurm/phase2_env.sh
"${PHASE2_PYTHON}" src/phase2_build.py --through test --acknowledge-test
"${PHASE2_PYTHON}" src/phase2_train.py --evaluate-test

# With the test year open, redraw the section 8.4 money plot -- it can now carry
# the observed overlay -- and recompute the per-lead cost-loss curves.
"${PHASE2_PYTHON}" src/phase2_forecast.py
"${PHASE2_PYTHON}" src/phase2_value.py

# Finally, the publication pack: the headline metric matrix and the five
# figures, now carrying the test column and the observed truth overlay.
"${PHASE2_PYTHON}" src/phase2_report.py
