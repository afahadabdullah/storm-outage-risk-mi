#!/bin/bash
# Submit downloads, preprocessing, training and application one stage at a
# time. Each sbatch call blocks until that stage succeeds, so a failure cannot
# leave a screenful of permanently pending dependency jobs.
# The final test is deliberately NOT submitted: it is manual, once, after the
# validation review freezes every model and hyperparameter decision.
set -euo pipefail

mkdir -p logs/slurm
source slurm/phase2_env.sh

# Preflight with the exact interpreter the compute jobs will use.
echo "== preflight =="
"${PHASE2_PYTHON}" src/doctor.py --phase 2
"${PHASE2_PYTHON}" -m pytest -q tests/
"${PHASE2_PYTHON}" -m ruff check src tests
echo

active_jobs=$(squeue --noheader --user="${USER}" --format='%A %j %T %R' | \
  awk '$2 ~ /^storm-p2-/')
if [[ -n "${active_jobs}" ]]; then
  printf 'Existing Phase 2 jobs must be resolved before a sequential run:\n%s\n' \
    "${active_jobs}" >&2
  exit 2
fi

run_stage() {
  local label="$1"
  local script="$2"
  printf '\n== %s ==\n' "${label}"
  if sbatch --wait --export=ALL "${script}"; then
    printf 'completed: %s\n' "${label}"
  else
    local status=$?
    printf 'FAILED: %s (sbatch exit %d). No later stage was submitted.\n' \
      "${label}" "${status}" >&2
    exit "${status}"
  fi
}

run_stage "outage download" slurm/phase2_download_outages.sbatch
run_stage "ERA5 monthly download" slurm/phase2_download_era5.sbatch
run_stage "canopy download" slurm/phase2_download_static.sbatch
run_stage "2018-2022 build" slurm/phase2_build.sbatch
run_stage "training and validation" slurm/phase2_train.sbatch

# GEFS is only needed by the forecast stage, and only if section 8 is in scope
# for this pass. Set PHASE2_WITH_FORECAST=0 to skip both -- downloading 31
# members x 4 leads for bytes nothing reads is pure waste.
with_forecast="${PHASE2_WITH_FORECAST:-1}"
if [[ "${with_forecast}" == "1" ]]; then
  run_stage "GEFS download" slurm/phase2_download_gefs.sbatch
  run_stage "application stages" slurm/phase2_apply.sbatch
else
  printf '\nGEFS and application stages skipped (PHASE2_WITH_FORECAST=0).\n'
fi

printf '\nSequential Phase 2 pipeline completed.\n'
printf 'Final test NOT submitted. Review these first:\n'
printf '  data/processed/phase2_validation_metrics.json\n'
printf '  data/processed/phase2_cv_metrics.csv        (all three CV schemes present?)\n'
printf '  data/processed/phase2_cox_ph_test.csv\n'
printf '  data/processed/phase2_coverage_exclusions.json\n'
printf '  figures/phase2_occurrence_validation_*.png\n'
printf '\nThen freeze decisions in git and run:\n'
printf '  sbatch slurm/phase2_final_test.sbatch\n'
