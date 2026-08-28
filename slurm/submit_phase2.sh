#!/bin/bash
# Submit downloads, preprocessing, training and application one stage at a
# time. The controller watches each job to keep terminal feedback useful, then
# confirms Slurm accounting reports success before submitting the next stage.
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
  local submission job_id queue_status status accounting state exit_code
  local attempt
  printf '\n== %s ==\n' "${label}"
  submission=$(sbatch --parsable --export=ALL "${script}")
  job_id="${submission%%;*}"
  if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
    printf 'FAILED: could not read a Slurm job ID for %s.\n' "${label}" >&2
    printf '  sbatch returned: %s\n' "${submission}" >&2
    exit 1
  fi
  printf 'submitted: %s\n' "${job_id}"
  printf 'logs: logs/slurm/*-%s*.out  (tail -f that file)\n' "${job_id}"

  while :; do
    queue_status=$(squeue --noheader --jobs="${job_id}" \
      --format='%T | elapsed %M | limit %l | %R' 2>/dev/null || true)
    [[ -z "${queue_status}" ]] && break
    while IFS= read -r status; do
      printf '[%s] job %s: %s\n' "$(date '+%H:%M:%S')" "${job_id}" "${status}"
    done <<< "${queue_status}"
    sleep 30
  done

  # A completed job can disappear from squeue a few seconds before sacct sees
  # it. Do not guess that this means success: fail closed if accounting cannot
  # confirm a clean completion, so no later stage is submitted incorrectly.
  accounting=""
  for attempt in {1..12}; do
    accounting=$(sacct --array --noheader --parsable2 --jobs="${job_id}" \
      --format=JobID,State,ExitCode 2>/dev/null | \
      awk -F'|' -v id="${job_id}" '
        $1 == id || index($1, id "_") == 1 {
          found = 1
          if ($2 != "COMPLETED" || $3 != "0:0") {
            failure = $2 "|" $3
          }
        }
        END {
          if (found) {
            print failure ? failure : "COMPLETED|0:0"
          }
        }')
    [[ -n "${accounting}" ]] && break
    sleep 5
  done
  if [[ -z "${accounting}" ]]; then
    printf 'FAILED: job %s left the queue but sacct did not report its result.\n' \
      "${job_id}" >&2
    printf 'No later stage was submitted; inspect logs/slurm/*-%s*.{out,err}.\n' \
      "${job_id}" >&2
    exit 1
  fi

  state="${accounting%%|*}"
  exit_code="${accounting#*|}"
  if [[ "${state}" == "COMPLETED" && "${exit_code}" == "0:0" ]]; then
    printf 'completed: %s\n' "${label}"
  else
    printf 'FAILED: %s (Slurm state %s, exit %s). No later stage was submitted.\n' \
      "${label}" "${state}" "${exit_code}" >&2
    printf 'logs: logs/slurm/*-%s*.{out,err}\n' "${job_id}" >&2
    exit 1
  fi
}

run_stage "outage download" slurm/phase2_download_outages.sbatch
run_stage "ERA5 regional ARCO cache" slurm/phase2_download_arco.sbatch
run_stage "ERA5 residual monthly downloads" slurm/phase2_download_era5.sbatch
run_stage "canopy download" slurm/phase2_download_static.sbatch
run_stage "configured training/validation build" slurm/phase2_build.sbatch
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
printf '  data/processed/phase2_results_matrix.csv\n'
printf '  data/processed/phase2_gefs_case_matrix.csv\n'
printf '  data/processed/phase2_county_skill.csv\n'
printf '  figures/phase2_skill_summary.{png,pdf}\n'
printf '  figures/phase2_county_diagnostics.{png,pdf}\n'
printf '  figures/phase2_gefs_case_studies.{png,pdf}\n'
printf '  docs/phase2_technical_memo.html\n'
printf '  docs/phase2_technical_memo.md\n'
printf '\nThen freeze decisions in git and run:\n'
printf '  sbatch slurm/phase2_final_test.sbatch\n'
