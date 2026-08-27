#!/bin/bash
# Submit downloads, preprocessing, training and the application stages.
# The final test is deliberately NOT submitted: it is manual, once, after the
# validation review freezes every model and hyperparameter decision.
set -euo pipefail

mkdir -p logs/slurm

# Preflight in the foreground. Submitting a nine-job chain that will all fail
# identically on a missing lazy import is the most expensive kind of typo.
echo "== preflight =="
python src/doctor.py --phase 2
python -m pytest -q tests/
echo

# TIGER counties are fetched by BOTH the outage and canopy jobs. The write is
# atomic now, but making canopy wait on outages removes the duplicate download
# entirely and costs nothing -- canopy is not on the critical path.
outage_job=$(sbatch --parsable slurm/phase2_download_outages.sbatch)
era5_job=$(sbatch --parsable slurm/phase2_download_era5.sbatch)
static_job=$(sbatch --parsable --dependency="afterok:${outage_job}" \
  slurm/phase2_download_static.sbatch)

build_job=$(sbatch --parsable \
  --dependency="afterok:${outage_job}:${era5_job}:${static_job}" \
  slurm/phase2_build.sbatch)
train_job=$(sbatch --parsable --dependency="afterok:${build_job}" \
  slurm/phase2_train.sbatch)

# GEFS is only needed by the forecast stage, and only if section 8 is in scope
# for this pass. Set PHASE2_WITH_FORECAST=0 to skip both -- downloading 31
# members x 4 leads for bytes nothing reads is pure waste.
with_forecast="${PHASE2_WITH_FORECAST:-1}"
if [[ "${with_forecast}" == "1" ]]; then
  gefs_job=$(sbatch --parsable slurm/phase2_download_gefs.sbatch)
  apply_job=$(sbatch --parsable \
    --dependency="afterok:${train_job}:${gefs_job}" slurm/phase2_apply.sbatch)
else
  gefs_job="skipped"
  apply_job="skipped (PHASE2_WITH_FORECAST=0)"
fi

printf '\noutages=%s\nera5=%s\nstatic=%s\ngefs=%s\nbuild=%s\ntrain=%s\napply=%s\n' \
  "${outage_job}" "${era5_job}" "${static_job}" "${gefs_job}" \
  "${build_job}" "${train_job}" "${apply_job}"
printf '\nFinal test NOT submitted. Review these first:\n'
printf '  data/processed/phase2_validation_metrics.json\n'
printf '  data/processed/phase2_cv_metrics.csv        (all three CV schemes present?)\n'
printf '  data/processed/phase2_cox_ph_test.csv\n'
printf '  data/processed/phase2_coverage_exclusions.json\n'
printf '  figures/phase2_occurrence_validation_*.png\n'
printf '\nThen freeze decisions in git and run:\n'
printf '  sbatch slurm/phase2_final_test.sbatch\n'
