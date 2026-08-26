#!/bin/bash
# Submit downloads, preprocessing, and CPU-only training. Final test is manual.
set -euo pipefail

mkdir -p logs/slurm
outage_job=$(sbatch --parsable slurm/phase2_download_outages.sbatch)
era5_job=$(sbatch --parsable slurm/phase2_download_era5.sbatch)
static_job=$(sbatch --parsable slurm/phase2_download_static.sbatch)
gefs_job=$(sbatch --parsable slurm/phase2_download_gefs.sbatch)
build_job=$(sbatch --parsable \
  --dependency="afterok:${outage_job}:${era5_job}:${static_job}" \
  slurm/phase2_build.sbatch)
train_job=$(sbatch --parsable --dependency="afterok:${build_job}" \
  slurm/phase2_train.sbatch)

printf 'outages=%s\nera5=%s\nstatic=%s\ngefs=%s\nbuild=%s\ntrain=%s\n' \
  "${outage_job}" "${era5_job}" "${static_job}" "${gefs_job}" \
  "${build_job}" "${train_job}"
printf 'Final test not submitted. Review validation and freeze decisions, then run:\n'
printf '  sbatch slurm/phase2_final_test.sbatch\n'
