# Phase 2 runbook

Phase 2 is CPU-only. The workload is tabular boosting, survival analysis,
rolling statistics, NetCDF I/O, and sparse county aggregation. GPU allocation
does not materially shorten the critical path.

## Frozen design

- Train: 2018-01-01 through 2021-12-31.
- Calibration/validation: 2022 only.
- Final test: 2023 only, opened once after all choices are frozen.
- Outage baseline: rolling 30-day 10th percentile. December 2017 is downloaded
  only as lookback context and is never a model row.
- Occurrence: LightGBM plus isotonic calibration fitted on 2022.
- Magnitude: NGBoost Normal density in `log1p(customer_hours)` space, with
  LightGBM quantiles as an automatic fallback.
- Duration: Weibull AFT primary and Cox PH secondary. The Cox Schoenfeld test is
  written to `data/processed/phase2_cox_ph_test.csv`.
- Validation: storm-blocked, leave-one-county-out, and forward-year occurrence
  CV; proper probabilistic scores for the frozen 2022 validation year.

## Cluster preflight

Activate the project environment before submitting. Slurm exports the current
environment by default on the target cluster.

```bash
cd /panfs/ccds02/nobackup/people/afahad/project/storm-outage-risk-mi
git pull --ff-only origin main
conda activate /panfs/ccds02/nobackup/people/afahad/envs/storm-outage-risk.
make doctor
make test
mkdir -p logs/slurm
```

The CDS token in `~/.cdsapirc` must work from compute nodes. Data live under
`data/raw`, so the repository should remain on project or scratch storage.

## Submit download, build, and training

```bash
make phase2-submit
```

This submits annual EAGLE-I downloads, a throttled 72-task ERA5 monthly array,
official 2021 NLCD tree-canopy county statistics, and 2023 GEFS case-study
downloads. The 2018–2022 feature build waits for its required downloads, and
CPU model training waits for the build. GEFS is independent of hindcast
training, so a GEFS failure does not block the ERA5 model fit.

Monitor with:

```bash
squeue -u "$USER"
tail -f logs/slurm/storm-p2-train-*.out
```

Every downloader is cache-aware. A failed monthly task can be restarted:

```bash
python src/phase2_download.py --only era5 --years 2020 --months 7
```

## Validation review and final test

Review these before opening 2023:

```text
data/processed/phase2_validation_metrics.json
data/processed/phase2_cv_metrics.csv
data/processed/phase2_cox_ph_test.csv
figures/phase2_occurrence_validation_2022.png
```

Freeze model and hyperparameter decisions in Git. Only then submit:

```bash
sbatch slurm/phase2_final_test.sbatch
```

That job builds the full table, loads the frozen model without refitting,
scores 2023, and writes `models/TEST_YEAR_OPENED.txt`. A normal second attempt
is refused. `--force-test` exists only to recover an interrupted run.

## Direct commands without Slurm

```bash
make phase2-download-outages
make phase2-download-era5
make phase2-download-canopy
make phase2-download-gefs
make phase2                 # build through 2022, train, calibrate, validate
make phase2-build-test      # only after decisions are frozen
make phase2-test            # one-time 2023 score
```
