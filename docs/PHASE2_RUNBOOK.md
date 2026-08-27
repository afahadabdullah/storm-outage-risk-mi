# Phase 2 runbook

Phase 2 is CPU-only. The workload is tabular boosting, survival analysis,
rolling statistics, NetCDF I/O, and sparse county aggregation. GPU allocation
does not materially shorten the critical path.

Measured on a 151k-row table with 26 features: one LightGBM fit is ~1.6 s, so
83-fold leave-one-county-out CV is ~2 minutes and the rolling 30-day baseline
over 3.7M county-hours is under a minute. The 18-hour Slurm walls are generous
by design — a job approaching one is a job that is wrong, not slow.

## Frozen design

- Train: 2018-01-01 through 2021-12-31.
- Calibration/validation: 2022 only.
- Final test: 2023 only, opened once after all choices are frozen.
- Outage baseline: rolling 30-day 10th percentile. December 2017 is downloaded
  only as lookback context and is never a model row.
- Reporting cohort: fixed by the **train + validation** years. A county whose
  coverage changes in the test year is reported, never silently dropped — the
  cohort is stored in the model bundle and re-checked when 2023 is opened.
- Occurrence: LightGBM plus isotonic calibration fitted on 2022. Reported
  validation metrics use **out-of-fold** calibrated probabilities within 2022;
  the in-sample number is kept beside them so the optimism stays visible.
- Magnitude: NGBoost Normal density in `log1p(customer_hours)` space, with
  LightGBM quantiles as an automatic fallback. Which one actually ran is
  recorded in the metrics as `magnitude_model_actually_used`.
- Duration: Weibull AFT primary and Cox PH secondary. The Cox Schoenfeld test is
  written to `data/processed/phase2_cox_ph_test.csv`.
- Reference models (spec §6.1, §6.2): logistic GLM and negative-binomial GLM on
  a reduced feature set, as skill floors.
- Baselines (spec §7.3): county-specific climatology, county-specific
  persistence, and the gust > 20 m/s threshold rule.
- Validation: storm-blocked, leave-one-county-out, and forward-year occurrence
  CV; proper probabilistic scores for the frozen 2022 validation year.

### Storm blocking

A storm day is one where at least `storm_min_county_frac` (default 0.10) of the
reporting counties are in event. Defining it as "any county anywhere in event"
saturates at 83 counties, fuses the whole record into one CV group and makes
`GroupKFold` unconstructible. The training log prints how many days clear the
threshold and how many episodes resulted — **read those two lines.** If almost
no days clear it, or almost all do, adjust `storm_min_county_frac` in
`config/phase2.yaml` before trusting any storm-blocked number.

## Cluster preflight

The Phase 2 Makefile and every Slurm job use this interpreter directly:

```text
/panfs/ccds02/nobackup/people/afahad/envs/storm-outage-risk./bin/python
```

The batch scripts deliberately do not use login shells: a login shell reset the
activated Conda environment and caused every job to fail at `import pandas`.

```bash
cd /panfs/ccds02/nobackup/people/afahad/project/storm-outage-risk-mi
git pull --ff-only origin main
conda activate /panfs/ccds02/nobackup/people/afahad/envs/storm-outage-risk.
make phase2-preflight        # doctor --phase 2, lint, tests
mkdir -p logs/slurm
```

`make doctor-phase2` promotes the packages Phase 2 imports lazily —
`properscoring`, `ngboost`, `statsmodels`, `cfgrib`, `cdsapi`, `boto3` — from
optional to required. They are imported *after* the model is fitted, so a node
missing one used to burn the whole training job. `phase2_train.py` also
re-checks them in its first second.

Confirm the CDS token works **from a compute node**, not just the login node:

```bash
srun --pty -t 00:10:00 --mem=4G bash -lc \
  '/panfs/ccds02/nobackup/people/afahad/envs/storm-outage-risk./bin/python -c "import cdsapi, ngboost, lifelines, properscoring, statsmodels; print(\"ok\")"'
```

Data live under `data/raw`, so the repository should remain on project or
scratch storage.

## Submit download, build, train and apply

```bash
make phase2-submit
```

The controller runs preflight, submits one stage at a time, prints its job ID,
log location, queue state, and elapsed time every 30 seconds, confirms its
success through Slurm accounting, and does not submit the next stage until the
current one succeeds. The order is annual EAGLE-I outages, one regional ARCO
cache, the residual ERA5 monthly array, 2021 NLCD tree-canopy statistics, the
2018–2022 build, training/validation, GEFS, and the application stages. The
ERA5 array permits five monthly tasks at once; no other Phase 2 stage overlaps
it. A failed stage stops the controller immediately.

Because the controller remains attached while waiting, run it in a persistent
`tmux`, `screen`, or JupyterHub terminal. If that terminal is terminated, the
already-submitted job continues but the next stage is not submitted.

If the §8 forecast work is out of scope for this pass, skip the GEFS download
and the apply job entirely — downloading 31 members × 4 leads for bytes nothing
reads is pure waste:

```bash
PHASE2_WITH_FORECAST=0 make phase2-submit
```

Only the current stage should appear in the queue. Monitor with:

```bash
squeue -u "$USER"
tail -f logs/slurm/storm-p2-train-*.out
```

Every downloader is cache-aware. A failed monthly task can be restarted:

```bash
/panfs/ccds02/nobackup/people/afahad/envs/storm-outage-risk./bin/python \
  src/phase2_download.py --only era5 --years 2020 --months 7
```

The ARCO job reads seven fields from ECMWF's geo-chunked Zarr store and saves
one Michigan-only 2018–2023 cache. This matches the live store's roughly
7.7-year time chunks and avoids re-reading the same cloud chunks in every
month. After that cache completes, the 72-task monthly array (two concurrent)
requests only the five unavailable fields—CAPE, soil water layers 1/2,
snowfall and snow depth—from CDS, merges the corresponding local ARCO month,
and atomically publishes one 12-field NetCDF. Residual CDS requests can still
queue.

For manual submission, preserve this order:

```bash
sbatch --parsable slurm/phase2_download_arco.sbatch
# After that job reports COMPLETED:
sbatch --parsable slurm/phase2_download_era5.sbatch
```

ARCO requires `zarr`, `fsspec`, and `aiohttp`. For an existing pip environment:

```bash
/panfs/ccds02/nobackup/people/afahad/envs/storm-outage-risk./bin/python -m pip \
  install 'zarr==2.18.7' 'fsspec==2025.7.0' 'aiohttp==3.10.11' \
  'decorator==5.2.1'

Run `python -m pip check` with the same interpreter after updating an existing
environment. `fsspec` is pinned to 2025.7.0 because that exact version is
required by the commonly installed `gcsfs==2025.7.0`; ECMWF ARCO itself is
accessed over authenticated HTTPS and does not require `gcsfs`.
```

## Validation review

Review these before opening 2023:

```text
data/processed/phase2_validation_metrics.json
data/processed/phase2_cv_metrics.csv
data/processed/phase2_cox_ph_test.csv
data/processed/phase2_coverage_exclusions.json
data/processed/phase2_composed_metrics.json
data/processed/phase2_uncertainty_by_lead.csv
figures/phase2_occurrence_validation_2022.png
figures/phase2_forecast_*.png
figures/phase2_cost_loss_value.png
```

Specific things to check rather than glance at:

- **All three CV schemes present** in `phase2_cv_metrics.csv` — `storm_blocked`,
  `leave_one_county_out`, and four `forward_year` folds (2019–2022). A missing
  scheme is a finding, not a formatting issue.
- **Leave-one-county-out vs storm-blocked.** If LOCO is much worse, the model is
  memorising county-specific baselines. Spec §7.1 says name it; do.
- **`occurrence_brier_skill_vs_climatology`** is now measured against the
  *county* rate. `..._vs_global_climatology` is reported alongside it; the gap
  between them is itself the county-memorisation diagnostic.
- **`magnitude_model_actually_used`** — if it says `lgbm_quantile`, NGBoost
  failed to import or fit and the write-up must say so.
- **`phase2_coverage_exclusions.json`** — the excluded counties are a limitation
  to state (spec §11), and `test_year_coverage_gaps` lists counties whose 2023
  coverage changed without altering the cohort.
- **Composed rank histogram** in `phase2_composed_metrics.json` — flat means a
  reliable ensemble, U-shaped means under-dispersed.
- **`bias_correction_active_day*`** gates in `logs/phase2_gates.json` — spec §8.2
  is not optional, and an inert quantile mapping is the failure they catch.

## Application stages (spec §6.4, §8, §9)

These run against the frozen bundle and the validation-scope table. None of them
read the test year, so they can be built and reviewed while 2023 is still
sealed.

```bash
make phase2-compose     # 6.4  Monte Carlo over occurrence x magnitude x duration
make phase2-forecast    # 8    GEFS, quantile mapping, 31 x 100 realizations
make phase2-value       # 9    cost-loss value, break-even cost, EVPI
make phase2-apply       # all three
```

`make phase2-forecast-synthetic` derives stand-in members from ERA5 to exercise
the plumbing with no download. **It implies no forecast skill** and its output
must never appear in the write-up.

Two design notes worth carrying into the technical note:

- The quantile-mapping transfer function is fitted **once per case study** on all
  leads and members pooled, then applied unchanged to each lead. Re-fitting per
  lead would force every lead's ensemble onto the same climatological
  distribution and erase exactly the lead-dependent dispersion the §8.3
  decomposition exists to measure.
- The GEFS side of that mapping is pooled over members and forecast hours per
  cell and season, because the case-study downloads are the only GEFS sample
  this project has. A mapping trained on the GEFSv12 reforecast archive would be
  better and is the obvious upgrade. Say so rather than leaving it implicit.

The §8.4 money plot is drawn **without** the observed overlay until the test
year is formally opened. That is deliberate; rerun `make phase2-forecast` after
the final test to add the truth line.

## Final test

Freeze model and hyperparameter decisions in Git and tag the commit. Only then:

```bash
sbatch slurm/phase2_final_test.sbatch
```

That job builds `phase2_merged_test.parquet` — a **separate** file from the
validation-scope table, so a refit remains possible afterwards without repeating
the 72-file ERA5 aggregation — loads the frozen model without refitting, scores
2023, writes `models/TEST_YEAR_OPENED.txt`, then redraws the money plot and the
cost-loss curves now that the observed outcomes may be read.

The scorer refuses if the reporting-county cohort in the test build differs from
the cohort frozen into the bundle. Scoring a frozen model against a different
study population is not a held-out test; read the message before reaching for
`--force-test`.

A normal second attempt is refused. `--force-test` exists only to recover an
interrupted run. The marker is written **after** scoring succeeds, so a crash no
longer burns the single attempt.

Whatever comes out, report it. If the numbers disappoint, diagnose and write the
diagnosis — do not go back and tune (spec §7.4).

## Direct commands without Slurm

```bash
make doctor-phase2
make phase2-preflight
make phase2-download-outages
make phase2-download-era5
make phase2-download-canopy
make phase2-download-gefs
make phase2                 # build through 2022, train, calibrate, validate
make phase2-apply           # composition, forecast, decision value
make phase2-build-test      # only after decisions are frozen
make phase2-test            # one-time 2023 score
```

## Testing without any data

`tests/phase2_fixture.py` writes a miniature but structurally faithful input set
— six counties, a 45-cell ERA5 grid, twelve monthly NetCDFs with the CDS
coordinate names, two annual EAGLE-I CSVs — that runs the whole chain in about
twenty seconds with no downloads and no credentials:

```bash
/panfs/ccds02/nobackup/people/afahad/envs/storm-outage-risk./bin/python \
  tests/phase2_fixture.py /tmp/p2scratch
```

`tests/test_phase2_logic.py` uses it, and includes a regression test at
statewide scale that fails on the old storm-blocking definition. If that test
ever passes on both implementations, it has stopped testing anything.
