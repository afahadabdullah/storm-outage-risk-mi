# Storm-Driven Outage Risk and Forecast Value — Michigan

Probabilistic county-day model linking weather hazard to electricity
distribution outage **consequence**, driven by real forecast ensembles, and
converted into a decision-economic answer.

> **Status: Phase 1 passed; the Phase 2 full-study workflow is ready to run. No
> validated results yet.**
> Phase 1 proves the plumbing: every join lands, every unit is what you think it
> is, every array has the shape you expect, and data flows from raw CSV to a
> dollar figure without manual intervention. The models fitted at this stage are
> statistically meaningless — five days gives too few independent storm
> observations for validation, regardless of the exact event count —
> and they are deleted at the handoff. No skill score, no coefficient and no
> dollar figure in this repository refers to anything real until Phase 2's
> validation stage has run against the held-out test year.

---

## Quickstart

```bash
mamba env create --prefix /panfs/ccds02/nobackup/people/afahad/envs/storm-outage-risk \
  -f env/environment.yml
conda activate /panfs/ccds02/nobackup/people/afahad/envs/storm-outage-risk
make doctor                  # packages, ~/.cdsapirc, egress, disk
make phase1-synthetic        # ~5 s, no credentials, no downloads
make test
```

`phase1-synthetic` generates stand-in data with the same schemas, units and
dtypes the real fetchers produce, then runs the entire pipeline through it. It
exists so that plumbing bugs are found in seconds rather than after a three-hour
CDS queue. Every number it produces is fictional.

**Running on a remote box?** `docs/RUNBOOK.md` is the ordered version of what
follows, with wait times, credentials, tmux and triage.

### The real run

```bash
# The CDS queue is the only unpredictable wait in the project, and it cannot be
# queued until the window is chosen, which needs the outage data first. That
# constraint fixes the order below.
make fetch                            # EAGLE-I annual CSV + MCC + TIGER counties
/panfs/ccds02/nobackup/people/afahad/envs/storm-outage-risk/bin/python \
  src/select_window.py --write       # pick the 5-day window FROM THE DATA
nohup make era5-only &                # queue ERA5, then go do something else
/panfs/ccds02/nobackup/people/afahad/envs/storm-outage-risk/bin/python \
  src/02_fetch_weather.py --only gefs
/panfs/ccds02/nobackup/people/afahad/envs/storm-outage-risk/bin/python \
  src/02_fetch_weather.py --only canopy
make phase1                           # raw -> decision number, one command
```

`make phase1` writes **`docs/phase1_report.md`** — the section 7 go/no-go table
and the section 8 measurements table, both filled in from the run. Phase 2 starts
only when every criterion in that report says PASS.

### Full Phase 2 run

The full study is CPU-only and uses frozen 2018–2021 / 2022 / 2023 splits:

```bash
make phase2-submit       # sequential downloads -> build -> train/validate
# Review validation and freeze all choices, then exactly once:
sbatch slurm/phase2_final_test.sbatch
```

See [`docs/PHASE2_RUNBOOK.md`](docs/PHASE2_RUNBOOK.md) for outputs, restart
commands, resource requests, and the test-year lock.

---

## Layout

```
config/region.yaml       the only file to edit for a new region; frozen after day 2
config/phase1.yaml       TEMPORARY overrides; `make phase1-diff` lists them all
config/phase2.yaml       full-study execution controls; frozen splits stay in region.yaml
src/doctor.py            preflight for a fresh machine
src/select_window.py     step 0  window chosen from data, not from memory
src/01_fetch_outage.py   step 1  EAGLE-I + MCC denominator + TIGER counties
src/02_fetch_weather.py  step 2  ERA5 (CDS), GEFS (AWS byte-range), NLCD canopy
src/03_build_events.py   step 3  normalise, remove baseline, threshold -> events
src/04_build_features.py step 4  ERA5 -> county-day, and the correlation gate
src/05_fit_models.py     step 5  occurrence / magnitude / duration, all distributions
src/06_compose_mc.py     step 6  Monte Carlo through all three stages
src/07_forecast_cases.py step 7  GEFS, quantile-mapped, one lead time
src/08_decision_value.py step 8  cost-loss value and break-even inspection cost
src/phase2_download.py   annual/monthly/case-study data acquisition
src/phase2_build.py      rolling-baseline multi-year events and weather features
src/phase2_train.py      frozen fit, calibration, blocked CV, and final test
src/run_phase2.py        build + train/validate runner; never opens test year
src/common/geo.py        the cell-to-county weight matrix, built once, in EPSG:5070
tests/                   the phase 1 assertions, promoted to a test module
```

Numbered scripts follow the project spec's file names, so they are executed
rather than imported; anything shared between them lives in `src/common/`.

## Design decisions worth knowing before reading the code

**One time reference.** Everything is UTC, converted once at ingestion. A silent
local/UTC mismatch shifts the storm four to five hours and quietly destroys the
weather–outage correlation; the model just looks weak and you spend a day
blaming the features.

**Two aggregations, on purpose.** Smooth fields (precipitation, soil moisture,
temperature) get an area-weighted county mean. Tail fields (gust, CAPE) get the
max over intersecting cells. Damage is a tail phenomenon and a county-mean gust
destroys the signal. Both matrices are built once, in the equal-area CRS — area
maths in EPSG:4326 is wrong by a latitude-dependent factor that in a north–south
state like Michigan is large enough to matter and small enough to look plausible.

**Everything outputs a distribution.** Occurrence, magnitude and duration are
each sampled, never multiplied as point estimates. Composing medians produces
output that looks entirely reasonable and carries no uncertainty at all, which is
why `tests/` asserts per-row spread on the conditional draws.

**Gate criterion 6 is the real gate.** `corr(gust_max, customer_hours / MCC) > 0.3`
across county-days with an event. This respects storms that reach counties at
different local times; the state-wide peak day remains a diagnostic. Criteria
1–5 and 7–10 test whether the code runs; 6 tests whether public county-level
outage records carry a recoverable weather signal at all. If it fails while the
rest pass, the problem is scientific, and the diagnostic order is: timezone
alignment, then event-day identity across the two datasets, then utility
reporting coverage in the hardest-hit counties, then whether ERA5 resolved this
storm's gusts.

## Phase 1 discipline

1. **Do not interpret any Phase 1 result.** Not the AUC, not the hazard ratios,
   not the break-even cost.
2. **Do not tune anything.** Tuning on a smoke-test slice contaminates Phase 2
   before it starts.
3. **Do not keep any Phase 1 artifact.** `make clean-phase1` deletes the fitted
   models and figures. Timing and volume measurements are the one thing that
   carries forward.
4. **Touch the test year exactly once**, at the end of Phase 2, after every model
   and hyperparameter decision is frozen. If the results disappoint, report them
   and diagnose why. Do not go back and tune.

## Limitations

Stated up front, before anyone asks — the full list is in §11 of the project
spec. The load-bearing ones: county aggregation hides network structure; EAGLE-I
reports consequence, not cause, so this is **not** fragility modeling; canopy
percentage is exposure, not vegetation-management condition; ERA5 under-resolves
convective gusts at 0.25°; there are no asset age, material or inspection
records; and restoration time confounds damage severity with crew logistics.

## Data sources

EAGLE-I county outage data (ORNL, figshare `10.6084/m9.figshare.24237376`);
ERA5 via the Copernicus Climate Data Store; GEFS via `noaa-gefs-pds` on AWS;
NLCD 2021 tree canopy (MRLC); TIGER/Line counties (US Census); ICE Calculator
(LBNL/DOE) for interruption costs.
