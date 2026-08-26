# Phase 1 report -- Michigan

Generated 2026-08-26 19:00 UTC in 0.1 min.
Window `2019-07-19` .. `2019-07-24` (5 days).

**Nothing in this report is a result.** Five days gives 20-60 events across one
state. No coefficient, AUC, CRPS or dollar figure below means anything. The only
question this report answers is whether the code executes and the outputs have
valid structure.

## Section 7 -- go / no-go

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | Every FIPS joins (zero unmatched on both sides) | **PASS** | unmatched EAGLE-I FIPS: none; counties with no outage record: none (acceptable only if the utility genuinely does not report there) |
| 2 | Timezones aligned (outage peak within 1 h of ERA5 gust peak) | **PASS** | tz=UTC -- one reference, UTC, set at ingest; ERA5 tz = UTC; outage peak day 2019-07-21 vs gust peak day 2019-07-21 (0 h apart) |
| 3 | Event table non-empty (20-60 events, 3 verified by eye) | **FAIL** | EMPTY EVENT TABLE -- window selection failed, restart at section 3; min customer_hours = 88.0; every event ends after it starts; 64 of 69 events uncensored; 69 events (expected 20- |
| 4 | Units confirmed (m/s, metres, Kelvin asserted in code) | **PASS** | i10fg units = 'm s**-1' -- NOT knots, NOT mph; tp units = 'm' -- ERA5 precip is METRES accumulated, not mm; t2m units = 'K'; t2m in [270.4, 287.5] K |
| 5 | Area weights valid (sum to 1.0 per county, equal-area CRS) | **PASS** | weight row sums in [1.00000000, 1.00000000], computed in EPSG:5070 |
| 6 | HAZARD-CONSEQUENCE CORRELATION POSITIVE (gust_max vs customer_hours > 0.3) | **PASS** | corr=0.741 on 2019-07-21 across 24 counties (gate: >0.3). If this fails after criteria 1-5 pass the problem is SCIENTIFIC, not mechanical: check timezone alignment first, then whet |
| 7 | All three model stages execute (no exceptions, valid shapes, no NaNs) | **FAIL** | proba shape (120,) vs 120 rows; proba in [0.2250, 0.8432]; no NaN predictions; lgbm_quantile: 7 quantile columns for 7 quantiles; lgbm_quantile: finite predictions; lgbm_quantile:  |
| 8 | Monte Carlo produces spread (per-row std > 0) | **PASS** | (120, 50) == (120, 50); min sample 0.000; 0 rows with zero spread (zero spread means point estimates were composed instead of sampled) |
| 9 | Bias correction active (mapped GEFS mean shifts toward ERA5 climatology) | **PASS** | |raw-ERA5| 3.432 -> |mapped-ERA5| 0.021 m/s (if this does not shrink, the mapping is inert and the forecast stage is silently broken) |
| 10 | End-to-end single command (`make phase1` runs clean) | **PASS** | `make phase1-synthetic` completed in 0.1 min |
| 11 | Volumes and timings recorded (section 8 table filled in) | **FAIL** | section 8 table filled in from this run |

**Failed checks**

- `03_events` / event_count_in_expected_range: 69 events (expected 20-60; thousands means baseline removal is not working, zero means the window missed the storm)
- `05_models` / no_quantile_crossing: lgbm_quantile: 31 crossing pairs (independent quantile fits can cross; the composition step sorts each row, and Phase 2 should fit a monotone model instead)
- `08_value` / cost_loss_varies_with_ratio: value changes with C/L (a flat curve means the threshold rule is not binding, or every probability is on one side of every ratio)
- `run_phase1` / measurements_recorded: section 8 table filled in from this run

**Overall: NO-GO -- do not start Phase 2**

Criterion 6 is the real gate. Criteria 1-5 and 7-10 test whether the code runs;
6 tests whether the premise holds in this data -- that public county-level
outage records carry a recoverable weather signal.

## Section 8 -- measurements

| Quantity | Phase 1 measured | Phase 2 projected |
|---|---|---|
| ERA5 download | _not measured_ | -- |
| ERA5 -> county aggregation | 0.0 s | 0.0 h |
| Event detection | 0.1 s | 0.0 h |
| Feature build | 0.2 s | 0.0 h |
| Model fit (all stages) | 0.9 s | 0.0 h |
| Monte Carlo | 0.1 s | 0.2 h |
| Peak memory | 0.29 GB | -- |
| Processed data on disk | 0.000 GB | 0.1 GB |

If the projected ERA5 aggregation exceeds a few hours, precompute the
cell-to-county weight matrix once as a sparse array and apply it as a single
matrix multiply per timestep (already done in `src/common/geo.py`). If projected
memory exceeds available RAM, switch to Dask with time chunking now, while the
pipeline is small.

## Section 9 -- handoff checklist

- [ ] Revert every `config/phase1.yaml` override: `make phase1-diff` lists them
      (24 keys).
- [ ] Restore the 30-day rolling baseline (`baseline_method: rolling`).
- [ ] `make clean-phase1` -- delete every fitted model and figure. They are
      contaminated by a relaxed threshold and a one-day validation split.
- [ ] Keep: download code, join logic, weight matrix, assertion suite, this table.
- [ ] `make test` -- the assertions are now a test module that runs on every
      Phase 2 execution.
- [ ] Confirm `region.yaml` splits are the frozen 2018-2021 / 2022 / 2023
      boundaries and that nothing in Phase 1 touched 2023.
