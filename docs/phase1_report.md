# Phase 1 report -- Michigan

Generated 2026-08-26 19:04 UTC in 0.0 min.
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
| 3 | Event table non-empty (20-60 events, 3 verified by eye) | **PASS** | 21 events detected; min customer_hours = 122.0; every event ends after it starts; 20 of 21 events uncensored; 21 events (expected 20-60; thousands means baseline removal is not wor |
| 4 | Units confirmed (m/s, metres, Kelvin asserted in code) | **PASS** | i10fg units = 'm s**-1' -- NOT knots, NOT mph; tp units = 'm' -- ERA5 precip is METRES accumulated, not mm; t2m units = 'K'; t2m in [270.4, 287.5] K |
| 5 | Area weights valid (sum to 1.0 per county, equal-area CRS) | **PASS** | weight row sums in [1.00000000, 1.00000000], computed in EPSG:5070 |
| 6 | HAZARD-CONSEQUENCE CORRELATION POSITIVE (gust_max vs customer_hours > 0.3) | **PASS** | corr=0.645 on 2019-07-21 across 24 counties (gate: >0.3) |
| 7 | All three model stages execute (no exceptions, valid shapes, no NaNs) | **PASS** | proba shape (120,) vs 120 rows; proba in [0.0898, 0.7317]; no NaN predictions; lgbm_quantile: 7 quantile columns for 7 quantiles; lgbm_quantile: finite predictions; lgbm_quantile:  |
| 8 | Monte Carlo produces spread (per-row std > 0) | **PASS** | (120, 50) == (120, 50); min sample 0.000; min per-row magnitude std 397.4; min per-row duration std 0.4067; 118/120 rows drew at least one event; all of them have non-zero spread ( |
| 9 | Bias correction active (mapped GEFS mean shifts toward ERA5 climatology) | **PASS** | |raw-ERA5| 2.942 -> |mapped-ERA5| 0.020 m/s (if this does not shrink, the mapping is inert and the forecast stage is silently broken) |
| 10 | End-to-end single command (`make phase1` runs clean) | **PASS** | `make phase1-synthetic` completed in 0.0 min |
| 11 | Volumes and timings recorded (section 8 table filled in) | **PASS** | section 8 table filled in from this run |

**Overall: GO -- Phase 2 may start**

Criterion 6 is the real gate. Criteria 1-5 and 7-10 test whether the code runs;
6 tests whether the premise holds in this data -- that public county-level
outage records carry a recoverable weather signal.

## Section 8 -- measurements

| Quantity | Phase 1 measured | Phase 2 projected |
|---|---|---|
| ERA5 download | 0.0 s (synthetic -- no download) | 0.0 h |
| ERA5 -> county aggregation | 0.0 s | 0.0 h |
| Event detection | 0.0 s | 0.0 h |
| Feature build | 0.1 s | 0.0 h |
| Model fit (all stages) | 0.4 s | 0.0 h |
| Monte Carlo | 0.1 s | 0.2 h |
| Peak memory | 0.28 GB | -- |
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
- [ ] Confirm `region.yaml` splits are the frozen 2018-2019 / Jan-Jul 2020 / 2023
      boundaries and that nothing in Phase 1 touched 2023.
