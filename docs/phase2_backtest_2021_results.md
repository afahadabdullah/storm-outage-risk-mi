# Archived Phase 2 split experiment: 2021 available months

> **Superseded design.** The current frozen model trains through 2022-06-30,
> so 2021 is now training data. The numbers below document an earlier split
> experiment only and must not be reported as out-of-sample performance for the
> retrained model.

## Scope

This is an exploratory temporal backtest of the already frozen Phase 2 bundle,
not the final held-out test. It leaves the configured 2023 final-test pathway
sealed. The local ERA5 cache contained a contiguous 2021 window from
**2021-01-01 through 2021-05-31**; no later month was included.

The generated, reproducible run artifacts are intentionally excluded from Git:

```text
data/processed/phase2_backtest_2021_metrics.json
data/processed/phase2_backtest_2021_skill_matrix.csv
data/processed/phase2_backtest_2021_county_skill.csv
figures/phase2_backtest_2021_20210101_20210531_diagnostics.png
figures/phase2_backtest_2021_20210101_20210531_maps.png
```

## Results

| Metric | Validation run | 2021 Jan--May backtest | Reading |
|---|---:|---:|---|
| Brier skill score vs county climatology | 0.101 | 0.084 | Positive skill persists, although it is modestly lower. |
| Average precision | 0.327 | 0.249 | Ranking precision is lower in this new temporal window. |
| ROC AUC | 0.752 | 0.760 | Discrimination is essentially unchanged. |
| Magnitude CRPS skill score | 0.113 | 0.230 | Conditional magnitude forecasts improve on their climatological reference. |

The whole-window result is encouraging: the model retains positive probability
skill against the demanding county-specific climatology and its AUC is stable.
It is not evidence of uniformly strong skill: the BSS is only 0.084 and average
precision fell relative to validation.

The monthly matrix is positive in every available month. Occurrence BSS falls
from 0.12 in January--February to 0.04 in May, while AUC ranges from 0.67 to
0.78. Magnitude CRPS skill remains positive (0.17--0.27). That late-window BSS
decline is a useful monitoring signal, not a reason to retune on this backtest.

## Diagnostic interpretation

- The reliability curve follows the diagonal closely over the probability range
  represented in this five-month sample. Calibration is reassuring, but there
  are few very-high-probability forecasts, so tail calibration remains uncertain.
- The county maps show mostly positive BSS across the Lower Peninsula, with
  several near-zero or negative counties, particularly in the Upper Peninsula.
  County results should be read with care because each county has only 151
  evaluated days and event counts vary substantially.
- The probability-bias map is generally close to zero, with geographically
  mixed over- and under-prediction. This supports inspecting county effects in
  later data rather than claiming spatial calibration is solved.

## Reporting decision

Report this as a **Jan--May 2021 retrospective backtest**. Do not call it a
full-year test and do not use it for further model or threshold tuning. If the
remaining 2021 weather months are later cached, rerun the same frozen bundle to
extend the evaluation window; keep this result as the first available-month
record.
