# Phase 2 results — Michigan

**Abdullah Al Fahad · NASA Goddard Space Flight Center**

This reviewed result record reports the frozen model after the one-time 2023
test. Training used 2018-01-01 through 2022-06-30; calibration/validation used
2022-07-01 through 2022-12-31; the final test used all of 2023.

## Headline metric matrix

| Metric | Better | Validation | 2023 test |
|---|---|---:|---:|
| Occurrence Brier score | lower | 0.0926 | **0.0795** |
| Brier skill vs county climatology | higher | +0.101 | **+0.075** |
| Average precision | higher | 0.339 | **0.277** |
| ROC AUC | higher | 0.753 | **0.746** |
| Occurrence log loss | lower | 0.3196 | **0.2838** |
| Logistic GLM Brier | lower | 0.0999 | **0.0840** |
| Gust > 20 m/s rule Brier | lower | 0.1418 | **0.1118** |
| County climatology Brier | lower | 0.1030 | **0.0859** |
| Magnitude CRPS (customer-hours) | lower | 16,011 | **39,164** |
| Magnitude CRPS skill vs climatology | higher | +0.083 | **+0.084** |
| Magnitude log score | lower | 9.242 | **9.003** |
| Magnitude median RMSE | lower | 146,422 | **501,575** |
| Restoration concordance | higher | 0.663 | **0.619** |
| Restoration median MAE (hours) | lower | 4.42 | **5.22** |
| County persistence MAE (hours) | lower | 4.57 | **4.20** |

## GEFS case-study matrix

| Case | Lead | Median | 10–90% interval | Observed | Meteorological variance |
|---|---:|---:|---:|---:|---:|
| Aug 2023 wind event | day −5 | 53,113 | 13,677–364,016 | 25,577 | 9.6% |
| Aug 2023 wind event | day −3 | 52,981 | 14,710–179,218 | 25,577 | 11.1% |
| Aug 2023 wind event | day −2 | 76,129 | 18,787–433,563 | 25,577 | 26.8% |
| Aug 2023 wind event | day −1 | 54,824 | 14,710–223,482 | 25,577 | 36.0% |
| Feb 2023 ice storm | day −5 | 2,454 | 0–15,818 | 38,402 | 11.5% |
| Feb 2023 ice storm | day −3 | 4,272 | 0–20,890 | 38,402 | 18.1% |
| Feb 2023 ice storm | day −2 | 3,184 | 0–14,660 | 38,402 | 9.0% |
| Feb 2023 ice storm | day −1 | 3,462 | 0–16,146 | 38,402 | 6.7% |

All four August wind-event intervals contained the observation. Every February
ice-storm interval was below the observation; this is a case-specific
underprediction diagnostic, not a replacement for annual verification.

![Probabilistic statewide customer-hour forecasts by GEFS lead for the August 2023 wind event, with observed customer-hours](../figures/phase2_gefs_lead_trajectories.png)

## Publication figures

![Frozen-model skill summary](../figures/phase2_skill_summary.png)

![County-level diagnostic maps](../figures/phase2_county_diagnostics.png)

![GEFS forecast distributions](../figures/phase2_gefs_case_studies.png)

![ERA5 hazard fields on the case-study days](../figures/phase2_case_hazards.png)

![Relative economic value across cost-loss ratios](../figures/phase2_cost_loss_value.png)

## Interpretation

The 2023 occurrence Brier score remained better than the logistic GLM, county
climatology, and gust-threshold references. Magnitude CRPS skill remained
positive, but the larger 2023 magnitude error and lower restoration concordance
identify uncertainty that should be addressed with more event and utility
operations data. Customer-hour-to-dollar values are interruption-cost proxies,
not direct infrastructure damage or realized savings.

## Forecast-triggered impact by lead

At `C/L = 0.10` and an assumed 20% consequence reduction, the non-zero
forecast-triggered scenarios are shown below. The labels above the coverage
bars give the triggered counties out of 80.

![August 2023 wind-event forecast-triggered action by lead](../figures/phase2_forecast_triggered_impact_by_lead.png)

The February ice-storm rows are omitted because no counties crossed the trigger
and the potential avoided proxy was zero. These are counterfactual potential
impacts, not observed savings.

Generated from the final frozen artifacts by `make phase2-report`; the README
is the results-first scientific summary.
