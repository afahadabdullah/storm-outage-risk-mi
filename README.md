# Storm-driven outage risk and forecast value — Michigan

**Abdullah Al Fahad · NASA Goddard Space Flight Center**

This repository develops a probabilistic county-day model of weather-conditioned
electricity-distribution outage consequence and the operational value of
ensemble forecasts. It estimates outage occurrence, customer-hours, and
restoration duration; it is not an asset-fragility or physical damage model.

## Workflow overview

![Storm-driven outage risk workflow: ERA5, EAGLE-I, NLCD, and GEFS inputs are aggregated to county-day features; a three-part probabilistic model produces county risk and decision-value inputs.](docs/assets/project-workflow.png)

This workflow figure is a methods schematic. Its maps, curves, density shapes,
and annotated medians illustrate the analysis design; the empirical results are
the metric tables and generated figures below.

## Final 2023 results

The frozen model was fit on January 2018–June 2022, calibrated and verified on
July–December 2022, and scored once on the full 2023 holdout. The 2023 score is
the primary result below; validation is shown for context.

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

The frozen occurrence model outperformed the logistic GLM, county climatology,
and gust-threshold reference on Brier score. Magnitude skill remained positive,
while the larger 2023 magnitude error and lower restoration concordance show
where uncertainty remains greatest.

### GEFS 2023 case studies

These are separate operational forecast evaluations of the frozen model. They
do not replace the year-long test.

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

All August wind-event intervals contained the observation. Every February
ice-storm interval was below the observation, identifying a case-specific
underprediction that should guide future hazard/magnitude diagnostics.

## Publication figures

The PNGs are the GitHub-visible versions; matching vector PDFs are produced by
the reporting job for manuscript use.

![Frozen-model skill summary: reliability, precision–recall, magnitude PIT, reference models, hazard regimes, and cross-validation.](figures/phase2_skill_summary.png)

![County-level diagnostic maps: event rate, forecast probability, bias, Brier skill, event count, and magnitude CRPS.](figures/phase2_county_diagnostics.png)

![GEFS forecast distributions for the two 2023 case studies and four lead times.](figures/phase2_gefs_case_studies.png)

![ERA5 hazard fields on the two frozen-model GEFS case-study days.](figures/phase2_case_hazards.png)

![Relative economic value across cost-loss ratios.](figures/phase2_cost_loss_value.png)

For the complete generated result record, see [the Phase 2 result memo](docs/phase2_results.md) and the [GitHub-readable technical memo](docs/phase2_technical_memo.md). The animated HTML memo is retained for local/browser review, but GitHub does not execute its JavaScript.

## Economic impact and forecast value

The model's loss unit is customer-hours: one customer without service for one
hour. The configured interruption-cost proxy is

\[
\$/{\rm customer\ hour}=0.88(\$4.00)+0.12(\$180.00)=\$25.12.
\]

This is a documented placeholder based on the current residential/commercial
mix, not measured Michigan damage, repair expense, or utility financial loss.
The memo converts observed customer-hours to that proxy and shows potential
avoided impact when a forecast-triggered action is assumed to reduce realized
consequence by 10%, 20%, or 30%. Those are counterfactual scenarios, not
observed savings. Direct asset damage and realized action value require
utility-specific asset, action-cost, work-order, and intervention-outcome data.

Forecast value is evaluated in two complementary ways:

- **Dimensionless cost–loss value:** act when calibrated probability
  \(p>C/L\), then compare forecast decisions with climatology and perfect
  information across configured cost–loss ratios.
- **Action-impact scenarios:** identify counties triggered by the selected
  probability threshold, report the observed loss covered by those counties,
  and apply the explicit mitigation-effectiveness assumption to estimate
  potential avoided customer-hours and cost proxy.

The animated memo includes the cost–loss curve and forecast-triggered action
plot. Its dollar annotation remains marked as a proxy until the ICE Calculator
inputs and utility action costs are replaced with region-specific values.

## Scientific scope and methods

### Inputs and county-day features

- **EAGLE-I** county outage records provide hourly outage consequence and event
  construction; MCC data supply customer denominators.
- **ERA5** hourly reanalysis supplies wind, gust, precipitation, temperature,
  soil moisture, CAPE, snow, and related predictors over Michigan.
- **TIGER/Line** county geometry supports equal-area spatial aggregation.
- **NLCD tree canopy** supplies a broad exposure proxy, not vegetation-
  management condition.
- **GEFS** ensemble forecasts drive the two 2023 case studies.

ERA5's 0.25° grid is aggregated to counties in equal-area EPSG:5070. Smooth
fields use area-weighted means; localized hazards use spatial maxima. Daily
features include maximum gust, precipitation totals, threshold-exceedance
hours, antecedent wetness, freezing-rain and wet-snow proxies, canopy
interactions, and seasonal indicators.

### Probabilistic risk model

For county \(c\) and day \(t\), the model estimates a predictive distribution,
not a deterministic warning or physical asset-failure probability:

1. **Occurrence:** \(p_{ct}=P(Y_{ct}=1\mid X_{ct})\), fitted with LightGBM and
   isotonic probability calibration.
2. **Magnitude:** a conditional distribution of customer-hours, using NGBoost
   in log-space with LightGBM quantiles as fallback.
3. **Restoration:** conditional restoration duration from a Weibull AFT model,
   with Cox diagnostics.

Monte Carlo composition draws occurrence, conditional magnitude, and duration
jointly. This preserves zero inflation, heavy tails, and uncertainty for
customer-hours and decision-value calculations.

### Verification language

- **Brier score** measures squared error of event probabilities; lower is better.
- **Brier skill** compares Brier score with county climatology; positive means
  improvement over that reference.
- **Average precision and ROC AUC** measure rare-event ranking, not calibration.
- **CRPS** evaluates a full predictive distribution; lower is better.
- **PIT/rank histograms** diagnose ensemble dispersion.
- **Concordance and MAE** evaluate restoration-duration predictions.

The validation reliability curve uses out-of-fold calibrated probabilities.
Storm-blocked, leave-one-county-out, and forward-year folds are retained to
reduce leakage from correlated weather and outage observations.

## Data split and safeguards

| Period | Role |
|---|---|
| 2018-01-01 through 2022-06-30 | Model fitting |
| 2022-07-01 through 2022-12-31 | Calibration and validation |
| 2023-01-01 through 2023-12-31 | One-time final test |

All timestamps use UTC. County-grid weights are calculated in an equal-area
CRS. No 2023 outcome is used for tuning, calibration, bias correction, or
model selection. The earlier January–May 2021 retrospective is superseded
because 2021 is now training data; forward-year CV is the appropriate temporal
diagnostic.

## Reproducible commands and outputs

After inputs are cached and the model is frozen:

```bash
make phase2                 # build through 2022 and train/validate
make phase2-apply           # composition, GEFS cases, decision value, report
make phase2-build-test      # open/build 2023, once decisions are frozen
make phase2-test            # score frozen model on 2023 exactly once
make phase2-techmemo        # HTML + GitHub Markdown memo
```

The final-test Slurm job runs the report and both memo formats automatically.
The report writes `phase2_results_matrix.csv`, `phase2_gefs_case_matrix.csv`,
`phase2_county_skill.csv`, 300-dpi PNGs, vector PDFs, and the Markdown/HTML
technical memos. Commit the reviewed Markdown memo and `figures/phase2_*.png`
files when publishing the result on GitHub.

## Limitations and provenance

County aggregation hides feeder topology, asset age, conductor type, local
vegetation management, and utility operations. EAGLE-I records consequence, not
causal mechanism. ERA5 can under-resolve convective gusts at 0.25°, and the
GEFS case-study bias correction is limited by the small available case sample.
The interruption-cost and asset-count assumptions must be replaced before
external economic claims.

Data provenance: EAGLE-I is from ORNL/figshare (`10.6084/m9.figshare.24237376`);
ERA5 is from Copernicus Climate Data Store and ECMWF ARCO; GEFS is from NOAA's
public cloud archive; NLCD canopy is from MRLC; county geometry is from U.S.
Census TIGER/Line; interruption-cost assumptions follow the LBNL/DOE ICE
Calculator framework.

For the operational runbook, environment setup, gates, and artifact checks,
see [`docs/PHASE2_RUNBOOK.md`](docs/PHASE2_RUNBOOK.md).
