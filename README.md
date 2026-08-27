# Storm-Driven Outage Risk and Forecast Value — Michigan

This repository develops a probabilistic, county-day framework for estimating
electricity-distribution outage consequence from weather hazards and for
quantifying the operational value of real forecast ensembles. The study area is
Michigan; the outcome is customer outage consequence, not asset fragility or
the physical cause of damage.

## Scientific objective

The project asks two linked questions:

1. Given observed weather, what is the distribution of outage occurrence,
   customer-hours, and restoration time for each county-day?
2. Given an ensemble forecast, how does advance knowledge of that distribution
   change a cost-sensitive preparation decision?

The analysis preserves uncertainty throughout. It models outage occurrence,
magnitude, and restoration as separate distributions, composes them by Monte
Carlo simulation, and reports risk and decision value rather than a single
deterministic outage estimate.

## Workflow

![Storm-driven outage risk workflow: ERA5, EAGLE-I, NLCD, and GEFS inputs are aggregated to county-day features; a three-part probabilistic model produces county risk and decision-value inputs. The small maps and curves are schematic, not reported results.](docs/assets/project-workflow.png)

Historical hazards and outage records are converted from hourly gridded and
county data into a common county-day table. The frozen core temporal design
fits on 2018–2019, calibrates and validates on January–July 2020, and reserves
2023 for a single final evaluation. The separate 2021 retrospective backtest is
diagnostic only; it does not tune or refit the model. GEFS ensemble forecasts
then drive the 2023 case-study scenarios at lead days 5, 3, 2, and 1.

The workflow image is a method schematic: its maps, curves, density shapes,
and annotated medians illustrate the types of inputs and outputs, rather than
empirical estimates from a completed run. Replace those panel graphics with
generated model output before using the figure to report results.

For a detailed interpretation of the model, verification metrics, risk terms,
and current economic assumptions, see the
[atmospheric-science overview](docs/ATMOSPHERIC_SCIENTIST_OVERVIEW.md).

## Data

| Data product | Coverage | Role in the analysis |
|---|---|---|
| EAGLE-I county outage records | 2017–2023 | Hourly outage consequence and event construction; 2017 supplies baseline context. |
| Monthly County Customer (MCC) data | Available reporting years | Customer denominator for outage fractions and customer-hours. |
| ERA5 hourly reanalysis | 2018–2023, Michigan bbox | Historical wind, gust, precipitation, temperature, soil, convective, and snow predictors. Seven fields are read from ECMWF ARCO; five residual fields are retrieved from CDS. |
| TIGER/Line counties | 2023 geography | County geometry and land area for spatial aggregation and exposure features. |
| NLCD tree canopy | 2021 | County canopy exposure and weather–vegetation interactions. |
| GEFS ensemble forecasts | Two 2023 case studies | Forecast scenarios for lead-time, calibration, and decision-value analysis. |
| ICE Calculator inputs | Study assumptions | Interruption-cost inputs for cost-loss and value-of-information calculations. |

## Methods

### Outcome construction

Outage records are normalized to UTC and converted to county-hour maxima.
Customer outages are divided by MCC denominators. A rolling 30-day, 10th
percentile baseline separates excess outage from routine background activity;
contiguous excess periods form outage events.

### Weather and spatial features

ERA5 fields are aggregated from a 0.25° grid to counties in the equal-area
EPSG:5070 coordinate system. Area-weighted means represent smooth fields such
as precipitation, soil moisture, and temperature. Spatial maxima represent
tail hazards such as gust and CAPE. Daily features include gust thresholds,
precipitation totals, freezing-rain and wet-snow proxies, antecedent
precipitation, canopy interactions, and seasonal indicators.

### Probabilistic outage model

Three linked models estimate:

- **Occurrence:** probability of any outage event on a county-day.
- **Magnitude:** conditional distribution of customer-hours.
- **Restoration:** conditional duration of an outage event.

The three distributions are sampled jointly through Monte Carlo composition to
produce a county-level distribution of customers affected, customer-hours, and
cost. Calibration and storm-aware cross-validation evaluate the models without
treating nearby days from the same storm as independent evidence.

### Encouraging retrospective results

With its parameters frozen, the Phase 2 model retained useful skill in the
available 2021 backtest window (January--May): it achieved an AUC of **0.760**,
showing strong discrimination between event and non-event county-days; a Brier
skill score of **0.084** against the demanding county-specific climatology; and
a magnitude CRPS skill score of **0.230**. In practical terms, the model
continues to rank higher-risk county-days well, improves probabilistic outage
estimates beyond each county's historical baseline, and produces more useful
conditional outage-magnitude distributions than climatology alone. This is a
retrospective backtest, not the sealed 2023 final evaluation; full details are
in [the 2021 backtest results](docs/phase2_backtest_2021_results.md).

### Forecast and decision analysis

For the two 2023 case studies, GEFS members provide day-5, day-3, day-2, and
day-1 forecast scenarios. Forecast features are aligned with the historical
feature definitions and bias-corrected before the trained risk model is
applied. The resulting predictive distributions feed a cost-loss calculation
that compares preparation costs, expected outage costs, break-even inspection
costs, and the value of perfect information.

## Scientific safeguards

- **One time standard:** all timestamps are converted to UTC at ingestion.
- **Geographically valid aggregation:** county-grid weights are calculated in
  an equal-area CRS rather than latitude/longitude degrees.
- **No test-year tuning:** 2023 is reserved for final evaluation after model
  choices are fixed from the earlier years.
- **Distributional predictions:** uncertainty from occurrence, magnitude, and
  duration is propagated rather than collapsed into point estimates.
- **Event-aware validation:** storm-grouped and county-aware folds reduce
  leakage from correlated weather and outage observations.

## Outputs

The repository produces reproducible county-hour and county-day tables,
event records, calibrated validation metrics, fitted model bundles, probabilistic
outage-risk scenarios, forecast maps, and decision-economic summaries. The
workflow is designed so that every intermediate artifact can be inspected,
validated, and regenerated from configuration and raw inputs.

## Repository guide

```text
config/       study geography, temporal splits, data sources, and model controls
data/         raw, interim, and processed artifacts
docs/         runbooks, methodological notes, and workflow figures
env/          reproducible environment specifications
slurm/        CPU Slurm jobs for downloading, preprocessing, training, and evaluation
src/          ingestion, event construction, feature engineering, modeling, forecasting,
              decision analysis, and shared geospatial utilities
tests/        data-contract, feature, modeling, and workflow assertions
Makefile      reproducible commands and validation targets
```

Create the specified environment and inspect the available reproducible
commands with:

```bash
mamba env create --prefix /panfs/ccds02/nobackup/people/afahad/envs/storm-outage-risk. \
  -f env/environment.yml
conda activate /panfs/ccds02/nobackup/people/afahad/envs/storm-outage-risk.
make test
make help
```

## Limitations

County aggregation hides distribution-network topology and local asset
condition. EAGLE-I records outage consequence rather than causal mechanism, so
the framework is not a fragility model. ERA5 can under-resolve convective gusts
at 0.25°, tree-canopy percentage is not a measure of vegetation-management
condition, and restoration duration also reflects crew logistics. The dataset
does not include asset age, conductor material, inspection history, or
utility-specific operational practices.

## Data provenance

EAGLE-I county outage data are from ORNL/figshare
(`10.6084/m9.figshare.24237376`); ERA5 is provided by the Copernicus Climate
Data Store and ECMWF ARCO; GEFS is accessed through NOAA's public cloud
archive; NLCD tree canopy is from MRLC; county geometry is from the U.S. Census
TIGER/Line program; and interruption-cost assumptions draw on the LBNL/DOE ICE
Calculator.
