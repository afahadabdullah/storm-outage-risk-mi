# Atmospheric-science overview

## What the project does

This is a probabilistic, county-day weather-to-outage-consequence model for
Michigan. Its central question is:

> Given the atmospheric state in a county on a day, what is the probability,
> magnitude, and likely duration of an electricity-outage event, and how does
> that risk estimate change when the atmospheric input is an ensemble forecast?

It is an empirical consequence model, not a network fragility model. It does
not identify the pole, feeder, or tree that fails, nor does it establish that a
specific weather variable caused a particular outage. It estimates the
historical statistical relationship between atmospheric predictors, county
exposure proxies, and observed outage consequences.

## Inputs and county-day features

- **ERA5** supplies historical hourly fields, including wind and gust,
  precipitation, temperature, snow, CAPE, and soil moisture.
- **EAGLE-I** supplies county outage observations, including customers out over
  time. These records are normalized by Monthly County Customer (MCC) counts
  and converted to event outcomes.
- **NLCD tree canopy** is a county-scale exposure proxy. It can represent broad
  vegetation exposure, but not utility vegetation-management practice or the
  condition of individual trees.
- **GEFS** supplies the ensemble members for the 2023 forecast case studies.

ERA5's 0.25-degree grid is aggregated to county polygons in an equal-area
coordinate system. Area-weighted means are used for spatially smooth fields;
spatial maxima are used for localized hazard fields such as gust. Hourly
weather becomes county-day predictors such as daily maximum gust, accumulated
precipitation, threshold-exceedance hours, antecedent wetness, canopy
interactions, and seasonal indicators.

## What “risk” means here

For county \(c\) and day \(t\), risk is a **predictive distribution** of outage
consequence, not a deterministic warning and not a physical probability of
asset failure. The model uses a three-part *hurdle model*:

1. **Occurrence**

   \[
   p_{ct}=P(Y_{ct}=1\mid X_{ct}).
   \]

   This is the probability of an outage event on the county-day, conditional on
   predictors \(X_{ct}\). It is fitted with LightGBM and isotonic probability
   calibration.

2. **Magnitude**

   Conditional on an event, the model estimates a distribution of
   **customer-hours**. One customer-hour is one customer without service for one
   hour; for example, 10,000 customers interrupted for two hours is 20,000
   customer-hours. The primary magnitude model is an NGBoost Normal density in
   \(\log(1+\text{customer-hours})\) space, with LightGBM quantiles as a
   fallback.

3. **Restoration duration**

   Conditional on an event, the model estimates restoration time in hours. A
   Weibull accelerated-failure-time survival model is primary, with a Cox
   proportional-hazards fit used as a diagnostic cross-check.

The three components are sampled with Monte Carlo simulation. Each realization
first draws whether an event occurs, then draws its conditional magnitude and
duration. This retains the zero-inflated and heavy-tailed character of outage
consequence, and yields medians, intervals, exceedance probabilities, and
decision-relevant loss distributions.

## Why separate occurrence, magnitude, and duration?

Outage data are strongly non-Gaussian: most county-days have no material event,
most events are modest, and a small number of storms produce very large
consequences. A single regression of mean outage impact would mix these
different processes. The hurdle model separates *whether* an event occurs from
*how large* and *how persistent* it is when it does occur.

## Evaluation language

- **Brier score:** mean squared error of event probabilities. It rewards both
  sharp and accurate probabilities.
- **Brier skill score (BSS):** Brier-score improvement relative to a reference,
  here principally a county's own climatological event frequency. Positive BSS
  means improvement over that reference; zero means no improvement.
- **Calibration / reliability:** among all county-days assigned a probability
  near 0.30, events should occur near 30% of the time.
- **ROC-AUC:** ranking discrimination between event and non-event days. It does
  not itself assess probability calibration.
- **Average precision:** rare-event discrimination metric that emphasizes
  whether the high-risk predictions identify actual events.
- **CRPS:** a proper score for a full predictive distribution, analogous to a
  distribution-aware version of absolute error. Lower is better.
- **CRPSS:** CRPS skill score relative to a reference distribution. Positive is
  an improvement over the reference.
- **PIT or rank histogram:** checks ensemble dispersion. A U shape commonly
  suggests under-dispersion; a dome shape suggests over-dispersion.
- **Concordance index:** for restoration time, tests whether events that lasted
  longer generally received longer predicted durations.
- **Quantile calibration:** checks interval coverage; a nominal 90% predictive
  interval should contain roughly 90% of suitable observations.

The core frozen temporal design fits on 2018–2019, calibrates and validates on
January–July 2020, and reserves 2023 for one final evaluation. A 2021 score is
a separately labelled retrospective diagnostic and is not used for tuning.

## Ensemble forecast and forecast value

Each GEFS member is a plausible atmospheric evolution. Gust and precipitation
are quantile-mapped toward the ERA5 climatology, then transformed into the same
county-day features used for model fitting. Each member is passed through the
frozen hurdle model, preserving both ensemble and conditional-outage
uncertainty. The forecast products therefore describe a distribution of
county-level outage consequence at day-5, day-3, day-2, and day-1 lead.

The GEFS bias correction is a necessary but limited case-study procedure: its
GEFS-side distribution is pooled across the available members and forecast
hours. A reforecast archive would provide a more rigorous calibration sample.

## Cost–loss ratio: what \(C/L\) means

In the classical binary decision model:

- \(C\) is the cost of taking protective action;
- \(L\) is the loss incurred if an event occurs and no action is taken;
- \(\alpha=C/L\) is the dimensionless cost–loss ratio.

For a calibrated event probability \(p\), the standard decision rule is

\[
\text{act if } p>\alpha.
\]

For example, if preparation costs $10,000 and the avoided loss is $100,000,
then \(C/L=0.10\); action is economically justified when the event probability
exceeds 10% under this simple model.

\(C/L\) must be non-negative. A negative value would mean that acting pays the
decision maker even when no event occurs, which is not a normal cost–loss
decision and makes the threshold rule meaningless. It does **not** mean that a
negative ratio is evidence of forecast value. As \(C/L\) approaches zero from
above, action becomes rational for nearly any nonzero event probability because
the action is nearly free. The configured ratios are 0.01 through 0.50.

Relative economic value compares the expense of a forecast-based decision with
climatology and perfect information, all in units of \(L\). The cost–loss curve
therefore does not need a dollar value for \(C\) or \(L\); it evaluates the
quality of probability-supported decisions across plausible ratios.

## How dollar values are currently calculated

The dollar calculations are separate from the dimensionless \(C/L\) curves.
They use a configured interruption-cost rate, currently marked in the code and
configuration as a **placeholder** pending actual LBNL/DOE ICE Calculator
output for Michigan and the intended customer mix:

\[
\begin{aligned}
\$ / \text{customer-hour}
&=0.88\times\$4.00+0.12\times\$180.00\\
&=\$25.12\ \text{per customer-hour}.
\end{aligned}
\]

The inputs are 88% residential customers at $4/customer-hour and 12%
commercial customers at $180/customer-hour. They are assumptions, not measured
Michigan outage costs; no dollar result should be interpreted as final until
they are replaced.

For the break-even inspection calculation, the implementation computes, by
county:

\[
\text{avoided dollars}
=\frac{\text{mean simulated customer-minutes}\times\delta}{60}
\times\$25.12,
\]

where \(\delta\) is an assumed 10%, 20%, or 30% reduction in failure hazard.
It then divides by an asset-count proxy:

\[
\text{assets}=\frac{\text{MCC customers}}{12}.
\]

The result is a **maximum defensible program cost per proxy asset**, not an
estimated actual inspection cost. The `12 customers per asset` denominator is
also a placeholder; real utility pole, span, feeder, or work-order data should
replace it.

The output labelled `evpi_usd` is an uncertainty-spread proxy: it converts the
difference between the mean and 10th percentile of total simulated
customer-hours to dollars. It is not a full decision-theoretic expected value
of perfect information, because it does not explicitly optimize a specified
action, action cost, and loss function under perfect versus imperfect
information. A formal EVPI analysis should be added before treating that value
as a decision ceiling.

## Scientific limits

- ERA5 can under-resolve convective gusts and local precipitation extremes at
  0.25-degree resolution.
- County aggregation hides feeder topology, asset age, vegetation-management
  practice, and utility operations.
- EAGLE-I records outage consequence rather than causal failure mechanism.
- Tree canopy is an exposure proxy, not a measure of maintainable vegetation
  risk.
- Restoration duration includes logistics, crews, and mutual assistance as well
  as weather impact.

The appropriate scientific claim is therefore: the project estimates
statistically calibrated, county-day outage-consequence risk conditioned on
atmospheric predictors and broad exposure proxies.
