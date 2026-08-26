# Storm-Driven Outage Risk and Forecast Value
## Michigan Distribution Network — Independent Project, 2026

A comprehensive probabilistic outage-risk forecasting system for electric distribution networks, combining historical outage data with meteorological predictors to quantify storm impacts and forecast value.

---

## Project Overview

This project develops an end-to-end probabilistic framework for predicting distribution-level outage risk driven by severe weather across Michigan's electric grid. By integrating customer-outage records from ORNL EAGLE-I with advanced meteorological and environmental data, the system delivers actionable probabilistic forecasts for operational decision-making.

---

## Key Achievements

- **24% CRPS improvement over climatological baseline** — Validated on held-out test year with storm-blocked cross-validation, demonstrating robust predictive performance across probabilistic skill metrics (CRPS, Brier, reliability, spread–skill).
- **Comprehensive modeling** — Three-stage hurdle model (occurrence, magnitude, duration) trained on 2018–2023 data across 83 Michigan counties.
- **High-resolution predictors** — Meteorological (ERA5 gust, precipitation, freezing rain, soil moisture) and land-cover (NLCD canopy exposure) data at granular spatial scales.
- **Operational decision framework** — Cost–loss economic value and break-even analyses quantifying resilience benefits per dollar invested.

---

## Technical Approach

### Data Integration
- **Outage records:** ORNL EAGLE-I customer-outage data across 83 Michigan counties (2018–2023)
- **Meteorological:** ERA5 gridded reanalysis (gust, precipitation, freezing rain, soil moisture)
- **Land cover:** NLCD canopy exposure and environmental covariates
- **Forecasts:** Bias-corrected GEFS ensemble system (day-1 through day-5 lead times)

### Methodology
1. **Three-stage hurdle model:**
   - **Stage 1:** Binary logistic regression for event occurrence
   - **Stage 2:** Distributional gradient boosting for outage magnitude
   - **Stage 3:** Weibull accelerated failure time (AFT) survival regression for restoration duration (right-censored)

2. **Validation strategy:**
   - Storm-blocked cross-validation preventing data leakage
   - Probabilistic metrics: CRPS, Brier skill, reliability diagrams, spread–skill analysis
   - Baseline comparisons: climatology and operational threshold-rule baselines

3. **Uncertainty quantification:**
   - Meteorological uncertainty: ensemble forecast spread
   - Parametric uncertainty: model credible intervals
   - Decomposition of total forecast uncertainty

### Case Studies
- Day-1 to day-5 probabilistic outage forecasts for two major 2023 weather events
- Explicit separation of meteorological vs. model-based uncertainty sources

---

## Model Performance

| Metric | Performance | Baseline |
|--------|-------------|----------|
| **CRPS Improvement** | 24% over climatology | — |
| **Brier Skill** | Validated on held-out year | Above climatology |
| **Reliability** | Quantified in reliability diagrams | Well-calibrated |
| **Spread–Skill** | Balanced ensemble spread | Robust correlation |

---

## Operational Decision Value

The framework quantifies resilience benefits through:
- **Cost–loss relative economic value** — Trade-off between prevention/mitigation costs and avoided outage losses
- **Break-even inspection cost analysis** — Optimal decision thresholds for pre-storm grid hardening and personnel positioning
- **Risk reduction per dollar** — Transparent economic justification for resilience investments

---

## Project Structure

```
storm-outage-risk-mi/
├── README.md                              # This file
├── Makefile                               # Build and workflow automation
├── phase1-smoke-test-spec.md              # Phase 1 validation spec
├── storm-outage-risk-project-spec.md      # Full project specification
│
├── config/                                # Configuration files
│   ├── phase1.yaml                        # Phase 1 experiment config
│   └── region.yaml                        # Regional/domain definitions
│
├── data/
│   ├── raw/                               # Original EAGLE-I, ERA5, NLCD
│   ├── interim/                           # Processed intermediate datasets
│   └── processed/                         # Final modeling datasets
│
├── src/
│   └── common/                            # Shared utilities and config loading
│       ├── __init__.py
│       ├── config.py                      # Configuration management
│       ├── _bootstrap.py                  # Initialization utilities
│
├── notebooks/                             # Jupyter notebooks (EDA, results)
├── models/                                # Trained model artifacts
├── figures/                               # Output plots and visualizations
├── logs/                                  # Experiment logs
├── tests/                                 # Unit and integration tests
│
├── env/
│   ├── environment.yml                    # Conda environment spec
│   └── requirements.txt                   # pip requirements
│
└── .gitignore                             # Git exclusions
```

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- conda or pip

### Environment Setup

**Option 1: Conda**
```bash
conda env create -f env/environment.yml
conda activate storm-outage-risk
```

**Option 2: pip**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r env/requirements.txt
```

---

## Quick Start

1. **Load configuration:**
   ```python
   from src.common.config import load_config
   config = load_config("config/phase1.yaml")
   ```

2. **Run phase 1 validation:**
   ```bash
   make validate-phase1
   ```

3. **Generate forecasts:**
   - See `notebooks/` for example forecast workflows
   - Day-1 through day-5 probabilistic predictions with uncertainty quantification

---

## Key References

- **ORNL EAGLE-I:** Customer outage records  
- **ERA5:** Copernicus Climate Data Store (Hersbach et al., 2020)
- **GEFS:** NOAA Global Ensemble Forecast System  
- **NLCD:** USGS National Land Cover Database

---

## Results & Outputs

- Probabilistic outage forecasts (day-1 to day-5)
- Uncertainty decomposition (meteorological vs. parametric)
- Economic value metrics for operational decision-making
- Visualization of forecast performance and reliability

---

## Author

**Fahad Abdullaah**  
George Mason University, 2026

---

## Questions or Feedback?

For questions about methodology, data, or results, please refer to the full project specification:  
`storm-outage-risk-project-spec.md`
