# =============================================================================
# storm-outage-risk-mi
#   make phase1-synthetic   full pipeline on generated data -- no credentials
#   make phase1             full pipeline on real data (needs ~/.cdsapirc)
# =============================================================================
ENV_PREFIX := /panfs/ccds02/nobackup/people/afahad/envs/storm-outage-risk.
PY      := $(ENV_PREFIX)/bin/python

# On shared machines ~/.local/lib/pythonX.Y/site-packages shadows the conda env
# and you silently run a DIFFERENT scipy/numpy than the one you installed. That
# is how a pinned environment produces an unpinned failure.
export PYTHONNOUSERSITE := 1
CFG     ?= config/region.yaml
P1      ?= config/phase1.yaml
P2      ?= config/phase2.yaml
SRC      = src

.DEFAULT_GOAL := help
.PHONY: help env env-pip env-lock doctor doctor-phase2 window fetch weather events features models compose forecast \
        value phase1 phase1-synthetic phase1-diff gates gates-synthetic test lint \
        clean-phase1 clean-synthetic era5-only phase2-download phase2-download-outages \
        phase2-download-era5 phase2-download-gefs phase2-download-canopy phase2-build \
        phase2-train phase2 phase2-build-test phase2-test phase2-submit \
        phase2-compose phase2-forecast phase2-forecast-synthetic phase2-value \
        phase2-apply phase2-preflight

help:
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n",$$1,$$2}'

env: ## create the conda environment (ranges; then run `make env-lock`)
	mamba env create --prefix $(ENV_PREFIX) -f env/environment.yml || \
	  conda env create --prefix $(ENV_PREFIX) -f env/environment.yml

env-pip: ## alternative: a uv/pip virtualenv from exact pins
	uv venv --python 3.11 $(ENV_PREFIX) && \
	  uv pip install --python $(PY) -r env/requirements.txt

env-lock: ## capture the exact solve -- THIS is the reproducible artifact
	conda env export --prefix $(ENV_PREFIX) --no-builds | \
	  grep -v '^prefix:' > env/environment.lock.yml
	@echo "wrote env/environment.lock.yml -- commit it alongside any results"

doctor: ## preflight a fresh machine: packages, ~/.cdsapirc, network, disk
	@$(PY) $(SRC)/doctor.py --phase 1

doctor-phase2: ## same, but the Phase 2 lazy imports are REQUIRED, not optional
	@$(PY) $(SRC)/doctor.py --phase 2

# ---- individual steps -------------------------------------------------------
window: ## step 0: pick the 5-day window from data (spec section 3)
	$(PY) $(SRC)/select_window.py --config $(CFG) --phase1 $(P1) --write

fetch: ## step 1: EAGLE-I outages + MCC + TIGER counties
	$(PY) $(SRC)/01_fetch_outage.py --config $(CFG) --phase1 $(P1)

weather: ## step 2: ERA5 (start this FIRST -- CDS queue) + GEFS + canopy
	$(PY) $(SRC)/02_fetch_weather.py --config $(CFG) --phase1 $(P1)

era5-only: ## step 2a: queue the ERA5 request and nothing else
	$(PY) $(SRC)/02_fetch_weather.py --config $(CFG) --phase1 $(P1) --only era5

events:   ; $(PY) $(SRC)/03_build_events.py   --config $(CFG) --phase1 $(P1)
features: ; $(PY) $(SRC)/04_build_features.py --config $(CFG) --phase1 $(P1)
models:   ; $(PY) $(SRC)/05_fit_models.py     --config $(CFG) --phase1 $(P1)
compose:  ; $(PY) $(SRC)/06_compose_mc.py     --config $(CFG) --phase1 $(P1)
forecast: ; $(PY) $(SRC)/07_forecast_cases.py --config $(CFG) --phase1 $(P1)
value:    ; $(PY) $(SRC)/08_decision_value.py --config $(CFG) --phase1 $(P1)

# ---- end to end -------------------------------------------------------------
phase1: ## gate criterion 10: one command, raw -> decision number
	$(PY) $(SRC)/run_phase1.py --config $(CFG) --phase1 $(P1)

phase1-synthetic: ## same pipeline on generated data: proves plumbing with no downloads
	$(PY) $(SRC)/run_phase1.py --config $(CFG) --phase1 $(P1) --synthetic

gates: ## reprint the go/no-go table from the last real run
	$(PY) $(SRC)/run_phase1.py --config $(CFG) --phase1 $(P1) --report-only

gates-synthetic: ## reprint the go/no-go table from the last synthetic run
	$(PY) $(SRC)/run_phase1.py --config $(CFG) --phase1 $(P1) --report-only --synthetic

phase1-diff: ## section 9.1: every key phase1.yaml overrides, for the revert checklist
	@$(PY) -c "import yaml;a=yaml.safe_load(open('$(CFG)'));b=yaml.safe_load(open('$(P1)'));\
	[print(f'{k:26s} phase1={b[k]!r:28s} region={a.get(k,\"<unset>\")!r}') for k in b if a.get(k)!=b[k]]"

test: ## the assertion suite, promoted to pytest (section 9.5)
	$(PY) -m pytest -q tests/

lint: ; $(PY) -m ruff check $(SRC) tests

clean-phase1: ## section 9.3: delete every contaminated phase 1 artifact
	rm -rf models/phase1_* figures/phase1_* data/processed/phase1_*
	@echo "phase 1 models and figures deleted. download cache in data/raw kept."

clean-synthetic: ## remove the generated stand-in data entirely
	rm -rf data/*/_synthetic models/_synthetic figures/_synthetic logs/_synthetic \
	       config/_phase1_synthetic.yaml

# ---- Phase 2: full 2018-2023 study (CPU only) -------------------------------
phase2-download: ## all full-study inputs (large; prefer phase2-submit on Slurm)
	$(PY) src/phase2_download.py --config $(CFG) --phase2 $(P2) --only all

phase2-download-outages: ## EAGLE-I 2017 baseline buffer + 2018-2023 study years
	$(PY) src/phase2_download.py --config $(CFG) --phase2 $(P2) --only outages

phase2-download-era5: ## monthly ERA5: Michigan-sliced ARCO + 5 residual CDS fields
	$(PY) src/phase2_download.py --config $(CFG) --phase2 $(P2) --only era5

phase2-download-gefs: ## 2023 cases, day-5/-3/-2/-1, 31 members
	$(PY) src/phase2_download.py --config $(CFG) --phase2 $(P2) --only gefs

phase2-download-canopy: ## required static USFS/NLCD canopy layer
	$(PY) src/phase2_download.py --config $(CFG) --phase2 $(P2) --only canopy

phase2-build: ## build 2018-2022 features; does not open 2023 outcomes
	$(PY) src/phase2_build.py --config $(CFG) --phase2 $(P2) --through validation

phase2-train: ## fit 2018-2021, calibrate and validate on 2022
	$(PY) src/phase2_train.py --config $(CFG) --phase2 $(P2)

phase2: ## preprocess + train/validate; downloads must already be cached
	$(PY) src/run_phase2.py --config $(CFG) --phase2 $(P2)

# ---- Phase 2 application stages (spec sections 6.4, 8, 9) -------------------
# None of these touch the test year. They run against the frozen bundle and the
# validation-scope table, so they can be built and reviewed while the test year
# is still sealed.
phase2-compose: ## section 6.4: Monte Carlo composition of the three models
	$(PY) src/phase2_compose.py --config $(CFG) --phase2 $(P2)

phase2-forecast: ## section 8: GEFS case studies, bias correction, 31x100 realizations
	$(PY) src/phase2_forecast.py --config $(CFG) --phase2 $(P2)

phase2-forecast-synthetic: ## same plumbing on stand-in members; NO skill implied
	$(PY) src/phase2_forecast.py --config $(CFG) --phase2 $(P2) --synthetic-gefs

phase2-value: ## section 9: cost-loss value, break-even inspection cost, EVPI
	$(PY) src/phase2_value.py --config $(CFG) --phase2 $(P2)

phase2-apply: phase2-compose phase2-forecast phase2-value ## all three application stages

phase2-preflight: ## everything that must be green before submitting anything
	$(MAKE) doctor-phase2
	$(MAKE) lint
	$(MAKE) test

phase2-build-test: ## explicitly open/build the held-out test year
	$(PY) src/phase2_build.py --config $(CFG) --phase2 $(P2) --through test --acknowledge-test

phase2-test: ## score frozen model on the test year exactly once
	$(PY) src/phase2_train.py --config $(CFG) --phase2 $(P2) --evaluate-test

phase2-submit: ## sequential Slurm pipeline; each stage is watched and must succeed
	bash slurm/submit_phase2.sh
