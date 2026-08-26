# =============================================================================
# storm-outage-risk-mi
#   make phase1-synthetic   full pipeline on generated data -- no credentials
#   make phase1             full pipeline on real data (needs ~/.cdsapirc)
# =============================================================================
PY      ?= python
CFG     ?= config/region.yaml
P1      ?= config/phase1.yaml
SRC      = src

.DEFAULT_GOAL := help
.PHONY: help env env-pip env-lock doctor window fetch weather events features models compose forecast \
        value phase1 phase1-synthetic phase1-diff gates gates-synthetic test lint \
        clean-phase1 clean-synthetic era5-only

help:
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n",$$1,$$2}'

env: ## create the conda environment (ranges; then run `make env-lock`)
	mamba env create -f env/environment.yml || conda env create -f env/environment.yml

env-pip: ## alternative: a uv/pip virtualenv from exact pins
	uv venv --python 3.11 .venv && uv pip install -r env/requirements.txt

env-lock: ## capture the exact solve -- THIS is the reproducible artifact
	conda env export --no-builds | grep -v '^prefix:' > env/environment.lock.yml
	@echo "wrote env/environment.lock.yml -- commit it alongside any results"

doctor: ## preflight a fresh machine: packages, ~/.cdsapirc, network, disk
	@$(PY) $(SRC)/doctor.py

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
	pytest -q tests/

lint: ; ruff check $(SRC) tests

clean-phase1: ## section 9.3: delete every contaminated phase 1 artifact
	rm -rf models/phase1_* figures/phase1_* data/processed/phase1_*
	@echo "phase 1 models and figures deleted. download cache in data/raw kept."

clean-synthetic: ## remove the generated stand-in data entirely
	rm -rf data/*/_synthetic models/_synthetic figures/_synthetic logs/_synthetic \
	       config/_phase1_synthetic.yaml
