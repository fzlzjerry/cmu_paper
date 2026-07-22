SHELL := /usr/bin/bash
.SHELLFLAGS := --noprofile --norc -eu -o pipefail -c

unexport BASH_ENV
unexport ENV
unexport LD_LIBRARY_PATH
unexport LD_PRELOAD
unexport PYTHONHOME
unexport PYTHONPATH

.PHONY: preflight preflight-unit

PHASE2_PYTHON := /usr/bin/python3
PHASE2_ENV := PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=src
PHASE2_VALIDATE := $(PHASE2_ENV) $(PHASE2_PYTHON) scripts/validate_phase2.py
PHASE2_CLI := $(PHASE2_ENV) $(PHASE2_PYTHON) -m kvbench

.PHONY: bootstrap test checks format-check lint typecheck config-check
.PHONY: provenance-check scope-check immutable-check package-lock-check
.PHONY: test-cuda test-graph smoke pilot full-scan profile-subset
.PHONY: fit figures reproduce

preflight:
	@/usr/bin/env -i PATH=/usr/bin:/bin LC_ALL=C LANG=C TZ=UTC \
		/usr/bin/bash --noprofile --norc scripts/preflight.sh

preflight-unit:
	@PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tests/unit -p 'test_preflight_unit.py' -v

bootstrap:
	@test -x $(PHASE2_PYTHON)
	@$(PHASE2_ENV) $(PHASE2_PYTHON) -c 'import sys; assert sys.version_info >= (3, 11); import kvbench'
	@$(PHASE2_VALIDATE) package-lock
	@echo '{"status":"ready","action":"verified_only","installed":false}'

format-check:
	@$(PHASE2_VALIDATE) format

lint:
	@$(PHASE2_VALIDATE) lint

typecheck:
	@$(PHASE2_VALIDATE) annotations

config-check:
	@$(PHASE2_VALIDATE) configs

provenance-check:
	@$(PHASE2_VALIDATE) provenance

scope-check:
	@$(PHASE2_VALIDATE) scope

immutable-check:
	@$(PHASE2_VALIDATE) immutable

package-lock-check:
	@$(PHASE2_VALIDATE) package-lock

checks: format-check lint typecheck config-check provenance-check scope-check immutable-check package-lock-check

test: checks
	@$(PHASE2_ENV) $(PHASE2_PYTHON) -m unittest discover -s tests/schema -p 'test_*.py' -v
	@$(PHASE2_ENV) $(PHASE2_PYTHON) -m unittest discover -s tests/unit -p 'test_phase2_*.py' -v

test-cuda:
	@echo '{"error":"phase_not_implemented","target":"test-cuda","phase":"3+"}' >&2
	@exit 2

test-graph:
	@echo '{"error":"phase_not_implemented","target":"test-graph","phase":"4+"}' >&2
	@exit 2

smoke: export KVBENCH_METHOD := $(METHOD)
smoke:
	@$(PHASE2_VALIDATE) method
	@$(PHASE2_CLI) run --plan configs/plans/smoke.yaml --dry-run
	@echo '{"status":"validation_only","target":"smoke","requested_method":"$(METHOD)","plan_scope":"all_methods","timing_collected":false}'

pilot:
	@$(PHASE2_CLI) run --plan configs/plans/pilot.yaml --dry-run
	@echo '{"status":"validation_only","target":"pilot","timing_collected":false}'

full-scan:
	@$(PHASE2_CLI) run --plan configs/plans/full_scan.yaml --dry-run
	@echo '{"status":"validation_only","target":"full-scan","timing_collected":false}'

profile-subset:
	@$(PHASE2_CLI) run --plan configs/plans/profiler_subset.yaml --dry-run
	@echo '{"status":"validation_only","target":"profile-subset","profiler_executed":false}'

fit:
	@echo '{"error":"phase_not_implemented","target":"fit","phase":"17"}' >&2
	@exit 2

figures:
	@echo '{"error":"phase_not_implemented","target":"figures","phase":"17+"}' >&2
	@exit 2

reproduce: export KVBENCH_RUN_ID := $(RUN_ID)
reproduce:
	@$(PHASE2_VALIDATE) run-id
	@echo '{"error":"phase_not_implemented","target":"reproduce","phase":"18"}' >&2
	@exit 2
