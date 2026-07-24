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
PHASE3_PYTHON := .venv/bin/python
PHASE3_SITE := $(CURDIR)/.phase3/site-packages
PHASE3_ENV := /usr/bin/env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=$(PHASE3_SITE):src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
PHASE3_CLI := $(PHASE3_ENV) $(PHASE3_PYTHON) -m kvbench
PHASE3_REMEDIATION_UNIT_TESTS := tests.unit.test_allocation_attribution tests.unit.test_gqa_device_dispatch tests.unit.test_gqa_taxonomy tests.unit.test_process_supervision
TURBOQUANT_REFERENCE_VENV := .reference/turboquant-v0.25.1
TURBOQUANT_REFERENCE_PYTHON := $(TURBOQUANT_REFERENCE_VENV)/bin/python
TURBOQUANT_REFERENCE_SOURCE := .reference/vllm-source-v0.25.1
TURBOQUANT_REFERENCE_ENV := /usr/bin/env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=$(CURDIR)/src:$(CURDIR) CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_CUDA_ARCH_LIST=12.0 CUDAARCHS=120 CMAKE_CUDA_ARCHITECTURES=120 VLLM_NO_USAGE_STATS=1 HF_HUB_DISABLE_TELEMETRY=1
MEASUREMENT_IMAGE ?= kvbench-measurement:phase6a
MEASUREMENT_IMAGE_CONFIG_DIGEST ?=
MEASUREMENT_BASE_IMAGE_DIGEST := sha256:0eee3094c71518ad31d011a594ae6ed6de72959ee07e318cb31cffe71690e90c
MEASUREMENT_GPU_UUID := GPU-75bd273e-6b20-0d22-1b0b-5fbb6fb0025b
R2_ARTIFACT := $(PHASE2_ENV) $(PHASE2_PYTHON) scripts/r2_artifact.py

.PHONY: bootstrap bootstrap-phase3 test checks format-check lint hot-path-check typecheck config-check
.PHONY: provenance-check scope-check immutable-check package-lock-check
.PHONY: phase3-package-lock-check test-cuda test-graph test-allocation
.PHONY: smoke pilot full-scan profile-subset
.PHONY: fit figures reproduce
.PHONY: reference-turboquant validate-reference-turboquant
.PHONY: measurement-container verify-measurement-container preflight-container
.PHONY: publish-artifact-r2 verify-artifact-r2
.PHONY: phase6a-source-safety

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

bootstrap-phase3:
	@$(PHASE3_ENV) $(PHASE3_PYTHON) scripts/bootstrap_phase3.py verify

format-check:
	@$(PHASE2_VALIDATE) format

lint:
	@$(PHASE2_VALIDATE) lint

hot-path-check:
	@$(PHASE2_VALIDATE) hot-path

typecheck:
	@$(PHASE2_VALIDATE) annotations
	@$(PHASE2_VALIDATE) phase3-annotations

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

phase3-package-lock-check:
	@$(PHASE2_VALIDATE) phase3-package-lock

checks: format-check lint hot-path-check typecheck config-check provenance-check scope-check immutable-check package-lock-check phase3-package-lock-check

test: checks
	@$(PHASE2_ENV) $(PHASE2_PYTHON) -m unittest discover -s tests/schema -p 'test_*.py' -v
	@$(PHASE2_ENV) $(PHASE2_PYTHON) -m unittest discover -s tests/unit -p 'test_phase2_*.py' -v
	@$(PHASE3_ENV) $(PHASE3_PYTHON) -m unittest discover -s tests/unit -p 'test_phase3_*.py' -v
	@$(PHASE3_ENV) $(PHASE3_PYTHON) -m unittest $(PHASE3_REMEDIATION_UNIT_TESTS) -v
	@$(PHASE3_ENV) $(PHASE3_PYTHON) -m unittest discover -s tests/unit -p 'test_phase4_*.py' -v
	@$(PHASE3_ENV) $(PHASE3_PYTHON) -m unittest discover -s tests/unit -p 'test_phase5_*.py' -v
	@$(PHASE3_ENV) $(PHASE3_PYTHON) -m unittest tests.unit.test_measurement_container tests.unit.test_phase6a_governance tests.unit.test_preflight_unit tests.unit.test_r2_artifact -v
	@$(PHASE2_VALIDATE) immutable

test-cuda: phase3-package-lock-check immutable-check
	@set +e; $(PHASE3_ENV) $(PHASE3_PYTHON) -m unittest discover -s tests/cuda -p 'test_phase*_*.py' -v; test_status=$$?; $(PHASE2_VALIDATE) immutable; immutable_status=$$?; if (( test_status != 0 )); then exit $$test_status; fi; exit $$immutable_status

test-graph: phase3-package-lock-check immutable-check
	@set +e; $(PHASE3_ENV) $(PHASE3_PYTHON) -m unittest discover -s tests/graph -p 'test_phase*_*.py' -v; test_status=$$?; $(PHASE2_VALIDATE) immutable; immutable_status=$$?; if (( test_status != 0 )); then exit $$test_status; fi; exit $$immutable_status

test-allocation: test-cuda

reference-turboquant:
	@$(PHASE2_PYTHON) reference/turboquant/bootstrap_environment.py prepare --venv $(TURBOQUANT_REFERENCE_VENV) --source $(TURBOQUANT_REFERENCE_SOURCE)
	@$(TURBOQUANT_REFERENCE_ENV) $(TURBOQUANT_REFERENCE_PYTHON) reference/turboquant/generate_fixtures.py --venv $(TURBOQUANT_REFERENCE_VENV) --source $(TURBOQUANT_REFERENCE_SOURCE)
	@$(TURBOQUANT_REFERENCE_ENV) $(PHASE2_PYTHON) reference/turboquant/validate_fixtures.py

validate-reference-turboquant:
	@$(TURBOQUANT_REFERENCE_ENV) $(PHASE2_PYTHON) reference/turboquant/validate_fixtures.py

phase6a-source-safety:
	@test -z "$$(git status --porcelain=v1 --untracked-files=all)" || { echo '{"status":"BLOCKED","reason":"source_tree_not_clean"}' >&2; exit 2; }
	@if git ls-files --error-unmatch .env >/dev/null 2>&1; then echo '{"status":"FAIL","reason":"env_tracked"}' >&2; exit 2; fi
	@git check-ignore -q .env || { echo '{"status":"FAIL","reason":"env_not_ignored"}' >&2; exit 2; }
	@if test -e .env || test -L .env; then \
		test -f .env && test ! -L .env || { echo '{"status":"FAIL","reason":"env_unsafe"}' >&2; exit 2; }; \
		mode="$$(stat -c '%a' .env)"; (( (8#$$mode & 077) == 0 )) || { echo '{"status":"FAIL","reason":"env_permissions_too_broad"}' >&2; exit 2; }; \
	fi
	@grep -Fxq '.env' .dockerignore && grep -Fxq '.env.*' .dockerignore || { echo '{"status":"FAIL","reason":"env_not_excluded_from_docker"}' >&2; exit 2; }

measurement-container: phase6a-source-safety
	@command -v docker >/dev/null || { echo '{"status":"BLOCKED","reason":"docker_cli_unavailable"}' >&2; exit 2; }
	@docker info >/dev/null || { echo '{"status":"BLOCKED","reason":"docker_daemon_unavailable"}' >&2; exit 2; }
	@task_root="$$(mktemp -d /tmp/kvbench-phase6a-build.XXXXXX)"; \
		trap 'chmod -R u+w "$$task_root" 2>/dev/null || true; rm -rf -- "$$task_root"' EXIT; \
		mkdir "$$task_root/context"; \
		head="$$(git rev-parse HEAD)"; \
		git archive --format=tar "$$head" | tar -xf - -C "$$task_root/context"; \
		test ! -e "$$task_root/context/.env"; \
		DOCKER_BUILDKIT=1 docker build --pull --platform linux/amd64 \
			--iidfile "$$task_root/image.id" \
			--label "org.opencontainers.image.revision=$$head" \
			--tag "$(MEASUREMENT_IMAGE)" \
			--file "$$task_root/context/docker/measurement.Dockerfile" \
			"$$task_root/context"; \
		image_id="$$(tr -d '\r\n' < "$$task_root/image.id")"; \
		[[ "$$image_id" =~ ^sha256:[0-9a-f]{64}$$ ]] || { echo '{"status":"FAIL","reason":"invalid_image_config_digest"}' >&2; exit 2; }; \
		docker image inspect "$$image_id" > "$$task_root/image-inspect.json"; \
		docker history --no-trunc --format '{{json .}}' "$$image_id" > "$$task_root/image-history.jsonl"; \
		PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c 'import os,sys; from pathlib import Path; from preflight.run_preflight import configured_secret_value_names,validate_container_image_inspect; raw=Path(sys.argv[1]).read_bytes(); _,result=validate_container_image_inspect(Path(sys.argv[1]),expected_image_config_digest=sys.argv[2],expected_base_image_digest=sys.argv[3],expected_revision=sys.argv[4],secret_environment=os.environ); history=Path(sys.argv[5]).read_bytes(); raise SystemExit(0 if result["status"] == "PASS" and not configured_secret_value_names(history,os.environ) else 2)' "$$task_root/image-inspect.json" "$$image_id" "$(MEASUREMENT_BASE_IMAGE_DIGEST)" "$$head" "$$task_root/image-history.jsonl" || { echo '{"status":"FAIL","reason":"image_identity_or_secret_validation_failed"}' >&2; exit 2; }; \
		docker image save --output "$$task_root/image-save.tar" "$$image_id"; \
		PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c 'import os,sys; from pathlib import Path; from preflight.run_preflight import validate_docker_image_save_archive; result=validate_docker_image_save_archive(Path(sys.argv[1]),secret_environment=os.environ); raise SystemExit(0 if result["status"] == "PASS" else 2)' "$$task_root/image-save.tar" || { echo '{"status":"FAIL","reason":"image_layer_secret_validation_failed"}' >&2; exit 2; }; \
		printf '{"status":"BUILT_UNCERTIFIED","image_reference":"%s","image_config_digest":"%s","base_image_digest":"%s"}\n' "$(MEASUREMENT_IMAGE)" "$$image_id" "$(MEASUREMENT_BASE_IMAGE_DIGEST)"

verify-measurement-container: phase6a-source-safety
	@PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m unittest tests.unit.test_measurement_container -v
	@[[ "$(MEASUREMENT_IMAGE_CONFIG_DIGEST)" =~ ^sha256:[0-9a-f]{64}$$ ]] || { echo '{"status":"BLOCKED","reason":"MEASUREMENT_IMAGE_CONFIG_DIGEST_required"}' >&2; exit 2; }
	@test -f preflight/measurement-container-system-packages.expected.json && test ! -L preflight/measurement-container-system-packages.expected.json && git ls-files --error-unmatch preflight/measurement-container-system-packages.expected.json >/dev/null || { echo '{"status":"BLOCKED","reason":"expected_container_system_lock_absent"}' >&2; exit 2; }
	@command -v docker >/dev/null || { echo '{"status":"BLOCKED","reason":"docker_cli_unavailable"}' >&2; exit 2; }
	@docker info >/dev/null || { echo '{"status":"BLOCKED","reason":"docker_daemon_unavailable"}' >&2; exit 2; }
	@task_root="$$(mktemp -d /tmp/kvbench-phase6a-verify.XXXXXX)"; \
		trap 'chmod -R u+w "$$task_root" 2>/dev/null || true; rm -rf -- "$$task_root"' EXIT; \
		head="$$(git rev-parse HEAD)"; \
		image_id="$(MEASUREMENT_IMAGE_CONFIG_DIGEST)"; \
		docker image inspect "$$image_id" > "$$task_root/image-inspect.json"; \
		docker history --no-trunc --format '{{json .}}' "$$image_id" > "$$task_root/image-history.jsonl"; \
		PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c 'import os,sys; from pathlib import Path; from preflight.run_preflight import configured_secret_value_names,validate_container_image_inspect; _,result=validate_container_image_inspect(Path(sys.argv[1]),expected_image_config_digest=sys.argv[2],expected_base_image_digest=sys.argv[3],expected_revision=sys.argv[4],secret_environment=os.environ); history=Path(sys.argv[5]).read_bytes(); raise SystemExit(0 if result["status"] == "PASS" and not configured_secret_value_names(history,os.environ) else 2)' "$$task_root/image-inspect.json" "$$image_id" "$(MEASUREMENT_BASE_IMAGE_DIGEST)" "$$head" "$$task_root/image-history.jsonl" || { echo '{"status":"FAIL","reason":"image_identity_or_secret_validation_failed"}' >&2; exit 2; }; \
		docker image save --output "$$task_root/image-save.tar" "$$image_id"; \
		PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c 'import os,sys; from pathlib import Path; from preflight.run_preflight import validate_docker_image_save_archive; result=validate_docker_image_save_archive(Path(sys.argv[1]),secret_environment=os.environ); raise SystemExit(0 if result["status"] == "PASS" else 2)' "$$task_root/image-save.tar" || { echo '{"status":"FAIL","reason":"image_layer_secret_validation_failed"}' >&2; exit 2; }; \
		docker run --rm --network=none --gpus "device=$(MEASUREMENT_GPU_UUID)" --entrypoint /usr/bin/nvidia-smi "$$image_id" --query-gpu=uuid,name,compute_cap --format=csv,noheader,nounits > "$$task_root/gpu.csv"; \
		PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c 'import csv,sys; from pathlib import Path; rows=list(csv.reader(Path(sys.argv[1]).read_text().splitlines())); expected=[sys.argv[2],"NVIDIA RTX PRO 6000 Blackwell Workstation Edition","12.0"]; raise SystemExit(0 if len(rows) == 1 and [item.strip() for item in rows[0]] == expected else 2)' "$$task_root/gpu.csv" "$(MEASUREMENT_GPU_UUID)" || { echo '{"status":"FAIL","reason":"exact_gpu_runtime_probe_failed"}' >&2; exit 2; }; \
		printf '{"status":"IDENTITY_AND_RUNTIME_READY_NOT_CERTIFIED","image_config_digest":"%s","base_image_digest":"%s"}\n' "$$image_id" "$(MEASUREMENT_BASE_IMAGE_DIGEST)"

preflight-container: verify-measurement-container
	@task_root="$$(mktemp -d /tmp/kvbench-phase6a-container.XXXXXX)"; \
		cid=""; preserve=1; \
		cleanup() { if [[ -n "$$cid" ]]; then docker rm -f "$$cid" >/dev/null 2>&1 || true; fi; if (( preserve == 0 )); then chmod -R u+w "$$task_root" 2>/dev/null || true; rm -rf -- "$$task_root"; else printf '{"status":"FAILED_LAUNCH_EVIDENCE_PRESERVED","path":"%s"}\n' "$$task_root" >&2; fi; }; \
		trap cleanup EXIT; \
		head="$$(git rev-parse HEAD)"; \
		image_id="$(MEASUREMENT_IMAGE_CONFIG_DIGEST)"; \
		git clone --quiet --no-local --no-checkout "$(CURDIR)" "$$task_root/source"; \
		git -C "$$task_root/source" checkout --quiet --detach "$$head"; \
		git -C "$$task_root/source" remote remove origin; \
		test "$$(git -C "$$task_root/source" rev-parse HEAD)" = "$$head"; \
		test -z "$$(git -C "$$task_root/source" status --porcelain=v1 --untracked-files=all)"; \
		test ! -e "$$task_root/source/.env"; \
		mkdir -p "$$task_root/source/artifacts/phase6a/container_g0"; \
		mkdir -p "$(CURDIR)/artifacts/phase6a/container_g0"; \
		docker image inspect "$$image_id" > "$$task_root/image-inspect.json"; \
		touch "$$task_root/runtime-inspect.json"; \
		chmod 0444 "$$task_root/image-inspect.json"; \
		cid="$$(docker create --read-only --network=none --pid=host \
			--gpus "device=$(MEASUREMENT_GPU_UUID)" \
			--tmpfs /tmp:rw,nosuid,nodev,size=8g \
			--tmpfs /root:rw,nosuid,nodev,size=2g \
			--mount "type=bind,src=$$task_root/source,dst=/workspace,readonly" \
			--mount "type=bind,src=$(CURDIR)/artifacts/phase6a/container_g0,dst=/workspace/artifacts/phase6a/container_g0" \
			--mount "type=bind,src=$$task_root/image-inspect.json,dst=/run/kvbench/image-inspect.json,readonly" \
			--mount "type=bind,src=$$task_root/runtime-inspect.json,dst=/run/kvbench/runtime-inspect.json,readonly" \
			--workdir /workspace "$$image_id" \
			/opt/kvbench/.venv/bin/python preflight/run_preflight.py \
			--measurement-container \
			--container-image-reference "$(MEASUREMENT_IMAGE)" \
			--container-image-config-digest "$$image_id" \
			--container-base-image-digest "$(MEASUREMENT_BASE_IMAGE_DIGEST)" \
			--container-image-inspect-json /run/kvbench/image-inspect.json \
			--container-runtime-inspect-json /run/kvbench/runtime-inspect.json)"; \
		[[ "$$cid" =~ ^[0-9a-f]{64}$$ ]] || { echo '{"status":"FAIL","reason":"invalid_runtime_container_id"}' >&2; exit 2; }; \
		chmod 0644 "$$task_root/runtime-inspect.json"; \
		docker container inspect "$$cid" > "$$task_root/runtime-inspect.json"; \
		chmod 0444 "$$task_root/runtime-inspect.json"; \
		PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c 'import os,sys; from pathlib import Path; from preflight.run_preflight import validate_container_runtime_inspect; _,result=validate_container_runtime_inspect(Path(sys.argv[1]),expected_image_config_digest=sys.argv[2],expected_hostname=sys.argv[3][:12],secret_environment=os.environ); raise SystemExit(0 if result["status"] == "PASS" else 2)' "$$task_root/runtime-inspect.json" "$$image_id" "$$cid" || { echo '{"status":"FAIL","reason":"runtime_identity_or_secret_validation_failed"}' >&2; exit 2; }; \
		docker start --attach "$$cid"; \
		preserve=0

publish-artifact-r2:
	@test -n "$(ARTIFACT)" || { echo '{"status":"FAIL","reason":"ARTIFACT_required"}' >&2; exit 2; }
	@$(R2_ARTIFACT) publish "$(ARTIFACT)"

verify-artifact-r2:
	@test -n "$(ROOT_SHA256)" || { echo '{"status":"FAIL","reason":"ROOT_SHA256_required"}' >&2; exit 2; }
	@[[ "$(ROOT_SHA256)" =~ ^[0-9a-f]{64}$$ ]] || { echo '{"status":"FAIL","reason":"ROOT_SHA256_invalid"}' >&2; exit 2; }
	@$(R2_ARTIFACT) verify "$(ROOT_SHA256)"

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
