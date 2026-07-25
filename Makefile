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
PHASE6_AUTHORIZED_IMAGE_CONFIG_DIGEST := sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e
R2_ARTIFACT := /usr/bin/env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=$(CURDIR):$(CURDIR)/src $(PHASE2_PYTHON) scripts/r2_artifact.py

.PHONY: bootstrap bootstrap-phase3 test checks format-check lint hot-path-check typecheck config-check
.PHONY: provenance-check scope-check immutable-check package-lock-check
.PHONY: phase3-package-lock-check test-cuda test-graph test-allocation
.PHONY: smoke pilot full-scan profile-subset
.PHONY: fit figures reproduce
.PHONY: reference-turboquant validate-reference-turboquant
.PHONY: measurement-container observe-measurement-container-lock
.PHONY: measurement-container-lock-review verify-measurement-container
.PHONY: preflight-container phase6a-bf16-container-parity
.PHONY: publish-artifact-r2 verify-artifact-r2
.PHONY: phase6a-source-safety admit-turboquant validate-admission-turboquant
.PHONY: remediate-b018-turboquant

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
	@$(PHASE3_ENV) $(PHASE3_PYTHON) -m unittest tests.unit.test_measurement_container tests.unit.test_phase6a_bf16_parity tests.unit.test_phase6a_governance tests.unit.test_preflight_unit tests.unit.test_r2_artifact -v
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

observe-measurement-container-lock: phase6a-source-safety
	@[[ "$(MEASUREMENT_IMAGE_CONFIG_DIGEST)" =~ ^sha256:[0-9a-f]{64}$$ ]] || { echo '{"status":"BLOCKED","reason":"MEASUREMENT_IMAGE_CONFIG_DIGEST_required"}' >&2; exit 2; }
	@test ! -e preflight/measurement-container-system-packages.lock.json && test ! -L preflight/measurement-container-system-packages.lock.json || { echo '{"status":"BLOCKED","reason":"observed_container_system_lock_exists"}' >&2; exit 2; }
	@test ! -e preflight/measurement-container-system-packages.expected.json && test ! -L preflight/measurement-container-system-packages.expected.json || { echo '{"status":"BLOCKED","reason":"expected_container_system_lock_exists"}' >&2; exit 2; }
	@command -v docker >/dev/null || { echo '{"status":"BLOCKED","reason":"docker_cli_unavailable"}' >&2; exit 2; }
	@docker info >/dev/null || { echo '{"status":"BLOCKED","reason":"docker_daemon_unavailable"}' >&2; exit 2; }
	@task_root="$$(mktemp -d /tmp/kvbench-phase6a-lock.XXXXXX)"; \
		trap 'chmod -R u+w "$$task_root" 2>/dev/null || true; rm -rf -- "$$task_root"' EXIT; \
		mkdir "$$task_root/source"; \
		head="$$(git rev-parse HEAD)"; \
		image_id="$(MEASUREMENT_IMAGE_CONFIG_DIGEST)"; \
		git archive --format=tar "$$head" | tar -xf - -C "$$task_root/source"; \
		test ! -e "$$task_root/source/.env"; \
		docker image inspect "$$image_id" > "$$task_root/image-inspect.json"; \
		docker history --no-trunc --format '{{json .}}' "$$image_id" > "$$task_root/image-history.jsonl"; \
		PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c 'import os,sys; from pathlib import Path; from preflight.run_preflight import configured_secret_value_names,validate_container_image_inspect; _,result=validate_container_image_inspect(Path(sys.argv[1]),expected_image_config_digest=sys.argv[2],expected_base_image_digest=sys.argv[3],expected_revision=sys.argv[4],secret_environment=os.environ); history=Path(sys.argv[5]).read_bytes(); raise SystemExit(0 if result["status"] == "PASS" and not configured_secret_value_names(history,os.environ) else 2)' "$$task_root/image-inspect.json" "$$image_id" "$(MEASUREMENT_BASE_IMAGE_DIGEST)" "$$head" "$$task_root/image-history.jsonl" || { echo '{"status":"FAIL","reason":"candidate_image_identity_or_secret_validation_failed"}' >&2; exit 2; }; \
		chmod 0444 "$$task_root/image-inspect.json"; \
		docker run --rm --read-only --network=none \
			--tmpfs /tmp:rw,nosuid,nodev,size=1g \
			--mount "type=bind,src=$$task_root/source,dst=/workspace,readonly" \
			--mount "type=bind,src=$$task_root/image-inspect.json,dst=/run/kvbench/image-inspect.json,readonly" \
			--workdir /workspace \
			--entrypoint /opt/kvbench/.venv/bin/python "$$image_id" \
			-m preflight.run_preflight \
			--observe-measurement-container-system-lock \
			--container-image-reference "$(MEASUREMENT_IMAGE)" \
			--container-image-config-digest "$$image_id" \
			--container-base-image-digest "$(MEASUREMENT_BASE_IMAGE_DIGEST)" \
			--container-image-inspect-json /run/kvbench/image-inspect.json \
			--container-source-revision "$$head" \
			> "$$task_root/observed.json"; \
		PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c 'import os,sys; from pathlib import Path; from preflight.run_preflight import configured_secret_value_names,json_bytes,load_measurement_container_system_lock,publish_atomic_exclusive; source=Path(sys.argv[1]); payload,result=load_measurement_container_system_lock(source,expected_image_config_digest=sys.argv[3],expected_source_revision=sys.argv[4]); raw=json_bytes(payload) if payload is not None else b""; valid=payload is not None and result["status"] == "PASS" and not configured_secret_value_names(raw,os.environ); valid and publish_atomic_exclusive(Path(sys.argv[2]),raw); raise SystemExit(0 if valid else 2)' "$$task_root/observed.json" "preflight/measurement-container-system-packages.lock.json" "$$image_id" "$$head" || { echo '{"status":"FAIL","reason":"observed_container_system_lock_invalid"}' >&2; exit 2; }; \
		printf '{"status":"OBSERVED_UNTRUSTED_REVIEW_REQUIRED","path":"preflight/measurement-container-system-packages.lock.json","candidate_image_config_digest":"%s","candidate_source_revision":"%s"}\n' "$$image_id" "$$head"

measurement-container-lock-review: phase6a-source-safety
	@for lock_path in preflight/measurement-container-system-packages.lock.json preflight/measurement-container-system-packages.expected.json; do \
		test -f "$$lock_path" && test ! -L "$$lock_path" && git ls-files --error-unmatch "$$lock_path" >/dev/null || { echo '{"status":"BLOCKED","reason":"reviewed_container_system_locks_absent_or_untracked"}' >&2; exit 2; }; \
	done
	@cmp -s preflight/measurement-container-system-packages.lock.json preflight/measurement-container-system-packages.expected.json || { echo '{"status":"BLOCKED","reason":"reviewed_container_system_locks_differ"}' >&2; exit 2; }
	@PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c 'import os; from pathlib import Path; from preflight.run_preflight import configured_secret_value_names,load_measurement_container_system_lock; paths=[Path("preflight/measurement-container-system-packages.lock.json"),Path("preflight/measurement-container-system-packages.expected.json")]; loaded=[load_measurement_container_system_lock(path) for path in paths]; raw=[path.read_bytes() for path in paths]; valid=all(payload is not None and result["status"] == "PASS" for payload,result in loaded) and raw[0] == raw[1] and all(not configured_secret_value_names(item,os.environ) for item in raw); raise SystemExit(0 if valid else 2)' || { echo '{"status":"FAIL","reason":"reviewed_container_system_lock_invalid"}' >&2; exit 2; }
	@printf '{"status":"REVIEWED_LOCK_BYTES_EQUAL","authority":"rebuild_and_exact_runtime_verification_still_required"}\n'

verify-measurement-container: measurement-container-lock-review
	@PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m unittest tests.unit.test_measurement_container -v
	@[[ "$(MEASUREMENT_IMAGE_CONFIG_DIGEST)" =~ ^sha256:[0-9a-f]{64}$$ ]] || { echo '{"status":"BLOCKED","reason":"MEASUREMENT_IMAGE_CONFIG_DIGEST_required"}' >&2; exit 2; }
	@command -v docker >/dev/null || { echo '{"status":"BLOCKED","reason":"docker_cli_unavailable"}' >&2; exit 2; }
	@docker info >/dev/null || { echo '{"status":"BLOCKED","reason":"docker_daemon_unavailable"}' >&2; exit 2; }
	@task_root="$$(mktemp -d /tmp/kvbench-phase6a-verify.XXXXXX)"; \
		trap 'chmod -R u+w "$$task_root" 2>/dev/null || true; rm -rf -- "$$task_root"' EXIT; \
		head="$$(git rev-parse HEAD)"; \
		image_id="$(MEASUREMENT_IMAGE_CONFIG_DIGEST)"; \
		docker image inspect "$$image_id" > "$$task_root/image-inspect.json"; \
		docker history --no-trunc --format '{{json .}}' "$$image_id" > "$$task_root/image-history.jsonl"; \
		PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c 'import os,re,sys; from pathlib import Path; from preflight.run_preflight import configured_secret_value_names,validate_container_image_inspect; _,result=validate_container_image_inspect(Path(sys.argv[1]),expected_image_config_digest=sys.argv[2],expected_base_image_digest=sys.argv[3],expected_revision=None,secret_environment=os.environ); history=Path(sys.argv[4]).read_bytes(); revision=result.get("source_revision"); valid=result["status"] == "PASS" and isinstance(revision,str) and re.fullmatch(r"[0-9a-f]{40}",revision) is not None and not configured_secret_value_names(history,os.environ); valid and Path(sys.argv[5]).write_text(revision+"\n",encoding="ascii"); raise SystemExit(0 if valid else 2)' "$$task_root/image-inspect.json" "$$image_id" "$(MEASUREMENT_BASE_IMAGE_DIGEST)" "$$task_root/image-history.jsonl" "$$task_root/image-revision" || { echo '{"status":"FAIL","reason":"image_identity_or_secret_validation_failed"}' >&2; exit 2; }; \
		image_revision="$$(tr -d '\r\n' < "$$task_root/image-revision")"; \
		git cat-file -e "$$image_revision^{commit}"; \
		git merge-base --is-ancestor "$$image_revision" "$$head"; \
		git diff --quiet "$$image_revision" "$$head" -- docker/measurement.Dockerfile preflight/requirements-e00.txt preflight/requirements-phase3.txt; \
		for lock_path in preflight/measurement-container-system-packages.lock.json preflight/measurement-container-system-packages.expected.json; do \
			git cat-file -e "$$image_revision:$$lock_path"; \
			test "$$(git rev-parse "$$image_revision:$$lock_path")" = "$$(git rev-parse "HEAD:$$lock_path")"; \
		done; \
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
		PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c 'import re,sys; from pathlib import Path; from preflight.run_preflight import validate_container_image_inspect; _,result=validate_container_image_inspect(Path(sys.argv[1]),expected_image_config_digest=sys.argv[2],expected_base_image_digest=sys.argv[3],expected_revision=None); revision=result.get("source_revision"); valid=result["status"] == "PASS" and isinstance(revision,str) and re.fullmatch(r"[0-9a-f]{40}",revision) is not None; valid and Path(sys.argv[4]).write_text(revision+"\n",encoding="ascii"); raise SystemExit(0 if valid else 2)' "$$task_root/image-inspect.json" "$$image_id" "$(MEASUREMENT_BASE_IMAGE_DIGEST)" "$$task_root/image-revision"; \
		image_revision="$$(tr -d '\r\n' < "$$task_root/image-revision")"; \
		git cat-file -e "$$image_revision^{commit}"; \
		git merge-base --is-ancestor "$$image_revision" "$$head"; \
		git diff --quiet "$$image_revision" "$$head" -- docker/measurement.Dockerfile preflight/requirements-e00.txt preflight/requirements-phase3.txt; \
		for lock_path in preflight/measurement-container-system-packages.lock.json preflight/measurement-container-system-packages.expected.json; do \
			git cat-file -e "$$image_revision:$$lock_path"; \
			test "$$(git rev-parse "$$image_revision:$$lock_path")" = "$$(git rev-parse "HEAD:$$lock_path")"; \
		done; \
		docker history --no-trunc --format '{{json .}}' "$$image_id" > "$$task_root/image-history.jsonl"; \
		docker image save --output "$$task_root/image-save.tar" "$$image_id"; \
		scan_status=0; \
		PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c 'import os,sys; from pathlib import Path; from preflight.run_preflight import configured_secret_value_names,json_bytes,sha256_bytes,validate_docker_image_save_archive; history=Path(sys.argv[1]).read_bytes(); history_secrets=configured_secret_value_names(history,os.environ); layers=validate_docker_image_save_archive(Path(sys.argv[2]),secret_environment=os.environ); record={"schema_version":"kvbench-measurement-container-build-scan-1.0.0","image_config_digest":sys.argv[3],"image_history":{"sha256":sha256_bytes(history),"size_bytes":len(history),"line_count":len(history.splitlines()),"configured_secret_variables":history_secrets},"image_layers":layers}; Path(sys.argv[4]).write_bytes(json_bytes(record)); valid=layers["status"] == "PASS" and not history_secrets and layers["model_weight_path_count"] == 0; raise SystemExit(0 if valid else 2)' "$$task_root/image-history.jsonl" "$$task_root/image-save.tar" "$$image_id" "$$task_root/image-build-scan.json" || scan_status=$$?; \
		rm -f -- "$$task_root/image-save.tar"; \
		if (( scan_status != 0 )); then rm -f -- "$$task_root/image-history.jsonl"; echo '{"status":"FAIL","reason":"image_build_scan_failed"}' >&2; exit 2; fi; \
		touch "$$task_root/runtime-inspect.json"; \
		chmod 0444 "$$task_root/image-inspect.json" "$$task_root/image-history.jsonl" "$$task_root/image-build-scan.json"; \
		cid="$$(docker create --read-only --network=none --pid=host \
			--gpus "device=$(MEASUREMENT_GPU_UUID)" \
			--tmpfs /tmp:rw,nosuid,nodev,size=8g \
			--tmpfs /root:rw,nosuid,nodev,size=2g \
			--mount "type=bind,src=$$task_root/source,dst=/workspace,readonly" \
			--mount "type=bind,src=$(CURDIR)/artifacts/phase6a/container_g0,dst=/workspace/artifacts/phase6a/container_g0" \
			--mount "type=bind,src=$$task_root/image-inspect.json,dst=/run/kvbench/image-inspect.json,readonly" \
			--mount "type=bind,src=$$task_root/runtime-inspect.json,dst=/run/kvbench/runtime-inspect.json,readonly" \
			--mount "type=bind,src=$$task_root/image-history.jsonl,dst=/run/kvbench/image-history.jsonl,readonly" \
			--mount "type=bind,src=$$task_root/image-build-scan.json,dst=/run/kvbench/image-build-scan.json,readonly" \
			--workdir /workspace "$$image_id" \
			/opt/kvbench/.venv/bin/python preflight/run_preflight.py \
			--measurement-container \
			--container-image-reference "$(MEASUREMENT_IMAGE)" \
			--container-image-config-digest "$$image_id" \
			--container-base-image-digest "$(MEASUREMENT_BASE_IMAGE_DIGEST)" \
			--container-image-inspect-json /run/kvbench/image-inspect.json \
			--container-runtime-inspect-json /run/kvbench/runtime-inspect.json \
			--container-image-history-jsonl /run/kvbench/image-history.jsonl \
			--container-image-build-scan-json /run/kvbench/image-build-scan.json \
			--container-source-revision "$$image_revision")"; \
		[[ "$$cid" =~ ^[0-9a-f]{64}$$ ]] || { echo '{"status":"FAIL","reason":"invalid_runtime_container_id"}' >&2; exit 2; }; \
		chmod 0644 "$$task_root/runtime-inspect.json"; \
		docker container inspect "$$cid" > "$$task_root/runtime-inspect.json"; \
		chmod 0444 "$$task_root/runtime-inspect.json"; \
		PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c 'import os,sys; from pathlib import Path; from preflight.run_preflight import validate_container_runtime_inspect; _,result=validate_container_runtime_inspect(Path(sys.argv[1]),expected_image_config_digest=sys.argv[2],expected_hostname=sys.argv[3][:12],secret_environment=os.environ); raise SystemExit(0 if result["status"] == "PASS" else 2)' "$$task_root/runtime-inspect.json" "$$image_id" "$$cid" || { echo '{"status":"FAIL","reason":"runtime_identity_or_secret_validation_failed"}' >&2; exit 2; }; \
		docker start --attach "$$cid"; \
		preserve=0

phase6a-bf16-container-parity: verify-measurement-container
	@[[ "$(MEASUREMENT_IMAGE_CONFIG_DIGEST)" =~ ^sha256:[0-9a-f]{64}$$ ]] || { echo '{"status":"BLOCKED","reason":"MEASUREMENT_IMAGE_CONFIG_DIGEST_required"}' >&2; exit 2; }
	@test -n "$(CONTAINER_G0_ARTIFACT)" || { echo '{"status":"BLOCKED","reason":"CONTAINER_G0_ARTIFACT_required"}' >&2; exit 2; }
	@task_root="$$(mktemp -d /tmp/kvbench-phase6a-parity.XXXXXX)"; \
		cid=""; preserve=1; \
		cleanup() { if [[ -n "$$cid" ]]; then docker rm -f "$$cid" >/dev/null 2>&1 || true; fi; if (( preserve == 0 )); then chmod -R u+w "$$task_root" 2>/dev/null || true; rm -rf -- "$$task_root"; else printf '{"status":"FAILED_PARITY_LAUNCH_EVIDENCE_PRESERVED","path":"%s"}\n' "$$task_root" >&2; fi; }; \
		trap cleanup EXIT; \
		head="$$(git rev-parse HEAD)"; image_id="$(MEASUREMENT_IMAGE_CONFIG_DIGEST)"; \
		git clone --quiet --no-local --no-checkout "$(CURDIR)" "$$task_root/source"; \
		git -C "$$task_root/source" checkout --quiet --detach "$$head"; \
		git -C "$$task_root/source" remote remove origin; \
		test "$$(git -C "$$task_root/source" rev-parse HEAD)" = "$$head"; \
		test -z "$$(git -C "$$task_root/source" status --porcelain=v1 --untracked-files=all)"; \
		test ! -e "$$task_root/source/.env"; \
		mkdir -p "$$task_root/source/artifacts/phase6a/bf16_parity"; \
		mkdir -p "$(CURDIR)/artifacts/phase6a/bf16_parity"; \
		g0="$$(realpath "$(CONTAINER_G0_ARTIFACT)")"; \
		model_root="$$(realpath /root/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct)"; \
		model_snapshot="$$model_root/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"; \
		test -d "$$g0" && test -d "$$model_root" && test -d "$$model_snapshot"; \
		overall=0; \
		for graph_mode in eager cuda_graph; do \
			cid="$$(docker create --read-only --network=none --pid=host \
				--gpus "device=$(MEASUREMENT_GPU_UUID)" \
				--tmpfs /tmp:rw,nosuid,nodev,size=8g \
				--tmpfs /root:rw,nosuid,nodev,size=2g \
				--mount "type=bind,src=$$task_root/source,dst=/workspace,readonly" \
				--mount "type=bind,src=$$task_root/source,dst=/home/rockrock/cmu_paper,readonly" \
				--mount "type=bind,src=$(CURDIR)/artifacts/phase6a/bf16_parity,dst=/workspace/artifacts/phase6a/bf16_parity" \
				--mount "type=bind,src=$$g0,dst=/run/kvbench/container-g0,readonly" \
				--mount "type=bind,src=$$model_root,dst=/root/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct,readonly" \
				--env PYTHONPATH=/opt/kvbench/.phase3/site-packages:/workspace/src:/workspace \
				--workdir /workspace "$$image_id" \
				/opt/kvbench/.venv/bin/python scripts/phase6a_bf16_parity.py \
				--graph-mode "$$graph_mode" \
				--image-reference "$(MEASUREMENT_IMAGE)" \
				--image-config-digest "$$image_id" \
				--container-g0-artifact /run/kvbench/container-g0)"; \
			[[ "$$cid" =~ ^[0-9a-f]{64}$$ ]]; \
			if ! docker start --attach "$$cid"; then overall=1; fi; \
			docker rm -f "$$cid" >/dev/null; cid=""; \
		done; \
		(( overall == 0 )) || exit 1; \
		preserve=0

remediate-b018-turboquant: verify-measurement-container
	@test "$(MEASUREMENT_IMAGE_CONFIG_DIGEST)" = "$(PHASE6_AUTHORIZED_IMAGE_CONFIG_DIGEST)" || { echo '{"status":"BLOCKED","reason":"authorized_image_digest_required"}' >&2; exit 2; }
	@test -z "$$(git status --porcelain=v1 --untracked-files=all)" || { echo '{"status":"BLOCKED","reason":"source_tree_not_clean"}' >&2; exit 2; }
	@task_root="$$(mktemp -d /tmp/kvbench-phase6-b018.XXXXXX)"; \
		cid=""; preserve=1; \
		cleanup() { if [[ -n "$$cid" ]]; then docker rm -f "$$cid" >/dev/null 2>&1 || true; fi; if (( preserve == 0 )); then chmod -R u+w "$$task_root" 2>/dev/null || true; rm -rf -- "$$task_root"; else printf '{"status":"FAILED_B018_LAUNCH_EVIDENCE_PRESERVED","path":"%s"}\n' "$$task_root" >&2; fi; }; \
		trap cleanup EXIT; \
		head="$$(git rev-parse HEAD)"; image_id="$(MEASUREMENT_IMAGE_CONFIG_DIGEST)"; \
		git clone --quiet --no-local --no-checkout "$(CURDIR)" "$$task_root/source"; \
		git -C "$$task_root/source" checkout --quiet --detach "$$head"; \
		git -C "$$task_root/source" remote remove origin; \
		test "$$(git -C "$$task_root/source" rev-parse HEAD)" = "$$head"; \
		test -z "$$(git -C "$$task_root/source" status --porcelain=v1 --untracked-files=all)"; \
		mkdir -p "$$task_root/source/artifacts/phase6" "$(CURDIR)/artifacts/phase6"; \
		test ! -e "$$task_root/source/.env"; \
		cid="$$(docker create --read-only --network=none --pid=host \
			--gpus "device=$(MEASUREMENT_GPU_UUID)" \
			--tmpfs /tmp:rw,nosuid,nodev,size=8g \
			--tmpfs /root:rw,exec,nosuid,nodev,size=8g \
			--mount "type=bind,src=$$task_root/source,dst=/home/rockrock/cmu_paper,readonly" \
			--mount "type=bind,src=$(CURDIR)/artifacts/phase6,dst=/home/rockrock/cmu_paper/artifacts/phase6" \
			--env PYTHONDONTWRITEBYTECODE=1 \
			--env PYTHONNOUSERSITE=1 \
			--env PYTHONPATH=/opt/kvbench/.phase3/site-packages:/home/rockrock/cmu_paper/src:/home/rockrock/cmu_paper \
			--env HF_HUB_OFFLINE=1 \
			--env TRANSFORMERS_OFFLINE=1 \
			--env HF_HUB_DISABLE_TELEMETRY=1 \
			--env TOKENIZERS_PARALLELISM=false \
			--env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
			--env TRITON_CACHE_DIR=/root/.triton \
			--env KVBENCH_AUTHORIZED_IMAGE_DIGEST="$$image_id" \
			--env KVBENCH_EXECUTION_ENVIRONMENT=measurement_container \
			--workdir /home/rockrock/cmu_paper \
			--entrypoint /opt/kvbench/.venv/bin/python "$$image_id" \
			-m scripts.phase6_turboquant_admission \
			--b018-sanitizer-only)"; \
		[[ "$$cid" =~ ^[0-9a-f]{64}$$ ]]; \
		docker start --attach "$$cid"; \
		docker rm -f "$$cid" >/dev/null; cid=""; \
		preserve=0

admit-turboquant: verify-measurement-container
	@test "$(MEASUREMENT_IMAGE_CONFIG_DIGEST)" = "$(PHASE6_AUTHORIZED_IMAGE_CONFIG_DIGEST)" || { echo '{"status":"BLOCKED","reason":"authorized_image_digest_required"}' >&2; exit 2; }
	@test -z "$$(git status --porcelain=v1 --untracked-files=all)" || { echo '{"status":"BLOCKED","reason":"source_tree_not_clean"}' >&2; exit 2; }
	@task_root="$$(mktemp -d /tmp/kvbench-phase6-admission.XXXXXX)"; \
		cid=""; preserve=1; \
		cleanup() { if [[ -n "$$cid" ]]; then docker rm -f "$$cid" >/dev/null 2>&1 || true; fi; if (( preserve == 0 )); then chmod -R u+w "$$task_root" 2>/dev/null || true; rm -rf -- "$$task_root"; else printf '{"status":"FAILED_PHASE6_LAUNCH_EVIDENCE_PRESERVED","path":"%s"}\n' "$$task_root" >&2; fi; }; \
		trap cleanup EXIT; \
		head="$$(git rev-parse HEAD)"; image_id="$(MEASUREMENT_IMAGE_CONFIG_DIGEST)"; \
		git clone --quiet --no-local --no-checkout "$(CURDIR)" "$$task_root/source"; \
		git -C "$$task_root/source" checkout --quiet --detach "$$head"; \
		git -C "$$task_root/source" remote remove origin; \
		test "$$(git -C "$$task_root/source" rev-parse HEAD)" = "$$head"; \
		test -z "$$(git -C "$$task_root/source" status --porcelain=v1 --untracked-files=all)"; \
		chmod -R a-w "$$task_root/source/docs/evidence/e00"; \
		test -z "$$(find "$$task_root/source/docs/evidence/e00" -perm /222 -print -quit)"; \
		mkdir "$$task_root/source/.venv"; \
		ln -s /opt/kvbench/.venv/bin "$$task_root/source/.venv/bin"; \
		ln -s /opt/kvbench/.venv/lib "$$task_root/source/.venv/lib"; \
		ln -s /opt/kvbench/.venv/pyvenv.cfg "$$task_root/source/.venv/pyvenv.cfg"; \
		mkdir "$$task_root/source/.phase3"; \
		ln -s /opt/kvbench/.phase3/site-packages "$$task_root/source/.phase3/site-packages"; \
		mkdir -p "$$task_root/source/artifacts/phase6" "$(CURDIR)/artifacts/phase6"; \
		test ! -e "$$task_root/source/.env"; \
		model_root="$$(realpath /root/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct)"; \
		model_snapshot="$$model_root/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"; \
		test -d "$$model_root" && test -d "$$model_snapshot"; \
		cid="$$(docker create --read-only --network=none --pid=host \
			--gpus "device=$(MEASUREMENT_GPU_UUID)" \
			--tmpfs /tmp:rw,nosuid,nodev,size=8g \
			--tmpfs /root:rw,exec,nosuid,nodev,size=8g \
			--mount "type=bind,src=$$task_root/source,dst=/workspace,readonly" \
			--mount "type=bind,src=$$task_root/source,dst=/home/rockrock/cmu_paper,readonly" \
			--mount "type=bind,src=$(CURDIR)/artifacts/phase6,dst=/home/rockrock/cmu_paper/artifacts/phase6" \
			--mount "type=bind,src=$$model_root,dst=/root/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct,readonly" \
			--env PYTHONDONTWRITEBYTECODE=1 \
			--env PYTHONNOUSERSITE=1 \
			--env PYTHONPATH=/opt/kvbench/.phase3/site-packages:/home/rockrock/cmu_paper/src:/home/rockrock/cmu_paper \
			--env HF_HUB_OFFLINE=1 \
			--env TRANSFORMERS_OFFLINE=1 \
			--env HF_HUB_DISABLE_TELEMETRY=1 \
			--env TOKENIZERS_PARALLELISM=false \
			--env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
			--env TRITON_CACHE_DIR=/root/.triton \
			--env KVBENCH_AUTHORIZED_IMAGE_DIGEST="$$image_id" \
			--env KVBENCH_EXECUTION_ENVIRONMENT=measurement_container \
			--workdir /home/rockrock/cmu_paper \
			--entrypoint /usr/bin/bash "$$image_id" \
			--noprofile --norc -eu -o pipefail -c \
			'make PHASE2_PYTHON=/opt/kvbench/.venv/bin/python PHASE3_PYTHON=/opt/kvbench/.venv/bin/python test-cuda && make PHASE2_PYTHON=/opt/kvbench/.venv/bin/python PHASE3_PYTHON=/opt/kvbench/.venv/bin/python test-graph && /opt/kvbench/.venv/bin/python scripts/phase6_turboquant_admission.py')"; \
		[[ "$$cid" =~ ^[0-9a-f]{64}$$ ]]; \
		docker start --attach "$$cid"; \
		docker rm -f "$$cid" >/dev/null; cid=""; \
		preserve=0

validate-admission-turboquant:
	@$(PHASE3_ENV) $(PHASE3_PYTHON) -m scripts.phase6_turboquant_admission --validate-only

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
