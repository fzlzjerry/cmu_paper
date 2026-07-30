"""Focused tests for the self-reference-safe Phase 11 R2 outer bundle."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from kvbench.adapters.kvquant import KVQUANT_ADAPTER_VERSION
from kvbench.runtime.artifacts import AppendOnlyArtifactStore
from kvbench.schema import (
    ClaimEligibility,
    MeasurementScope,
    QualityExecutionState,
    QualityValidationState,
    RunStatus,
    canonical_json_bytes,
)
from kvbench.schema.phase11 import (
    PHASE11_BOUNDED_POINT_SIGNATURES,
    PHASE11_CONFIGURATIONS,
    PHASE11_DECISIONS,
    PHASE11_EXTENSION_SHA256,
    Phase11RunManifest,
    Phase11RunPoint,
)
from preflight.run_preflight import json_bytes, sha256_file
from scripts.phase11_r2_outer_bundle import (
    BUNDLED_INNER_RECEIPT_PATH,
    BUNDLED_METHOD_ADMISSION_PATH,
    BUNDLED_METHOD_CHECKSUM_PATH,
    BUNDLED_PASS_REPORT_PATH,
    INNER_PUBLISH_STDERR_RELATIVE,
    INNER_PUBLISH_STDOUT_RELATIVE,
    INNER_RECEIPT_RELATIVE,
    INNER_REFERENCE_PATH,
    INNER_VERIFY_STDERR_RELATIVE,
    INNER_VERIFY_STDOUT_RELATIVE,
    METHOD_ADMISSION_CHECKSUM_RELATIVE,
    METHOD_ADMISSION_RELATIVE,
    OUTER_RECEIPT_RELATIVE,
    OUTER_PUBLISH_STDERR_RELATIVE,
    OUTER_PUBLISH_STDOUT_RELATIVE,
    OUTER_VERIFY_STDERR_RELATIVE,
    OUTER_VERIFY_STDOUT_RELATIVE,
    PASS_REPORT_RELATIVE,
    Phase11OuterBundleError,
    _validate_inner_bundle,
    assemble_publication_receipt,
    build_outer_bundle,
    validate_outer_bundle,
    validate_outer_publication_receipt,
)
from scripts.phase11_kvquant_admission import (
    Phase11KVQuantDriverError,
    _GIT_BOUND_SOURCE_PATHS,
    _RUNTIME_CALIBRATION_PATH,
    _expected_execution_changed_paths,
    _validate_inner_records,
    derive_phase11_method_admission_report,
    write_phase11_method_admission_report,
)
from scripts.r2_artifact import publication_order, validate_local_artifact
from scripts.r2_artifact import STATUS_VARIABLES
from tests.unit.test_phase11_kvquant_admission import (
    _authority,
    _report,
)


EXECUTION_GIT_SHA = "7" * 40
OUTER_GIT_SHA = "8" * 40
INNER_RUN_ID = "phase11-admission-inner-test"
LOCK_ID = "phase11-test-indefinite-lock"


def _test_git_binding() -> dict[str, object]:
    hashes = {
        "scripts/phase11_kvquant_admission.py": "2" * 64,
        "scripts/validate_kvquant_long_context_patch.py": "4" * 64,
        "src/kvbench/adapters/kvquant.py": "b" * 64,
        "src/kvbench/runtime/kvquant_cache.py": "c" * 64,
        "src/kvbench/runtime/bf16_endpoint.py": "d" * 64,
        "src/kvbench/runtime/kvquant_session.py": "e" * 64,
        "src/kvbench/schema/phase11.py": "5" * 64,
        "tests/cuda/test_phase11_kvquant_cuda.py": "f" * 64,
        "tests/graph/test_phase11_kvquant_graph.py": "1" * 64,
        "tests/cuda/phase11_kvquant_sanitizer_probe.py": "3" * 64,
    }
    return {
        "schema_version": "kvbench-phase11-git-source-binding-1.0.0",
        "git_sha": EXECUTION_GIT_SHA,
        "records": [
            {
                "path": path,
                "git_blob_oid": f"{index + 1:x}" * 40,
                "content_sha256": hashes[path],
            }
            for index, path in enumerate(_GIT_BOUND_SOURCE_PATHS)
        ],
        "all_match": True,
    }


def _point_payload(
    *,
    index: int,
    signature: tuple,
    run_id: str,
    allocation_audits: list[dict[str, object]],
) -> dict[str, object]:
    configuration, runner_kind, graph_mode, context, output_steps = signature
    return {
        "schema_version": "kvbench-phase11-kvquant-point-1.0.0",
        "run_id": run_id,
        "configuration": configuration,
        "runner_kind": runner_kind.value,
        "graph_mode": graph_mode.value,
        "batch_size": 1,
        "context_length": context,
        "output_steps": output_steps,
        "quality_status": "unvalidated",
        "claim_eligibility": "performance_only",
        "performance_claim_eligible": False,
        "measurement_scope": "measurement_container_admission",
        "speedup_calculated": False,
        "runner": {
            "point_index": index,
            "output_finite": True,
            "performance_claim_eligible": False,
            "speedup_calculated": False,
            "r_hbm": None,
        },
        "allocation_audits": allocation_audits,
    }


def _initial_manifest() -> Phase11RunManifest:
    return Phase11RunManifest(
        schema_version=Phase11RunManifest.SCHEMA_VERSION,
        artifact_schema_version=(
            Phase11RunManifest.ARTIFACT_SCHEMA_VERSION
        ),
        run_id=INNER_RUN_ID,
        status=RunStatus.CREATED,
        created_at_utc="2026-07-30T01:00:00Z",
        started_at_utc=None,
        finished_at_utc=None,
        run_kind="phase11_admission",
        git_sha=EXECUTION_GIT_SHA,
        git_dirty=False,
        authority=_authority(),
        bounded_point_count=9,
        measurement_scope=(
            MeasurementScope.MEASUREMENT_CONTAINER_ADMISSION
        ),
        quality_status=QualityValidationState.UNVALIDATED,
        claim_eligibility=ClaimEligibility.PERFORMANCE_ONLY,
        quality_execution=QualityExecutionState.LOCKED,
        performance_claim_eligible=False,
        performance_data_frozen=False,
        quality_benchmark_executed=False,
        speedup_calculated=False,
        r_hbm=None,
        full_scan_state="CLOSED",
        g2_kvq_state="NOT_EVALUATED_PUBLICATION_PENDING",
        global_g2_g5_state="NOT_EVALUATED",
        inventory_path=None,
        failure_reason=None,
    )


def _terminal_manifest(initial: Phase11RunManifest) -> Phase11RunManifest:
    return dataclasses.replace(
        initial,
        status=RunStatus.COMPLETED,
        started_at_utc="2026-07-30T01:01:00Z",
        finished_at_utc="2026-07-30T01:02:00Z",
        inventory_path="artifact_inventory.json",
    )


def _make_inner_bundle(
    repository: Path,
    *,
    bad_record: bool = False,
    credential_field: bool = False,
    environment_drift: bool = False,
    gqa_drift: bool = False,
) -> tuple[Path, tuple[Phase11RunPoint, ...], dict[str, str], dict[str, str]]:
    initial = _initial_manifest()
    store = AppendOnlyArtifactStore(
        repository / "artifacts" / "phase11",
        formal_evidence_roots=(
            repository / "docs" / "evidence",
            repository / "artifacts" / "quality",
            repository / "artifacts" / "profiler",
        ),
    )
    run = store.create(initial.run_id, initial)
    run.start()
    points: list[Phase11RunPoint] = []
    point_payloads: list[dict[str, object]] = []
    point_records: list[dict[str, object]] = []
    raw_audits: dict[str, list[dict[str, object]]] = {
        configuration: [] for configuration in PHASE11_CONFIGURATIONS
    }
    for index, signature in enumerate(PHASE11_BOUNDED_POINT_SIGNATURES):
        run_id = f"phase11-point-{index:02d}"
        configuration, runner_kind, graph_mode, context, output_steps = (
            signature
        )
        allocation_audits: list[dict[str, object]] = []
        for step in range(output_steps):
            evidence_root = (
                f"allocation/operations/{run_id}/step-{step:04d}"
            )
            raw_hashes: dict[str, str] = {}
            for raw_index in range(9):
                basename = f"raw-{raw_index:02d}.json"
                raw_payload = {
                    "schema_version": "phase11-test-raw-allocation-1.0.0",
                    "point": run_id,
                    "step": step,
                    "index": raw_index,
                }
                data = json_bytes(
                    [raw_payload] if raw_index == 0 else raw_payload
                )
                run.write_bytes(f"{evidence_root}/{basename}", data)
                raw_hashes[basename] = hashlib.sha256(data).hexdigest()
            audit: dict[str, object] = {
                "passed": True,
                "criterion": {
                    "cache_growth_count": 0,
                    "dynamic_sparse_allocation_count": 0,
                    "complete_prefix_allocation_count": 0,
                    "gqa_expanded_allocation_count": 0,
                    "unknown_allocation_count": 0,
                    "persistent_allocated_delta": 0,
                    "persistent_reserved_delta": 0,
                },
                "raw_evidence_root": evidence_root,
                "raw_evidence_sha256": raw_hashes,
            }
            allocation_audits.append(audit)
            raw_audits[configuration].append(audit)
        payload = _point_payload(
            index=index,
            signature=signature,
            run_id=run_id,
            allocation_audits=allocation_audits,
        )
        digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        point = Phase11RunPoint(
            run_id=run_id,
            configuration=configuration,
            runner_kind=runner_kind,
            graph_mode=graph_mode,
            batch_size=1,
            context_length=context,
            output_steps=output_steps,
            status=RunStatus.COMPLETED,
            manifest_sha256=digest,
            quality_status=QualityValidationState.UNVALIDATED,
            claim_eligibility=ClaimEligibility.PERFORMANCE_ONLY,
            performance_claim_eligible=False,
            measurement_scope=(
                MeasurementScope.MEASUREMENT_CONTAINER_ADMISSION
            ),
            speedup_calculated=False,
        )
        relative = f"grid/{index:02d}-{run_id}/point.json"
        run.write_json(relative, payload)
        points.append(point)
        point_payloads.append(payload)
        point_records.append(
            {
                "index": index,
                "run_id": run_id,
                "path": relative,
                "sha256": digest,
            }
        )
    if bad_record:
        point_records[0]["sha256"] = "0" * 64
    base_report = _report()
    method_fingerprints = dict(base_report.method_fingerprints)
    cache_fingerprints = dict(base_report.cache_layout_fingerprints)
    run.write_json(
        "validation/bounded-grid.json",
        {
            "schema_version": "kvbench-phase11-bounded-grid-1.0.0",
            "points": [point.to_dict() for point in points],
            "point_records": point_records,
            "attempted": 9,
            "passed": 9,
            "failed": 0,
            "capacity_infeasible": 0,
            "method_fingerprints": method_fingerprints,
            "cache_layout_fingerprints": cache_fingerprints,
            "quality_status": "unvalidated",
            "performance_claim_eligible": False,
            "measurement_scope": "measurement_container_admission",
            "speedup_calculated": False,
        },
    )
    required_payloads = {
        "accounting/contexts.json": {
            "schema_version": "kvbench-phase11-accounting-set-1.0.0",
            "active_logical_basis": (
                "source-faithful-key-zero-occupancy-fixed-value-12"
            ),
            "composite_endpoint_and_cache_accounting": True,
            "endpoint_rope_scratch_bytes_per_record": 163_840,
            "records": [
                item.to_dict() for item in base_report.byte_accounting
            ],
        },
        "allocation/audit.json": {
            "schema_version": "kvbench-phase11-allocation-set-1.0.0",
            "raw_audits": raw_audits,
            "records": [
                item.to_dict() for item in base_report.allocation_evidence
            ],
        },
        "config/authority.json": _authority().to_dict(),
        "environment/container_identity.json": {
            "schema_version": "kvbench-phase11-container-runtime-1.0.0",
            "container_digest": _authority().authorized_container_digest,
            "execution_environment": "measurement_container",
            "torch": "2.12.1+cu130",
            "triton": "3.7.1",
            "cuda_runtime": "13.0",
            "compute_capability": "12.0",
            "gpu_name": (
                "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"
            ),
            "gpu_uuid": (
                "GPU-75bd273e-6b20-0d22-1b0b-5fbb6fb0025b"
            ),
            "native_host_cuda_rejected": True,
            "image_changed": False,
            "packages_installed": False,
            "network_enabled": False,
            "credentials_passed": False,
            "calibration_validation": {
                "schema_version": (
                    "kvbench-phase11-calibration-runtime-validation-1.0.0"
                ),
                "calibration_id": _authority().calibration_id,
                "mount_path": str(_RUNTIME_CALIBRATION_PATH),
                "expected_root_sha256": _authority().calibration_root,
                "observed_root_sha256": _authority().calibration_root,
                "object_count": 68,
                "complete_marker_valid": True,
                "inventory_valid": True,
                "checksum_ledger_valid": True,
                "repository_lifecycle_valid": True,
                "validated_before_cuda": True,
            },
            "git_source_binding": _test_git_binding(),
            "source_validation": {
                "status": "PASS",
                "decision": (
                    "docs/decisions/"
                    "0027-kvquant-deterministic-long-context-value-decode.md"
                ),
                "decision_status": "Accepted",
                "patched_commit": _authority().corrected_commit,
                "patched_tree": _authority().corrected_tree,
                "aggregate_patch_sha256": (
                    _authority().aggregate_patch_sha256
                ),
                "aggregate_changed_paths": list(
                    _expected_execution_changed_paths()
                ),
                "parent_commit": (
                    "0d9df350bd1788284e1ce76a8bf6e886beca5efa"
                ),
                "parent_tree": (
                    "a85cf7bf093982a4bf89c33d4e6794d9a85f846d"
                ),
                "parent_relative_changed_paths": [
                    "deployment/kvquant/quant_cuda.cpp",
                    "deployment/kvquant/quant_cuda_kernel.cu",
                ],
                "source_contract": "PASS",
                "reconstruction": {"status": "PASS"},
            },
            "authority_extension_path": "/authority/quant_cuda.so",
            "authority_extension_sha256": PHASE11_EXTENSION_SHA256,
            "fresh_build_extension_path": "/fresh/quant_cuda.so",
            "fresh_build_extension_sha256": PHASE11_EXTENSION_SHA256,
            "fresh_build_byte_identical_to_authority": True,
            "nvcc_cuda_object_byte_reproducible": False,
            "fresh_build_source_equivalent": True,
            "fresh_build_code_objects": {
                "schema_version": (
                    "kvbench-phase11-sm120-code-object-check-1.0.0"
                ),
                "records": [
                    {
                        "label": label,
                        "command_argv": ["cuobjdump"],
                        "returncode": 0,
                        "stdout_sha256": digest * 64,
                        "stderr_sha256": "0" * 64,
                        "passed": True,
                    }
                    for label, digest in (
                        ("sm_120_cubin", "1"),
                        ("compute_120_ptx", "2"),
                    )
                ],
                "native_sm120": True,
                "sm_120_cubin": True,
                "compute_120_ptx": True,
            },
        },
        "execution-path/audit.json": {
            "schema_version": (
                "kvbench-phase11-execution-path-set-1.0.0"
            ),
            "adapter_version": KVQUANT_ADAPTER_VERSION,
            "adapter_source_sha256": base_report.adapter_source_sha256,
            "cache_source_sha256": "c" * 64,
            "endpoint_source_sha256": "d" * 64,
            "session_source_sha256": "e" * 64,
            "fixture_test_sha256": "f" * 64,
            "graph_test_sha256": "1" * 64,
            "records": [
                item.to_dict()
                for item in base_report.execution_path_evidence
            ],
        },
        "gqa/audit.json": {
            "schema_version": "kvbench-phase11-gqa-audit-1.0.0",
            "passed": True,
            "query_heads": 32,
            "kv_heads": 8,
            "groups": 4,
            "mapping": "query_head//4",
            "native_kv_storage": True,
            "query_head_sized_cache": False,
            "repeat_kv": False,
        },
        "numerical/fixture-conformance.json": (
            base_report.fixture_evidence.to_dict()
        ),
        "validation/admission-candidate.json": {
            "schema_version": (
                "kvbench-phase11-kvquant-local-admission-candidate-1.0.0"
            ),
            "status": "LOCAL_CHECKS_PASS_PUBLICATION_PENDING",
            "git_sha": EXECUTION_GIT_SHA,
            "container_digest": _authority().authorized_container_digest,
            "fixture_conformance": True,
            "byte_accounting": True,
            "execution_path": True,
            "native_gqa": True,
            "allocation": True,
            "cuda_graph": True,
            "compute_sanitizer": True,
            "bounded_admission_grid": True,
            "immutable_checksums": "pending_finalization",
            "durable_publication": "PENDING_HOST_SIDE",
            "clean_retrieval": "PENDING_HOST_SIDE",
            "final_method_admission_report": "PENDING_HOST_SIDE",
            "g2_kvq": "NOT_EVALUATED_PUBLICATION_PENDING",
            "global_g2_g5": "NOT_EVALUATED",
            "full_scan": "CLOSED",
            "quality_execution": "LOCKED",
            "performance_data_frozen": False,
            "performance_claim_eligible": False,
            "speedup_calculated": False,
            "r_hbm": None,
            "historical_evidence_unchanged": True,
            "existing_methods_unchanged": True,
            "measurement_container_unchanged": True,
        },
        "validation/cuda-graph.json": {
            "schema_version": "kvbench-phase11-cuda-graph-set-1.0.0",
            "records": [
                item.to_dict() for item in base_report.graph_evidence
            ],
        },
        "validation/sanitizer.json": (
            base_report.sanitizer_evidence.to_dict()
        ),
        "validation/checks/fixture-conformance.json": {
            "schema_version": "phase11-test-exact-check-1.0.0",
            "passed": True,
        },
        "validation/checks/cuda-graph.json": {
            "schema_version": "phase11-test-exact-check-1.0.0",
            "passed": True,
        },
        "validation/sanitizer-runs.json": {
            "schema_version": "phase11-test-sanitizer-runs-1.0.0",
            "passed": True,
        },
    }
    if credential_field:
        required_payloads["validation/credential-leak.json"] = {
            "aws_secret_access_key": "not-a-real-secret"
        }
    if environment_drift:
        required_payloads["environment/container_identity.json"][
            "packages_installed"
        ] = True
    if gqa_drift:
        required_payloads["gqa/audit.json"]["repeat_kv"] = True
    for relative, payload in required_payloads.items():
        run.write_json(relative, payload)
    final = run.finalize(_terminal_manifest(initial))
    return (
        final,
        tuple(points),
        method_fingerprints,
        cache_fingerprints,
    )


def _receipt_payload(
    source: Path,
    root_sha256: str,
    object_count: int,
    *,
    schema: str = (
        "kvbench-phase11-kvquant-admission-r2-publication-1.0.0"
    ),
    source_git_sha: str = EXECUTION_GIT_SHA,
) -> dict[str, object]:
    uri = (
        "r2://kvbench-artifacts/kvbench/sha256/"
        f"{root_sha256}/"
    )
    artifact = validate_local_artifact(source)
    order_bytes = "".join(
        f"{item.relative_path}\n" for item in publication_order(artifact)
    ).encode()
    return {
        "schema_version": schema,
        "recorded_at_utc": "2026-07-30T01:06:00Z",
        "admission_status": "PASS",
        "artifact_status": "completed",
        "source_git_sha": source_git_sha,
        "source_run_id": source.name,
        "local_validation": {
            "valid": True,
            "complete": True,
            "status": "completed",
            "root_sha256": root_sha256,
            "object_count": object_count,
            "complete_marker_valid": True,
            "inventory_valid": True,
            "checksum_ledger_valid": True,
            "root_digest_valid": True,
            "bundle_validation_valid": True,
        },
        "publication": {
            "result": "PASS",
            "provider": "cloudflare_r2",
            "root_sha256": root_sha256,
            "uri": uri,
            "object_count": object_count,
            "uploaded_count": object_count,
            "verified_existing_count": 0,
            "content_addressed": True,
            "conditional_writes": True,
            "complete_last": True,
            "publication_order_sha256": hashlib.sha256(
                order_bytes
            ).hexdigest(),
            "published_at_utc": "2026-07-30T01:04:00Z",
        },
        "clean_retrieval": {
            "result": "PASS",
            "provider": "cloudflare_r2",
            "root_sha256": root_sha256,
            "uri": uri,
            "object_count": object_count,
            "destination_initially_empty": True,
            "complete_marker_valid": True,
            "inventory_valid": True,
            "checksum_ledger_valid": True,
            "root_digest_valid": True,
            "bundle_validation_valid": True,
            "unexpected_objects": False,
            "retrieved_at_utc": "2026-07-30T01:05:00Z",
        },
        "bucket_lock": {
            "provider": "cloudflare_r2",
            "bucket": "kvbench-artifacts",
            "endpoint": (
                "https://<R2_ACCOUNT_ID>.r2.cloudflarestorage.com"
            ),
            "endpoint_class": "cloudflare_r2_s3",
            "bucket_exists": True,
            "verification_result": "PASS",
            "enabled": True,
            "public_state_result": "PASS",
            "managed_r2_dev_enabled": False,
            "public_r2_dev": False,
            "custom_domain_count": 0,
            "enabled_custom_domain_count": 0,
            "public_custom_domain": False,
            "lock_rule_id": LOCK_ID,
            "lock_rule_name": LOCK_ID,
            "lock_scope": "exact",
            "covered_prefix": "kvbench/sha256/",
            "lock_prefix": "kvbench/sha256/",
            "retention_type": "Indefinite",
            "retention_condition": "Indefinite",
            "bucket_public": False,
            "verified_at_utc": "2026-07-30T01:03:00Z",
        },
        "credential_values_recorded": False,
        "env_file_read": False,
    }


def _write_governance(
    repository: Path,
    source: Path,
    points: tuple[Phase11RunPoint, ...],
    method_fingerprints: dict[str, str],
    cache_fingerprints: dict[str, str],
) -> None:
    inner = validate_local_artifact(source)
    receipt_path = repository / INNER_RECEIPT_RELATIVE
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    publish_output, publish_stderr, verify_output, verify_stderr = (
        _write_raw_r2_tool_outputs(
            repository,
            source,
            kind="inner",
        )
    )
    receipt_path.write_bytes(
        json_bytes(
            assemble_publication_receipt(
                artifact_root=source,
                publish_output=publish_output,
                publish_stderr=publish_stderr,
                verify_output=verify_output,
                verify_stderr=verify_stderr,
                receipt_kind="inner",
                source_run_id=INNER_RUN_ID,
                source_git_sha=EXECUTION_GIT_SHA,
                recorded_at_utc="2026-07-30T01:06:00Z",
                repository_root=repository,
            )
        )
    )
    report = derive_phase11_method_admission_report(
        bundle_path=source,
        publication_receipt_path=receipt_path,
        created_at_utc="2026-07-30T01:07:00Z",
        repository_root=repository,
    )
    report_path = repository / METHOD_ADMISSION_RELATIVE
    report_path.write_bytes(json_bytes(report.to_dict()))
    checksum_path = repository / METHOD_ADMISSION_CHECKSUM_RELATIVE
    checksum_path.write_text(
        f"{sha256_file(report_path)}  {report_path.name}\n",
        encoding="utf-8",
    )
    phase_report = repository / PASS_REPORT_RELATIVE
    phase_report.parent.mkdir(parents=True, exist_ok=True)
    phase_report.write_text(
        "# PHASE 11 REPORT\n\n"
        "Status: PASS\n"
        "Working tree: clean\n"
        f"Algorithm identifier: {report.authority.method_identifier}\n"
        "Execution-source identifier: "
        f"{report.authority.execution_source_identifier}\n"
        f"Decisions: {', '.join(PHASE11_DECISIONS)}\n"
        f"Aggregate patch SHA: {report.authority.aggregate_patch_sha256}\n"
        f"Corrected commit: {report.authority.corrected_commit}\n"
        f"Corrected tree: {report.authority.corrected_tree}\n"
        f"Extension SHA: {report.authority.extension_sha256}\n"
        f"Calibration ID: {report.authority.calibration_id}\n"
        f"Calibration root: {report.authority.calibration_root}\n"
        "Historical Phase 10 root: "
        f"{report.authority.historical_fixture_root}\n"
        f"Corrected fixture ID: {report.authority.fixture_id}\n"
        f"Corrected fixture root: {report.authority.fixture_root}\n"
        "Adapter location: src/kvbench/adapters/kvquant.py\n"
        "Supported configurations: kvq4, kvq3, kvq2\n"
        "Boundary semantics: pre-RoPE Key quantization; "
        "attention-ready sink Key\n"
        "Static cache: PASS\n"
        "Fixture conformance: 9/9 PASS\n"
        "Execution-path and GQA audit: PASS\n"
        "Eager allocation: PASS\n"
        "CUDA Graph: PASS\n"
        "Sanitizer: PASS\n"
        "Bounded admission: 9/9 PASS\n"
        "Admission run IDs: "
        f"{', '.join(point.run_id for point in report.bounded_runs)}\n"
        f"MethodAdmissionReport SHA-256: {sha256_file(report_path)}\n"
        f"Inner R2 URI: {report.r2_uri}\n"
        "G2-KVQ: PASS\n"
        "Global G2: NOT EVALUATED\n"
        "G3: NOT EVALUATED\n"
        "G4: NOT EVALUATED\n"
        "G5: NOT EVALUATED\n"
        "Full Scan: CLOSED\n"
        "Quality execution: LOCKED\n"
        "PERFORMANCE_DATA_FROZEN: absent\n"
        "Performance claim eligible: false\n"
        "Speedup calculated: no\n"
        "r_hbm: null\n"
        "Historical evidence changed: no\n"
        "Existing methods changed: no\n"
        "Measurement Container changed: no\n"
        "Phase 12 started: no\n",
        encoding="utf-8",
    )


def _outer_receipt_payload(
    repository: Path,
    artifact: Path,
    root_sha256: str,
    object_count: int,
) -> dict[str, object]:
    if (
        validate_local_artifact(artifact).root_sha256 != root_sha256
        or len(validate_local_artifact(artifact).files) != object_count
    ):
        raise AssertionError("outer test artifact identity differs")
    publish_output, publish_stderr, verify_output, verify_stderr = (
        _write_raw_r2_tool_outputs(
            repository,
            artifact,
            kind="outer",
        )
    )
    return assemble_publication_receipt(
        artifact_root=artifact,
        publish_output=publish_output,
        publish_stderr=publish_stderr,
        verify_output=verify_output,
        verify_stderr=verify_stderr,
        receipt_kind="outer",
        source_run_id=artifact.name,
        source_git_sha=OUTER_GIT_SHA,
        recorded_at_utc="2026-07-30T01:06:00Z",
        repository_root=repository,
    )


def _r2_tool_outputs(
    artifact_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    artifact = validate_local_artifact(artifact_path)
    receipt = _receipt_payload(
        artifact_path,
        artifact.root_sha256,
        len(artifact.files),
    )
    identity = {
        "provider": "cloudflare_r2",
        "endpoint_class": "cloudflare_r2_s3",
        "endpoint": "https://<R2_ACCOUNT_ID>.r2.cloudflarestorage.com",
        "bucket": "kvbench-artifacts",
        "prefix": "kvbench/sha256",
        "region": "auto",
    }
    publication = dict(receipt["publication"])
    publication.pop("result")
    publication.pop("content_addressed")
    publication.pop("conditional_writes")
    retrieval = dict(receipt["clean_retrieval"])
    retrieval.pop("result")
    retrieval.pop("destination_initially_empty")
    retrieval.pop("root_digest_valid")
    retrieval.pop("bundle_validation_valid")
    retrieval["verification_result"] = "PASS"
    common = {
        "status": "PASS",
        "required_variables": {
            name: "PRESENT" for name in STATUS_VARIABLES
        },
        "r2": identity,
        "bucket_lock": receipt["bucket_lock"],
    }
    return (
        {**common, "publish": publication},
        {**common, "verify": retrieval},
    )


def _write_raw_r2_tool_outputs(
    repository: Path,
    artifact_path: Path,
    *,
    kind: str,
) -> tuple[Path, Path, Path, Path]:
    publish, verify = _r2_tool_outputs(artifact_path)
    relatives = (
        (
            INNER_PUBLISH_STDOUT_RELATIVE,
            INNER_PUBLISH_STDERR_RELATIVE,
            INNER_VERIFY_STDOUT_RELATIVE,
            INNER_VERIFY_STDERR_RELATIVE,
        )
        if kind == "inner"
        else (
            OUTER_PUBLISH_STDOUT_RELATIVE,
            OUTER_PUBLISH_STDERR_RELATIVE,
            OUTER_VERIFY_STDOUT_RELATIVE,
            OUTER_VERIFY_STDERR_RELATIVE,
        )
    )
    paths = tuple(repository / relative for relative in relatives)
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    paths[0].write_bytes(json_bytes(publish))
    paths[1].write_bytes(b"")
    paths[2].write_bytes(json_bytes(verify))
    paths[3].write_bytes(b"")
    return paths


def _make_writable(root: Path) -> None:
    if not root.exists():
        return
    root.chmod(0o755)
    for path in root.rglob("*"):
        if path.is_file():
            path.chmod(0o644)
        elif path.is_dir():
            path.chmod(0o755)


class Phase11R2OuterBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.semantic_replay = mock.patch(
            "scripts.phase11_kvquant_admission."
            "_validate_phase11_allocator_semantically"
        )
        self.semantic_replay_mock = self.semantic_replay.start()
        self.raw_process_validation = mock.patch(
            "scripts.phase11_kvquant_admission."
            "_validate_exact_and_sanitizer_raw_evidence"
        )
        self.raw_process_validation_mock = (
            self.raw_process_validation.start()
        )
        self.git_source_binding = mock.patch(
            "scripts.phase11_kvquant_admission._git_source_binding",
            return_value=_test_git_binding(),
        )
        self.git_source_binding_mock = self.git_source_binding.start()
        self.temporary = tempfile.TemporaryDirectory()
        self.repository = Path(self.temporary.name) / "repository"
        self.repository.mkdir()
        (
            self.source,
            self.points,
            self.method_fingerprints,
            self.cache_fingerprints,
        ) = _make_inner_bundle(self.repository)
        _write_governance(
            self.repository,
            self.source,
            self.points,
            self.method_fingerprints,
            self.cache_fingerprints,
        )
        self.output_root = (
            self.repository / "artifacts" / "phase11_r2_outer"
        )

    def tearDown(self) -> None:
        _make_writable(self.repository)
        self.temporary.cleanup()
        self.semantic_replay.stop()
        self.raw_process_validation.stop()
        self.git_source_binding.stop()

    def _build(self, run_id: str = "phase11-r2-outer-test") -> Path:
        final, validation = build_outer_bundle(
            repository_root=self.repository,
            source_bundle=self.source,
            output_root=self.output_root,
            run_id=run_id,
            source_git_sha=OUTER_GIT_SHA,
        )
        self.assertEqual(validation.run_id, run_id)
        self.assertEqual(validation.admission_run_count, 9)
        return final

    def test_pass_report_uses_phase11r_path_not_immutable_blocked_path(
        self,
    ) -> None:
        self.assertEqual(
            PASS_REPORT_RELATIVE.as_posix(),
            "docs/phase_reports/phase11r-kvquant-measurement-adapter.md",
        )
        self.assertEqual(
            BUNDLED_PASS_REPORT_PATH.as_posix(),
            "reports/phase11r-kvquant-measurement-adapter.md",
        )
        self.assertNotEqual(
            PASS_REPORT_RELATIVE.as_posix(),
            "docs/phase_reports/phase11-kvquant-measurement-adapter.md",
        )

    def test_inner_governance_scan_accepts_list_shaped_allocator_trace(
        self,
    ) -> None:
        closure = _validate_inner_bundle(self.source)
        self.assertEqual(len(closure.points), 9)

    def test_report_writer_is_derived_and_refuses_overwrite(self) -> None:
        repository = self.repository / "writer-repository"
        repository.mkdir()
        source, _, _, _ = _make_inner_bundle(repository)
        receipt_path = repository / INNER_RECEIPT_RELATIVE
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        publish, publish_stderr, verify, verify_stderr = (
            _write_raw_r2_tool_outputs(repository, source, kind="inner")
        )
        receipt_path.write_bytes(
            json_bytes(
                assemble_publication_receipt(
                    artifact_root=source,
                    publish_output=publish,
                    publish_stderr=publish_stderr,
                    verify_output=verify,
                    verify_stderr=verify_stderr,
                    receipt_kind="inner",
                    source_run_id=INNER_RUN_ID,
                    source_git_sha=EXECUTION_GIT_SHA,
                    recorded_at_utc="2026-07-30T01:06:00Z",
                    repository_root=repository,
                )
            )
        )
        report_path = repository / METHOD_ADMISSION_RELATIVE
        checksum_path = repository / METHOD_ADMISSION_CHECKSUM_RELATIVE
        result = write_phase11_method_admission_report(
            bundle_path=source,
            publication_receipt_path=receipt_path,
            report_path=report_path,
            checksum_path=checksum_path,
            created_at_utc="2026-07-30T01:07:00Z",
            repository_root=repository,
        )
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(
            checksum_path.read_text(encoding="utf-8"),
            f"{sha256_file(report_path)}  {report_path.name}\n",
        )
        with self.assertRaisesRegex(
            Phase11KVQuantDriverError,
            "already exist",
        ):
            write_phase11_method_admission_report(
                bundle_path=source,
                publication_receipt_path=receipt_path,
                report_path=report_path,
                checksum_path=checksum_path,
                created_at_utc="2026-07-30T01:07:00Z",
                repository_root=repository,
            )

    def test_exact_inner_runs_reports_and_self_reference_boundary(self) -> None:
        final = self._build()
        validation = validate_outer_bundle(
            final,
            repository_root=self.repository,
            source_bundle=self.source,
        )
        inner = validate_local_artifact(self.source)
        self.assertEqual(validation.inner_root_sha256, inner.root_sha256)
        self.assertEqual(
            publication_order(validate_local_artifact(final))[-1].relative_path,
            "COMPLETE",
        )
        binding = json.loads(
            (final / INNER_REFERENCE_PATH).read_text(encoding="utf-8")
        )
        self.assertEqual(binding["run_id"], INNER_RUN_ID)
        self.assertEqual(binding["root_sha256"], inner.root_sha256)
        self.assertEqual(len(binding["point_records"]), 9)
        self.assertEqual(
            (final / BUNDLED_METHOD_ADMISSION_PATH).read_bytes(),
            (self.repository / METHOD_ADMISSION_RELATIVE).read_bytes(),
        )
        self.assertEqual(
            (final / BUNDLED_METHOD_CHECKSUM_PATH).read_bytes(),
            (
                self.repository / METHOD_ADMISSION_CHECKSUM_RELATIVE
            ).read_bytes(),
        )
        self.assertEqual(
            (final / BUNDLED_INNER_RECEIPT_PATH).read_bytes(),
            (self.repository / INNER_RECEIPT_RELATIVE).read_bytes(),
        )
        self.assertEqual(
            (final / BUNDLED_PASS_REPORT_PATH).read_bytes(),
            (self.repository / PASS_REPORT_RELATIVE).read_bytes(),
        )
        self.assertFalse((final / OUTER_RECEIPT_RELATIVE).exists())

    def test_existing_run_and_invalid_inner_point_fail_closed(self) -> None:
        final = self._build()
        root_before = validate_local_artifact(final).root_sha256
        with self.assertRaises(Phase11OuterBundleError):
            self._build()
        self.assertEqual(
            validate_local_artifact(final).root_sha256,
            root_before,
        )
        other_repository = Path(self.temporary.name) / "other"
        other_repository.mkdir()
        bad_source, _, _, _ = _make_inner_bundle(
            other_repository,
            bad_record=True,
        )
        with self.assertRaisesRegex(
            Phase11OuterBundleError,
            "point record",
        ):
            _validate_inner_bundle(bad_source)

    def test_credentials_and_report_global_gate_drift_are_rejected(self) -> None:
        other_repository = Path(self.temporary.name) / "credential"
        other_repository.mkdir()
        bad_source, _, _, _ = _make_inner_bundle(
            other_repository,
            credential_field=True,
        )
        with self.assertRaisesRegex(
            Phase11OuterBundleError,
            "credential",
        ):
            _validate_inner_bundle(bad_source)

        report_path = self.repository / METHOD_ADMISSION_RELATIVE
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["gates"]["global_g2"] = "PASS"
        report_path.write_bytes(json_bytes(report))
        checksum_path = (
            self.repository / METHOD_ADMISSION_CHECKSUM_RELATIVE
        )
        checksum_path.write_text(
            f"{sha256_file(report_path)}  {report_path.name}\n",
            encoding="utf-8",
        )
        with self.assertRaises(Phase11OuterBundleError):
            self._build("phase11-r2-outer-global-drift")

        _write_governance(
            self.repository,
            self.source,
            self.points,
            self.method_fingerprints,
            self.cache_fingerprints,
        )
        phase_report = self.repository / PASS_REPORT_RELATIVE
        phase_report.write_text(
            phase_report.read_text(encoding="utf-8").replace(
                "Speedup calculated: no",
                "Speedup calculated: yes",
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            Phase11OuterBundleError,
            "final report governance",
        ):
            self._build("phase11-r2-outer-report-claim-drift")

    def test_authority_environment_and_gqa_wrappers_are_bound(self) -> None:
        for label, options in (
            ("environment", {"environment_drift": True}),
            ("gqa", {"gqa_drift": True}),
        ):
            with self.subTest(label=label):
                repository = Path(self.temporary.name) / f"bad-{label}"
                repository.mkdir()
                source, _, _, _ = _make_inner_bundle(
                    repository,
                    **options,
                )
                with self.assertRaises(Phase11KVQuantDriverError):
                    _validate_inner_records(source)

    def test_every_allocator_operation_requires_semantic_replay(self) -> None:
        self.semantic_replay_mock.reset_mock()
        _validate_inner_records(self.source)
        self.assertEqual(self.semantic_replay_mock.call_count, 12)
        self.semantic_replay_mock.side_effect = Phase11KVQuantDriverError(
            "semantic replay tamper"
        )
        with self.assertRaisesRegex(
            Phase11KVQuantDriverError,
            "semantic replay tamper",
        ):
            _validate_inner_records(self.source)
        self.semantic_replay_mock.side_effect = None

    def test_raw_test_and_sanitizer_records_are_mandatory(self) -> None:
        self.raw_process_validation_mock.reset_mock()
        _validate_inner_records(self.source)
        self.raw_process_validation_mock.assert_called_once()
        self.raw_process_validation_mock.side_effect = (
            Phase11KVQuantDriverError("raw process evidence tamper")
        )
        with self.assertRaisesRegex(
            Phase11KVQuantDriverError,
            "raw process evidence tamper",
        ):
            _validate_inner_records(self.source)
        self.raw_process_validation_mock.side_effect = None

    def test_non_derived_report_field_is_rejected_even_with_new_checksum(
        self,
    ) -> None:
        report_path = self.repository / METHOD_ADMISSION_RELATIVE
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        payload["adapter_source_sha256"] = "0" * 64
        report_path.write_bytes(json_bytes(payload))
        checksum_path = (
            self.repository / METHOD_ADMISSION_CHECKSUM_RELATIVE
        )
        checksum_path.write_text(
            f"{sha256_file(report_path)}  {report_path.name}\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            Phase11OuterBundleError,
            "not derived exactly",
        ):
            self._build("phase11-r2-outer-derived-report-drift")

    def test_receipt_assembler_binds_order_timestamps_and_clean_retrieval(
        self,
    ) -> None:
        publish, publish_stderr, verify, verify_stderr = (
            _write_raw_r2_tool_outputs(
                self.repository,
                self.source,
                kind="inner",
            )
        )
        verify_payload = json.loads(verify.read_text(encoding="utf-8"))
        verify_payload["bucket_lock"]["verified_at_utc"] = (
            "2026-07-30T01:04:30Z"
        )
        verify.write_bytes(json_bytes(verify_payload))
        payload = assemble_publication_receipt(
            artifact_root=self.source,
            publish_output=publish,
            publish_stderr=publish_stderr,
            verify_output=verify,
            verify_stderr=verify_stderr,
            receipt_kind="inner",
            source_run_id=INNER_RUN_ID,
            source_git_sha=EXECUTION_GIT_SHA,
            recorded_at_utc="2026-07-30T01:06:00Z",
            repository_root=self.repository,
        )
        publish_payload = json.loads(publish.read_text(encoding="utf-8"))
        self.assertEqual(
            payload["publication"]["publication_order_sha256"],
            publish_payload["publish"]["publication_order_sha256"],
        )
        self.assertEqual(
            payload["clean_retrieval"]["retrieved_at_utc"],
            "2026-07-30T01:05:00Z",
        )
        self.assertEqual(
            payload["bucket_lock"]["verified_at_utc"],
            "2026-07-30T01:03:00Z",
        )
        bad_publish = json.loads(publish.read_text(encoding="utf-8"))
        bad_publish["publish"]["publication_order_sha256"] = "0" * 64
        publish.write_bytes(json_bytes(bad_publish))
        with self.assertRaisesRegex(
            Phase11OuterBundleError,
            "publisher or clean-retrieval",
        ):
            assemble_publication_receipt(
                artifact_root=self.source,
                publish_output=publish,
                publish_stderr=publish_stderr,
                verify_output=verify,
                verify_stderr=verify_stderr,
                receipt_kind="inner",
                source_run_id=INNER_RUN_ID,
                source_git_sha=EXECUTION_GIT_SHA,
                recorded_at_utc="2026-07-30T01:06:00Z",
                repository_root=self.repository,
            )
        publish.write_bytes(json_bytes(publish_payload))
        with self.assertRaisesRegex(
            Phase11OuterBundleError,
            "does not bind",
        ):
            assemble_publication_receipt(
                artifact_root=self.source,
                publish_output=publish,
                publish_stderr=publish_stderr,
                verify_output=verify,
                verify_stderr=verify_stderr,
                receipt_kind="inner",
                source_run_id=INNER_RUN_ID,
                source_git_sha=EXECUTION_GIT_SHA,
                recorded_at_utc="2026-07-30T01:04:30Z",
                repository_root=self.repository,
            )

    def test_receipt_assembler_rejects_stable_bucket_lock_drift(
        self,
    ) -> None:
        publish, publish_stderr, verify, verify_stderr = (
            _write_raw_r2_tool_outputs(
                self.repository,
                self.source,
                kind="inner",
            )
        )
        verify_payload = json.loads(verify.read_text(encoding="utf-8"))
        verify_payload["bucket_lock"]["lock_rule_id"] = "different-lock"
        verify.write_bytes(json_bytes(verify_payload))
        with self.assertRaisesRegex(
            Phase11OuterBundleError,
            "identities differ",
        ):
            assemble_publication_receipt(
                artifact_root=self.source,
                publish_output=publish,
                publish_stderr=publish_stderr,
                verify_output=verify,
                verify_stderr=verify_stderr,
                receipt_kind="inner",
                source_run_id=INNER_RUN_ID,
                source_git_sha=EXECUTION_GIT_SHA,
                recorded_at_utc="2026-07-30T01:06:00Z",
                repository_root=self.repository,
            )

    def test_receipt_assembler_rejects_nonempty_r2_stderr(self) -> None:
        for stderr_position, operation in ((1, "publish"), (3, "verify")):
            publish, publish_stderr, verify, verify_stderr = (
                _write_raw_r2_tool_outputs(
                    self.repository,
                    self.source,
                    kind="inner",
                )
            )
            stderr_paths = (
                publish,
                publish_stderr,
                verify,
                verify_stderr,
            )
            stderr_paths[stderr_position].write_text(
                "unexpected tool output\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                Phase11OuterBundleError,
                rf"raw R2 {operation} stderr must be empty",
            ):
                assemble_publication_receipt(
                    artifact_root=self.source,
                    publish_output=publish,
                    publish_stderr=publish_stderr,
                    verify_output=verify,
                    verify_stderr=verify_stderr,
                    receipt_kind="inner",
                    source_run_id=INNER_RUN_ID,
                    source_git_sha=EXECUTION_GIT_SHA,
                    recorded_at_utc="2026-07-30T01:06:00Z",
                    repository_root=self.repository,
                )

    def test_receipt_validator_rejects_hash_bound_nonempty_stderr(
        self,
    ) -> None:
        receipt_path = self.repository / INNER_RECEIPT_RELATIVE
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        stderr_path = self.repository / INNER_VERIFY_STDERR_RELATIVE
        stderr_path.write_text(
            "unexpected verifier output\n",
            encoding="utf-8",
        )
        payload["raw_tool_evidence"]["verify"]["stderr_sha256"] = (
            sha256_file(stderr_path)
        )
        receipt_path.write_bytes(json_bytes(payload))
        with self.assertRaisesRegex(
            Phase11OuterBundleError,
            "raw R2 verify stderr must be empty",
        ):
            derive_phase11_method_admission_report(
                bundle_path=self.source,
                publication_receipt_path=receipt_path,
                created_at_utc="2026-07-30T01:07:00Z",
                repository_root=self.repository,
            )

    def test_method_checksum_and_required_symlink_fail_closed(self) -> None:
        checksum_path = (
            self.repository / METHOD_ADMISSION_CHECKSUM_RELATIVE
        )
        checksum_path.write_text(
            f"{'0' * 64}  {METHOD_ADMISSION_RELATIVE.name}\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            Phase11OuterBundleError,
            "checksum differs",
        ):
            self._build("phase11-r2-outer-bad-checksum")

        _write_governance(
            self.repository,
            self.source,
            self.points,
            self.method_fingerprints,
            self.cache_fingerprints,
        )
        phase_report = self.repository / PASS_REPORT_RELATIVE
        phase_report.unlink()
        phase_report.symlink_to(self.repository / METHOD_ADMISSION_RELATIVE)
        with self.assertRaisesRegex(
            Phase11OuterBundleError,
            "absent or unsafe",
        ):
            self._build("phase11-r2-outer-symlink")

    def test_external_receipt_is_exact_and_rejects_claim_drift(self) -> None:
        final = self._build("phase11-r2-outer-published")
        validation = validate_outer_bundle(
            final,
            repository_root=self.repository,
            source_bundle=self.source,
        )
        receipt_path = self.repository / OUTER_RECEIPT_RELATIVE
        receipt_path.write_bytes(
            json_bytes(
                _outer_receipt_payload(
                    self.repository,
                    final,
                    validation.root_sha256,
                    validation.object_count,
                )
            )
        )
        publication = validate_outer_publication_receipt(
            final,
            receipt_path=receipt_path,
            repository_root=self.repository,
            source_bundle=self.source,
        )
        self.assertEqual(publication.root_sha256, validation.root_sha256)
        self.assertFalse((final / OUTER_RECEIPT_RELATIVE).exists())

        for field, value in (
            ("speedup_calculated", True),
            ("r_hbm", 1.0),
            ("global_g2", "PASS"),
            ("performance_claim_eligible", True),
            ("phase12_started", True),
        ):
            with self.subTest(field=field):
                payload = _outer_receipt_payload(
                    self.repository,
                    final,
                    validation.root_sha256,
                    validation.object_count,
                )
                payload[field] = value
                receipt_path.write_bytes(json_bytes(payload))
                with self.assertRaises(Phase11OuterBundleError):
                    validate_outer_publication_receipt(
                        final,
                        receipt_path=receipt_path,
                        repository_root=self.repository,
                        source_bundle=self.source,
                    )
        payload = _outer_receipt_payload(
            self.repository,
            final,
            validation.root_sha256,
            validation.object_count,
        )
        payload["aws_access_key_id"] = "not-a-real-key"
        receipt_path.write_bytes(json_bytes(payload))
        with self.assertRaisesRegex(
            Phase11OuterBundleError,
            "credential",
        ):
            validate_outer_publication_receipt(
                final,
                receipt_path=receipt_path,
                repository_root=self.repository,
                source_bundle=self.source,
            )
        alias = self.repository / "outer-symlink"
        alias.symlink_to(final, target_is_directory=True)
        receipt_path.write_bytes(
            json_bytes(
                _outer_receipt_payload(
                    self.repository,
                    final,
                    validation.root_sha256,
                    validation.object_count,
                )
            )
        )
        with self.assertRaisesRegex(
            Phase11OuterBundleError,
            "artifact path is a symlink",
        ):
            validate_outer_publication_receipt(
                alias,
                receipt_path=receipt_path,
                repository_root=self.repository,
                source_bundle=self.source,
            )

    def test_clean_retrieval_validates_and_tamper_fails(self) -> None:
        final = self._build()
        retrieved = self.repository / "retrieved-empty-destination"
        shutil.copytree(final, retrieved, copy_function=shutil.copy2)
        self.assertEqual(
            validate_outer_bundle(
                retrieved,
                repository_root=self.repository,
                source_bundle=self.source,
            ).root_sha256,
            validate_outer_bundle(
                final,
                repository_root=self.repository,
                source_bundle=self.source,
            ).root_sha256,
        )
        _make_writable(retrieved)
        report = retrieved / BUNDLED_METHOD_ADMISSION_PATH
        report.write_bytes(report.read_bytes() + b"tamper\n")
        with self.assertRaises(RuntimeError):
            validate_outer_bundle(
                retrieved,
                repository_root=self.repository,
                source_bundle=self.source,
            )


if __name__ == "__main__":
    unittest.main()
