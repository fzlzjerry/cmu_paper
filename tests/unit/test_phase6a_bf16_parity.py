"""Offline regressions for the bounded Phase 6A BF16 parity wrapper."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import stat
import tempfile
import unittest

from kvbench.runtime.phase3_audit_operation import Phase3AuditOperationKey
from kvbench.runtime.phase3_raw_audit_evidence import (
    PHASE3_RAW_AUDIT_OPERATION_SCHEMA_VERSION,
    RAW_AUDIT_STATUS_COMPLETED,
    Phase3RawAuditFile,
    Phase3RawAuditOperationRecord,
    Phase3RawAuditRunIndex,
)
from kvbench.schema import GraphMode, RunnerKind, canonical_json_bytes
from kvbench.schema.phase3 import (
    PHASE3_BF16_VARIANT_FINGERPRINT,
    PHASE3_FIXED_PLAN_PATH,
    PHASE3_GPU_UUID,
    PHASE3_PLAN_FINGERPRINTS,
    Phase3ProcessPoint,
)
from preflight.run_preflight import json_bytes
from scripts.phase6a_bf16_parity import (
    Phase6AParityError,
    _container_raw_audit_identities,
    _validate_context,
    _phase3_provenance_for_legacy_replay,
    publish_parity_lane,
    publish_parity_setup_failure,
    validate_container_g0_manifest,
    validate_finalized_parity_artifact,
    validate_parity_result,
)
from scripts.r2_artifact import validate_local_artifact
from tests.schema.test_phase3_schema import backend_payload

DIGEST = "sha256:" + "a" * 64
SHA = "b" * 40
REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _context() -> dict[str, object]:
    backend = backend_payload()
    from kvbench.schema.phase3 import BF16BackendIdentity

    backend_model = BF16BackendIdentity.from_dict(backend)
    return {
        "execution_git_sha": SHA,
        "image_reference": "kvbench-measurement:phase6a",
        "image_config_digest": DIGEST,
        "container_g0": {
            "run_id": "e00-20260725T000000.000000Z-deadbeef0000-abcdef12",
            "gpu_uuid": PHASE3_GPU_UUID,
            "gpu_name": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
            "compute_capability": "12.0",
            "image_config_digest": DIGEST,
            "artifact_root_sha256": "c" * 64,
            "manifest_sha256": "d" * 64,
            "artifact_directory": "/run/kvbench/container-g0/example",
        },
        "runtime_gpu": {
            "uuid": PHASE3_GPU_UUID,
            "name": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
            "compute_capability": "12.0",
        },
        "model_identity": {
            "model_id": "meta-llama/Llama-3.1-8B-Instruct",
            "model_revision": REVISION,
            "tokenizer_revision": REVISION,
            "frozen_identity_sha256": "e" * 64,
            "snapshot_file_ledger_sha256": "f" * 64,
            "load_receipt_sha256": "1" * 64,
            "tokenizer_runtime_sha256": "2" * 64,
            "parameter_runtime_sha256": "3" * 64,
            "num_query_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "weight_dtype": "bfloat16",
        },
        "backend_identity": {
            "backend": backend,
            "fingerprint": backend_model.fingerprint(),
            "attention_implementation": "kvbench_bf16_flash",
            "fallback_permitted": False,
        },
        "method_identity": {
            "method": "bf16",
            "method_config_id": "bf16",
            "method_config_fingerprint": PHASE3_BF16_VARIANT_FINGERPRINT,
            "adapter_version": "kvbench-bf16-method-adapter-1.0.0",
            "adapter_implementation_path": "src/kvbench/adapters/bf16.py",
            "adapter_implementation_sha256": "5" * 64,
        },
        "source_identity": {
            "source_identity_sha256": "6" * 64,
            "execution_source_identity_sha256": "7" * 64,
        },
}


def _comparison(*, atol: float = 0.02) -> dict[str, object]:
    return {
        "passed": True,
        "finite": True,
        "max_absolute_error": 0.0,
        "max_relative_error": 0.0,
        "atol": atol,
        "rtol": 0.02,
    }


def _graph_validation() -> dict[str, object]:
    return {
        "passed": True,
        "prefix_length": 128,
        "graph": {
            "captured": True,
            "output_data_ptr": 1,
            "capture_stream_id": 2,
            "fallback": False,
        },
        "eager_replay_comparison": _comparison(),
        "replay_outputs_exact": True,
        "replay_copies_independent": True,
        "eager_checksum": "8" * 64,
        "first_replay_checksum": "9" * 64,
        "second_replay_checksum": "9" * 64,
        "cache_pointers_stable": True,
        "historical_cache_unchanged": True,
        "replay_allocation": {
            "audit_available": True,
            "passed": True,
            "allocation_event_count": 0,
            "allocation_event_bytes": 0,
            "event_counts": {},
            "allocated_before": 1,
            "allocated_after": 1,
            "allocated_delta": 0,
            "reserved_before": 1,
            "reserved_after": 1,
            "reserved_delta": 0,
            "peak_allocated": 1,
            "peak_reserved": 1,
            "failure_reason": None,
            "instrumented_duration_reported_as_timing": False,
        },
        "timing_collected": False,
        "performance_claim_eligible": False,
    }


def _full_model() -> dict[str, object]:
    fixed = [
        {
            "mode": "fixed_l",
            "step": step,
            "position": 128,
            "reference_checksum": "7" * 64,
            "observed_checksum": "8" * 64,
            "comparison": _comparison(atol=0.125),
        }
        for step in range(3)
    ]
    growing = [
        {
            "mode": "growing_context",
            "step": step,
            "position": 128 + step,
            "reference_checksum": str(step + 1) * 64,
            "observed_checksum": str(step + 4) * 64,
            "comparison": _comparison(atol=0.125),
        }
        for step in range(3)
    ]
    return {
        "passed": True,
        "reference_implementation": "transformers_eager_dynamic_cache",
        "reference_cache_type": "DynamicCache",
        "reference_implementation_restored": True,
        "tolerance_atol": 0.125,
        "tolerance_rtol": 0.02,
        "fixed_repeat_exact": True,
        "fixed_historical_cache_unchanged": True,
        "fixed_steps": fixed,
        "growing_steps": growing,
        "timing_collected": False,
        "performance_claim_eligible": False,
    }


def _result(mode: str) -> dict[str, object]:
    raw_output = "8" * 64 if mode == "eager" else "9" * 64
    operation: dict[str, object] = {
        "operation_fingerprint_sha256": "c" * 64,
        "dispatch_audit_sha256": "d" * 64,
        "allocation_audit_sha256": "e" * 64,
        "gqa_verdict": "gqa_nonmaterialization_verified",
        "gqa_reasons": [],
        "device_kernel_families": {
            "gqa": "pytorch_flash::flash_fwd_splitkv",
            "mha_control": "pytorch_flash::flash_fwd_splitkv",
        },
        "allocation_criterion_id": (
            "phase3_eager_attributed_ephemeral_v1"
            if mode == "eager"
            else "phase3_graph_zero_allocation_v1"
        ),
        "allocation_event_count": 1066 if mode == "eager" else 0,
        "allocation_class_counts": (
            {
                "context_scaled_workspace": 64,
                "fixed_output": 1,
                "fixed_shared_activation": 937,
                "framework_bookkeeping": 64,
            }
            if mode == "eager"
            else {}
        ),
        "allocation_failure_reasons": [],
        "allocation_join_sha256": "f" * 64,
        "paired_allocator_control_sha256": {
            "gqa": "1" * 64,
            "mha_control": "2" * 64,
        },
        "split_k_pair_multiplicity": (
            [{"num_splits": 2, "pair_count": 1}]
            if mode == "eager"
            else []
        ),
        "operation_output_sha256": raw_output,
        "operation_output_finite": True,
    }
    value: dict[str, object] = {
        "passed": True,
        "runner": "fixed_l",
        "graph_mode": mode,
        "batch_size": 1,
        "context_length": 128,
        "output_steps": 1,
        "backend_fallback": False,
        "adapter_config_fingerprint": "a" * 64,
        "cache_layout_fingerprint": "3" * 64,
        "allocated_cache_bytes": 17_072_128,
        "logical_bf16_bytes": 16_908_288,
        "byte_breakdown": {
            "data_bytes": 16_908_288,
            "workspace_bytes": 163_840,
            "padding_bytes": 0,
            "scale_bytes": 0,
            "zero_point_bytes": 0,
            "metadata_bytes": 0,
        },
        "byte_breakdown_sums_to_allocated": True,
        "numerical": {
            "passed": True,
            "small_tensor": {
                "passed": True,
                "reference": "explicit_fp32_gqa_attention",
                "atol": 0.02,
                "rtol": 0.02,
                "records": [
                    {
                        "batch_size": batch,
                        "context_length": length,
                        "mode": record_mode,
                        "boundary_first_finite": True,
                        "boundary_last_finite": True,
                        "comparison": _comparison(),
                    }
                    for batch in (1, 2)
                    for length in (7, 17)
                    for record_mode in (
                        "causal_gqa",
                        "decode_gqa",
                        "causal_mha",
                    )
                ],
                "timing_collected": False,
            },
            "full_model": _full_model(),
            "full_model_graph": (
                _graph_validation() if mode == "cuda_graph" else None
            ),
            "timing_collected": False,
        },
        "cache_geometry": {
            "cache_shape": [32, 1, 8, 129, 128],
            "num_query_heads": 32,
            "num_kv_heads": 8,
            "uses_kv_head_geometry": True,
            "measured_storage_bytes": 16_908_288,
            "predicted_kv_head_bytes": 16_908_288,
            "forbidden_query_head_bytes": 67_633_152,
            "query_head_storage_detected": False,
        },
        "raw_audit_semantics": {
            "semantic_validation_passed": True,
            "scientific_completion_passed": True,
            "transport_terminal_eligible": False,
            "semantic_operations": [operation],
            "raw_audit_index_sha256": "b" * 64,
            "adapter_runtime_fingerprint": "a" * 64,
            "legacy_replay_projection_applied": True,
            "source_revalidated_after_execution": True,
        },
        "output_checksum_join": {
            "passed": True,
            "trusted_reference_observed_sha256": "8" * 64,
            "raw_audit_output_sha256": raw_output,
            **(
                {"scenario_output_sha256": "8" * 64}
                if mode == "eager"
                else {
                    "scenario_eager_sha256": "8" * 64,
                    "scenario_first_replay_sha256": "9" * 64,
                    "scenario_second_replay_sha256": "9" * 64,
                }
            ),
        },
        "eager_allocation_criterion_passed": (
            True if mode == "eager" else None
        ),
    }
    if mode == "eager":
        value.update(
            {
                "output_finite": True,
                "output_sha256": "8" * 64,
                "cache_pointers_stable": True,
                "historical_cache_unchanged": True,
            }
        )
    if mode == "cuda_graph":
        value["graph_validation"] = _graph_validation()
    return value


def _complete_result(
    mode: str,
    run_id: str,
    stage: Path,
) -> dict[str, object]:
    context = _context()
    backend_identity = context["backend_identity"]
    assert isinstance(backend_identity, dict)
    backend_fingerprint = backend_identity["fingerprint"]
    assert isinstance(backend_fingerprint, str)
    runtime_gpu = context["runtime_gpu"]
    container_g0 = context["container_g0"]
    assert isinstance(runtime_gpu, dict)
    assert isinstance(container_g0, dict)
    raw_identities = _container_raw_audit_identities(
        runtime_gpu=runtime_gpu,
        container_g0=container_g0,
        image_config_digest=DIGEST,
    )
    point = Phase3ProcessPoint(
        point_id=f"fixed_l-b1-l128-{mode}-r1",
        runner_kind=RunnerKind.FIXED_L,
        graph_mode=GraphMode(mode),
        batch_size=1,
        context_length=128,
        output_steps=1,
        process_replicate=1,
        stability_member=False,
    )
    operation = Phase3AuditOperationKey.from_point(
        run_id=run_id,
        point=point,
        decode_step=0,
        cache_layout_fingerprint="3" * 64,
        execution_git_sha=SHA,
        plan_fingerprint=PHASE3_PLAN_FINGERPRINTS[
            PHASE3_FIXED_PLAN_PATH
        ],
        hardware_identity_sha256=raw_identities[
            "hardware_identity_sha256"
        ],
        software_identity_sha256=raw_identities[
            "software_identity_sha256"
        ],
        model_identity_sha256="e" * 64,
        backend_identity_sha256=backend_fingerprint,
        source_identity_sha256="6" * 64,
    )
    payloads = {
        "b011_audit": ("b011.audit.json", b"{\"status\":\"PASS\"}"),
        "b011_gqa_chrome_trace": (
            "gqa.geometry.chrome.json",
            b"{\"traceEvents\":[]}",
        ),
        "b011_mha_chrome_trace": (
            "mha.geometry.chrome.json",
            b"{\"traceEvents\":[]}",
        ),
        "b012_allocation_audit": (
            "b012.allocation.json",
            b"{\"status\":\"PASS\"}",
        ),
        "b012_allocator_snapshot": ("allocator.snapshot.json", b"{}"),
        "b012_allocator_trace": ("allocator.trace.json", b"{}"),
        "phase3_session_provenance": (
            "session.provenance.json",
            b"{\"status\":\"complete\"}",
        ),
    }
    raw_root = stage / "raw" / "audits"
    raw_root.mkdir(parents=True, mode=0o700)
    declarations = []
    for kind, (name, payload) in sorted(payloads.items()):
        relative = f"step-0000/{name}"
        target = raw_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        declarations.append(
            Phase3RawAuditFile.from_bytes(
                path=relative,
                kind=kind,
                payload=payload,
            )
        )
    record = Phase3RawAuditOperationRecord(
        schema_version=PHASE3_RAW_AUDIT_OPERATION_SCHEMA_VERSION,
        operation=operation,
        status=RAW_AUDIT_STATUS_COMPLETED,
        failure_reason=None,
        files=tuple(declarations),
    )
    index = Phase3RawAuditRunIndex.create((record,))
    index_raw = json_bytes(index.to_dict())
    (stage / "raw-audit-index.json").write_bytes(index_raw)
    result = _result(mode)
    semantics = result["raw_audit_semantics"]
    assert isinstance(semantics, dict)
    semantics["raw_audit_index_sha256"] = hashlib.sha256(
        index_raw
    ).hexdigest()
    semantic_operations = semantics["semantic_operations"]
    assert isinstance(semantic_operations, list)
    semantic_operation = semantic_operations[0]
    assert isinstance(semantic_operation, dict)
    semantic_operation["operation_fingerprint_sha256"] = (
        operation.operation_fingerprint_sha256
    )
    return result


def _g0() -> dict[str, object]:
    return {
        "schema_version": "e00-manifest-1.1.0",
        "run": {
            "id": "e00-20260725T000000.000000Z-deadbeef0000-abcdef12",
            "gate": "G0",
            "status": "PASS",
            "completed": True,
            "benchmark_timing_collected": False,
        },
        "gate": {
            "aggregate_status": "PASS",
            "checks": [
                {
                    "name": "measurement_container_environment_verified",
                    "status": "PASS",
                }
            ],
        },
        "execution_environment": {
            "kind": "measurement_container",
            "verification_status": "PASS",
            "performance_claim_eligible": False,
            "container": {
                "runtime": "docker",
                "image_config_digest": DIGEST,
                "digest_status": (
                    "verified_against_sanitized_image_inspect"
                ),
            },
        },
        "gpu": {
            "collection_status": "PASS",
            "full_name": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
            "uuid": PHASE3_GPU_UUID,
            "compute_capability": {"major": 12, "minor": 0, "text": "12.0"},
        },
    }


class Phase6AParityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(
            tempfile.mkdtemp(prefix="kvbench-phase6a-parity-test-")
        )
        self.addCleanup(self._cleanup)

    def _cleanup(self) -> None:
        if not self.root.exists():
            return
        for path in sorted(
            self.root.rglob("*"),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            try:
                if path.is_dir() and not path.is_symlink():
                    path.chmod(0o700)
                elif not path.is_symlink():
                    path.chmod(0o600)
            except OSError:
                pass
        self.root.chmod(0o700)
        shutil.rmtree(self.root)

    def test_extended_adapter_provenance_projects_without_mutation(self) -> None:
        payload = {
            "schema_version": "kvbench-phase3-endpoint-session-1.0.0",
            "receipt_sha256": "1" * 64,
            "cache_pointers": {},
            "cache_layout_fingerprint": "2" * 64,
            "operation_fingerprints": [],
            "dispatch_audit_sha256": [],
            "allocation_audit_sha256": [],
            "audit_output_sha256": [],
            "audit_output_finite": [],
            "graph_retained": False,
            "prefix_sha256": "3" * 64,
            "history_chain_sha256": "4" * 64,
            "method_name": "bf16",
            "adapter_version": "kvbench-bf16-method-adapter-1.0.0",
            "adapter_config_fingerprint": "a" * 64,
        }
        raw = canonical_json_bytes(payload)
        observed, projected = _phase3_provenance_for_legacy_replay(
            raw,
            expected_adapter_config_fingerprint="a" * 64,
        )
        self.assertEqual(observed, payload)
        self.assertEqual(raw, canonical_json_bytes(payload))
        legacy = json.loads(projected)
        self.assertNotIn("method_name", legacy)
        self.assertNotIn("adapter_version", legacy)
        self.assertNotIn("adapter_config_fingerprint", legacy)
        with self.assertRaises(Phase6AParityError):
            _phase3_provenance_for_legacy_replay(
                raw,
                expected_adapter_config_fingerprint="b" * 64,
            )

    def test_producer_validator_owns_its_error_type(self) -> None:
        with self.assertRaises(Phase6AParityError):
            validate_finalized_parity_artifact(
                self.root,
                {},
                [],
                set(),
            )

    def test_setup_failure_is_complete_immutable_and_nonclaim(self) -> None:
        published = publish_parity_setup_failure(
            graph_mode="eager",
            execution_git_sha=SHA,
            image_reference="kvbench-measurement:phase6a",
            image_config_digest=DIGEST,
            container_g0_artifact=Path("/run/kvbench/container-g0"),
            error=RuntimeError("synthetic_setup_failure"),
            output_root=self.root / "setup-failure",
            run_id="phase6a-bf16-parity-eager-setup-failure",
            validation_environ={},
        )
        self.assertEqual(published.status, "FAIL")
        validate_local_artifact(published.directory, environ={})
        manifest = json.loads(
            (published.directory / "manifest.json").read_text()
        )
        self.assertIs(manifest["setup_completed"], False)
        self.assertIs(manifest["performance_claim_eligible"], False)
        self.assertIsNone(manifest["result_path"])
        self.assertIn(
            "synthetic_setup_failure",
            (published.directory / "failure.json").read_text(),
        )

    def test_g0_requires_exact_pass_digest_and_gpu_uuid(self) -> None:
        identity = validate_container_g0_manifest(
            _g0(),
            expected_image_config_digest=DIGEST,
        )
        self.assertEqual(identity["gpu_uuid"], PHASE3_GPU_UUID)
        for mutation in ("digest", "uuid", "status"):
            manifest = copy.deepcopy(_g0())
            if mutation == "digest":
                manifest["execution_environment"]["container"][
                    "image_config_digest"
                ] = "sha256:" + "f" * 64
            elif mutation == "uuid":
                manifest["gpu"]["uuid"] = (
                    "GPU-00000000-0000-0000-0000-000000000000"
                )
            else:
                manifest["gate"]["checks"][0]["status"] = "FAIL"
            with self.subTest(mutation=mutation), self.assertRaises(
                Phase6AParityError
            ):
                validate_container_g0_manifest(
                    manifest,
                    expected_image_config_digest=DIGEST,
                )

    def test_exact_eager_and_graph_results_validate(self) -> None:
        validate_parity_result("eager", _result("eager"))
        validate_parity_result("cuda_graph", _result("cuda_graph"))
        bad = copy.deepcopy(_result("eager"))
        bad["context_length"] = 129
        with self.assertRaises(Phase6AParityError):
            validate_parity_result("eager", bad)

    def test_nested_result_fields_and_numeric_types_fail_closed(self) -> None:
        mutations: list[tuple[str, object]] = []

        bad = copy.deepcopy(_result("eager"))
        bad["numerical"]["latency_ms"] = 1.0
        mutations.append(("numerical_extra", bad))

        bad = copy.deepcopy(_result("eager"))
        bad["numerical"]["small_tensor"]["records"][0][
            "latency_ms"
        ] = 1.0
        mutations.append(("record_extra", bad))

        bad = copy.deepcopy(_result("eager"))
        bad["cache_geometry"]["measured_storage_bytes"] = None
        mutations.append(("cache_none", bad))

        bad = copy.deepcopy(_result("eager"))
        bad["byte_breakdown"]["workspace_bytes"] = True
        mutations.append(("breakdown_bool", bad))

        bad = copy.deepcopy(_result("eager"))
        bad["batch_size"] = True
        mutations.append(("batch_bool", bad))

        bad = copy.deepcopy(_result("cuda_graph"))
        bad["graph_validation"]["replay_allocation"][
            "allocation_event_count"
        ] = False
        mutations.append(("graph_count_bool", bad))

        bad = copy.deepcopy(_result("eager"))
        bad["output_checksum_join"]["latency_ms"] = 1.0
        mutations.append(("checksum_extra", bad))

        bad = copy.deepcopy(_result("eager"))
        bad["numerical"]["small_tensor"]["records"][0]["comparison"][
            "max_absolute_error"
        ] = 0
        mutations.append(("comparison_integer", bad))

        bad = copy.deepcopy(_result("eager"))
        bad["raw_audit_semantics"]["semantic_operations"][0][
            "operation_output_sha256"
        ] = int("8" * 64)
        mutations.append(("output_digest_integer", bad))

        bad = copy.deepcopy(_result("eager"))
        bad["raw_audit_semantics"]["semantic_operations"][0][
            "allocation_event_count"
        ] = 0
        bad["raw_audit_semantics"]["semantic_operations"][0][
            "allocation_class_counts"
        ] = {}
        mutations.append(("eager_allocation_summary", bad))

        bad = copy.deepcopy(_result("cuda_graph"))
        bad["raw_audit_semantics"]["semantic_operations"][0][
            "split_k_pair_multiplicity"
        ] = [{"num_splits": 2, "pair_count": 1}]
        mutations.append(("graph_split_k", bad))

        bad = copy.deepcopy(_result("cuda_graph"))
        bad["graph_validation"]["eager_replay_comparison"]["atol"] = 0.01
        mutations.append(("graph_tolerance", bad))

        bad = copy.deepcopy(_result("cuda_graph"))
        allocation = bad["graph_validation"]["replay_allocation"]
        allocation["allocated_after"] = allocation["allocated_before"] + 1
        allocation["allocated_delta"] = 1
        mutations.append(("graph_positive_delta", bad))

        for name, mutation in mutations:
            with self.subTest(name=name), self.assertRaises(
                Phase6AParityError
            ):
                validate_parity_result(
                    (
                        "cuda_graph"
                        if name
                        in {
                            "graph_count_bool",
                            "graph_split_k",
                            "graph_tolerance",
                            "graph_positive_delta",
                        }
                        else "eager"
                    ),
                    mutation,
                )
        bad = copy.deepcopy(_result("eager"))
        bad["raw_audit_semantics"]["semantic_operations"][0][
            "gqa_verdict"
        ] = "unknown"
        with self.assertRaises(Phase6AParityError):
            validate_parity_result("eager", bad)
        bad = copy.deepcopy(_result("cuda_graph"))
        bad["graph_validation"]["replay_allocation"][
            "allocation_event_count"
        ] = 1
        with self.assertRaises(Phase6AParityError):
            validate_parity_result("cuda_graph", bad)
        bad = copy.deepcopy(_result("eager"))
        bad["latency_ms"] = 1.0
        with self.assertRaises(Phase6AParityError):
            validate_parity_result("eager", bad)

    def test_backend_schema_error_is_normalized(self) -> None:
        context = _context()
        backend = context["backend_identity"]
        assert isinstance(backend, dict)
        backend["backend"] = {}
        with self.assertRaises(Phase6AParityError):
            _validate_context(context)

    def test_pass_artifact_is_complete_immutable_and_nonclaim(self) -> None:
        def executor(
            mode: str,
            run_id: str,
            stage: Path,
        ) -> dict[str, object]:
            return _complete_result(mode, run_id, stage)

        published = publish_parity_lane(
            graph_mode="eager",
            context=_context(),
            executor=executor,
            output_root=self.root / "artifacts",
            run_id="phase6a-bf16-parity-eager-test-a",
            validation_environ={},
        )
        self.assertEqual(published.status, "PASS")
        validated = validate_local_artifact(published.directory, environ={})
        self.assertEqual(validated.root_sha256, published.root_sha256)
        manifest = json.loads(
            (published.directory / "manifest.json").read_text()
        )
        self.assertEqual(manifest["quality_status"], "unvalidated")
        self.assertEqual(manifest["claim_eligibility"], "performance_only")
        self.assertIs(manifest["performance_claim_eligible"], False)
        self.assertEqual(
            manifest["measurement_scope"],
            "measurement_container_parity",
        )
        self.assertIs(manifest["timing_collected"], False)
        self.assertIs(manifest["nsight_executed"], False)
        self.assertIs(manifest["performance_profiling_executed"], False)
        self.assertIs(manifest["untimed_admission_trace_collected"], True)
        self.assertIs(manifest["quality_benchmark_executed"], False)
        self.assertIs(manifest["performance_data_frozen"], False)
        self.assertEqual(
            manifest["container"]["image_config_digest"],
            DIGEST,
        )
        self.assertIs(
            manifest["container"]["floating_tag_authoritative"],
            False,
        )
        complete = json.loads(
            (published.directory / "COMPLETE").read_text()
        )
        self.assertIs(complete["written_last"], True)
        for path in (published.directory, *published.directory.rglob("*")):
            if not path.is_symlink():
                self.assertEqual(
                    path.stat().st_mode
                    & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH),
                    0,
                )
        self.assertNotIn("latency", json.dumps(manifest).lower())

    def test_failed_eager_is_preserved_and_graph_still_runs(self) -> None:
        observed: list[str] = []

        def executor(
            mode: str,
            run_id: str,
            stage: Path,
        ) -> dict[str, object]:
            observed.append(mode)
            if mode == "eager":
                trace = stage / "raw" / "gqa.geometry.chrome.json"
                trace.parent.mkdir(parents=True)
                trace.write_text("{}", encoding="utf-8")
                raise RuntimeError("synthetic_eager_failure")
            return _complete_result(mode, run_id, stage)

        eager = publish_parity_lane(
            graph_mode="eager",
            context=_context(),
            executor=executor,
            output_root=self.root / "artifacts",
            validation_environ={},
        )
        graph = publish_parity_lane(
            graph_mode="cuda_graph",
            context=_context(),
            executor=executor,
            output_root=self.root / "artifacts",
            validation_environ={},
        )
        self.assertEqual(observed, ["eager", "cuda_graph"])
        self.assertEqual([eager.status, graph.status], ["FAIL", "PASS"])
        for item in (eager, graph):
            directory = item.directory
            validate_local_artifact(directory, environ={})
            manifest = json.loads((directory / "manifest.json").read_text())
            if item.status == "FAIL":
                self.assertEqual(manifest["failure_path"], "failure.json")
                self.assertIs(
                    manifest["untimed_admission_trace_collected"],
                    False,
                )
                self.assertFalse((directory / "result.json").exists())
                self.assertIn(
                    "synthetic_eager_failure",
                    (directory / "failure.json").read_text(),
                )

    def test_make_uses_two_processes_and_complete_model_cache(self) -> None:
        makefile = (REPOSITORY_ROOT / "Makefile").read_text(encoding="utf-8")
        target = makefile.split(
            "phase6a-bf16-container-parity:", 1
        )[1].split("publish-artifact-r2:", 1)[0]
        self.assertIn("for graph_mode in eager cuda_graph", target)
        self.assertIn('--graph-mode "$$graph_mode"', target)
        self.assertIn("if ! docker start --attach", target)
        self.assertIn("models--meta-llama--Llama-3.1-8B-Instruct", target)
        self.assertIn("src=$$model_root,dst=/root/.cache/huggingface", target)
        self.assertNotIn("src=$$model_snapshot", target)

    def test_parity_reuses_preflight_artifact_lifecycle(self) -> None:
        source = (
            REPOSITORY_ROOT / "scripts" / "phase6a_bf16_parity.py"
        ).read_text(encoding="utf-8")
        self.assertIn("finalize_stage(stage=stage, final=final", source)
        for duplicate in (
            "def _write_exclusive(",
            "def _reserve(",
            "def _freeze(",
            "def _finalize(",
        ):
            self.assertNotIn(duplicate, source)

    def test_existing_run_is_never_replaced(self) -> None:
        output = self.root / "artifacts"
        run_id = "phase6a-bf16-parity-eager-test-reuse"
        first = publish_parity_lane(
            graph_mode="eager",
            context=_context(),
            executor=_complete_result,
            output_root=output,
            run_id=run_id,
            validation_environ={},
        )
        before = (first.directory / "checksums.sha256").read_bytes()
        with self.assertRaises(Phase6AParityError):
            publish_parity_lane(
                graph_mode="eager",
                context=_context(),
                executor=_complete_result,
                output_root=output,
                run_id=run_id,
                validation_environ={},
            )
        self.assertEqual(
            (first.directory / "checksums.sha256").read_bytes(),
            before,
        )
        self.assertEqual(
            validate_local_artifact(
                first.directory,
                environ={},
            ).root_sha256,
            first.root_sha256,
        )

    def test_context_rejects_image_and_model_drift(self) -> None:
        for field in ("image", "revision"):
            context = copy.deepcopy(_context())
            if field == "image":
                context["container_g0"]["image_config_digest"] = (
                    "sha256:" + "0" * 64
                )
            else:
                context["model_identity"]["tokenizer_revision"] = "0" * 40
            with self.subTest(field=field), self.assertRaises(
                Phase6AParityError
            ):
                publish_parity_lane(
                    graph_mode="eager",
                    context=context,
                    executor=lambda mode, identifier, stage: _result(mode),
                    output_root=self.root / field,
                    run_id=f"phase6a-bf16-parity-eager-{field}",
                    validation_environ={},
                )


if __name__ == "__main__":
    unittest.main()
