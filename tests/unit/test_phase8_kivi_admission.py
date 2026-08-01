"""Focused CPU-only tests for the strict Phase 8 admission join."""

from __future__ import annotations

import dataclasses
import copy
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from kvbench.adapters.kivi import (
    KIVI_BGEMV2_HOST_STUB_OFFSET,
    KIVI_BGEMV4_HOST_STUB_OFFSET,
    KIVIMethodAdapter,
    KIVI_NEW_PACK_SHA256,
)
from kvbench.runtime.kivi_admission import (
    KIVIAdmissionError,
    OFFICIAL_KIVI_HOST_STUB_OFFSETS,
    PHASE8_ADMISSION_GRID,
    PHASE8_DECISION_0026_ENDPOINT_COMMIT,
    PHASE8_HISTORICAL_ADAPTER_VERSION,
    Phase8HistoricalSourceAuthority,
    audit_kivi_execution_path,
    build_phase8_method_admission_report,
    derive_phase8_admission_evidence,
    _supervision_passed,
    require_authorized_kivi_environment,
    require_exact_phase8_grid,
    summarize_phase8_accounting,
)
from kvbench.runtime.allocation_attribution import (
    cuda_allocator_rounded_minimum,
    instantiate_decision_0013_phase8_kivi_rules,
    preserve_allocator_evidence,
)
from kvbench.runtime.artifacts import phase8_artifact_store, sha256_file
from kvbench.runtime.kivi_allocation import (
    KIVIAllocationBinding,
    _derived_operation_summary,
    _geometry,
    derive_kivi_allocation_attribution,
    raw_file_sha256,
)
from kvbench.runtime.kivi_session import (
    build_kivi_operation_keys,
    phase8_kivi_backend_fingerprint,
)
from kvbench.schema import (
    ClaimEligibility,
    GraphMode,
    MeasurementScope,
    QualityExecutionState,
    QualityValidationState,
    RunnerKind,
    RunStatus,
)
from kvbench.schema.phase8 import (
    PHASE8_ADMISSION_CHECK_IDS,
    PHASE8_AUTHORIZED_CONTAINER_DIGEST,
    PHASE8_BASE_TREE,
    PHASE8_DECISION_0018_PATCH_SHA256,
    PHASE8_EXTENSION_SHA256,
    PHASE8_FIXTURE_ROOT_DIGEST,
    PHASE8_OFFICIAL_COMMIT,
    PHASE8_PATCHED_TREE,
    Phase8ByteAccounting,
    Phase8ByteBreakdown,
    Phase8RunManifest,
)
from scripts.r2_artifact import validate_local_artifact
from scripts.phase8_kivi_admission import validate_local_admission


def _synthetic_historical_source_authority(
    *,
    repository_root: Path,
    execution_git_sha: str,
    manifest_adapter_sha256: str,
) -> Phase8HistoricalSourceAuthority:
    adapter = sha256_file(
        repository_root / "src/kvbench/adapters/kivi.py"
    )
    if adapter != manifest_adapter_sha256:
        raise KIVIAdmissionError("synthetic adapter authority differs")
    return Phase8HistoricalSourceAuthority(
        execution_git_sha=execution_git_sha,
        current_git_sha=execution_git_sha,
        adapter_source_sha256=adapter,
        cache_source_sha256=sha256_file(
            repository_root / "src/kvbench/runtime/kivi_cache.py"
        ),
        endpoint_source_sha256=sha256_file(
            repository_root / "src/kvbench/runtime/bf16_endpoint.py"
        ),
        endpoint_transition_commit=PHASE8_DECISION_0026_ENDPOINT_COMMIT,
    )


def _synthetic_git_object_query(
    repository_root: Path,
    *arguments: str,
    binary: bool = False,
) -> bytes | str:
    if len(arguments) == 3 and arguments[:2] == ("cat-file", "blob"):
        _, separator, relative_path = arguments[2].partition(":")
        if not separator:
            raise KIVIAdmissionError("synthetic Git blob path is absent")
        payload = (repository_root / relative_path).read_bytes()
        return payload if binary else payload.decode("ascii").strip()
    raise KIVIAdmissionError("unexpected synthetic Git object query")


def _breakdown() -> Phase8ByteBreakdown:
    return Phase8ByteBreakdown(
        quantized_historical_k_payload=16,
        quantized_historical_v_payload=16,
        k_scales=4,
        k_zeros=4,
        v_scales=4,
        v_zeros=4,
        other_metadata=0,
        residual_k=16,
        residual_v=16,
        fp16_staging=16,
        quantization_staging=16,
        padding_alignment=0,
        persistent_workspace=16,
        value_rollover_shift_scratch=16,
        block_group_rounding=0,
    )


def _accounting(
    capacity: int,
    *,
    active_context: int,
) -> Phase8ByteAccounting:
    breakdown = _breakdown()
    allocated = breakdown.total
    logical = capacity * 1024
    return Phase8ByteAccounting(
        capacity=capacity,
        active_context=active_context,
        allocated_bytes=allocated,
        predicted_allocated_bytes=allocated,
        active_storage_bytes=allocated,
        logical_bf16_allocated_bytes=logical,
        logical_bf16_active_bytes=active_context * 1024,
        rho_alloc=allocated / logical,
        r_alloc=logical / allocated,
        predicted_relative_error=0.0,
        temporary_peak_bytes=0,
        breakdown=breakdown,
        r_hbm=None,
    )


def _manifests() -> tuple[Phase8RunManifest, ...]:
    bits = {
        "k4v4": (4, 4),
        "k2v4": (2, 4),
        "k2v2": (2, 2),
        "k4v2": (4, 2),
    }
    records: list[Phase8RunManifest] = []
    for index, point in enumerate(PHASE8_ADMISSION_GRID):
        capacity = point.context_length + point.output_steps
        active_context = (
            point.context_length
            if point.runner_kind is RunnerKind.FIXED_L
            else capacity
        )
        k_bits, v_bits = bits[point.configuration]
        records.append(
            Phase8RunManifest(
                schema_version=Phase8RunManifest.SCHEMA_VERSION,
                artifact_schema_version=(
                    Phase8RunManifest.ARTIFACT_SCHEMA_VERSION
                ),
                run_id=f"phase8-kivi-admission-{index:02d}",
                status=RunStatus.COMPLETED,
                git_sha="8" * 40,
                git_dirty=False,
                created_at_utc="2026-07-27T00:00:00Z",
                started_at_utc="2026-07-27T00:00:01Z",
                finished_at_utc="2026-07-27T00:01:00Z",
                runner_kind=point.runner_kind,
                graph_mode=point.graph_mode,
                method_configuration=point.configuration,
                k_bits=k_bits,
                v_bits=v_bits,
                method_config_fingerprint="1" * 64,
                method_fingerprint=f"{index + 1:064x}",
                adapter_version=PHASE8_HISTORICAL_ADAPTER_VERSION,
                adapter_source_sha256="a" * 64,
                official_base_commit=PHASE8_OFFICIAL_COMMIT,
                official_base_tree=PHASE8_BASE_TREE,
                patched_tree=PHASE8_PATCHED_TREE,
                decision_0018_patch_sha256=(
                    PHASE8_DECISION_0018_PATCH_SHA256
                ),
                extension_sha256=PHASE8_EXTENSION_SHA256,
                fixture_root_digest=PHASE8_FIXTURE_ROOT_DIGEST,
                group_size=32,
                residual_length=32,
                dtype_boundary=(
                    "bf16_to_fp16_official_kivi_to_bf16"
                ),
                cache_layout_fingerprint=f"{index + 20:064x}",
                authorized_container_digest=(
                    PHASE8_AUTHORIZED_CONTAINER_DIGEST
                ),
                batch_size=1,
                context_length=point.context_length,
                output_steps=point.output_steps,
                capacity=capacity,
                accounting=_accounting(
                    capacity,
                    active_context=active_context,
                ),
                quality_status=QualityValidationState.UNVALIDATED,
                claim_eligibility=ClaimEligibility.PERFORMANCE_ONLY,
                performance_claim_eligible=False,
                measurement_scope=(
                    MeasurementScope.MEASUREMENT_CONTAINER_ADMISSION
                ),
                quality_execution=QualityExecutionState.LOCKED,
                quality_benchmark_executed=False,
                performance_data_frozen=False,
                speedup_calculated=False,
                full_scan_state="CLOSED",
                inventory_path="artifact_inventory.json",
                failure_reason=None,
            )
        )
    return tuple(records)


def _passing_audit():
    source = """
def decode():
    for query_head in range(32):
        kv_head = query_head // KIVI_GQA_GROUP_SIZE
"""
    kernels = (
        "bgemv2_kernel_outer_dim(__half const*)",
        "bgemv4_kernel_outer_dim(__half const*)",
        "aten::bmm",
    )
    return audit_kivi_execution_path(
        kernel_names=kernels,
        repeated_kernel_names=kernels,
        runtime_event_names=("cudaLaunchKernel",),
        temporary_shapes={
            "query_workspace": (1, 32, 1, 128),
            "key_staging": (1, 8, 1, 128),
            "value_staging": (1, 8, 1, 128),
        },
        adapter_hot_path_source=source,
        observed_extension_sha256=PHASE8_EXTENSION_SHA256,
        observed_new_pack_sha256=KIVI_NEW_PACK_SHA256,
        official_commit=PHASE8_OFFICIAL_COMMIT,
        official_base_tree=PHASE8_BASE_TREE,
        patched_tree=PHASE8_PATCHED_TREE,
        decision_0018_patch_sha256=(
            PHASE8_DECISION_0018_PATCH_SHA256
        ),
        fixture_root_digest=PHASE8_FIXTURE_ROOT_DIGEST,
        host_stub_offsets=OFFICIAL_KIVI_HOST_STUB_OFFSETS,
        backend_fallback_observed=False,
        cache_growth_observed=False,
    )


def _initial_manifest(
    manifest: Phase8RunManifest,
) -> Phase8RunManifest:
    return dataclasses.replace(
        manifest,
        status=RunStatus.CREATED,
        started_at_utc=None,
        finished_at_utc=None,
        inventory_path=None,
        failure_reason=None,
    )


def _supervision(
    *,
    argv: tuple[str, ...],
    stdout: bytes = b"",
    stderr: bytes = b"",
) -> dict[str, object]:
    return {
        "schema_version": (
            "kvbench-generic-supervised-command-result-1.0.0"
        ),
        "identity": {
            "pid": 1234,
            "start_time_ticks": 5678,
            "parent_pid": 4321,
        },
        "command": {
            "argv": list(argv),
            "working_directory": "/home/rockrock/cmu_paper",
            "environment_sha256": "e" * 64,
            "command_fingerprint": "f" * 64,
            "shell": False,
        },
        "timeout": {
            "timeout_seconds": 900.0,
            "timed_out": False,
            "terminate_requested": False,
            "kill_requested": False,
        },
        "returncode": 0,
        "pidfd": {
            "supported": True,
            "opened": True,
            "descriptor": 7,
            "closed": True,
        },
        "direct_child": {
            "verified": True,
            "expected_parent_pid": 4321,
            "parent_pid_verified": True,
            "start_time_ticks_verified": True,
            "process_handle_retained": True,
        },
        "final_reap": {"completed": True, "count": 1},
        "stdout": {
            "bytes": len(stdout),
            "sha256": hashlib.sha256(stdout).hexdigest(),
        },
        "stderr": {
            "bytes": len(stderr),
            "sha256": hashlib.sha256(stderr).hexdigest(),
        },
    }


def _allocator_memory_stats(
    *,
    count: int = 0,
    requested_bytes: int = 0,
    block_bytes: int = 0,
) -> dict[str, int]:
    return {
        "allocation.all.allocated": count,
        "requested_bytes.all.allocated": requested_bytes,
        "allocated_bytes.all.allocated": block_bytes,
        "allocation.all.freed": count,
        "requested_bytes.all.freed": requested_bytes,
        "allocated_bytes.all.freed": block_bytes,
        "segment.all.allocated": 0,
        "segment.all.freed": 0,
        "num_device_alloc": 0,
        "num_device_free": 0,
        "num_alloc_retries": 0,
        "num_ooms": 0,
    }


def _allocator_accounting(
    binding: KIVIAllocationBinding,
    *,
    role: str,
    timestamp: int,
) -> dict[str, object]:
    return {
        "schema_version": "kvbench-phase3-memory-accounting-2.0.0",
        "operation_fingerprint_sha256": (
            binding.operation_fingerprint_sha256
        ),
        "sample_role": role,
        "timestamp_ns": timestamp,
        "device": "cuda:0",
        "device_index": 0,
        "gpu_uuid": "GPU-phase8-admission-unit-test",
        "allocated_bytes": 1_024,
        "reserved_bytes": 2_048,
        "device_free_bytes": 8_192,
        "device_total_bytes": 10_240,
        "device_used_bytes": 2_048,
    }


def _allocator_trace(
    binding: KIVIAllocationBinding,
) -> tuple[list[dict[str, object]], int, int]:
    if binding.graph_mode == "cuda_graph":
        return [], 0, 0
    rules = instantiate_decision_0013_phase8_kivi_rules(
        geometry=_geometry(binding),
        backend_identity=binding.backend_identity,
        composition_binding_sha256=binding.identity_sha256,
    )
    trace: list[dict[str, object]] = []
    address = 0x100000
    requested_total = 0
    block_total = 0
    for policy in rules.permitted_allocation_policies:
        size = next(iter(policy.allowed_requested_bytes))
        block = cuda_allocator_rounded_minimum(size)
        python_frame = policy.required_python_frames[0]
        cpp_frame = policy.required_cpp_frames[0]
        for _ in range(policy.exact_count):
            trace.extend(
                (
                    {
                        "action": "alloc",
                        "addr": address,
                        "size": size,
                        "stream": 7,
                        "allocated_block_size": block,
                        "python_stack": [
                            {
                                "name": python_frame.function_name,
                                "filename": python_frame.source_suffix,
                                "line": 1,
                            }
                        ],
                        "cpp_stack": [
                            {
                                "name": cpp_frame.function_name,
                                "filename": cpp_frame.source_suffix,
                                "line": 1,
                            }
                        ],
                    },
                    {
                        "action": "free_requested",
                        "addr": address,
                        "size": size,
                        "stream": 7,
                    },
                    {
                        "action": "free_completed",
                        "addr": address,
                        "size": size,
                        "stream": 7,
                    },
                )
            )
            address += block + 0x1000
            requested_total += size
            block_total += block
    return trace, requested_total, block_total


def _write_allocator_operation(
    *,
    run: object,
    repository: Path,
    manifest: Phase8RunManifest,
    step: int,
) -> dict[str, object]:
    operation_key = build_kivi_operation_keys(
        configuration=manifest.method_configuration,
        runner_kind=manifest.runner_kind,
        graph_mode=manifest.graph_mode,
        starting_context=manifest.context_length,
        output_steps=manifest.output_steps,
    )[step]
    binding = KIVIAllocationBinding(
        configuration=manifest.method_configuration,
        runner_kind=manifest.runner_kind.value,
        graph_mode=manifest.graph_mode.value,
        historical_context=operation_key.historical_context,
        attended_context=operation_key.attended_context,
        operation_fingerprint_sha256=(
            operation_key.operation_fingerprint_sha256
        ),
        cache_layout_fingerprint=manifest.cache_layout_fingerprint,
        method_fingerprint=manifest.method_fingerprint,
        backend_identity=phase8_kivi_backend_fingerprint(),
        adapter_source_sha256=sha256_file(
            repository / "src/kvbench/adapters/kivi.py"
        ),
        cache_source_sha256=sha256_file(
            repository / "src/kvbench/runtime/kivi_cache.py"
        ),
        endpoint_source_sha256=sha256_file(
            repository / "src/kvbench/runtime/bf16_endpoint.py"
        ),
        authorized_container_digest=PHASE8_AUTHORIZED_CONTAINER_DIGEST,
        official_commit=PHASE8_OFFICIAL_COMMIT,
        patched_tree=PHASE8_PATCHED_TREE,
        decision_0018_patch_sha256=(
            PHASE8_DECISION_0018_PATCH_SHA256
        ),
        extension_sha256=PHASE8_EXTENSION_SHA256,
    )
    trace, requested_bytes, block_bytes = _allocator_trace(binding)
    snapshot = {"device_traces": [trace]}
    before_stats = _allocator_memory_stats()
    after_stats = _allocator_memory_stats(
        count=sum(item.get("action") == "alloc" for item in trace),
        requested_bytes=requested_bytes,
        block_bytes=block_bytes,
    )
    before_accounting = _allocator_accounting(
        binding,
        role="before",
        timestamp=10 + step * 2,
    )
    after_accounting = _allocator_accounting(
        binding,
        role="after",
        timestamp=11 + step * 2,
    )
    state_before = {
        "cache_pointers_sha256": "f" * 64,
        "active_context": binding.historical_context,
    }
    state_after = {
        **state_before,
        "active_context": (
            binding.attended_context
            if manifest.runner_kind is RunnerKind.GROWING_CONTEXT
            else binding.historical_context
        ),
    }
    witness = {
        "schema_version": (
            "kvbench-phase8-kivi-allocation-operation-witness-1.0.0"
        ),
        "binding_sha256": binding.identity_sha256,
        "operation_fingerprint_sha256": (
            binding.operation_fingerprint_sha256
        ),
        "state_before": state_before,
        "state_after": state_after,
        "measured_output": {"sha256": "d" * 64, "finite": True},
    }
    attribution, memory, criterion = derive_kivi_allocation_attribution(
        binding=binding,
        snapshot=snapshot,
        trace=tuple(trace),
        memory_stats_before=before_stats,
        memory_stats_after=after_stats,
        memory_accounting_before=before_accounting,
        memory_accounting_after=after_accounting,
        operation_witness=witness,
    )
    summary, expected_count, expected_bytes, unknown_count = (
        _derived_operation_summary(
            attribution=attribution,
            memory=memory,
            criterion=criterion,
            binding=binding,
        )
    )
    history = attribution.history_integrity
    assert history is not None
    audit_payload = {
        "schema_version": (
            "kvbench-phase8-kivi-allocation-attribution-1.0.0"
        ),
        "run_kind": "allocation_audit",
        "evidence_status": "complete",
        "execution_mode": binding.execution_mode,
        "binding": binding.to_dict(),
        "binding_sha256": binding.identity_sha256,
        "memory": memory.to_dict(),
        "attribution": attribution.to_dict(),
        "criterion": criterion.to_dict(),
        "expected_allocation_event_count": expected_count,
        "expected_allocation_event_bytes": expected_bytes,
        "observed_allocation_event_bytes": summary["raw"][
            "allocation_event_bytes"
        ],
        "unknown_allocation_count": unknown_count,
        "operation_witness": witness,
        "profiler_timing_reported": False,
        "instrumented_duration_reported_as_timing": False,
        "normal_benchmark_timing_eligible": False,
    }
    with tempfile.TemporaryDirectory(
        prefix="phase8-kivi-allocation-unit-"
    ) as temporary:
        evidence_directory = Path(temporary)
        files = preserve_allocator_evidence(
            evidence_directory,
            snapshot=snapshot,
            trace=trace,
            memory_stats_before=before_stats,
            memory_stats_after=after_stats,
            memory_accounting_before=before_accounting,
            memory_accounting_after=after_accounting,
            operation_witness=witness,
            expected_snapshot_sha256=history.raw_snapshot_sha256,
            expected_trace_sha256=history.raw_trace_sha256,
            audit_payload=audit_payload,
        )
        basenames = (
            files.snapshot_file,
            files.trace_file,
            files.memory_stats_before_file,
            files.memory_stats_after_file,
            files.memory_accounting_before_file,
            files.memory_accounting_after_file,
            files.operation_witness_file,
            files.audit_file,
            files.audit_sha256_file,
        )
        raw_payloads = {
            name: (evidence_directory / name).read_bytes()
            for name in basenames
        }
    evidence_root = f"allocation/operations/step-{step:04d}"
    for name, payload in raw_payloads.items():
        run.write_bytes(f"{evidence_root}/{name}", payload)
    return {
        **summary,
        "raw_files": files.to_dict(),
        "raw_evidence_root": evidence_root,
        "raw_evidence_sha256": {
            name: raw_file_sha256(payload)
            for name, payload in sorted(raw_payloads.items())
        },
    }


def _allocation(
    *,
    run: object,
    repository: Path,
    manifest: Phase8RunManifest,
) -> dict[str, object]:
    graph_required = manifest.graph_mode is GraphMode.CUDA_GRAPH
    operations: list[dict[str, object]] = []
    for step in range(manifest.output_steps):
        operations.append(
            _write_allocator_operation(
                run=run,
                repository=repository,
                manifest=manifest,
                step=step,
            )
        )
    return {
        "passed": True,
        "operation_allocations": operations,
        "unknown_allocation_count": 0,
        "all_ephemeral_allocations_attributed": (
            None if graph_required else True
        ),
        "graph_required": graph_required,
        "graph_passed": True,
        "pointers_stable": True,
        "outputs": [
            {"sha256": "d" * 64, "finite": True}
            for _ in range(manifest.output_steps)
        ],
    }


def _launcher_probe(
    manifest: Phase8RunManifest,
) -> dict[str, object]:
    records = [
        {
            "bits": bits,
            "kernel_family": f"bgemv{bits}_kernel_outer_dim",
            "group_size": 32,
            "num_query_heads": 32,
            "num_kv_heads": 8,
        }
        for bits in sorted({manifest.k_bits, manifest.v_bits})
    ]
    return {
        "schema_version": (
            "kvbench-phase8-kivi-runtime-launch-observation-1.0.0"
        ),
        "passed": True,
        "configuration": manifest.method_configuration,
        "runner_kind": manifest.runner_kind.value,
        "graph_mode": manifest.graph_mode.value,
        "observed_step": 0,
        "first_sequence": records,
        "second_sequence": records,
        "first_output_sha256": "d" * 64,
        "second_output_sha256": "d" * 64,
        "first_output_finite": True,
        "second_output_finite": True,
        "stable_post_warmup_sequence": True,
        "expected_bits": sorted({manifest.k_bits, manifest.v_bits}),
        "observed_bits": sorted({manifest.k_bits, manifest.v_bits}),
        "instrumented_audit_separate": True,
        "allocation_audit_instrumented": False,
        "normal_timing_instrumented": False,
        "host_synchronization_outside_hot_path": True,
    }


def _method_record(manifest: Phase8RunManifest) -> dict[str, object]:
    return {
        "schema_version": "kvbench-phase8-kivi-method-record-1.0.0",
        "method": "kivi",
        "method_config_id": "kivi",
        "method_config_path": "configs/methods/kivi.yaml",
        "method_config_file_sha256": manifest.method_config_fingerprint,
        "configuration": manifest.method_configuration,
        "variant_role": (
            "held_out"
            if manifest.method_configuration == "k4v2"
            else "primary"
        ),
        "admission_role": (
            "held_out_validation"
            if manifest.method_configuration == "k4v2"
            else "mandatory"
        ),
        "key_bits": manifest.k_bits,
        "value_bits": manifest.v_bits,
        "group_size": 32,
        "residual_length": 32,
        "method_config_fingerprint": manifest.method_config_fingerprint,
        "method_fingerprint": manifest.method_fingerprint,
        "adapter_version": manifest.adapter_version,
        "adapter_source_sha256": manifest.adapter_source_sha256,
        "cache_layout_fingerprint": manifest.cache_layout_fingerprint,
        "official_base_commit": PHASE8_OFFICIAL_COMMIT,
        "official_base_tree": PHASE8_BASE_TREE,
        "patched_tree": PHASE8_PATCHED_TREE,
        "decision_0018_patch_sha256": (
            PHASE8_DECISION_0018_PATCH_SHA256
        ),
        "extension_sha256": PHASE8_EXTENSION_SHA256,
        "fixture_root_digest": PHASE8_FIXTURE_ROOT_DIGEST,
        "dtype_boundary": "bf16_to_fp16_official_kivi_to_bf16",
        "gqa_mapping": "query_head // 4",
        "native_kv_head_storage": True,
        "r_hbm": None,
    }


def _point_payload(
    manifest: Phase8RunManifest,
    *,
    allocation: dict[str, object],
    native_gqa: bool = True,
) -> tuple[dict[str, object], dict[str, object]]:
    active_context = (
        manifest.context_length
        if manifest.runner_kind is RunnerKind.FIXED_L
        else manifest.capacity
    )
    launcher = _launcher_probe(manifest)
    point = {
        "schema_version": "kvbench-phase8-kivi-point-validation-1.0.0",
        "run_id": manifest.run_id,
        "git_sha": manifest.git_sha,
        "container_digest": PHASE8_AUTHORIZED_CONTAINER_DIGEST,
        "configuration": manifest.method_configuration,
        "runner_kind": manifest.runner_kind.value,
        "graph_mode": manifest.graph_mode.value,
        "batch_size": 1,
        "context_length": manifest.context_length,
        "output_steps": manifest.output_steps,
        "capacity": manifest.capacity,
        "allocation": allocation,
        "launcher_probe": launcher,
        "accounting": manifest.accounting.to_dict(),
        "runtime_committed_context": active_context,
        "byte_breakdown_sum": manifest.accounting.allocated_bytes,
        "allocated_bytes": manifest.accounting.allocated_bytes,
        "rho_alloc": manifest.accounting.rho_alloc,
        "r_alloc": manifest.accounting.r_alloc,
        "reciprocal_product_error": 0.0,
        "output_finite": True,
        "cache_pointers_stable": True,
        "historical_cache_unchanged": True,
        "native_gqa": native_gqa,
        "rollover_active_lengths": (
            [31, 32, 33, 34]
            if manifest.runner_kind is RunnerKind.GROWING_CONTEXT
            else None
        ),
        "speedup_calculated": False,
        "r_hbm": None,
        "quality_status": "unvalidated",
        "performance_claim_eligible": False,
        "measurement_scope": "measurement_container_admission",
        "passed": True,
    }
    return point, launcher


def _write_point_payloads(
    run: object,
    manifest: Phase8RunManifest,
    *,
    repository: Path,
    native_gqa: bool = True,
) -> tuple[dict[str, object], dict[str, object]]:
    allocation = _allocation(
        run=run,
        repository=repository,
        manifest=manifest,
    )
    point, launcher = _point_payload(
        manifest,
        allocation=allocation,
        native_gqa=native_gqa,
    )
    active_context = (
        manifest.context_length
        if manifest.runner_kind is RunnerKind.FIXED_L
        else manifest.capacity
    )
    geometry = {
        "native_kv_head_storage": True,
        "gqa_materialized": False,
        "num_query_heads": 32,
        "num_kv_heads": 8,
    }
    runner = {
        "cache_accounting": {
            "allocated_bytes": manifest.accounting.allocated_bytes,
            "predicted_tensor_bytes": (
                manifest.accounting.predicted_allocated_bytes
            ),
            "active_context": active_context,
        },
        "cache_byte_breakdown": (
            manifest.accounting.breakdown.to_dict()
        ),
        "cache_layout_fingerprint": manifest.cache_layout_fingerprint,
        "output_checksum": "d" * 64,
        "output_finite": True,
        "cache_pointers_stable": True,
        "historical_cache_unchanged": True,
        "gqa_cache_geometry": geometry,
        "measurement_scope": "measurement_container_admission",
        "speedup_calculated": False,
    }
    numerical = {
        "output_checksum": "d" * 64,
        "output_finite": True,
        "graph": (
            {
                "fallback": False,
                "consecutive_replay_outputs_exact": True,
            }
            if manifest.graph_mode is GraphMode.CUDA_GRAPH
            else None
        ),
        "eager_graph_comparison": (
            {"passed": True}
            if manifest.graph_mode is GraphMode.CUDA_GRAPH
            else None
        ),
        "decode_atol": 0.02,
        "decode_rtol": 0.02,
    }
    run.write_json("config/method.json", _method_record(manifest))
    run.write_json(
        "environment/container_identity.json",
        {
            "schema_version": "kvbench-phase8-container-runtime-1.0.0",
            "container_digest": PHASE8_AUTHORIZED_CONTAINER_DIGEST,
            "execution_environment": "measurement_container",
            "git_sha": manifest.git_sha,
            "image_mutated": False,
            "packages_installed": False,
            "network_enabled": False,
            "credentials_passed": False,
        },
    )
    run.write_json("raw/runner.json", runner)
    run.write_json("validation/point.json", point)
    run.write_json("allocation/full-model.json", point["allocation"])
    run.write_json(
        "execution-path/launcher-observation.json",
        launcher,
    )
    run.write_json("accounting/bytes.json", manifest.accounting.to_dict())
    run.write_json(
        "gqa/full-model.json",
        {
            "native_gqa": True,
            "mapping": "query_head // 4",
            "geometry": geometry,
        },
    )
    run.write_json("numerical/output.json", numerical)
    return point, launcher


def _write_exact_test(
    run: object,
    *,
    repository: Path,
    evidence_name: str,
    source_path: str,
    test_names: tuple[str, ...],
    arbitrary_result: bool = False,
) -> None:
    stdout = b""
    stderr = (
        "\n".join(
            [
                *(f"{name} ... ok" for name in test_names),
                "",
                f"Ran {len(test_names)} tests in 1.0s",
                "",
                "OK",
                "",
            ]
        )
    ).encode()
    prefix = f"validation/{evidence_name}"
    command = (
        "/opt/kvbench/.venv/bin/python",
        f"/workspace/{source_path}",
        "-v",
    )
    run.write_bytes(f"{prefix}/stdout.txt", stdout)
    run.write_bytes(f"{prefix}/stderr.txt", stderr)
    if arbitrary_result:
        run.write_bytes(f"{prefix}/result.json", b"arbitrary bytes")
        return
    run.write_json(
        f"{prefix}/result.json",
        {
            "schema_version": (
                "kvbench-phase8-exact-container-test-1.0.0"
            ),
            "evidence_name": evidence_name,
            "command": list(command),
            "source_path": source_path,
            "source_sha256": sha256_file(repository / source_path),
            "exit_code": 0,
            "timed_out": False,
            "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
            "container_digest": PHASE8_AUTHORIZED_CONTAINER_DIGEST,
            "performance_timing": False,
            "process_supervision": _supervision(
                argv=command,
                stdout=stdout,
                stderr=stderr,
            ),
            "passed": True,
        },
    )


def _write_sanitizer(run: object, *, repository: Path) -> None:
    probe = {
        "authority": {
            "official_commit": PHASE8_OFFICIAL_COMMIT,
            "official_base_tree": PHASE8_BASE_TREE,
            "patched_tree": PHASE8_PATCHED_TREE,
            "decision_0018_patch_sha256": (
                PHASE8_DECISION_0018_PATCH_SHA256
            ),
            "extension_sha256": PHASE8_EXTENSION_SHA256,
            "new_pack_sha256": KIVI_NEW_PACK_SHA256,
            "fixture_root_sha256": PHASE8_FIXTURE_ROOT_DIGEST,
        },
        "configurations": [
            {
                "configuration": "k4v4",
                "gemv_bits": [4],
                "rollover": "L31_to_L33",
                "active_context": 33,
                "key_history_tokens": 32,
                "key_residual_tokens": 1,
                "value_history_tokens": 1,
                "value_residual_tokens": 32,
                "method_fingerprint": "1" * 64,
                "finite": True,
                "token_movement": "exact",
            },
            {
                "configuration": "k2v2",
                "gemv_bits": [2],
                "active_context": 33,
                "method_fingerprint": "2" * 64,
                "finite": True,
                "token_movement": "exact",
            },
        ],
        "container_digest": PHASE8_AUTHORIZED_CONTAINER_DIGEST,
        "kernel_families": [
            "bgemv2_kernel_outer_dim",
            "bgemv4_kernel_outer_dim",
        ],
        "status": "pass",
    }
    stdout = (
        json.dumps(probe, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    stderr = (
        "LEAK SUMMARY: 0 bytes leaked in 0 allocations\n"
        "ERROR SUMMARY: 0 errors\n"
    ).encode()
    command = (
        "/usr/local/cuda-13.0/bin/compute-sanitizer",
        "--tool",
        "memcheck",
        "--error-exitcode",
        "99",
        "--target-processes",
        "application-only",
        "--leak-check",
        "full",
        "/opt/kvbench/.venv/bin/python",
        "/workspace/tests/cuda/phase8_kivi_sanitizer_probe.py",
        "--image-config-digest",
        PHASE8_AUTHORIZED_CONTAINER_DIGEST,
    )
    run.write_bytes("validation/sanitizer/stdout.txt", stdout)
    run.write_bytes("validation/sanitizer/stderr.txt", stderr)
    run.write_json(
        "validation/sanitizer/result.json",
        {
            "schema_version": (
                "kvbench-phase8-kivi-sanitizer-result-1.0.0"
            ),
            "probe_source_path": (
                "tests/cuda/phase8_kivi_sanitizer_probe.py"
            ),
            "probe_source_sha256": sha256_file(
                repository
                / "tests/cuda/phase8_kivi_sanitizer_probe.py"
            ),
            "adapter_source_sha256": sha256_file(
                repository / "src/kvbench/adapters/kivi.py"
            ),
            "extension_sha256": PHASE8_EXTENSION_SHA256,
            "container_digest": PHASE8_AUTHORIZED_CONTAINER_DIGEST,
            "tool_identity": {
                "path": "/usr/local/cuda-13.0/bin/compute-sanitizer",
                "sha256": "e" * 64,
                "exit_code": 0,
            },
            "command": list(command),
            "exit_code": 0,
            "timed_out": False,
            "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
            "memcheck_summaries_passed": True,
            "probe_passed": True,
            "rollover_covered": True,
            "kernel_families": [
                "bgemv2_kernel_outer_dim",
                "bgemv4_kernel_outer_dim",
            ],
            "performance_timing": False,
            "process_supervision": _supervision(
                argv=command,
                stdout=stdout,
                stderr=stderr,
            ),
            "passed": True,
        },
    )


def _copy_artifact_files(
    source: Path,
    run: object,
    *,
    prefix: str,
) -> None:
    for path in sorted(source.rglob("*")):
        if path.is_file():
            run.write_bytes(
                f"{prefix}/{path.relative_to(source).as_posix()}",
                path.read_bytes(),
            )


def _receipt_payload(
    *,
    source: Path,
    root_sha256: str,
    object_count: int,
) -> dict[str, object]:
    uri = (
        "r2://kvbench-artifacts/kvbench/sha256/"
        f"{root_sha256}/"
    )
    validation = {
        "complete_marker_valid": True,
        "inventory_valid": True,
        "checksum_ledger_valid": True,
        "root_digest_valid": True,
        "bundle_validation_valid": True,
    }
    return {
        "schema_version": (
            "kvbench-phase8-kivi-admission-r2-publication-1.0.0"
        ),
        "admission_status": "PASS",
        "artifact_status": "completed",
        "source_git_sha": "8" * 40,
        "source_run_id": source.name,
        "local_validation": {
            "valid": True,
            "complete": True,
            "status": "completed",
            "root_sha256": root_sha256,
            "object_count": object_count,
            **validation,
        },
        "publication": {
            "result": "PASS",
            "root_sha256": root_sha256,
            "uri": uri,
            "object_count": object_count,
            "content_addressed": True,
            "conditional_writes": True,
            "complete_last": True,
        },
        "clean_retrieval": {
            "result": "PASS",
            "root_sha256": root_sha256,
            "object_count": object_count,
            "destination_initially_empty": True,
            **validation,
            "unexpected_objects": False,
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
            "lock_rule_id": "phase8-test-lock",
            "lock_rule_name": "phase8-test-lock",
            "lock_scope": "exact",
            "covered_prefix": "kvbench/sha256/",
            "lock_prefix": "kvbench/sha256/",
            "retention_type": "Indefinite",
            "retention_condition": "Indefinite",
            "bucket_public": False,
            "verified_at_utc": "2026-07-27T01:59:00Z",
        },
        "credential_values_recorded": False,
        "env_file_read": False,
    }


def _make_inner_bundle(
    repository: Path,
    *,
    arbitrary_fixture_result: bool = False,
    forged_native_gqa: bool = False,
    omit_graph_result: bool = False,
) -> tuple[Path, Path]:
    live = Path(__file__).resolve().parents[2]
    source_paths = (
        "src/kvbench/adapters/kivi.py",
        "src/kvbench/runtime/bf16_endpoint.py",
        "src/kvbench/runtime/kivi_cache.py",
        "configs/methods/kivi.yaml",
        "tests/cuda/test_phase8_kivi_cuda.py",
        "tests/graph/test_phase8_kivi_graph.py",
        "tests/cuda/phase8_kivi_sanitizer_probe.py",
    )
    for relative in source_paths:
        target = repository / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(live / relative, target)
    adapter_sha256 = sha256_file(
        repository / "src/kvbench/adapters/kivi.py"
    )
    manifests = tuple(
        dataclasses.replace(
            manifest,
            adapter_source_sha256=adapter_sha256,
            method_config_fingerprint=sha256_file(
                repository / "configs/methods/kivi.yaml"
            ),
        )
        for manifest in _manifests()
    )
    store = phase8_artifact_store(repository)
    bundle = store.create(
        manifests[0].run_id,
        _initial_manifest(manifests[0]),
    )
    bundle.start()
    points: list[dict[str, object]] = []
    launchers: list[dict[str, object]] = []
    point, launcher = _write_point_payloads(
        bundle,
        manifests[0],
        repository=repository,
        native_gqa=not forged_native_gqa,
    )
    points.append(point)
    launchers.append(launcher)
    embedded_ids: list[str] = []
    for manifest in manifests[1:]:
        run = store.create(
            manifest.run_id,
            _initial_manifest(manifest),
        )
        run.start()
        point, launcher = _write_point_payloads(
            run,
            manifest,
            repository=repository,
        )
        points.append(point)
        launchers.append(launcher)
        completed = run.finalize(manifest)
        _copy_artifact_files(
            completed,
            bundle,
            prefix=f"grid-runs/{manifest.run_id}",
        )
        embedded_ids.append(manifest.run_id)

    _write_exact_test(
        bundle,
        repository=repository,
        evidence_name="fixture-conformance",
        source_path="tests/cuda/test_phase8_kivi_cuda.py",
        test_names=(
            "test_all_four_frozen_configurations_store_append_and_rollover",
            "test_cache_position_requires_cuda_int64",
        ),
        arbitrary_result=arbitrary_fixture_result,
    )
    _write_exact_test(
        bundle,
        repository=repository,
        evidence_name="graph-harness",
        source_path="tests/graph/test_phase8_kivi_graph.py",
        test_names=(
            "test_mandatory_configs_capture_direct_decode_without_replay_allocation",
        ),
    )
    if omit_graph_result:
        result = (
            bundle.stage
            / "validation"
            / "graph-harness"
            / "result.json"
        )
        result.unlink()
    _write_sanitizer(bundle, repository=repository)
    first_names = tuple(
        record["kernel_family"]
        for launcher in launchers
        for record in launcher["first_sequence"]
    )
    second_names = tuple(
        record["kernel_family"]
        for launcher in launchers
        for record in launcher["second_sequence"]
    )
    audit = audit_kivi_execution_path(
        kernel_names=first_names,
        repeated_kernel_names=second_names,
        runtime_event_names=("cudaLaunchKernel",),
        temporary_shapes={
            "key_staging": (1, 8, 1, 128),
            "value_staging": (1, 8, 1, 128),
        },
        adapter_hot_path_source=(
            "kv_head = query_head // KIVI_GQA_GROUP_SIZE"
        ),
        observed_extension_sha256=PHASE8_EXTENSION_SHA256,
        observed_new_pack_sha256=KIVI_NEW_PACK_SHA256,
        official_commit=PHASE8_OFFICIAL_COMMIT,
        official_base_tree=PHASE8_BASE_TREE,
        patched_tree=PHASE8_PATCHED_TREE,
        decision_0018_patch_sha256=(
            PHASE8_DECISION_0018_PATCH_SHA256
        ),
        fixture_root_digest=PHASE8_FIXTURE_ROOT_DIGEST,
        host_stub_offsets=OFFICIAL_KIVI_HOST_STUB_OFFSETS,
        backend_fallback_observed=False,
        cache_growth_observed=False,
    )
    bundle.write_json(
        "validation/execution-path-static-precheck.json",
        {
            "schema_version": (
                "kvbench-phase8-kivi-static-execution-precheck-1.0.0"
            ),
            "passed": True,
            "forbidden_tokens_present": [],
            "launcher_observer_available": True,
            "authority_passed": True,
            "adapter_source_sha256": adapter_sha256,
            "instrumented_runtime_observation_required": True,
            "runtime_path_claimed_by_static_precheck": False,
        },
    )
    bundle.write_json(
        "validation/execution-path.json",
        {
            **audit.to_dict(),
            "adapter_source_path": "src/kvbench/adapters/kivi.py",
            "adapter_source_sha256": adapter_sha256,
            "host_stub_offsets": {
                str(bits): offset
                for bits, offset in OFFICIAL_KIVI_HOST_STUB_OFFSETS.items()
            },
            "runtime_launcher_probe_count": len(launchers),
            "runtime_observation_instrumented_separately": True,
            "normal_timing_instrumented": False,
        },
    )
    run_ids = [manifest.run_id for manifest in manifests]
    bundle.write_json(
        "validation/bounded-grid.json",
        {
            "schema_version": (
                "kvbench-phase8-kivi-bounded-grid-1.0.0"
            ),
            "plan": [
                {
                    "configuration": item.configuration,
                    "runner_kind": item.runner_kind.value,
                    "graph_mode": item.graph_mode.value,
                    "batch_size": 1,
                    "context_length": item.context_length,
                    "output_steps": item.output_steps,
                    "engineering_samples": 1,
                }
                for item in PHASE8_ADMISSION_GRID
            ],
            "run_ids": run_ids,
            "embedded_run_ids": embedded_ids,
            "bundle_root_point_run_id": run_ids[0],
            "points": points,
            "attempted": 10,
            "passed": 10,
            "failed": 0,
            "speedup_calculated": False,
            "performance_claim_eligible": False,
            "measurement_scope": "measurement_container_admission",
        },
    )
    local_checks = {
        check_id: True
        for check_id in (
            "fixture_conformance",
            "byte_accounting",
            "residual_rollover",
            "token_integrity",
            "static_cache",
            "no_measured_torch_cat",
            "direct_compressed_decode",
            "native_gqa",
            "no_unknown_allocation",
            "graph_capture_replay",
            "graph_zero_replay_allocation",
            "no_backend_fallback",
            "compute_sanitizer",
            "bounded_admission_grid",
        )
    }
    bundle.write_json(
        "validation/admission-candidate.json",
        {
            "schema_version": (
                "kvbench-phase8-kivi-admission-candidate-1.0.0"
            ),
            "status": "LOCAL_CHECKS_PASS_PUBLICATION_PENDING",
            "git_sha": "8" * 40,
            "container_digest": PHASE8_AUTHORIZED_CONTAINER_DIGEST,
            **local_checks,
            "immutable_checksums": "pending_finalization",
            "durable_publication": "pending_host_side",
            "clean_retrieval": "pending_host_side",
            "g2_kivi": "NOT_EVALUATED",
            "global_g2": "NOT_EVALUATED",
            "quality_execution": "LOCKED",
            "full_scan": "CLOSED",
            "performance_data_frozen": False,
            "quality_benchmark_executed": False,
            "performance_claim_eligible": False,
            "speedup_calculated": False,
            "r_hbm": None,
            "derivation": {"literal_gate_overrides": False},
        },
    )
    source = bundle.finalize(manifests[0])
    validated = validate_local_artifact(source, environ={})
    receipt = (
        repository
        / "docs"
        / "evidence"
        / "phase8"
        / "r2-admission-publication.json"
    )
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(
        json.dumps(
            _receipt_payload(
                source=source,
                root_sha256=validated.root_sha256,
                object_count=len(validated.files),
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return source, receipt


def _make_writable(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file():
            path.chmod(0o644)
        elif path.is_dir():
            path.chmod(0o755)
    root.chmod(0o755)


class Phase8KIVIAdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch(
            "kvbench.runtime.kivi_admission."
            "resolve_phase8_historical_source_authority",
            side_effect=_synthetic_historical_source_authority,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        git_patcher = mock.patch(
            "kvbench.runtime.kivi_admission._phase8_git",
            side_effect=_synthetic_git_object_query,
        )
        git_patcher.start()
        self.addCleanup(git_patcher.stop)

    def test_local_validation_explicitly_selects_bundle_and_records_history(
        self,
    ) -> None:
        temporary = Path(
            tempfile.mkdtemp(prefix="kvbench-phase8-selection-")
        )
        try:
            repository = temporary / "repository"
            repository.mkdir()
            selected, _ = _make_inner_bundle(repository)
            store = phase8_artifact_store(repository)
            base = _manifests()[0]
            histories = (
                (
                    "phase8-historical-completed-bundle",
                    RunStatus.COMPLETED,
                    None,
                    True,
                ),
                (
                    "phase8-historical-failed-run",
                    RunStatus.RUNTIME_FAILED,
                    "historical failure",
                    False,
                ),
            )
            for run_id, status, reason, is_bundle in histories:
                terminal = dataclasses.replace(
                    base,
                    run_id=run_id,
                    status=status,
                    failure_reason=reason,
                )
                run = store.create(run_id, _initial_manifest(terminal))
                run.start()
                run.write_json("config/method.json", {"method": "kivi"})
                run.write_json(
                    "environment/container_identity.json",
                    {"container": "historical"},
                )
                run.write_json("raw/runner.json", {"historical": True})
                run.write_json("validation/point.json", {"historical": True})
                if is_bundle:
                    run.write_json(
                        "validation/bounded-grid.json",
                        {"historical": True},
                    )
                run.finalize(terminal)

            result = validate_local_admission(selected)
            self.assertEqual(result["bundle_path"], str(selected))
            self.assertEqual(
                result["selection_kind"],
                "explicit_bundle_path",
            )
            self.assertEqual(
                result["historical_completed_bundle_run_ids"],
                ["phase8-historical-completed-bundle"],
            )
            self.assertEqual(result["historical_completed_run_ids"], [])
            self.assertEqual(
                result["historical_failed_run_ids"],
                ["phase8-historical-failed-run"],
            )
        finally:
            _make_writable(temporary)
            shutil.rmtree(temporary)

    def test_strict_supervision_accepts_actual_schema_and_fails_closed(
        self,
    ) -> None:
        argv = ("/opt/kvbench/.venv/bin/python", "probe.py", "-v")
        stdout = b"probe output\n"
        stderr = b""
        passing = _supervision(
            argv=argv,
            stdout=stdout,
            stderr=stderr,
        )
        expected = {
            "expected_argv": argv,
            "expected_stdout_sha256": hashlib.sha256(stdout).hexdigest(),
            "expected_stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        }
        self.assertTrue(_supervision_passed(passing, **expected))

        failures = {
            "schema": ("schema_version", "wrong"),
            "returncode": ("returncode", 1),
            "timeout": ("timeout.timed_out", True),
            "identity": ("identity.pid", 0),
            "parent_identity": (
                "direct_child.expected_parent_pid",
                9999,
            ),
            "direct_child": ("direct_child.verified", False),
            "final_reap": ("final_reap.count", 0),
            "stdout_binding": ("stdout.sha256", "0" * 64),
        }
        for label, (path, replacement) in failures.items():
            with self.subTest(label=label):
                candidate = copy.deepcopy(passing)
                components = path.split(".")
                target = candidate
                for component in components[:-1]:
                    nested = target[component]
                    self.assertIsInstance(nested, dict)
                    target = nested
                target[components[-1]] = replacement
                self.assertFalse(_supervision_passed(candidate, **expected))

    def test_grid_is_exactly_ten_points(self) -> None:
        records = require_exact_phase8_grid(_manifests())
        self.assertEqual(len(records), 10)
        self.assertEqual(
            sum(point.context_length == 4096 for point in PHASE8_ADMISSION_GRID),
            2,
        )
        self.assertEqual(
            sum(
                point.runner_kind.value == "growing_context"
                for point in PHASE8_ADMISSION_GRID
            ),
            1,
        )
        self.assertEqual(PHASE8_ADMISSION_GRID[-1].configuration, "k4v2")

    def test_grid_rejects_reordering_duplicate_and_incomplete_run(self) -> None:
        records = _manifests()
        with self.assertRaises(KIVIAdmissionError):
            require_exact_phase8_grid((records[1], records[0], *records[2:]))
        with self.assertRaises(KIVIAdmissionError):
            require_exact_phase8_grid(
                (*records[:-1], dataclasses.replace(records[-1], run_id=records[0].run_id))
            )
        with self.assertRaises(KIVIAdmissionError):
            require_exact_phase8_grid(
                (
                    dataclasses.replace(
                        records[0],
                        status=RunStatus.RUNTIME_FAILED,
                        finished_at_utc="2026-07-27T00:01:00Z",
                        inventory_path="artifact_inventory.json",
                        failure_reason="failed",
                    ),
                    *records[1:],
                )
            )
        fixed = records[0]
        fixed_capacity_accounting = dataclasses.replace(
            fixed.accounting,
            active_context=fixed.capacity,
            logical_bf16_active_bytes=fixed.capacity * 1024,
        )
        with self.assertRaises(KIVIAdmissionError):
            require_exact_phase8_grid(
                (
                    dataclasses.replace(
                        fixed,
                        accounting=fixed_capacity_accounting,
                    ),
                    *records[1:],
                )
            )
        growing_index = next(
            index
            for index, point in enumerate(PHASE8_ADMISSION_GRID)
            if point.runner_kind is RunnerKind.GROWING_CONTEXT
        )
        growing = records[growing_index]
        growing_start_accounting = dataclasses.replace(
            growing.accounting,
            active_context=growing.context_length,
            logical_bf16_active_bytes=growing.context_length * 1024,
        )
        with self.assertRaises(KIVIAdmissionError):
            require_exact_phase8_grid(
                (
                    *records[:growing_index],
                    dataclasses.replace(
                        growing,
                        accounting=growing_start_accounting,
                    ),
                    *records[growing_index + 1 :],
                )
            )

    def test_execution_path_binds_official_families_source_and_offsets(self) -> None:
        audit = _passing_audit()
        self.assertTrue(audit.passed)
        self.assertTrue(audit.two_bit_kernel_verified)
        self.assertTrue(audit.four_bit_kernel_verified)
        self.assertTrue(audit.host_stub_offsets_verified)
        self.assertFalse(audit.backend_fallback_detected)
        self.assertEqual(
            OFFICIAL_KIVI_HOST_STUB_OFFSETS,
            {
                2: KIVI_BGEMV2_HOST_STUB_OFFSET,
                4: KIVI_BGEMV4_HOST_STUB_OFFSET,
            },
        )

    def test_current_compressed_hot_path_has_no_false_positive(self) -> None:
        kernels = (
            "bgemv2_kernel_outer_dim",
            "bgemv4_kernel_outer_dim",
            "aten::bmm",
        )
        audit = audit_kivi_execution_path(
            kernel_names=kernels,
            repeated_kernel_names=kernels,
            runtime_event_names=("cudaLaunchKernel",),
            temporary_shapes={
                "key_staging": (1, 8, 1, 128),
                "value_staging": (1, 8, 1, 128),
                "key_kernel_output_fp16": (32, 1, 128),
                "decode_output": (1, 32, 128),
            },
            adapter_hot_path_source=inspect.getsource(
                KIVIMethodAdapter._decode_compressed
            ),
            observed_extension_sha256=PHASE8_EXTENSION_SHA256,
            observed_new_pack_sha256=KIVI_NEW_PACK_SHA256,
            official_commit=PHASE8_OFFICIAL_COMMIT,
            official_base_tree=PHASE8_BASE_TREE,
            patched_tree=PHASE8_PATCHED_TREE,
            decision_0018_patch_sha256=(
                PHASE8_DECISION_0018_PATCH_SHA256
            ),
            fixture_root_digest=PHASE8_FIXTURE_ROOT_DIGEST,
            host_stub_offsets=OFFICIAL_KIVI_HOST_STUB_OFFSETS,
            backend_fallback_observed=False,
            cache_growth_observed=False,
        )
        self.assertTrue(audit.passed, audit.reasons)

    def test_execution_path_detects_forbidden_mechanisms(self) -> None:
        source = """
kv_head = query_head // KIVI_GQA_GROUP_SIZE
torch.cat(parts)
repeat_interleave(cache)
tensor.cpu()
"""
        audit = audit_kivi_execution_path(
            kernel_names=(
                "bgemv2_kernel_outer_dim",
                "bgemv4_kernel_outer_dim",
                "flash_fwd",
            ),
            repeated_kernel_names=(
                "bgemv4_kernel_outer_dim",
                "bgemv2_kernel_outer_dim",
            ),
            runtime_event_names=("cudaStreamSynchronize",),
            temporary_shapes={
                "key_full_prefix": (1, 32, 128, 128),
            },
            adapter_hot_path_source=source,
            observed_extension_sha256="0" * 64,
            observed_new_pack_sha256="1" * 64,
            official_commit=PHASE8_OFFICIAL_COMMIT,
            official_base_tree=PHASE8_BASE_TREE,
            patched_tree=PHASE8_PATCHED_TREE,
            decision_0018_patch_sha256=(
                PHASE8_DECISION_0018_PATCH_SHA256
            ),
            fixture_root_digest=PHASE8_FIXTURE_ROOT_DIGEST,
            host_stub_offsets={2: 0, 4: 0},
            backend_fallback_observed=False,
            cache_growth_observed=False,
        )
        self.assertFalse(audit.passed)
        self.assertTrue(audit.measured_torch_cat_detected)
        self.assertTrue(audit.gqa_materialization_detected)
        self.assertTrue(audit.host_synchronization_detected)
        self.assertTrue(audit.full_prefix_temporary_detected)
        self.assertTrue(audit.query_head_sized_kv_temporary_detected)
        self.assertTrue(audit.backend_fallback_detected)

    def test_accounting_summary_is_canonical_and_has_no_hbm(self) -> None:
        summary = summarize_phase8_accounting(_manifests())
        self.assertEqual(len(summary["points"]), 10)
        self.assertIsNone(summary["r_hbm"])
        self.assertEqual(len(summary["summary_sha256"]), 64)
        self.assertTrue(
            all(
                point["reciprocal_product_error"] <= 1e-9
                for point in summary["points"]
            )
        )

    def test_authorized_environment_delegates_to_exact_decision_0016_gate(
        self,
    ) -> None:
        environment = {
            "KVBENCH_AUTHORIZED_IMAGE_DIGEST": (
                PHASE8_AUTHORIZED_CONTAINER_DIGEST
            ),
            "KVBENCH_EXECUTION_ENVIRONMENT": "measurement_container",
        }
        identity = {
            "container_digest": PHASE8_AUTHORIZED_CONTAINER_DIGEST,
            "execution_environment": "measurement_container",
        }
        with mock.patch.dict("os.environ", environment, clear=False), mock.patch(
            "kvbench.runtime.kivi_admission.require_authorized_cuda_environment",
            return_value=identity,
        ):
            observed = require_authorized_kivi_environment(
                PHASE8_AUTHORIZED_CONTAINER_DIGEST
            )
        self.assertTrue(observed["native_host_cuda_rejected"])
        with self.assertRaises(KIVIAdmissionError):
            require_authorized_kivi_environment("sha256:" + "0" * 64)

    def test_report_is_derived_from_finalized_inner_bundle(self) -> None:
        temporary = Path(
            tempfile.mkdtemp(prefix="kvbench-phase8-admission-")
        )
        try:
            repository = temporary / "repository"
            repository.mkdir()
            source, receipt = _make_inner_bundle(repository)
            derived = derive_phase8_admission_evidence(
                evidence_root=repository,
                inner_bundle_root=source,
                publication_receipt_path=receipt,
                creation_git_sha="8" * 40,
            )
            self.assertEqual(
                tuple(check.check_id for check in derived.checks),
                PHASE8_ADMISSION_CHECK_IDS,
            )
            self.assertTrue(
                all(check.status.value == "PASS" for check in derived.checks)
            )
            self.assertTrue(derived.execution_path_audit.passed)
            self.assertTrue(
                derived.durable_publication.retrieval_passed
            )
            raw_references = tuple(
                reference
                for reference in derived.evidence_references
                if "_allocation_raw_" in reference.evidence_id
            )
            self.assertEqual(len(raw_references), 13 * 9)
            self.assertTrue(
                all(
                    "/allocation/operations/" in reference.path
                    for reference in raw_references
                )
            )
            no_unknown = next(
                check
                for check in derived.checks
                if check.check_id == "no_unknown_allocation"
            )
            self.assertTrue(
                {
                    reference.evidence_id
                    for reference in raw_references
                }.issubset(no_unknown.evidence_ids)
            )
            report = build_phase8_method_admission_report(
                created_at_utc="2026-07-27T01:00:00Z",
                creation_git_sha="8" * 40,
                evidence_root=repository,
                inner_bundle_root=source,
                publication_receipt_path=receipt,
            )
            self.assertEqual(report.status.value, "PASS")
            self.assertEqual(report.gates.global_g2.value, "NOT_EVALUATED")
            self.assertFalse(report.performance_claim_eligible)
            self.assertIsNone(report.r_hbm)
            self.assertEqual(
                report.admitted_configurations,
                ("k4v4", "k2v4", "k2v2"),
            )
            self.assertEqual(len(report.method_fingerprints), 4)
            self.assertEqual(len(report.cache_layout_fingerprints), 4)
            self.assertNotIn(
                "checks",
                inspect.signature(
                    build_phase8_method_admission_report
                ).parameters,
            )
        finally:
            _make_writable(temporary)
            shutil.rmtree(temporary)

    def test_sha_bound_arbitrary_bytes_cannot_forge_pass(self) -> None:
        temporary = Path(
            tempfile.mkdtemp(prefix="kvbench-phase8-arbitrary-")
        )
        try:
            repository = temporary / "repository"
            repository.mkdir()
            source, receipt = _make_inner_bundle(
                repository,
                arbitrary_fixture_result=True,
            )
            with self.assertRaisesRegex(
                KIVIAdmissionError,
                "fixture-conformance/result.json",
            ):
                derive_phase8_admission_evidence(
                    evidence_root=repository,
                    inner_bundle_root=source,
                    publication_receipt_path=receipt,
                    creation_git_sha="8" * 40,
                )
        finally:
            _make_writable(temporary)
            shutil.rmtree(temporary)

    def test_missing_structured_file_cannot_forge_pass(self) -> None:
        temporary = Path(
            tempfile.mkdtemp(prefix="kvbench-phase8-missing-")
        )
        try:
            repository = temporary / "repository"
            repository.mkdir()
            source, receipt = _make_inner_bundle(
                repository,
                omit_graph_result=True,
            )
            with self.assertRaises(KIVIAdmissionError):
                derive_phase8_admission_evidence(
                    evidence_root=repository,
                    inner_bundle_root=source,
                    publication_receipt_path=receipt,
                    creation_git_sha="8" * 40,
                )
        finally:
            _make_writable(temporary)
            shutil.rmtree(temporary)

    def test_candidate_booleans_cannot_override_point_failure(self) -> None:
        temporary = Path(
            tempfile.mkdtemp(prefix="kvbench-phase8-forged-")
        )
        try:
            repository = temporary / "repository"
            repository.mkdir()
            source, receipt = _make_inner_bundle(
                repository,
                forged_native_gqa=True,
            )
            with self.assertRaisesRegex(
                KIVIAdmissionError,
                "point evidence differs",
            ):
                derive_phase8_admission_evidence(
                    evidence_root=repository,
                    inner_bundle_root=source,
                    publication_receipt_path=receipt,
                    creation_git_sha="8" * 40,
                )
        finally:
            _make_writable(temporary)
            shutil.rmtree(temporary)

    def test_inner_receipt_binds_clean_bundle_not_future_report(self) -> None:
        temporary = Path(
            tempfile.mkdtemp(prefix="kvbench-phase8-receipt-")
        )
        try:
            repository = temporary / "repository"
            repository.mkdir()
            source, receipt = _make_inner_bundle(repository)
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            payload["retrieved_report_valid"] = True
            receipt.write_text(
                json.dumps(payload, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                KIVIAdmissionError,
                "cannot validate the MethodAdmissionReport",
            ):
                derive_phase8_admission_evidence(
                    evidence_root=repository,
                    inner_bundle_root=source,
                    publication_receipt_path=receipt,
                    creation_git_sha="8" * 40,
                )
        finally:
            _make_writable(temporary)
            shutil.rmtree(temporary)


if __name__ == "__main__":
    unittest.main()
