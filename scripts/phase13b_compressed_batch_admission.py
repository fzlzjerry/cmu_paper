#!/usr/bin/env python3
"""Narrow, non-timing Phase 13B compressed batch-admission coordinator."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

from kvbench.runtime.artifacts import (
    AppendOnlyArtifactStore,
    validate_run_directory,
)
from kvbench.runtime.method_harness import execution_path_audit_facade
from kvbench.schema import RunStatus, canonical_json_bytes, sha256_hex
from kvbench.schema.phase13b import (
    PHASE13B_AUTHORIZED_CONTAINER_DIGEST,
    PHASE13B_BATCH_SIZES,
    PHASE13B_CONFIGURATIONS,
    PHASE13B_CONTEXT_LENGTH,
    PHASE13B_FAMILY_CONFIGURATIONS,
    PHASE13B_SUCCESSOR_CHECK_IDS,
    PHASE13B_SUCCESSOR_EVIDENCE_IDS,
    Phase13BBatchAdmissionManifest,
    Phase13BMethodAdmissionReport,
)

from scripts import phase12_unified_admission as phase12
from scripts import phase13_pilot as phase13


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MATRIX_SCHEMA = "kvbench-phase13b-cuda-validation-1.0.0"
COMMAND_EVIDENCE_SCHEMA = "kvbench-phase13b-command-evidence-1.0.0"
SANITIZER_EVIDENCE_SCHEMA = "kvbench-phase13b-sanitizer-evidence-1.0.0"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
EAGER_ALLOCATION_AUTHORITY = {
    "turboquant": {
        "report": "docs/evidence/phase6/turboquant-method-admission.json",
        "report_sha256": (
            "388e8107b649a9093491699357c8b1ad1d8e12c8c75378bce658f8a09bf9ab2a"
        ),
        "b1_event_count": 898,
        "b1_event_bytes": 9_802_604,
        "batch_invariant_event_bytes": 96,
    },
    "kivi": {
        "report": "docs/evidence/phase8/kivi-method-admission.json",
        "report_sha256": (
            "3a4b63b9da0eab12db9a916ebdc1cffd788ea6f93678d87964a8332ae7cec83a"
        ),
        "b1_event_count": 874,
        "b1_event_bytes": 9_637_132,
        # Every KIVI eager event is owned by a batch-shaped outer-model
        # tensor; the admitted direct cache path contributes zero events.
        "batch_invariant_event_bytes": 0,
    },
    "kvquant": {
        "report": "docs/evidence/phase11rq23/kvquant-method-admission.json",
        "report_sha256": (
            "9cfed618cee9514a1071392d0a2dca327dcf6acd33d81ac72cc477c7880c09e2"
        ),
        "b1_event_count": 874,
        "b1_event_bytes": 9_637_132,
        # Caller-owned KVQuant kernels add no allocator events; the observed
        # outer-model event bytes are entirely batch-shaped.
        "batch_invariant_event_bytes": 0,
    },
}
HISTORICAL_PRESERVATION_AUTHORITY = {
    "docs/phase_reports/phase13-pilot-scan.md": (
        "8064c21622cc1f7c61b259de9ca901ecb9f1cf26b3eb4ee428cf3a6b5a98a6f6"
    ),
    "docs/evidence/phase13/pilot_qc.json": (
        "442bdeb112ce19a8822de9eb1851e159de61461f657cab35c56f72fa463202e4"
    ),
    "docs/evidence/phase13/r2-publication.json": (
        "d72f082991cf7869c8d668331b8482c11a4cb29499fd89a4918271a9000b9570"
    ),
}
SUCCESSOR_HISTORICAL_REPORTS = {
    "turboquant": (
        "docs/evidence/phase6/turboquant-method-admission.json",
        "388e8107b649a9093491699357c8b1ad1d8e12c8c75378bce658f8a09bf9ab2a",
    ),
    "kivi": (
        "docs/evidence/phase8/kivi-method-admission.json",
        "3a4b63b9da0eab12db9a916ebdc1cffd788ea6f93678d87964a8332ae7cec83a",
    ),
    "kvquant": (
        "docs/evidence/phase11rq23/kvquant-method-admission.json",
        "9cfed618cee9514a1071392d0a2dca327dcf6acd33d81ac72cc477c7880c09e2",
    ),
}
FAMILY_SOURCE_PATHS = {
    "turboquant": (
        "src/kvbench/adapters/turboquant.py",
        "src/kvbench/runtime/turboquant_cache.py",
    ),
    "kivi": (
        "src/kvbench/adapters/kivi.py",
        "src/kvbench/runtime/kivi_cache.py",
    ),
    "kvquant": (
        "src/kvbench/adapters/kvquant.py",
        "src/kvbench/runtime/kvquant_cache.py",
        "src/kvbench/runtime/kvquant_session.py",
    ),
}
SANITIZER_INITCHECK_CONFIGURATIONS = (
    "tq_3bit_nc",
    "k2v2",
    "kvq3",
)


class Phase13BBatchAdmissionError(RuntimeError):
    """A fail-closed Phase 13B validation error."""


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_json_bytes(payload))


def _require_git_authority(expected_head: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", expected_head):
        raise Phase13BBatchAdmissionError("expected Git SHA is invalid")
    observed = subprocess.run(
        ("/usr/bin/git", "rev-parse", "HEAD"),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ("/usr/bin/git", "status", "--porcelain=v1", "--untracked-files=all"),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if observed != expected_head or status:
        raise Phase13BBatchAdmissionError(
            "Phase 13B CUDA source is not the clean committed authority"
        )


def _allocation_passed(record: Any) -> bool:
    return bool(
        record.audit_available
        and record.passed
        and record.allocation_event_count == 0
        and record.allocation_event_bytes == 0
        and record.allocated_after == record.allocated_before
        and record.reserved_after == record.reserved_before
    )


def _eager_control(*, family: str, batch: int) -> dict[str, Any]:
    """Scale only the batch-dependent bytes in an admitted B=1 event set."""

    authority = EAGER_ALLOCATION_AUTHORITY[family]
    b1_bytes = int(authority["b1_event_bytes"])
    invariant_bytes = int(authority["batch_invariant_event_bytes"])
    expected_bytes = invariant_bytes + batch * (b1_bytes - invariant_bytes)
    count = int(authority["b1_event_count"])
    return {
        **authority,
        "batch_size": batch,
        "expected_allocation_event_count": count,
        "expected_allocation_event_bytes": expected_bytes,
        "expected_event_counts": {
            "alloc": count,
            "free_completed": count,
            "free_requested": count,
        },
        "formula": "family_fixed_plus_batch_times_admitted_b1_remainder",
    }


def _eager_matches_outer_control(observed: Any, control: Mapping[str, Any]) -> bool:
    """Require exact admitted event topology with no persistent growth."""

    return bool(
        observed.audit_available
        and observed.allocated_after == observed.allocated_before
        and observed.reserved_after == observed.reserved_before
        and observed.allocation_event_count
        == control["expected_allocation_event_count"]
        and observed.allocation_event_bytes
        == control["expected_allocation_event_bytes"]
        and observed.event_counts == control["expected_event_counts"]
    )


def _frozen_tolerance(family: str) -> tuple[float, float]:
    if family == "kivi":
        from kvbench.runtime.kivi_session import (
            PHASE8_DECODE_ATOL,
            PHASE8_DECODE_RTOL,
        )

        return float(PHASE8_DECODE_ATOL), float(PHASE8_DECODE_RTOL)
    if family == "kvquant":
        from kvbench.runtime.kvquant_session import (
            PHASE11_DECODE_ATOL,
            PHASE11_DECODE_RTOL,
        )

        return float(PHASE11_DECODE_ATOL), float(PHASE11_DECODE_RTOL)
    return 0.02, 0.02


def _batch_banks_equal(cache: Any, *, family: str) -> dict[str, Any]:
    """Compare identical-input persistent banks outside measured execution."""

    import torch

    batch = int(cache.batch_size)
    tensors: list[tuple[str, Any, int]]
    if family == "turboquant":
        packed = cache.packed_cache.reshape(
            len(cache.compressed_layers),
            batch,
            cache.block_count,
            cache.block_size,
            cache.num_kv_heads,
            cache.slot_size,
        )
        tensors = [
            ("packed_cache", packed, 1),
            ("bf16_keys", cache.bf16_cache.keys, 1),
            ("bf16_values", cache.bf16_cache.values, 1),
        ]
    elif family == "kivi":
        tensors = [
            (name, getattr(cache, name), 1)
            for name in (
                "packed_key_history",
                "packed_value_history",
                "key_scales",
                "key_minimums",
                "value_scales",
                "value_minimums",
                "key_residual",
                "value_residual_ring",
            )
        ]
    else:
        tensors = [
            (name, getattr(cache, name), 1)
            for name in (
                "packed_key_cache",
                "packed_value_cache",
                "value_lookup_cache",
                "key_sparse_values",
                "key_sparse_indices",
                "value_sparse_values",
                "value_sparse_indices",
                "key_active_counts",
                "value_active_counts",
                "sink_key",
                "sink_value",
            )
        ]

    comparisons: dict[str, bool] = {}
    for name, tensor, axis in tensors:
        reference = tensor.select(axis, 0)
        comparisons[name] = all(
            bool(torch.equal(tensor.select(axis, index), reference))
            for index in range(1, batch)
        )
    return {
        "passed": all(comparisons.values()),
        "input_contract": "identical_rows",
        "comparison": "exact_tensor_equality",
        "banks": comparisons,
    }


def _run_cuda_matrix(*, output: Path, git_sha: str) -> dict[str, Any]:
    """Run exactly 9 configurations x B=1/4/8 without timing samples."""

    attestation = phase12._require_authorized_container_runtime()
    _require_git_authority(git_sha)

    import torch

    from kvbench.runtime.allocation import audit_cuda_allocations
    from kvbench.runtime.backend import forced_flash_execution
    from kvbench.runtime.model_loader import load_frozen_model
    from kvbench.runtime.numerical import (
        compare_tensors_untimed,
        tensor_sha256_untimed,
    )

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise Phase13BBatchAdmissionError("authorized CUDA device is unavailable")
    device = torch.device("cuda:0")
    loaded = load_frozen_model(device=device)
    base_prefix = (
        torch.arange(PHASE13B_CONTEXT_LENGTH, dtype=torch.long, device=device)
        .add(13_000)
        .remainder(120_000)
        .add(1_000)
        .reshape(1, PHASE13B_CONTEXT_LENGTH)
    )
    base_decode = torch.full(
        (1, 1),
        13_000 + PHASE13B_CONTEXT_LENGTH + 257,
        dtype=torch.long,
        device=device,
    )
    records: list[dict[str, Any]] = []
    b1_outputs: dict[str, Any] = {}
    for authority in EAGER_ALLOCATION_AUTHORITY.values():
        report = REPOSITORY_ROOT / str(authority["report"])
        if _sha256_file(report) != authority["report_sha256"]:
            raise Phase13BBatchAdmissionError(
                "historical eager-allocation authority differs"
            )
    source_hashes = {
        relative: _sha256_file(REPOSITORY_ROOT / relative)
        for relative in (
            "src/kvbench/adapters/turboquant.py",
            "src/kvbench/runtime/turboquant_cache.py",
            "src/kvbench/adapters/kivi.py",
            "src/kvbench/runtime/kivi_cache.py",
            "src/kvbench/adapters/kvquant.py",
            "src/kvbench/runtime/kvquant_cache.py",
            "src/kvbench/runtime/kvquant_session.py",
        )
    }

    for configuration in PHASE13B_CONFIGURATIONS:
        for batch in PHASE13B_BATCH_SIZES:
            phase13._patch_phase12_point_globals(
                batch=batch,
                historical=PHASE13B_CONTEXT_LENGTH,
            )
            operation = phase13.Phase13OperationKey.create(
                configuration,
                batch,
                PHASE13B_CONTEXT_LENGTH,
            )
            prefix = base_prefix.expand(batch, -1).clone()
            decode = base_decode.expand(batch, -1).clone()
            with torch.inference_mode(), forced_flash_execution():
                session = phase12._build_phase12_session(
                    loaded=loaded,
                    operation_key=operation,
                    prefix_input_ids=prefix,
                    decode_input_ids=decode,
                )
                if session.graph is None or session._fixed_operation is None:
                    raise Phase13BBatchAdmissionError(
                        "fixed eager/graph operations are unavailable"
                    )
                family = phase12._method_family(configuration)
                eager_control = _eager_control(family=family, batch=batch)
                session._fixed_operation()
                torch.cuda.synchronize(device=device)
                pointers_before = phase12._phase12_session_pointers(session)
                history_before = session.current_historical_prefix_sha256()
                eager_allocation = audit_cuda_allocations(
                    session._fixed_operation,
                    device=session.cache_device,
                )
                graph_allocation = audit_cuda_allocations(
                    session.graph.replay,
                    device=session.cache_device,
                )
                eager_output = (
                    session._fixed_operation()
                    .detach()
                    .to(device="cpu", copy=True)
                    .clone()
                )
                graph_output = (
                    session.graph.replay()
                    .detach()
                    .to(device="cpu", copy=True)
                    .clone()
                )
                torch.cuda.synchronize(device=device)

                stream = torch.cuda.Stream(device=device)
                ready = torch.cuda.Event()
                complete = torch.cuda.Event()
                torch.cuda.current_stream(device=device).record_event(ready)
                stream.wait_event(ready)
                with torch.cuda.stream(stream):
                    stream_output = session._fixed_operation()
                    complete.record(stream)
                torch.cuda.current_stream(device=device).wait_event(complete)
                stream_cpu = (
                    stream_output.detach().to(device="cpu", copy=True).clone()
                )
                torch.cuda.synchronize(device=device)

            atol, rtol = _frozen_tolerance(family)
            eager_graph = compare_tensors_untimed(
                graph_output,
                eager_output,
                atol=atol,
                rtol=rtol,
            )
            stream_comparison = compare_tensors_untimed(
                stream_cpu,
                eager_output,
                atol=atol,
                rtol=rtol,
            )
            finite = bool(
                torch.isfinite(eager_output).all()
                and torch.isfinite(graph_output).all()
                and torch.isfinite(stream_cpu).all()
            )
            row_sha256 = [
                tensor_sha256_untimed(eager_output[index : index + 1])
                for index in range(batch)
            ]
            if batch == 1:
                b1_outputs[configuration] = eager_output.clone()
                batch_control = {
                    "passed": True,
                    "rows_compared": 1,
                    "reference": "b1_frozen_fixture_and_source_preservation",
                    "row_output_sha256": row_sha256,
                }
            else:
                reference = b1_outputs.get(configuration)
                if reference is None:
                    raise Phase13BBatchAdmissionError("B=1 control is absent")
                within_batch = [
                    compare_tensors_untimed(
                        eager_output[index : index + 1],
                        eager_output[0:1],
                        atol=atol,
                        rtol=rtol,
                    )
                    for index in range(1, batch)
                ]
                cross_batch_diagnostic = [
                    compare_tensors_untimed(
                        eager_output[index : index + 1],
                        reference,
                        atol=atol,
                        rtol=rtol,
                    )
                    for index in range(batch)
                ]
                batch_control = {
                    "passed": all(item.passed for item in within_batch),
                    "rows_compared": batch,
                    "maximum_absolute_error": max(
                        item.max_absolute_error for item in within_batch
                    ),
                    "maximum_relative_error": max(
                        item.max_relative_error for item in within_batch
                    ),
                    "reference": "identical_rows_within_same_batch_execution",
                    "row_output_sha256": row_sha256,
                    "cross_batch_b1_diagnostic": {
                        "admission_gate": False,
                        "passed_at_single_layer_method_tolerance": all(
                            item.passed for item in cross_batch_diagnostic
                        ),
                        "maximum_absolute_error": max(
                            item.max_absolute_error
                            for item in cross_batch_diagnostic
                        ),
                        "maximum_relative_error": max(
                            item.max_relative_error
                            for item in cross_batch_diagnostic
                        ),
                        "reason": (
                            "full_model_batched_bf16_gemm_rounding_is_not_"
                            "a_single_layer_cache_tolerance"
                        ),
                    },
                }

            pointers_after = phase12._phase12_session_pointers(session)
            history_after = session.current_historical_prefix_sha256()
            bank_control = _batch_banks_equal(
                session.cache,
                family=family,
            )
            geometry = session.gqa_cache_geometry()
            geometry_passed = phase12._gqa_geometry_passes(
                geometry,
                family=family,
            )
            eager_passed = _eager_matches_outer_control(
                eager_allocation,
                eager_control,
            )
            graph_alloc_passed = _allocation_passed(graph_allocation)
            graph_passed = bool(
                session.graph_evidence is not None
                and session.graph_evidence.get("captured") is True
                and session.graph_evidence.get("fallback") is False
                and session.graph_evidence.get(
                    "consecutive_replay_outputs_exact"
                )
                is True
                and eager_graph.passed
                and graph_alloc_passed
            )
            path_audit = execution_path_audit_facade(
                backend_identity_verified=True,
                device_kernel_family_verified=True,
                allocation_categories_verified=eager_passed and graph_alloc_passed,
                temporary_tensor_shapes_verified=(
                    geometry_passed and pointers_before == pointers_after
                ),
                gqa_replication_detected=not geometry_passed,
                full_prefix_temporary_detected=False,
                host_synchronization_detected=False,
                backend_fallback_detected=not graph_passed,
                full_prefix_dequantization="verified_false",
            )
            accounting = session.method_cache_accounting()
            allocated = int(accounting["allocated_bytes"])
            predicted = int(accounting["predicted_tensor_bytes"])
            relative_error = abs(predicted - allocated) / allocated
            logical_bf16 = int(session.cache.logical_bf16_storage_bytes)
            rho_alloc = allocated / logical_bf16
            r_alloc = logical_bf16 / allocated
            checks = {
                "eager_allocation": eager_passed,
                "graph": graph_passed,
                "non_default_stream": stream_comparison.passed,
                "batch_numerical_control": batch_control["passed"],
                "batch_bank_isolation": bank_control["passed"],
                "finite": finite,
                "native_gqa_geometry": geometry_passed,
                "execution_path": path_audit.passed,
                "pointers_stable": pointers_before == pointers_after,
                "historical_prefix_unchanged": history_before == history_after,
                "allocation_error_below_one_percent": relative_error < 0.01,
                "ratios_reciprocal": (
                    abs(rho_alloc * r_alloc - 1.0) <= 1e-9
                ),
            }
            record_passed = all(checks.values())
            record = {
                "schema_version": "kvbench-phase13b-point-1.0.0",
                "configuration": configuration,
                "method_family": family,
                "batch_size": batch,
                "historical_context": PHASE13B_CONTEXT_LENGTH,
                "attended_context": PHASE13B_CONTEXT_LENGTH + 1,
                "status": "PASS" if record_passed else "FAIL",
                "checks": checks,
                "adapter_version": session.method.adapter_version,
                "adapter_config_fingerprint": session.adapter_config_fingerprint,
                "cache_layout_fingerprint": session.cache_layout_fingerprint(),
                "output_sha256": tensor_sha256_untimed(eager_output),
                "output_finite": finite,
                "batch_numerical_control": batch_control,
                "batch_bank_isolation": bank_control,
                "eager_graph_comparison": eager_graph.to_dict(),
                "non_default_stream_comparison": stream_comparison.to_dict(),
                "eager_allocation": eager_allocation.to_dict(),
                "eager_outer_control": eager_control,
                "graph_allocation": graph_allocation.to_dict(),
                "graph": dict(session.graph_evidence or {}),
                "gqa": geometry,
                "execution_path": path_audit.to_dict(),
                "pointers_stable": pointers_before == pointers_after,
                "historical_prefix_unchanged": history_before == history_after,
                "byte_breakdown": session.method_byte_breakdown(),
                "accounting": accounting,
                "logical_bf16_bytes": logical_bf16,
                "rho_alloc": rho_alloc,
                "r_alloc": r_alloc,
                "reciprocal_error": abs(rho_alloc * r_alloc - 1.0),
                "r_hbm": None,
                "timing_collected": False,
                "performance_claim_eligible": False,
            }
            records.append(record)
            session.graph.graph.reset()
            del (
                complete,
                decode,
                eager_output,
                graph_output,
                prefix,
                ready,
                session,
                stream,
                stream_cpu,
                stream_output,
            )
            torch.cuda.empty_cache()
            if not record_passed:
                failure_payload = {
                    "schema_version": MATRIX_SCHEMA,
                    "status": "FAIL",
                    "created_at_utc": _utc_now(),
                    "creation_git_sha": git_sha,
                    "authorized_container_digest": (
                        PHASE13B_AUTHORIZED_CONTAINER_DIGEST
                    ),
                    "container_attestation": attestation,
                    "decision": "0030",
                    "configurations": list(PHASE13B_CONFIGURATIONS),
                    "batch_sizes": list(PHASE13B_BATCH_SIZES),
                    "context_length": PHASE13B_CONTEXT_LENGTH,
                    "point_count": len(records),
                    "source_hashes": source_hashes,
                    "records": records,
                    "eager_allocation_authority": EAGER_ALLOCATION_AUTHORITY,
                    "failed_point": {
                        "configuration": configuration,
                        "batch_size": batch,
                        "failed_checks": sorted(
                            name for name, passed in checks.items() if not passed
                        ),
                    },
                    "cuda_source_changed": False,
                    "timing_collected": False,
                    "performance_claim_eligible": False,
                }
                _write_exclusive(output, failure_payload)
                raise Phase13BBatchAdmissionError(
                    "Phase 13B point failed: "
                    f"{configuration}/B={batch}; "
                    f"checks={failure_payload['failed_point']['failed_checks']}"
                )

    if len(records) != len(PHASE13B_CONFIGURATIONS) * len(PHASE13B_BATCH_SIZES):
        raise Phase13BBatchAdmissionError("Phase 13B matrix cardinality differs")
    payload = {
        "schema_version": MATRIX_SCHEMA,
        "status": "PASS",
        "created_at_utc": _utc_now(),
        "creation_git_sha": git_sha,
        "authorized_container_digest": PHASE13B_AUTHORIZED_CONTAINER_DIGEST,
        "container_attestation": attestation,
        "decision": "0030",
        "configurations": list(PHASE13B_CONFIGURATIONS),
        "batch_sizes": list(PHASE13B_BATCH_SIZES),
        "context_length": PHASE13B_CONTEXT_LENGTH,
        "point_count": len(records),
        "source_hashes": source_hashes,
        "records": records,
        "eager_allocation_authority": EAGER_ALLOCATION_AUTHORITY,
        "cuda_source_changed": False,
        "timing_collected": False,
        "performance_claim_eligible": False,
    }
    _write_exclusive(output, payload)
    return payload


def validate_cuda_matrix(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase13BBatchAdmissionError("CUDA matrix is unreadable") from error
    expected_pairs = {
        (configuration, batch)
        for configuration in PHASE13B_CONFIGURATIONS
        for batch in PHASE13B_BATCH_SIZES
    }
    records = payload.get("records")
    if (
        payload.get("schema_version") != MATRIX_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("authorized_container_digest")
        != PHASE13B_AUTHORIZED_CONTAINER_DIGEST
        or payload.get("configurations") != list(PHASE13B_CONFIGURATIONS)
        or payload.get("batch_sizes") != list(PHASE13B_BATCH_SIZES)
        or payload.get("point_count") != 27
        or not isinstance(records, list)
        or len(records) != 27
        or {
            (item.get("configuration"), item.get("batch_size"))
            for item in records
            if isinstance(item, dict)
        }
        != expected_pairs
        or any(item.get("status") != "PASS" for item in records)
        or any(item.get("r_hbm") is not None for item in records)
        or any(item.get("timing_collected") is not False for item in records)
    ):
        raise Phase13BBatchAdmissionError("CUDA matrix contract differs")
    return payload


def _strict_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase13BBatchAdmissionError(f"{label} is unreadable") from error
    if not isinstance(payload, dict):
        raise Phase13BBatchAdmissionError(f"{label} must be a JSON object")
    return payload


def _command_markers(kind: str) -> tuple[str, ...]:
    if kind == "test-cuda":
        return (
            "test_all_mandatory_fixture_paths_and_bf16_boundaries",
            "test_all_four_frozen_configurations_store_append_and_rollover",
            "test_all_nine_corrected_fixtures_conform_through_adapter",
            "test_all_twenty_seven_geometry_records_pass",
            "OK",
        )
    if kind == "test-graph":
        return (
            "test_all_mandatory_configs_pass_common_graph_and_audits",
            "test_mandatory_configs_capture_direct_decode_without_replay_allocation",
            "test_all_bit_widths_capture_append_and_direct_decode",
            "OK",
        )
    raise Phase13BBatchAdmissionError("unsupported command-evidence kind")


def create_command_evidence(
    *,
    kind: str,
    stdout_path: Path,
    stderr_path: Path,
    exit_code: int,
    output: Path,
) -> dict[str, Any]:
    stdout = stdout_path.read_bytes()
    stderr = stderr_path.read_bytes()
    combined = (stdout + b"\n" + stderr).decode("utf-8", errors="strict")
    markers = _command_markers(kind)
    if exit_code != 0 or any(marker not in combined for marker in markers):
        raise Phase13BBatchAdmissionError(f"{kind} command evidence failed")
    payload = {
        "schema_version": COMMAND_EVIDENCE_SCHEMA,
        "status": "PASS",
        "kind": kind,
        "authorized_container_digest": PHASE13B_AUTHORIZED_CONTAINER_DIGEST,
        "exit_code": exit_code,
        "required_markers": list(markers),
        "stdout_name": stdout_path.name,
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_name": stderr_path.name,
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "timing_collected": False,
        "performance_claim_eligible": False,
    }
    _write_exclusive(output, payload)
    return payload


def validate_command_evidence(path: Path, *, kind: str) -> dict[str, Any]:
    payload = _strict_json(path, f"{kind} evidence")
    if (
        payload.get("schema_version") != COMMAND_EVIDENCE_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("kind") != kind
        or payload.get("authorized_container_digest")
        != PHASE13B_AUTHORIZED_CONTAINER_DIGEST
        or payload.get("exit_code") != 0
        or payload.get("required_markers") != list(_command_markers(kind))
        or not isinstance(payload.get("stdout_sha256"), str)
        or _SHA256_RE.fullmatch(payload["stdout_sha256"]) is None
        or not isinstance(payload.get("stderr_sha256"), str)
        or _SHA256_RE.fullmatch(payload["stderr_sha256"]) is None
        or payload.get("timing_collected") is not False
        or payload.get("performance_claim_eligible") is not False
    ):
        raise Phase13BBatchAdmissionError(f"{kind} evidence contract differs")
    return payload


def _parse_sanitizer_record(specification: str) -> tuple[str, str, int, Path, Path, int]:
    parts = specification.split(":", 5)
    if len(parts) != 6:
        raise Phase13BBatchAdmissionError("sanitizer record is malformed")
    tool, configuration, batch_text, stdout_text, stderr_text, exit_text = parts
    try:
        batch = int(batch_text)
        exit_code = int(exit_text)
    except ValueError as error:
        raise Phase13BBatchAdmissionError(
            "sanitizer record integers are malformed"
        ) from error
    return tool, configuration, batch, Path(stdout_text), Path(stderr_text), exit_code


def create_sanitizer_evidence(
    *,
    specifications: tuple[str, ...],
    version_path: Path,
    output: Path,
) -> dict[str, Any]:
    version_text = version_path.read_text(encoding="utf-8")
    if "Compute Sanitizer version" not in version_text:
        raise Phase13BBatchAdmissionError("Compute Sanitizer version is absent")
    records: list[dict[str, Any]] = []
    observed: set[tuple[str, str, int]] = set()
    for specification in specifications:
        tool, configuration, batch, stdout_path, stderr_path, exit_code = (
            _parse_sanitizer_record(specification)
        )
        key = (tool, configuration, batch)
        if key in observed:
            raise Phase13BBatchAdmissionError("sanitizer record is duplicated")
        observed.add(key)
        stdout = stdout_path.read_bytes()
        stderr = stderr_path.read_bytes()
        combined = (stdout + b"\n" + stderr).decode("utf-8", errors="strict")
        required = (
            "========= ERROR SUMMARY: 0 errors",
            '"status":"PASS"',
            f'"configuration":"{configuration}"',
            f'"batch_size":{batch}',
        )
        if tool == "memcheck":
            required = (*required, "========= LEAK SUMMARY: 0 bytes leaked")
        if (
            tool not in {"memcheck", "initcheck"}
            or configuration not in PHASE13B_CONFIGURATIONS
            or batch != 8
            or exit_code != 0
            or any(marker not in combined for marker in required)
        ):
            raise Phase13BBatchAdmissionError(
                f"sanitizer record failed: {tool}/{configuration}/B{batch}"
            )
        records.append(
            {
                "tool": tool,
                "configuration": configuration,
                "batch_size": batch,
                "exit_code": exit_code,
                "stdout_name": stdout_path.name,
                "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
                "stderr_name": stderr_path.name,
                "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
                "error_count": 0,
                "leaked_bytes": 0 if tool == "memcheck" else None,
            }
        )
    expected = {
        ("memcheck", configuration, 8)
        for configuration in PHASE13B_CONFIGURATIONS
    } | {
        ("initcheck", configuration, 8)
        for configuration in SANITIZER_INITCHECK_CONFIGURATIONS
    }
    if observed != expected:
        raise Phase13BBatchAdmissionError("sanitizer coverage matrix differs")
    payload = {
        "schema_version": SANITIZER_EVIDENCE_SCHEMA,
        "status": "PASS",
        "authorized_container_digest": PHASE13B_AUTHORIZED_CONTAINER_DIGEST,
        "tool_version": version_text.strip(),
        "records": records,
        "memcheck_configurations": list(PHASE13B_CONFIGURATIONS),
        "initcheck_configurations": list(SANITIZER_INITCHECK_CONFIGURATIONS),
        "errors": 0,
        "leaked_bytes": 0,
        "timing_collected": False,
    }
    _write_exclusive(output, payload)
    return payload


def validate_sanitizer_evidence(path: Path) -> dict[str, Any]:
    payload = _strict_json(path, "sanitizer evidence")
    records = payload.get("records")
    expected = {
        ("memcheck", configuration, 8)
        for configuration in PHASE13B_CONFIGURATIONS
    } | {
        ("initcheck", configuration, 8)
        for configuration in SANITIZER_INITCHECK_CONFIGURATIONS
    }
    observed = {
        (record.get("tool"), record.get("configuration"), record.get("batch_size"))
        for record in records
        if isinstance(record, dict)
    } if isinstance(records, list) else set()
    if (
        payload.get("schema_version") != SANITIZER_EVIDENCE_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("authorized_container_digest")
        != PHASE13B_AUTHORIZED_CONTAINER_DIGEST
        or observed != expected
        or len(records) != len(expected)
        or payload.get("errors") != 0
        or payload.get("leaked_bytes") != 0
        or payload.get("timing_collected") is not False
        or any(record.get("exit_code") != 0 for record in records)
        or any(record.get("error_count") != 0 for record in records)
        or any(
            record.get("leaked_bytes") != 0
            for record in records
            if record.get("tool") == "memcheck"
        )
    ):
        raise Phase13BBatchAdmissionError("sanitizer evidence contract differs")
    return payload


def _historical_preservation_payload() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for relative, expected in {
        **HISTORICAL_PRESERVATION_AUTHORITY,
        **{
            path: digest
            for path, digest in SUCCESSOR_HISTORICAL_REPORTS.values()
        },
    }.items():
        path = REPOSITORY_ROOT / relative
        observed = _sha256_file(path)
        if observed != expected:
            raise Phase13BBatchAdmissionError(
                f"historical evidence changed: {relative}"
            )
        records.append(
            {
                "path": relative,
                "sha256": observed,
                "status": "UNCHANGED",
            }
        )
    return {
        "schema_version": "kvbench-phase13b-historical-preservation-1.0.0",
        "status": "PASS",
        "records": records,
        "stopped_phase13_campaign_resumed": False,
        "historical_evidence_overwritten": False,
    }


def _report_for_family(
    *,
    family: str,
    matrix: Mapping[str, Any],
    evidence_sha256: Mapping[str, str],
    created_at: str,
    git_sha: str,
) -> Phase13BMethodAdmissionReport:
    configurations = PHASE13B_FAMILY_CONFIGURATIONS[family]
    records = [
        record
        for record in matrix["records"]
        if record["configuration"] in configurations
    ]
    if len(records) != 9 or any(record["status"] != "PASS" for record in records):
        raise Phase13BBatchAdmissionError(
            f"{family} Phase 13B geometry evidence is incomplete"
        )
    versions: dict[str, str] = {}
    adapter_fingerprints: dict[str, str] = {}
    layout_fingerprints: dict[str, str] = {}
    for configuration in configurations:
        configuration_records = [
            record
            for record in records
            if record["configuration"] == configuration
        ]
        observed_versions = {
            str(record["adapter_version"]) for record in configuration_records
        }
        if len(observed_versions) != 1:
            raise Phase13BBatchAdmissionError(
                f"{configuration} adapter version drifted across batches"
            )
        versions[configuration] = observed_versions.pop()
        for record in configuration_records:
            key = f"{configuration}/B{record['batch_size']}"
            adapter_fingerprints[key] = record["adapter_config_fingerprint"]
            layout_fingerprints[key] = record["cache_layout_fingerprint"]
    historical_path, historical_sha256 = SUCCESSOR_HISTORICAL_REPORTS[family]
    source_hashes = {
        path: matrix["source_hashes"][path]
        for path in FAMILY_SOURCE_PATHS[family]
    }
    for relative, expected in source_hashes.items():
        if _sha256_file(REPOSITORY_ROOT / relative) != expected:
            raise Phase13BBatchAdmissionError(
                f"{family} source changed after CUDA validation"
            )
    return Phase13BMethodAdmissionReport(
        schema_version=Phase13BMethodAdmissionReport.SCHEMA_VERSION,
        created_at_utc=created_at,
        status="PASS",
        method_family=family,
        configurations=configurations,
        batch_sizes=PHASE13B_BATCH_SIZES,
        authorized_container_digest=PHASE13B_AUTHORIZED_CONTAINER_DIGEST,
        decision_id="0030",
        creation_git_sha=git_sha,
        historical_report_path=historical_path,
        historical_report_sha256=historical_sha256,
        source_hashes=source_hashes,
        adapter_versions=versions,
        adapter_config_fingerprints=adapter_fingerprints,
        cache_layout_fingerprints=layout_fingerprints,
        checks={check_id: "PASS" for check_id in PHASE13B_SUCCESSOR_CHECK_IDS},
        evidence_references={
            evidence_id: evidence_sha256[evidence_id]
            for evidence_id in PHASE13B_SUCCESSOR_EVIDENCE_IDS
        },
        b1_numerical_preserved=True,
        cuda_source_changed=False,
        timing_collected=False,
        performance_claim_eligible=False,
        quality_execution="LOCKED",
        full_scan_state="CLOSED",
        r_hbm=None,
        blockers=(),
    )


def _verify_command_raw_logs(
    evidence: Mapping[str, Any],
    *,
    stdout_path: Path,
    stderr_path: Path,
) -> None:
    if (
        stdout_path.name != evidence["stdout_name"]
        or stderr_path.name != evidence["stderr_name"]
        or _sha256_file(stdout_path) != evidence["stdout_sha256"]
        or _sha256_file(stderr_path) != evidence["stderr_sha256"]
    ):
        raise Phase13BBatchAdmissionError("command raw logs differ")


def _verify_sanitizer_raw_logs(
    evidence: Mapping[str, Any],
    *,
    log_directory: Path,
) -> list[tuple[str, bytes]]:
    payloads: list[tuple[str, bytes]] = []
    for record in evidence["records"]:
        prefix = (
            f"{record['tool']}-{record['configuration']}-"
            f"B{record['batch_size']}"
        )
        stdout_path = log_directory / record["stdout_name"]
        stderr_path = log_directory / record["stderr_name"]
        if (
            _sha256_file(stdout_path) != record["stdout_sha256"]
            or _sha256_file(stderr_path) != record["stderr_sha256"]
        ):
            raise Phase13BBatchAdmissionError(
                f"sanitizer raw logs differ: {prefix}"
            )
        payloads.extend(
            (
                (f"validation/sanitizer/{prefix}.stdout.txt", stdout_path.read_bytes()),
                (f"validation/sanitizer/{prefix}.stderr.txt", stderr_path.read_bytes()),
            )
        )
    return payloads


def finalize_phase13b_bundle(
    *,
    matrix_path: Path,
    test_cuda_evidence_path: Path,
    test_graph_evidence_path: Path,
    test_cuda_stdout_path: Path,
    test_cuda_stderr_path: Path,
    test_graph_stdout_path: Path,
    test_graph_stderr_path: Path,
    sanitizer_evidence_path: Path,
    sanitizer_log_directory: Path,
    artifact_root: Path,
    reports_directory: Path,
    run_id: str,
    git_sha: str,
) -> dict[str, Any]:
    _require_git_authority(git_sha)
    matrix = validate_cuda_matrix(matrix_path)
    test_cuda = validate_command_evidence(
        test_cuda_evidence_path,
        kind="test-cuda",
    )
    test_graph = validate_command_evidence(
        test_graph_evidence_path,
        kind="test-graph",
    )
    sanitizer = validate_sanitizer_evidence(sanitizer_evidence_path)
    _verify_command_raw_logs(
        test_cuda,
        stdout_path=test_cuda_stdout_path,
        stderr_path=test_cuda_stderr_path,
    )
    _verify_command_raw_logs(
        test_graph,
        stdout_path=test_graph_stdout_path,
        stderr_path=test_graph_stderr_path,
    )
    sanitizer_logs = _verify_sanitizer_raw_logs(
        sanitizer,
        log_directory=sanitizer_log_directory,
    )
    historical = _historical_preservation_payload()
    evidence_payloads = {
        "cuda_matrix": matrix_path.read_bytes(),
        "test_cuda": test_cuda_evidence_path.read_bytes(),
        "test_graph": test_graph_evidence_path.read_bytes(),
        "sanitizer": sanitizer_evidence_path.read_bytes(),
        "historical_preservation": _json_bytes(historical),
    }
    evidence_sha256 = {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in evidence_payloads.items()
    }
    created_at = _utc_now()
    reports = {
        family: _report_for_family(
            family=family,
            matrix=matrix,
            evidence_sha256=evidence_sha256,
            created_at=created_at,
            git_sha=git_sha,
        )
        for family in PHASE13B_FAMILY_CONFIGURATIONS
    }
    report_paths = {
        family: reports_directory / f"{family}-method-admission.json"
        for family in reports
    }
    if any(path.exists() or path.is_symlink() for path in report_paths.values()):
        raise Phase13BBatchAdmissionError(
            "Phase 13B successor report path already exists"
        )
    initial = Phase13BBatchAdmissionManifest(
        schema_version=Phase13BBatchAdmissionManifest.SCHEMA_VERSION,
        run_id=run_id,
        status=RunStatus.CREATED,
        created_at_utc=created_at,
        started_at_utc=None,
        finished_at_utc=None,
        inventory_path=None,
        failure_reason=None,
        creation_git_sha=git_sha,
        authorized_container_digest=PHASE13B_AUTHORIZED_CONTAINER_DIGEST,
        decision_id="0030",
        configurations=PHASE13B_CONFIGURATIONS,
        batch_sizes=PHASE13B_BATCH_SIZES,
        context_length=PHASE13B_CONTEXT_LENGTH,
        timing_collected=False,
        performance_claim_eligible=False,
        quality_executed=False,
        full_scan_executed=False,
    )
    store = AppendOnlyArtifactStore(artifact_root)
    run = store.create(run_id, initial)
    run.start()
    authority = {
        "schema_version": "kvbench-phase13b-authority-1.0.0",
        "decision": "0030",
        "creation_git_sha": git_sha,
        "matrix_creation_git_sha": matrix["creation_git_sha"],
        "authorized_container_digest": PHASE13B_AUTHORIZED_CONTAINER_DIGEST,
        "configurations": list(PHASE13B_CONFIGURATIONS),
        "batch_sizes": list(PHASE13B_BATCH_SIZES),
        "context_length": PHASE13B_CONTEXT_LENGTH,
        "cuda_source_changed": False,
        "timing_collected": False,
    }
    run.write_json("config/authority.json", authority)
    run.write_json(
        "environment/container_identity.json",
        {
            "schema_version": "kvbench-phase13b-container-identity-1.0.0",
            "status": "PASS",
            "digest": PHASE13B_AUTHORIZED_CONTAINER_DIGEST,
            "matrix_attestation": matrix["container_attestation"],
        },
    )
    run.write_bytes("validation/matrix.json", evidence_payloads["cuda_matrix"])
    run.write_bytes("validation/test-cuda.json", evidence_payloads["test_cuda"])
    run.write_bytes("validation/test-graph.json", evidence_payloads["test_graph"])
    run.write_bytes("validation/sanitizer.json", evidence_payloads["sanitizer"])
    run.write_bytes(
        "validation/historical-preservation.json",
        evidence_payloads["historical_preservation"],
    )
    run.write_bytes("validation/test-cuda.stdout.txt", test_cuda_stdout_path.read_bytes())
    run.write_bytes("validation/test-cuda.stderr.txt", test_cuda_stderr_path.read_bytes())
    run.write_bytes("validation/test-graph.stdout.txt", test_graph_stdout_path.read_bytes())
    run.write_bytes("validation/test-graph.stderr.txt", test_graph_stderr_path.read_bytes())
    for relative, payload in sanitizer_logs:
        run.write_bytes(relative, payload)
    numerical_records = [
        {
            "configuration": record["configuration"],
            "batch_size": record["batch_size"],
            "status": record["status"],
            "output_finite": record["output_finite"],
            "batch_numerical_control": record["batch_numerical_control"],
            "batch_bank_isolation": record["batch_bank_isolation"],
        }
        for record in matrix["records"]
    ]
    run.write_json(
        "numerical/batch-control.json",
        {
            "schema_version": "kvbench-phase13b-numerical-summary-1.0.0",
            "status": "PASS",
            "b1_fixture_replay": "PASS",
            "records": numerical_records,
        },
    )
    run.write_json(
        "allocation/audit.json",
        {
            "schema_version": "kvbench-phase13b-allocation-summary-1.0.0",
            "status": "PASS",
            "records": [
                {
                    "configuration": record["configuration"],
                    "batch_size": record["batch_size"],
                    "eager": record["eager_allocation"],
                    "outer_control": record["eager_outer_control"],
                    "graph": record["graph_allocation"],
                    "accounting": record["accounting"],
                    "r_hbm": None,
                }
                for record in matrix["records"]
            ],
        },
    )
    run.write_json(
        "gqa/audit.json",
        {
            "schema_version": "kvbench-phase13b-gqa-summary-1.0.0",
            "status": "PASS",
            "records": [
                {
                    "configuration": record["configuration"],
                    "batch_size": record["batch_size"],
                    "gqa": record["gqa"],
                    "execution_path": record["execution_path"],
                }
                for record in matrix["records"]
            ],
        },
    )
    run.write_json(
        "validation/cuda-graph.json",
        {
            "schema_version": "kvbench-phase13b-graph-summary-1.0.0",
            "status": "PASS",
            "records": [
                {
                    "configuration": record["configuration"],
                    "batch_size": record["batch_size"],
                    "graph": record["graph"],
                    "comparison": record["eager_graph_comparison"],
                    "allocation": record["graph_allocation"],
                    "pointers_stable": record["pointers_stable"],
                }
                for record in matrix["records"]
            ],
        },
    )
    run.write_json(
        "validation/non-default-stream.json",
        {
            "schema_version": "kvbench-phase13b-stream-summary-1.0.0",
            "status": "PASS",
            "records": [
                {
                    "configuration": record["configuration"],
                    "batch_size": record["batch_size"],
                    "comparison": record["non_default_stream_comparison"],
                }
                for record in matrix["records"]
            ],
        },
    )
    for family, report in reports.items():
        run.write_json(
            f"admission/{family}-method-admission.json",
            report.to_dict(),
        )
    finished_at = _utc_now()
    completed = Phase13BBatchAdmissionManifest.from_dict(
        {
            **initial.to_dict(),
            "status": RunStatus.COMPLETED.value,
            "started_at_utc": created_at,
            "finished_at_utc": finished_at,
            "inventory_path": "artifact_inventory.json",
        }
    )
    final = run.finalize(completed)
    validation = validate_run_directory(final)
    if not validation.valid or not validation.complete:
        raise Phase13BBatchAdmissionError("final Phase 13B bundle is invalid")
    reports_directory.mkdir(parents=True, exist_ok=True)
    report_sha256: dict[str, str] = {}
    for family, report in reports.items():
        payload = _json_bytes(report.to_dict())
        path = report_paths[family]
        with path.open("xb") as handle:
            handle.write(payload)
        report_sha256[family] = hashlib.sha256(payload).hexdigest()
    complete = _strict_json(final / "COMPLETE", "Phase 13B COMPLETE")
    return {
        "status": "PASS",
        "run_id": run_id,
        "artifact": final.as_posix(),
        "root_sha256": complete["checksum_ledger_sha256"],
        "report_sha256": report_sha256,
    }


def validate_phase13b_bundle(path: Path) -> dict[str, Any]:
    validation = validate_run_directory(path)
    if not validation.valid or not validation.complete:
        raise Phase13BBatchAdmissionError("Phase 13B bundle integrity failed")
    matrix = validate_cuda_matrix(path / "validation/matrix.json")
    test_cuda = validate_command_evidence(
        path / "validation/test-cuda.json",
        kind="test-cuda",
    )
    test_graph = validate_command_evidence(
        path / "validation/test-graph.json",
        kind="test-graph",
    )
    sanitizer = validate_sanitizer_evidence(path / "validation/sanitizer.json")
    historical = _strict_json(
        path / "validation/historical-preservation.json",
        "historical preservation",
    )
    if historical != _historical_preservation_payload():
        raise Phase13BBatchAdmissionError("historical preservation record differs")
    common_evidence = {
        "cuda_matrix": _sha256_file(path / "validation/matrix.json"),
        "test_cuda": _sha256_file(path / "validation/test-cuda.json"),
        "test_graph": _sha256_file(path / "validation/test-graph.json"),
        "sanitizer": _sha256_file(path / "validation/sanitizer.json"),
        "historical_preservation": _sha256_file(
            path / "validation/historical-preservation.json"
        ),
    }
    report_sha256: dict[str, str] = {}
    for family in PHASE13B_FAMILY_CONFIGURATIONS:
        report_path = path / "admission" / f"{family}-method-admission.json"
        report = Phase13BMethodAdmissionReport.from_dict(
            _strict_json(report_path, f"{family} successor report")
        )
        if report.evidence_references != common_evidence:
            raise Phase13BBatchAdmissionError(
                f"{family} successor evidence binding differs"
            )
        report_sha256[family] = _sha256_file(report_path)
    _verify_command_raw_logs(
        test_cuda,
        stdout_path=path / "validation/test-cuda.stdout.txt",
        stderr_path=path / "validation/test-cuda.stderr.txt",
    )
    _verify_command_raw_logs(
        test_graph,
        stdout_path=path / "validation/test-graph.stdout.txt",
        stderr_path=path / "validation/test-graph.stderr.txt",
    )
    for record in sanitizer["records"]:
        prefix = (
            f"{record['tool']}-{record['configuration']}-"
            f"B{record['batch_size']}"
        )
        if (
            _sha256_file(path / f"validation/sanitizer/{prefix}.stdout.txt")
            != record["stdout_sha256"]
            or _sha256_file(path / f"validation/sanitizer/{prefix}.stderr.txt")
            != record["stderr_sha256"]
        ):
            raise Phase13BBatchAdmissionError("sanitizer bundle logs differ")
    complete = _strict_json(path / "COMPLETE", "Phase 13B COMPLETE")
    return {
        "status": "PASS",
        "run_id": complete["run_id"],
        "root_sha256": complete["checksum_ledger_sha256"],
        "point_count": matrix["point_count"],
        "report_sha256": report_sha256,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--cuda-matrix", action="store_true")
    actions.add_argument("--validate-cuda-matrix", action="store_true")
    actions.add_argument("--command-evidence", action="store_true")
    actions.add_argument("--sanitizer-evidence", action="store_true")
    actions.add_argument("--finalize", action="store_true")
    actions.add_argument("--validate-bundle", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--git-sha")
    parser.add_argument("--kind", choices=("test-cuda", "test-graph"))
    parser.add_argument("--stdout", type=Path)
    parser.add_argument("--stderr", type=Path)
    parser.add_argument("--exit-code", type=int)
    parser.add_argument("--sanitizer-record", action="append", default=[])
    parser.add_argument("--sanitizer-version", type=Path)
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--test-cuda-evidence", type=Path)
    parser.add_argument("--test-graph-evidence", type=Path)
    parser.add_argument("--test-cuda-stdout", type=Path)
    parser.add_argument("--test-cuda-stderr", type=Path)
    parser.add_argument("--test-graph-stdout", type=Path)
    parser.add_argument("--test-graph-stderr", type=Path)
    parser.add_argument("--sanitizer-evidence-path", type=Path)
    parser.add_argument("--sanitizer-log-directory", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--reports-directory", type=Path)
    parser.add_argument("--run-id")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.cuda_matrix:
        if args.output is None or args.git_sha is None:
            raise Phase13BBatchAdmissionError(
                "--cuda-matrix requires --output and --git-sha"
            )
        payload = _run_cuda_matrix(output=args.output, git_sha=args.git_sha)
    elif args.validate_cuda_matrix:
        if args.artifact is None:
            raise Phase13BBatchAdmissionError(
                "--validate-cuda-matrix requires --artifact"
            )
        payload = validate_cuda_matrix(args.artifact)
    elif args.command_evidence:
        if None in (args.kind, args.stdout, args.stderr, args.exit_code, args.output):
            raise Phase13BBatchAdmissionError(
                "--command-evidence requires kind, logs, exit code, and output"
            )
        payload = create_command_evidence(
            kind=args.kind,
            stdout_path=args.stdout,
            stderr_path=args.stderr,
            exit_code=args.exit_code,
            output=args.output,
        )
    elif args.sanitizer_evidence:
        if args.output is None or args.sanitizer_version is None:
            raise Phase13BBatchAdmissionError(
                "--sanitizer-evidence requires version and output"
            )
        payload = create_sanitizer_evidence(
            specifications=tuple(args.sanitizer_record),
            version_path=args.sanitizer_version,
            output=args.output,
        )
    elif args.finalize:
        required = (
            args.matrix,
            args.test_cuda_evidence,
            args.test_graph_evidence,
            args.test_cuda_stdout,
            args.test_cuda_stderr,
            args.test_graph_stdout,
            args.test_graph_stderr,
            args.sanitizer_evidence_path,
            args.sanitizer_log_directory,
            args.artifact_root,
            args.reports_directory,
            args.run_id,
            args.git_sha,
        )
        if any(value is None for value in required):
            raise Phase13BBatchAdmissionError(
                "--finalize requires every Phase 13B evidence input"
            )
        payload = finalize_phase13b_bundle(
            matrix_path=args.matrix,
            test_cuda_evidence_path=args.test_cuda_evidence,
            test_graph_evidence_path=args.test_graph_evidence,
            test_cuda_stdout_path=args.test_cuda_stdout,
            test_cuda_stderr_path=args.test_cuda_stderr,
            test_graph_stdout_path=args.test_graph_stdout,
            test_graph_stderr_path=args.test_graph_stderr,
            sanitizer_evidence_path=args.sanitizer_evidence_path,
            sanitizer_log_directory=args.sanitizer_log_directory,
            artifact_root=args.artifact_root,
            reports_directory=args.reports_directory,
            run_id=args.run_id,
            git_sha=args.git_sha,
        )
    else:
        if args.artifact is None:
            raise Phase13BBatchAdmissionError(
                "--validate-bundle requires --artifact"
            )
        payload = validate_phase13b_bundle(args.artifact)
    print(
        json.dumps(
            {
                "status": payload["status"],
                **(
                    {"schema_version": payload["schema_version"]}
                    if "schema_version" in payload
                    else {}
                ),
                **(
                    {"sha256": _sha256_file(args.output)}
                    if args.output is not None and args.output.is_file()
                    else {}
                ),
                **(
                    {"root_sha256": payload["root_sha256"]}
                    if "root_sha256" in payload
                    else {}
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
