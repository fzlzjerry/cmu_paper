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

from kvbench.runtime.method_harness import execution_path_audit_facade
from kvbench.schema import canonical_json_bytes, sha256_hex
from kvbench.schema.phase13b import (
    PHASE13B_AUTHORIZED_CONTAINER_DIGEST,
    PHASE13B_BATCH_SIZES,
    PHASE13B_CONFIGURATIONS,
    PHASE13B_CONTEXT_LENGTH,
)

from scripts import phase12_unified_admission as phase12
from scripts import phase13_pilot as phase13


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MATRIX_SCHEMA = "kvbench-phase13b-cuda-validation-1.0.0"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


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


def _eager_matches_outer_control(observed: Any, control: Mapping[str, Any]) -> bool:
    """Require the compressed path to add no CUDA storage events."""

    return bool(
        observed.audit_available
        and observed.allocated_after == observed.allocated_before
        and observed.reserved_after == observed.reserved_before
        and observed.allocation_event_count
        == control["allocation_event_count"]
        and observed.allocation_event_bytes
        == control["allocation_event_bytes"]
        and observed.event_counts == control["event_counts"]
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
    outer_allocation_controls: dict[int, dict[str, Any]] = {}
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

    for batch in PHASE13B_BATCH_SIZES:
        phase13._patch_phase12_point_globals(
            batch=batch,
            historical=PHASE13B_CONTEXT_LENGTH,
        )
        control_operation = phase13.Phase13OperationKey.create(
            "bf16",
            batch,
            PHASE13B_CONTEXT_LENGTH,
        )
        control_prefix = base_prefix.expand(batch, -1).clone()
        control_decode = base_decode.expand(batch, -1).clone()
        with torch.inference_mode(), forced_flash_execution():
            control_session = phase12._build_phase12_session(
                loaded=loaded,
                operation_key=control_operation,
                prefix_input_ids=control_prefix,
                decode_input_ids=control_decode,
            )
            if control_session.graph is None or control_session._fixed_operation is None:
                raise Phase13BBatchAdmissionError(
                    "BF16 outer allocation control is unavailable"
                )
            control_audit = audit_cuda_allocations(
                control_session._fixed_operation,
                device=control_session.cache_device,
            )
        control_record = control_audit.to_dict()
        if (
            not control_audit.audit_available
            or control_audit.allocated_after != control_audit.allocated_before
            or control_audit.reserved_after != control_audit.reserved_before
        ):
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
                "point_count": 0,
                "source_hashes": source_hashes,
                "records": [],
                "outer_allocation_controls": {
                    **outer_allocation_controls,
                    batch: control_record,
                },
                "failed_point": {
                    "configuration": "bf16_outer_allocation_control",
                    "batch_size": batch,
                    "failed_checks": [
                        "audit_available_or_persistent_delta"
                    ],
                },
                "cuda_source_changed": False,
                "timing_collected": False,
                "performance_claim_eligible": False,
            }
            _write_exclusive(output, failure_payload)
            raise Phase13BBatchAdmissionError(
                f"BF16 outer allocation control failed for B={batch}"
            )
        outer_allocation_controls[batch] = control_record
        control_session.graph.graph.reset()
        del control_decode, control_prefix, control_session
        torch.cuda.empty_cache()

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

            family = phase12._method_family(configuration)
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
            if batch == 1:
                b1_outputs[configuration] = eager_output.clone()
                batch_control = {
                    "passed": True,
                    "rows_compared": 1,
                    "reference": "self_b1_and_frozen_method_tolerance",
                }
            else:
                reference = b1_outputs.get(configuration)
                if reference is None:
                    raise Phase13BBatchAdmissionError("B=1 control is absent")
                comparisons = [
                    compare_tensors_untimed(
                        eager_output[index : index + 1],
                        reference,
                        atol=atol,
                        rtol=rtol,
                    )
                    for index in range(batch)
                ]
                batch_control = {
                    "passed": all(item.passed for item in comparisons),
                    "rows_compared": batch,
                    "maximum_absolute_error": max(
                        item.max_absolute_error for item in comparisons
                    ),
                    "maximum_relative_error": max(
                        item.max_relative_error for item in comparisons
                    ),
                    "reference": "identical_input_b1_execution",
                }

            pointers_after = phase12._phase12_session_pointers(session)
            history_after = session.current_historical_prefix_sha256()
            geometry = session.gqa_cache_geometry()
            geometry_passed = phase12._gqa_geometry_passes(
                geometry,
                family=family,
            )
            eager_passed = _eager_matches_outer_control(
                eager_allocation,
                outer_allocation_controls[batch],
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
                "eager_graph_comparison": eager_graph.to_dict(),
                "non_default_stream_comparison": stream_comparison.to_dict(),
                "eager_allocation": eager_allocation.to_dict(),
                "eager_outer_control": outer_allocation_controls[batch],
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
                    "outer_allocation_controls": outer_allocation_controls,
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
        "outer_allocation_controls": outer_allocation_controls,
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--cuda-matrix", action="store_true")
    actions.add_argument("--validate-cuda-matrix", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--git-sha")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.cuda_matrix:
        if args.output is None or args.git_sha is None:
            raise Phase13BBatchAdmissionError(
                "--cuda-matrix requires --output and --git-sha"
            )
        payload = _run_cuda_matrix(output=args.output, git_sha=args.git_sha)
    else:
        if args.artifact is None:
            raise Phase13BBatchAdmissionError(
                "--validate-cuda-matrix requires --artifact"
            )
        payload = validate_cuda_matrix(args.artifact)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "schema_version": payload["schema_version"],
                "sha256": (
                    _sha256_file(args.output)
                    if args.cuda_matrix
                    else _sha256_file(args.artifact)
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
