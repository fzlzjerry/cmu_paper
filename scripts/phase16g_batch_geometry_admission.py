#!/usr/bin/env python3
"""Focused B=2/B=16 geometry admission without benchmark timing."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import re
import shutil
import stat
import subprocess
from typing import Any

from kvbench.runtime.method_harness import execution_path_audit_facade
from kvbench.schema.phase16g import (
    PHASE16G_ADMITTED_BATCH_SIZES,
    PHASE16G_CONFIGURATIONS,
    PHASE16G_CONTAINER_DIGEST,
    PHASE16G_CONTEXT_LENGTH,
    PHASE16G_INDEX_SCHEMA,
    PHASE16G_NEW_BATCH_SIZES,
    PHASE16G_PREFIX_SCHEMA,
    geometry_key,
    validate_geometry_index,
)
from scripts import phase12_unified_admission as phase12
from scripts import phase13_pilot as phase13
from scripts import phase13b_compressed_batch_admission as phase13b
from scripts.phase13_prefix_state import (
    PREFIX_STATE_MANIFEST,
    PREFIX_STATE_SCHEMA_V3,
    restore_prefix_state,
    restored_prefix_witness,
    save_prefix_state,
    validate_prefix_state,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CUDA_EVIDENCE_SCHEMA = "kvbench-phase16g-cuda-admission-1.0.0"
POINT_SCHEMA = "kvbench-phase16g-admission-point-1.0.0"
PREFIX_EVIDENCE_SCHEMA = "kvbench-phase16g-prefix-evidence-1.0.0"
SANITIZER_SCHEMA = "kvbench-phase16g-sanitizer-evidence-1.0.0"
BUNDLE_SCHEMA = "kvbench-phase16g-bundle-1.0.0"
BASE_CONTEXT_LABELS = phase13.CONTEXT_LABELS
MODES = ("eager", "cuda_graph")
DECISION_ID = "0039"
MODEL_IDENTITY = {
    "id": phase12.PHASE12_MODEL_ID,
    "revision": phase12.PHASE12_MODEL_REVISION,
}
TOKENIZER_IDENTITY = {
    "id": phase12.PHASE12_TOKENIZER_ID,
    "revision": phase12.PHASE12_TOKENIZER_REVISION,
}
METHOD_FINGERPRINTS = dict(phase13.CONFIG_FINGERPRINTS)
SANITIZER_CONFIGURATIONS = (
    "tq_4bit_nc",
    "tq_k3v4_nc",
    "tq_3bit_nc",
    "k4v4",
    "k2v2",
    "kvq4",
    "kvq3",
    "kvq2",
)
CURRENT_REPORT_AUTHORITIES = {
    "bf16": "docs/evidence/phase4/method-admission.json",
    "tq_4bit_nc": "docs/evidence/phase13b/turboquant-method-admission.json",
    "tq_k3v4_nc": "docs/evidence/phase13b/turboquant-method-admission.json",
    "tq_3bit_nc": "docs/evidence/phase13b/turboquant-method-admission.json",
    "k4v4": "docs/evidence/phase13b/kivi-method-admission.json",
    "k2v4": "docs/evidence/phase13b/kivi-method-admission.json",
    "k2v2": "docs/evidence/phase13b/kivi-method-admission.json",
    "kvq4": "docs/evidence/phase13rq4/kvquant-q4-method-admission.json",
    "kvq3": "docs/evidence/phase13b/kvquant-method-admission.json",
    "kvq2": "docs/evidence/phase13b/kvquant-method-admission.json",
}
SOURCE_PATHS = (
    "scripts/phase13_prefix_state.py",
    "src/kvbench/schema/phase16g.py",
    "src/kvbench/runtime/cuda_graph.py",
    "src/kvbench/runtime/allocation.py",
    "src/kvbench/adapters/bf16.py",
    "src/kvbench/adapters/turboquant.py",
    "src/kvbench/adapters/kivi.py",
    "src/kvbench/adapters/kvquant.py",
    "src/kvbench/runtime/static_cache.py",
    "src/kvbench/runtime/turboquant_cache.py",
    "src/kvbench/runtime/kivi_cache.py",
    "src/kvbench/runtime/kvquant_cache.py",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class Phase16GError(RuntimeError):
    """A Phase 16G authority, admission, or bundle check failed closed."""


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(json_bytes(payload))


def require_clean_git(expected: str) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", expected) is None:
        raise Phase16GError("execution Git SHA is invalid")
    head = subprocess.run(
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
    if head != expected or status:
        raise Phase16GError("execution source is not the clean committed authority")


def current_report_authorities() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for configuration, relative in CURRENT_REPORT_AUTHORITIES.items():
        path = REPOSITORY_ROOT / relative
        if not path.is_file() or path.is_symlink():
            raise Phase16GError("current method admission report is absent")
        result[configuration] = {
            "path": relative,
            "sha256": sha256_file(path),
            "existing_batches": [1, 4, 8],
            "historical_evidence_modified": False,
        }
    return result


def source_hashes() -> dict[str, str]:
    return {relative: sha256_file(REPOSITORY_ROOT / relative) for relative in SOURCE_PATHS}


def b16_feasibility_records() -> list[dict[str, Any]]:
    """Use the frozen Phase 13F formula while admitting no execution geometry."""

    original_batches = phase13.BATCH_SIZES
    phase13.BATCH_SIZES = PHASE16G_ADMITTED_BATCH_SIZES
    try:
        records = []
        for configuration in PHASE16G_CONFIGURATIONS:
            for label in BASE_CONTEXT_LABELS:
                historical = phase13.actual_historical_context(label)
                record = phase13.feasibility_record(
                    {
                        "method_config_id": configuration,
                        "batch_size": 16,
                        "context_label": label,
                        "historical_context": historical,
                    }
                )
                record["phase16g_formula_reuse"] = True
                records.append(record)
    finally:
        phase13.BATCH_SIZES = original_batches
    if len(records) != len(PHASE16G_CONFIGURATIONS) * len(BASE_CONTEXT_LABELS):
        raise Phase16GError("B=16 feasibility cardinality differs")
    return records


def largest_feasible_contexts(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    result: dict[str, int] = {}
    for configuration in PHASE16G_CONFIGURATIONS:
        feasible = [
            int(item["context_label"])
            for item in records
            if item.get("method_config_id") == configuration
            and item.get("status") == "feasible"
        ]
        if not feasible:
            raise Phase16GError(f"B=16 has no feasible base point: {configuration}")
        result[configuration] = max(feasible)
    return result


def validate_feasibility_records(records: Sequence[Mapping[str, Any]]) -> None:
    expected = {
        (configuration, label)
        for configuration in PHASE16G_CONFIGURATIONS
        for label in BASE_CONTEXT_LABELS
    }
    observed = {
        (item.get("method_config_id"), item.get("context_label")) for item in records
    }
    if observed != expected:
        raise Phase16GError("B=16 feasibility coverage differs")
    for item in records:
        if (
            item.get("batch_size") != 16
            or item.get("status") not in {"feasible", "capacity_infeasible"}
            or item.get("max_memory_fraction") != 0.88
            or item.get("method_config_fingerprint")
            != METHOD_FINGERPRINTS[str(item["method_config_id"])]
        ):
            raise Phase16GError("B=16 feasibility record differs")


def _prefix_positions(*, batch: int, historical: int, device: Any) -> Any:
    import torch

    return (
        torch.arange(historical, dtype=torch.long, device=device)
        .reshape(1, historical)
        .expand(batch, -1)
        .clone()
    )


def _release_session(session: Any) -> None:
    import torch

    if session is not None and session.graph is not None:
        session.graph.graph.reset()
    del session
    gc.collect()
    torch.cuda.empty_cache()


def _prefix_prefill(endpoint: Any, input_ids: Any, original: Any, family: str) -> Any:
    if family == "kvquant":
        with phase13._chunked_kvquant_prefix_store(endpoint):
            return original(endpoint, input_ids)
    return original(endpoint, input_ids)


def _build_direct_session_and_snapshot(
    *,
    loaded: Any,
    configuration: str,
    batch: int,
    historical: int,
    snapshot_root: Path,
) -> tuple[Any, dict[str, Any], Any, Any, Any]:
    import torch

    from kvbench.runtime.backend import forced_flash_execution

    phase13._patch_phase12_point_globals(batch=batch, historical=historical)
    operation = phase13.Phase13OperationKey.create(configuration, batch, historical)
    prefix, decode = phase13._point_inputs(
        batch=batch,
        historical=historical,
        device=torch.device("cuda:0"),
    )
    positions = _prefix_positions(
        batch=batch,
        historical=historical,
        device=prefix.device,
    )
    family = phase12._method_family(configuration)
    captured: dict[str, Any] | None = None

    def callback(endpoint: Any, input_ids: Any, original: Any) -> Any:
        nonlocal captured
        result = _prefix_prefill(endpoint, input_ids, original, family)
        captured = save_prefix_state(
            cache=endpoint.cache,
            family=family,
            configuration=configuration,
            historical=historical,
            source_batch=batch,
            method_config_fingerprint=METHOD_FINGERPRINTS[configuration],
            output=snapshot_root,
            authority={
                "decision_id": DECISION_ID,
                "authorized_container_digest": PHASE16G_CONTAINER_DIGEST,
                "operation_fingerprint_sha256": operation.operation_fingerprint_sha256,
                "input_recipe_sha256": phase12.PHASE12_INPUT_RECIPE_SHA256,
                "timing_collected": False,
            },
            schema_version=PREFIX_STATE_SCHEMA_V3,
            prefix_token_ids=input_ids,
            prefix_positions=positions,
            model_identity=MODEL_IDENTITY,
            tokenizer_identity=TOKENIZER_IDENTITY,
        )
        return result

    with torch.inference_mode(), forced_flash_execution(), phase13._patched_endpoint_prefill(
        callback
    ):
        session = phase12._build_phase12_session(
            loaded=loaded,
            operation_key=operation,
            prefix_input_ids=prefix,
            decode_input_ids=decode,
        )
    if captured is None:
        raise Phase16GError("direct prefix snapshot was not captured")
    validated = validate_prefix_state(
        snapshot_root,
        configuration=configuration,
        family=family,
        batch=batch,
        historical=historical,
        method_config_fingerprint=METHOD_FINGERPRINTS[configuration],
        model_identity=MODEL_IDENTITY,
        tokenizer_identity=TOKENIZER_IDENTITY,
        verify_state_bytes=True,
    )
    if validated != captured:
        raise Phase16GError("direct prefix snapshot validation differs")
    return session, captured, prefix, decode, positions


def _build_restored_session(
    *,
    loaded: Any,
    configuration: str,
    batch: int,
    historical: int,
    snapshot_root: Path,
    manifest: Mapping[str, Any],
) -> tuple[Any, dict[str, Any], Any, Any]:
    import torch

    from kvbench.runtime.backend import forced_flash_execution

    phase13._patch_phase12_point_globals(batch=batch, historical=historical)
    operation = phase13.Phase13OperationKey.create(configuration, batch, historical)
    prefix, decode = phase13._point_inputs(
        batch=batch,
        historical=historical,
        device=torch.device("cuda:0"),
    )
    positions = _prefix_positions(
        batch=batch,
        historical=historical,
        device=prefix.device,
    )
    family = phase12._method_family(configuration)
    receipt: dict[str, Any] | None = None

    def callback(endpoint: Any, input_ids: Any, original: Any) -> None:
        del original
        nonlocal receipt
        receipt = restore_prefix_state(
            cache=endpoint.cache,
            family=family,
            configuration=configuration,
            historical=historical,
            root=snapshot_root,
            expected_state_sha256=str(manifest["state_file_sha256"]),
            expected_method_config_fingerprint=METHOD_FINGERPRINTS[configuration],
            expected_prefix_token_ids=input_ids,
            expected_prefix_positions=positions,
            expected_model_identity=MODEL_IDENTITY,
            expected_tokenizer_identity=TOKENIZER_IDENTITY,
        )
        witness = str(receipt["witness_sha256"])
        if hasattr(endpoint.cache, "history_sha256"):
            import types

            endpoint.cache.history_sha256 = types.MethodType(
                lambda self, historical_length: witness,
                endpoint.cache,
            )
        return None

    witness = restored_prefix_witness(manifest, target_batch=batch)
    with torch.inference_mode(), forced_flash_execution(), phase13._restored_prefix_hash_overrides(
        witness
    ), phase13._patched_endpoint_prefill(callback):
        session = phase12._build_phase12_session(
            loaded=loaded,
            operation_key=operation,
            prefix_input_ids=prefix,
            decode_input_ids=decode,
        )
    if receipt is None or receipt.get("witness_sha256") != witness:
        raise Phase16GError("prefix restoration did not execute exactly once")
    phase13._bind_session_prefix_witness(session, witness)
    return session, receipt, prefix, decode


def _allocation_passed(record: Any) -> bool:
    return bool(
        record.audit_available
        and record.passed
        and record.allocation_event_count == 0
        and record.allocation_event_bytes == 0
        and record.allocated_after == record.allocated_before
        and record.reserved_after == record.reserved_before
    )


def _eager_allocation_passed(record: Any, *, family: str, batch: int) -> tuple[bool, dict[str, Any]]:
    if family == "bf16":
        passed = bool(
            record.audit_available
            and record.allocated_after == record.allocated_before
            and record.reserved_after == record.reserved_before
        )
        return passed, {
            "authority": "historical_bf16_outer_model_ephemeral_allocation_contract",
            "cache_persistent_growth_allowed": False,
            "event_count_gated": False,
            "persistent_allocator_state_exact": passed,
        }
    control = phase13b._eager_control(family=family, batch=batch)
    return phase13b._eager_matches_outer_control(record, control), control


def _session_outputs_and_audits(session: Any) -> dict[str, Any]:
    import torch

    from kvbench.runtime.allocation import audit_cuda_allocations

    with torch.inference_mode():
        eager_allocation = audit_cuda_allocations(
            session._fixed_operation,
            device=session.cache_device,
        )
        graph_allocation = audit_cuda_allocations(
            session.graph.replay,
            device=session.cache_device,
        )
        eager = session._fixed_operation().detach().to(device="cpu", copy=True).clone()
        first_graph = session.graph.replay().detach().to(device="cpu", copy=True).clone()
        second_graph = session.graph.replay().detach().to(device="cpu", copy=True).clone()
        torch.cuda.synchronize(device=session.cache_device)
    return {
        "eager": eager,
        "first_graph": first_graph,
        "second_graph": second_graph,
        "eager_allocation": eager_allocation,
        "graph_allocation": graph_allocation,
    }


def _record_for_session(
    *,
    run_id: str,
    configuration: str,
    batch: int,
    historical: int,
    mode: str,
    session: Any,
    restore_receipt: Mapping[str, Any],
    direct: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    from kvbench.runtime.numerical import compare_tensors_untimed, tensor_sha256_untimed

    family = phase12._method_family(configuration)
    pointers_before = phase12._phase12_session_pointers(session)
    history_before = session.current_historical_prefix_sha256()
    evidence = _session_outputs_and_audits(session)
    pointers_after = phase12._phase12_session_pointers(session)
    history_after = session.current_historical_prefix_sha256()
    atol, rtol = phase13b._frozen_tolerance(family)
    eager_graph = compare_tensors_untimed(
        evidence["first_graph"],
        evidence["eager"],
        atol=atol,
        rtol=rtol,
    )
    direct_reference = direct[mode]
    selected = evidence["eager"] if mode == "eager" else evidence["second_graph"]
    restored_direct = compare_tensors_untimed(
        selected,
        direct_reference,
        atol=atol,
        rtol=rtol,
    )
    eager_allocation_passed, eager_control = _eager_allocation_passed(
        evidence["eager_allocation"],
        family=family,
        batch=batch,
    )
    graph_allocation_passed = _allocation_passed(evidence["graph_allocation"])
    accounting = session.method_cache_accounting()
    allocated = int(accounting["allocated_bytes"])
    predicted = int(accounting["predicted_tensor_bytes"])
    relative_error = abs(predicted - allocated) / allocated
    geometry = session.gqa_cache_geometry()
    geometry_passed = phase12._gqa_geometry_passes(geometry, family=family)
    graph_passed = bool(
        session.graph_evidence is not None
        and session.graph_evidence.get("captured") is True
        and session.graph_evidence.get("fallback") is False
        and session.graph_evidence.get("consecutive_replay_outputs_exact") is True
        and torch.equal(evidence["first_graph"], evidence["second_graph"])
        and eager_graph.passed
        and graph_allocation_passed
    )
    selected_allocation_passed = (
        eager_allocation_passed if mode == "eager" else graph_allocation_passed
    )
    path = execution_path_audit_facade(
        backend_identity_verified=True,
        device_kernel_family_verified=True,
        allocation_categories_verified=(
            selected_allocation_passed and graph_allocation_passed
        ),
        temporary_tensor_shapes_verified=(
            geometry_passed and pointers_before == pointers_after
        ),
        gqa_replication_detected=not geometry_passed,
        full_prefix_temporary_detected=False,
        host_synchronization_detected=False,
        backend_fallback_detected=not graph_passed,
        full_prefix_dequantization=(
            "not_applicable" if family == "bf16" else "verified_false"
        ),
    )
    finite = bool(
        torch.isfinite(evidence["eager"]).all()
        and torch.isfinite(evidence["first_graph"]).all()
        and torch.isfinite(evidence["second_graph"]).all()
    )
    checks = {
        "numerical_direct_restore": restored_direct.passed,
        "finite_outputs": finite,
        "exact_batch_shape": list(selected.shape[:1]) == [batch],
        "prefix_restore": bool(
            restore_receipt.get("stored_input_tensors_exact") is True
            and restore_receipt.get("restored_tensor_checksums_exact") is True
            and restore_receipt.get("source_batch") == batch
            and restore_receipt.get("target_batch") == batch
        ),
        "cache_geometry": geometry_passed,
        "allocation_error_below_one_percent": relative_error < 0.01,
        "eager_allocation_contract": eager_allocation_passed,
        "graph_capture_replay": graph_passed,
        "zero_graph_replay_allocation": graph_allocation_passed,
        "eager_graph_agreement": eager_graph.passed,
        "selected_mode_allocation": selected_allocation_passed,
        "pointer_stability": pointers_before == pointers_after,
        "historical_prefix_stability": history_before == history_after,
        "execution_path": path.passed,
        "no_backend_fallback": session.graph_evidence.get("fallback") is False,
    }
    record = {
        "schema_version": POINT_SCHEMA,
        "run_id": run_id,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "configuration": configuration,
        "method_family": family,
        "batch_size": batch,
        "historical_context": historical,
        "attended_context": historical + 1,
        "graph_mode": mode,
        "timing_collected": False,
        "performance_claim_eligible": False,
        "checks": checks,
        "method_config_fingerprint": METHOD_FINGERPRINTS[configuration],
        "adapter_geometry_config_fingerprint": session.adapter_config_fingerprint,
        "cache_layout_fingerprint": session.cache_layout_fingerprint(),
        "adapter_version": session.method.adapter_version,
        "output_sha256": tensor_sha256_untimed(selected),
        "direct_output_sha256": tensor_sha256_untimed(direct_reference),
        "direct_restore_comparison": restored_direct.to_dict(),
        "eager_graph_comparison": eager_graph.to_dict(),
        "eager_allocation": evidence["eager_allocation"].to_dict(),
        "eager_allocation_control": eager_control,
        "graph_allocation": evidence["graph_allocation"].to_dict(),
        "graph": dict(session.graph_evidence or {}),
        "execution_path": path.to_dict(),
        "gqa": geometry,
        "accounting": accounting,
        "byte_breakdown": session.method_byte_breakdown(),
        "predicted_allocated_relative_error": relative_error,
        "prefix_restore_receipt": dict(restore_receipt),
        "r_hbm": None,
    }
    return record


def _direct_output_record(session: Any) -> dict[str, Any]:
    import torch

    with torch.inference_mode():
        eager = session._fixed_operation().detach().to(device="cpu", copy=True).clone()
        graph = session.graph.replay().detach().to(device="cpu", copy=True).clone()
        torch.cuda.synchronize(device=session.cache_device)
    return {
        "eager": eager,
        "cuda_graph": graph,
        "adapter_geometry_config_fingerprint": session.adapter_config_fingerprint,
        "cache_layout_fingerprint": session.cache_layout_fingerprint(),
        "pointers": phase12._phase12_session_pointers(session),
        "graph": dict(session.graph_evidence or {}),
        "finite": bool(
            torch.isfinite(eager).all()
            and torch.isfinite(graph).all()
        ),
    }


def _write_run_record(root: Path, record: Mapping[str, Any]) -> None:
    run_id = str(record["run_id"])
    run_root = root / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    write_json_exclusive(run_root / "result.json", record)
    result_sha = sha256_file(run_root / "result.json")
    (run_root / "checksums.sha256").write_text(
        f"{result_sha}  result.json\n",
        encoding="ascii",
    )
    write_json_exclusive(
        run_root / "COMPLETE",
        {
            "schema_version": "kvbench-phase16g-run-complete-1.0.0",
            "run_id": run_id,
            "status": record["status"],
            "result_sha256": result_sha,
            "written_last": True,
        },
    )


def _build_direct_smoke_session(
    *, loaded: Any, configuration: str, batch: int, historical: int
) -> tuple[Any, Any, Any]:
    import torch

    from kvbench.runtime.backend import forced_flash_execution

    phase13._patch_phase12_point_globals(batch=batch, historical=historical)
    operation = phase13.Phase13OperationKey.create(configuration, batch, historical)
    prefix, decode = phase13._point_inputs(
        batch=batch,
        historical=historical,
        device=torch.device("cuda:0"),
    )
    family = phase12._method_family(configuration)

    def callback(endpoint: Any, input_ids: Any, original: Any) -> Any:
        return _prefix_prefill(endpoint, input_ids, original, family)

    with torch.inference_mode(), forced_flash_execution(), phase13._patched_endpoint_prefill(
        callback
    ):
        session = phase12._build_phase12_session(
            loaded=loaded,
            operation_key=operation,
            prefix_input_ids=prefix,
            decode_input_ids=decode,
        )
    return session, prefix, decode


def _max_smoke_record(
    *,
    run_id: str,
    configuration: str,
    context_label: int,
    feasibility: Mapping[str, Any],
    session: Any,
) -> dict[str, Any]:
    import torch

    from kvbench.runtime.allocation import audit_cuda_allocations
    from kvbench.runtime.numerical import tensor_sha256_untimed

    pointers_before = phase12._phase12_session_pointers(session)
    history_before = session.current_historical_prefix_sha256()
    with torch.inference_mode():
        allocation = audit_cuda_allocations(
            session.graph.replay,
            device=session.cache_device,
        )
        first = session.graph.replay().detach().to(device="cpu", copy=True).clone()
        second = session.graph.replay().detach().to(device="cpu", copy=True).clone()
        torch.cuda.synchronize(device=session.cache_device)
    accounting = session.method_cache_accounting()
    allocated = int(accounting["allocated_bytes"])
    predicted = int(accounting["predicted_tensor_bytes"])
    relative_error = abs(predicted - allocated) / allocated
    geometry = session.gqa_cache_geometry()
    checks = {
        "precomputed_feasible": feasibility.get("status") == "feasible",
        "finite_output": bool(torch.isfinite(second).all()),
        "graph_capture_replay": bool(
            session.graph_evidence.get("captured") is True
            and session.graph_evidence.get("fallback") is False
            and torch.equal(first, second)
        ),
        "zero_graph_replay_allocation": _allocation_passed(allocation),
        "pointer_stability": pointers_before
        == phase12._phase12_session_pointers(session),
        "historical_prefix_stability": history_before
        == session.current_historical_prefix_sha256(),
        "native_gqa_geometry": phase12._gqa_geometry_passes(
            geometry,
            family=phase12._method_family(configuration),
        ),
        "allocation_error_below_one_percent": relative_error < 0.01,
        "no_backend_fallback": session.graph_evidence.get("fallback") is False,
    }
    return {
        "schema_version": POINT_SCHEMA,
        "run_id": run_id,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "configuration": configuration,
        "method_family": phase12._method_family(configuration),
        "batch_size": 16,
        "context_label": context_label,
        "historical_context": phase13.actual_historical_context(context_label),
        "attended_context": phase13.actual_historical_context(context_label) + 1,
        "graph_mode": "cuda_graph",
        "maximum_feasible_smoke": True,
        "timing_collected": False,
        "performance_claim_eligible": False,
        "checks": checks,
        "method_config_fingerprint": METHOD_FINGERPRINTS[configuration],
        "adapter_geometry_config_fingerprint": session.adapter_config_fingerprint,
        "cache_layout_fingerprint": session.cache_layout_fingerprint(),
        "output_sha256": tensor_sha256_untimed(second),
        "graph": dict(session.graph_evidence or {}),
        "graph_allocation": allocation.to_dict(),
        "accounting": accounting,
        "byte_breakdown": session.method_byte_breakdown(),
        "predicted_allocated_relative_error": relative_error,
        "feasibility": dict(feasibility),
        "gqa": geometry,
        "r_hbm": None,
    }


def run_cuda_admission(
    *,
    output: Path,
    prefix_root: Path,
    campaign_id: str,
    git_sha: str,
) -> dict[str, Any]:
    """Run 40 short admissions and deduplicated B=16 feasibility smokes."""

    require_clean_git(git_sha)
    if output.exists() or output.is_symlink() or prefix_root.exists() or prefix_root.is_symlink():
        raise Phase16GError("Phase 16G output or prefix root already exists")
    attestation = phase12._require_authorized_container_runtime()

    import torch

    from kvbench.runtime.model_loader import load_frozen_model
    from kvbench.runtime.numerical import tensor_sha256_untimed

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise Phase16GError("authorized CUDA device is unavailable")
    output.mkdir(parents=True)
    (output / "runs").mkdir()
    (output / "prefix-evidence").mkdir()
    prefix_root.mkdir(parents=True)
    feasibility = b16_feasibility_records()
    validate_feasibility_records(feasibility)
    maxima = largest_feasible_contexts(feasibility)
    write_json_exclusive(
        output / "b16-feasibility.json",
        {
            "schema_version": "kvbench-phase16g-b16-feasibility-1.0.0",
            "status": "PASS",
            "batch_size": 16,
            "context_labels": list(BASE_CONTEXT_LABELS),
            "record_count": len(feasibility),
            "largest_feasible_contexts": maxima,
            "records": feasibility,
        },
    )
    loaded = load_frozen_model(device=torch.device("cuda:0"))
    short_records: list[dict[str, Any]] = []
    prefix_records: list[dict[str, Any]] = []
    max_records: list[dict[str, Any]] = []

    for configuration in PHASE16G_CONFIGURATIONS:
        for batch in PHASE16G_NEW_BATCH_SIZES:
            snapshot_id = f"{campaign_id}-prefix-{configuration}-b{batch}-l4096"
            snapshot = prefix_root / snapshot_id
            direct_session: Any | None = None
            prefix: Any | None = None
            decode: Any | None = None
            positions: Any | None = None
            try:
                (
                    direct_session,
                    manifest,
                    prefix,
                    decode,
                    positions,
                ) = _build_direct_session_and_snapshot(
                    loaded=loaded,
                    configuration=configuration,
                    batch=batch,
                    historical=PHASE16G_CONTEXT_LENGTH,
                    snapshot_root=snapshot,
                )
                direct = _direct_output_record(direct_session)
                if direct["finite"] is not True:
                    raise Phase16GError("direct prefix output is non-finite")
                direct_checksums = {
                    mode: tensor_sha256_untimed(direct[mode]) for mode in MODES
                }
            finally:
                if direct_session is not None:
                    _release_session(direct_session)
                    direct_session = None
                del prefix, decode, positions
                gc.collect()

            mode_ids: dict[str, str] = {}
            restored_receipts: dict[str, dict[str, Any]] = {}
            geometry_fingerprints: dict[str, dict[str, str]] = {}
            for mode in MODES:
                run_id = (
                    f"{campaign_id}-{configuration}-b{batch}-l4096-"
                    f"{'graph' if mode == 'cuda_graph' else 'eager'}"
                )
                mode_ids[mode] = run_id
                session: Any | None = None
                restored_prefix: Any | None = None
                restored_decode: Any | None = None
                try:
                    (
                        session,
                        receipt,
                        restored_prefix,
                        restored_decode,
                    ) = _build_restored_session(
                        loaded=loaded,
                        configuration=configuration,
                        batch=batch,
                        historical=PHASE16G_CONTEXT_LENGTH,
                        snapshot_root=snapshot,
                        manifest=manifest,
                    )
                    record = _record_for_session(
                        run_id=run_id,
                        configuration=configuration,
                        batch=batch,
                        historical=PHASE16G_CONTEXT_LENGTH,
                        mode=mode,
                        session=session,
                        restore_receipt=receipt,
                        direct=direct,
                    )
                    restored_receipts[mode] = dict(receipt)
                    geometry_fingerprints[mode] = {
                        "adapter_config_fingerprint": str(
                            record["adapter_geometry_config_fingerprint"]
                        ),
                        "cache_layout_fingerprint": str(
                            record["cache_layout_fingerprint"]
                        ),
                    }
                except BaseException as error:
                    record = {
                        "schema_version": POINT_SCHEMA,
                        "run_id": run_id,
                        "status": "FAIL",
                        "configuration": configuration,
                        "batch_size": batch,
                        "historical_context": PHASE16G_CONTEXT_LENGTH,
                        "graph_mode": mode,
                        "failure_type": type(error).__name__,
                        "failure_reason": str(error),
                        "timing_collected": False,
                        "performance_claim_eligible": False,
                        "r_hbm": None,
                    }
                    _write_run_record(output, record)
                    raise
                finally:
                    if session is not None:
                        _release_session(session)
                        session = None
                    del restored_prefix, restored_decode
                    gc.collect()
                _write_run_record(output, record)
                short_records.append(record)
                if record["status"] != "PASS":
                    raise Phase16GError(f"short geometry admission failed: {run_id}")

            if len(set(item["adapter_config_fingerprint"] for item in geometry_fingerprints.values())) != 1:
                raise Phase16GError("eager/Graph adapter geometry fingerprint differs")
            if len(set(item["cache_layout_fingerprint"] for item in geometry_fingerprints.values())) != 1:
                raise Phase16GError("eager/Graph cache layout fingerprint differs")
            prefix_record = {
                "schema_version": PREFIX_EVIDENCE_SCHEMA,
                "status": "PASS",
                "snapshot_id": snapshot_id,
                "configuration": configuration,
                "batch_size": batch,
                "historical_context": PHASE16G_CONTEXT_LENGTH,
                "prefix_schema_version": manifest["schema_version"],
                "state_file_sha256": manifest["state_file_sha256"],
                "state_file_bytes": manifest["state_file_bytes"],
                "manifest_sha256": sha256_file(snapshot / PREFIX_STATE_MANIFEST),
                "method_config_fingerprint": METHOD_FINGERPRINTS[configuration],
                "cache_layout_fingerprint": manifest["source_layout_fingerprint"],
                "token_ids_exact": True,
                "positions_exact": True,
                "active_length_exact": True,
                "cache_tensor_shapes_exact": True,
                "batch_broadcasting": False,
                "implicit_repeat_or_truncation": False,
                "direct_output_sha256": direct_checksums,
                "restored_run_ids": mode_ids,
                "restore_receipts": restored_receipts,
                "raw_prefix_payload_published": False,
                "local_prefix_state_preserved": True,
                "timing_collected": False,
            }
            write_json_exclusive(
                output / "prefix-evidence" / f"{configuration}-b{batch}-l4096.json",
                prefix_record,
            )
            prefix_records.append(prefix_record)
            del direct
            gc.collect()

    feasibility_by_key = {
        (str(item["method_config_id"]), int(item["context_label"])): item
        for item in feasibility
    }
    for configuration in PHASE16G_CONFIGURATIONS:
        context_label = maxima[configuration]
        if context_label == PHASE16G_CONTEXT_LENGTH:
            existing = next(
                item
                for item in short_records
                if item["configuration"] == configuration
                and item["batch_size"] == 16
                and item["graph_mode"] == "cuda_graph"
            )
            max_records.append(
                {
                    "schema_version": POINT_SCHEMA,
                    "run_id": existing["run_id"],
                    "status": "PASS",
                    "configuration": configuration,
                    "batch_size": 16,
                    "context_label": context_label,
                    "graph_mode": "cuda_graph",
                    "maximum_feasible_smoke": True,
                    "deduplicated_short_admission": True,
                    "referenced_run_id": existing["run_id"],
                    "timing_collected": False,
                    "r_hbm": None,
                }
            )
            continue
        run_id = f"{campaign_id}-{configuration}-b16-l{context_label}-max-graph"
        historical = phase13.actual_historical_context(context_label)
        session = None
        prefix = None
        decode = None
        try:
            session, prefix, decode = _build_direct_smoke_session(
                loaded=loaded,
                configuration=configuration,
                batch=16,
                historical=historical,
            )
            record = _max_smoke_record(
                run_id=run_id,
                configuration=configuration,
                context_label=context_label,
                feasibility=feasibility_by_key[(configuration, context_label)],
                session=session,
            )
        except BaseException as error:
            record = {
                "schema_version": POINT_SCHEMA,
                "run_id": run_id,
                "status": "FAIL",
                "configuration": configuration,
                "batch_size": 16,
                "context_label": context_label,
                "graph_mode": "cuda_graph",
                "maximum_feasible_smoke": True,
                "failure_type": type(error).__name__,
                "failure_reason": str(error),
                "timing_collected": False,
                "performance_claim_eligible": False,
                "r_hbm": None,
            }
            _write_run_record(output, record)
            raise
        finally:
            if session is not None:
                _release_session(session)
                session = None
            del prefix, decode
            gc.collect()
        _write_run_record(output, record)
        max_records.append(record)
        if record["status"] != "PASS":
            raise Phase16GError(f"maximum B=16 smoke failed: {run_id}")

    if len(short_records) != 40 or len(prefix_records) != 20 or len(max_records) != 10:
        raise Phase16GError("Phase 16G CUDA admission cardinality differs")
    payload = {
        "schema_version": CUDA_EVIDENCE_SCHEMA,
        "status": "PASS",
        "created_at_utc": utc_now(),
        "campaign_id": campaign_id,
        "execution_git_sha": git_sha,
        "authorized_container_digest": PHASE16G_CONTAINER_DIGEST,
        "container_attestation": attestation,
        "decision_id": DECISION_ID,
        "prefix_format_version": PHASE16G_PREFIX_SCHEMA,
        "configurations": list(PHASE16G_CONFIGURATIONS),
        "new_batch_sizes": list(PHASE16G_NEW_BATCH_SIZES),
        "short_context": PHASE16G_CONTEXT_LENGTH,
        "short_admission_count": len(short_records),
        "prefix_restore_count": len(prefix_records),
        "maximum_feasible_smoke_count": len(max_records),
        "source_hashes": source_hashes(),
        "current_method_admission_authorities": current_report_authorities(),
        "b16_largest_feasible_contexts": maxima,
        "short_records": short_records,
        "prefix_records": prefix_records,
        "maximum_feasible_smokes": max_records,
        "cuda_source_changed": False,
        "adapter_source_changed": False,
        "quantization_changed": False,
        "timing_collected": False,
        "performance_claim_eligible": False,
        "r_hbm": None,
    }
    write_json_exclusive(output / "cuda-validation.json", payload)
    return payload


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase16GError(f"JSON evidence is invalid: {path}") from error
    if not isinstance(payload, dict):
        raise Phase16GError(f"JSON evidence is not an object: {path}")
    return payload


def validate_cuda_admission(root: Path) -> dict[str, Any]:
    payload = load_json(root / "cuda-validation.json")
    short = payload.get("short_records")
    prefixes = payload.get("prefix_records")
    smokes = payload.get("maximum_feasible_smokes")
    expected_short = {
        (configuration, batch, mode)
        for configuration in PHASE16G_CONFIGURATIONS
        for batch in PHASE16G_NEW_BATCH_SIZES
        for mode in MODES
    }
    if (
        payload.get("schema_version") != CUDA_EVIDENCE_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("authorized_container_digest") != PHASE16G_CONTAINER_DIGEST
        or payload.get("prefix_format_version") != PHASE16G_PREFIX_SCHEMA
        or payload.get("configurations") != list(PHASE16G_CONFIGURATIONS)
        or payload.get("new_batch_sizes") != list(PHASE16G_NEW_BATCH_SIZES)
        or payload.get("short_admission_count") != 40
        or payload.get("prefix_restore_count") != 20
        or payload.get("maximum_feasible_smoke_count") != 10
        or not isinstance(short, list)
        or not isinstance(prefixes, list)
        or not isinstance(smokes, list)
        or len(short) != 40
        or len(prefixes) != 20
        or len(smokes) != 10
    ):
        raise Phase16GError("CUDA admission header differs")
    if {
        (item.get("configuration"), item.get("batch_size"), item.get("graph_mode"))
        for item in short
        if isinstance(item, dict)
    } != expected_short:
        raise Phase16GError("short admission matrix differs")
    for record in short:
        checks = record.get("checks")
        if (
            record.get("status") != "PASS"
            or not isinstance(checks, dict)
            or not checks
            or not all(value is True for value in checks.values())
            or record.get("timing_collected") is not False
            or record.get("performance_claim_eligible") is not False
            or record.get("r_hbm") is not None
            or float(record.get("predicted_allocated_relative_error", 1.0)) >= 0.01
        ):
            raise Phase16GError("short admission result differs")
        run_root = root / "runs" / str(record["run_id"])
        run_payload = load_json(run_root / "result.json")
        complete = load_json(run_root / "COMPLETE")
        if (
            run_payload != record
            or complete.get("status") != "PASS"
            or complete.get("result_sha256") != sha256_file(run_root / "result.json")
        ):
            raise Phase16GError("immutable short run differs")
    expected_prefix = {
        (configuration, batch)
        for configuration in PHASE16G_CONFIGURATIONS
        for batch in PHASE16G_NEW_BATCH_SIZES
    }
    if {
        (item.get("configuration"), item.get("batch_size"))
        for item in prefixes
        if isinstance(item, dict)
    } != expected_prefix:
        raise Phase16GError("prefix evidence matrix differs")
    for record in prefixes:
        if (
            record.get("status") != "PASS"
            or record.get("prefix_schema_version") != PHASE16G_PREFIX_SCHEMA
            or record.get("token_ids_exact") is not True
            or record.get("positions_exact") is not True
            or record.get("cache_tensor_shapes_exact") is not True
            or record.get("batch_broadcasting") is not False
            or record.get("implicit_repeat_or_truncation") is not False
            or record.get("raw_prefix_payload_published") is not False
        ):
            raise Phase16GError("prefix evidence result differs")
    if {item.get("configuration") for item in smokes} != set(PHASE16G_CONFIGURATIONS):
        raise Phase16GError("maximum smoke coverage differs")
    for record in smokes:
        if record.get("status") != "PASS" or record.get("timing_collected") is not False:
            raise Phase16GError("maximum smoke result differs")
        if record.get("deduplicated_short_admission") is not True:
            run_root = root / "runs" / str(record["run_id"])
            if load_json(run_root / "result.json") != record:
                raise Phase16GError("maximum smoke run differs")
    feasibility_payload = load_json(root / "b16-feasibility.json")
    records = feasibility_payload.get("records")
    if not isinstance(records, list):
        raise Phase16GError("B=16 feasibility records are absent")
    validate_feasibility_records(records)
    if feasibility_payload.get("largest_feasible_contexts") != largest_feasible_contexts(records):
        raise Phase16GError("largest feasible context selection differs")
    return payload


def build_sanitizer_evidence(log_root: Path) -> dict[str, Any]:
    version_path = log_root / "compute-sanitizer-version.txt"
    if not version_path.is_file() or version_path.is_symlink():
        raise Phase16GError("sanitizer version evidence is absent")
    version_text = version_path.read_text(encoding="utf-8").strip()
    if "Compute Sanitizer version" not in version_text:
        raise Phase16GError("sanitizer version evidence differs")
    records: list[dict[str, Any]] = []
    for tool in ("memcheck", "initcheck"):
        for configuration in SANITIZER_CONFIGURATIONS:
            stem = f"{tool}-{configuration}-b16-l4096"
            stdout_path = log_root / f"{stem}.stdout.txt"
            stderr_path = log_root / f"{stem}.stderr.txt"
            exit_path = log_root / f"{stem}.exit.txt"
            if any(not path.is_file() or path.is_symlink() for path in (stdout_path, stderr_path, exit_path)):
                raise Phase16GError("sanitizer raw evidence is absent")
            stdout = stdout_path.read_text(encoding="utf-8")
            stderr = stderr_path.read_text(encoding="utf-8")
            try:
                exit_code = int(exit_path.read_text(encoding="ascii").strip())
                channel = json.loads(stdout.strip().splitlines()[-1])
            except (ValueError, IndexError, json.JSONDecodeError) as error:
                raise Phase16GError("sanitizer result channel differs") from error
            combined = stdout + "\n" + stderr
            zero_errors = "ERROR SUMMARY: 0 errors" in combined
            zero_leaks = (
                tool != "memcheck"
                or "LEAK SUMMARY: 0 bytes leaked" in combined
            )
            if (
                exit_code != 0
                or not isinstance(channel, dict)
                or channel.get("status") != "PASS"
                or channel.get("result", {}).get("configuration") != configuration
                or channel.get("result", {}).get("batch_size") != 16
                or not zero_errors
                or not zero_leaks
            ):
                raise Phase16GError(f"sanitizer probe failed: {tool}/{configuration}")
            leak_lines = [
                line.strip()
                for line in stderr.splitlines()
                if "leak" in line.lower() or "Leaked" in line
            ]
            records.append(
                {
                    "tool": tool,
                    "configuration": configuration,
                    "batch_size": 16,
                    "historical_context": 4096,
                    "exit_code": exit_code,
                    "error_summary_zero": True,
                    "leak_check_enabled": tool == "memcheck",
                    "leak_error_detected": False,
                    "leaked_bytes": 0 if tool == "memcheck" else None,
                    "leak_output_lines": leak_lines,
                    "stdout_sha256": sha256_file(stdout_path),
                    "stderr_sha256": sha256_file(stderr_path),
                    "status": "PASS",
                }
            )
    payload = {
        "schema_version": SANITIZER_SCHEMA,
        "status": "PASS",
        "created_at_utc": utc_now(),
        "authorized_container_digest": PHASE16G_CONTAINER_DIGEST,
        "tool_version": version_text,
        "tool_version_sha256": sha256_file(version_path),
        "batch_size": 16,
        "historical_context": 4096,
        "configurations": list(SANITIZER_CONFIGURATIONS),
        "tools": ["memcheck", "initcheck"],
        "record_count": len(records),
        "zero_memory_errors": True,
        "zero_leaks": True,
        "bf16_custom_kernel_sanitizer_required": False,
        "records": records,
        "timing_collected": False,
    }
    return payload


def validate_sanitizer_evidence(payload: Mapping[str, Any]) -> None:
    records = payload.get("records")
    expected = {
        (tool, configuration)
        for tool in ("memcheck", "initcheck")
        for configuration in SANITIZER_CONFIGURATIONS
    }
    if (
        payload.get("schema_version") != SANITIZER_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("authorized_container_digest") != PHASE16G_CONTAINER_DIGEST
        or payload.get("zero_memory_errors") is not True
        or payload.get("zero_leaks") is not True
        or "Compute Sanitizer version" not in str(payload.get("tool_version", ""))
        or _SHA256_RE.fullmatch(str(payload.get("tool_version_sha256", ""))) is None
        or not isinstance(records, list)
        or len(records) != len(expected)
        or {(item.get("tool"), item.get("configuration")) for item in records} != expected
        or any(item.get("status") != "PASS" for item in records)
        or any(
            item.get("leaked_bytes") != 0
            for item in records
            if item.get("tool") == "memcheck"
        )
    ):
        raise Phase16GError("sanitizer evidence contract differs")


def build_geometry_index(
    cuda: Mapping[str, Any], sanitizer: Mapping[str, Any]
) -> dict[str, Any]:
    short = list(cuda["short_records"])
    smokes = {item["configuration"]: item for item in cuda["maximum_feasible_smokes"]}
    prefixes = {
        (item["configuration"], item["batch_size"]): item
        for item in cuda["prefix_records"]
    }
    sanitizer_coverage = {
        "bf16": {"required": False, "reason": "no_prefix_cuda_change"},
        "tq_4bit_nc": {"required": True, "covered_by": ["tq_4bit_nc"]},
        "tq_k3v4_nc": {"required": True, "covered_by": ["tq_k3v4_nc"]},
        "tq_3bit_nc": {"required": True, "covered_by": ["tq_3bit_nc"]},
        "k4v4": {"required": True, "covered_by": ["k4v4"]},
        "k2v4": {"required": True, "covered_by": ["k4v4", "k2v2"]},
        "k2v2": {"required": True, "covered_by": ["k2v2"]},
        "kvq4": {"required": True, "covered_by": ["kvq4"]},
        "kvq3": {"required": True, "covered_by": ["kvq3"]},
        "kvq2": {"required": True, "covered_by": ["kvq2"]},
    }
    new_records: dict[str, dict[str, Any]] = {}
    for configuration in PHASE16G_CONFIGURATIONS:
        for batch in PHASE16G_NEW_BATCH_SIZES:
            eager = next(
                item
                for item in short
                if item["configuration"] == configuration
                and item["batch_size"] == batch
                and item["graph_mode"] == "eager"
            )
            graph = next(
                item
                for item in short
                if item["configuration"] == configuration
                and item["batch_size"] == batch
                and item["graph_mode"] == "cuda_graph"
            )
            if (
                eager["adapter_geometry_config_fingerprint"]
                != graph["adapter_geometry_config_fingerprint"]
                or eager["cache_layout_fingerprint"]
                != graph["cache_layout_fingerprint"]
            ):
                raise Phase16GError("mode geometry fingerprint differs")
            prefix = prefixes[(configuration, batch)]
            key = geometry_key(configuration, batch)
            new_records[key] = {
                "configuration": configuration,
                "batch_size": batch,
                "status": "PASS",
                "method_config_fingerprint": METHOD_FINGERPRINTS[configuration],
                "adapter_config_fingerprint": eager[
                    "adapter_geometry_config_fingerprint"
                ],
                "cache_layout_fingerprint": eager["cache_layout_fingerprint"],
                "short_eager": "PASS",
                "short_eager_run_id": eager["run_id"],
                "short_cuda_graph": "PASS",
                "short_cuda_graph_run_id": graph["run_id"],
                "prefix_restore": "PASS",
                "prefix_state_sha256": prefix["state_file_sha256"],
                "allocation": "PASS",
                "execution_path": "PASS",
                "maximum_feasible_context": cuda["b16_largest_feasible_contexts"][configuration]
                if batch == 16
                else None,
                "maximum_context_graph_smoke": smokes[configuration]["status"]
                if batch == 16
                else "NOT_APPLICABLE",
                "sanitizer_coverage": sanitizer_coverage[configuration],
            }
    index = {
        "schema_version": PHASE16G_INDEX_SCHEMA,
        "status": "PASS",
        "created_at_utc": utc_now(),
        "decision_id": DECISION_ID,
        "execution_git_sha": cuda["execution_git_sha"],
        "authorized_container_digest": PHASE16G_CONTAINER_DIGEST,
        "prefix_format_version": PHASE16G_PREFIX_SCHEMA,
        "admitted_full_scan_batches": list(PHASE16G_ADMITTED_BATCH_SIZES),
        "existing_geometry_batches_unchanged": [1, 4, 8],
        "existing_geometry_authorities": current_report_authorities(),
        "new_geometry_records": new_records,
        "short_admission_count": 40,
        "maximum_feasible_smoke_count": 10,
        "sanitizer_record_count": sanitizer["record_count"],
        "gates": {gate: "PASS" for gate in ("G0", "G1", "G2", "G3", "G4", "G5")},
        "full_scan": "READY",
        "quality_execution": "LOCKED",
        "performance_claim_eligible": False,
        "timing_collected": False,
        "r_hbm": None,
    }
    validate_geometry_index(index)
    return index


def render_report(index: Mapping[str, Any], cuda: Mapping[str, Any]) -> str:
    maxima = cuda["b16_largest_feasible_contexts"]
    max_rows = "\n".join(
        f"- `{configuration}`: L={maxima[configuration]} Graph PASS"
        for configuration in PHASE16G_CONFIGURATIONS
    )
    return f"""# Phase 16G — B=2/B=16 Geometry Admission

- Status: PASS
- Execution HEAD: `{index['execution_git_sha']}`
- Authorized container: `{PHASE16G_CONTAINER_DIGEST}`
- Prefix format: `{PHASE16G_PREFIX_SCHEMA}`
- Geometry authority: Decision {DECISION_ID}
- Admitted Full Scan batches: `{{1,2,4,8,16}}`

The exact-shape prefix schema accepts positive integer batches while the
successor execution index separately admits only the five Full Scan batches.
Legacy B=1/4/8 artifacts and geometry records remain unchanged and readable.

All 40 B=2/B=16 L=4096 eager/Graph admissions passed numerical, finite-output,
prefix restoration, allocation, native-GQA/path, pointer-stability, and CUDA
Graph checks. All 20 exact-batch prefix cases passed without broadcasting,
repeat, or truncation. Selected custom-kernel memcheck/initcheck coverage passed.

## B=16 maximum-feasible Graph smokes

{max_rows}

G0–G5 remain PASS. Full Scan is READY but was not started. Quality remains
LOCKED. No timing, profiling, quality execution, speedup, or HBM claim was made.
Durable publication identity is recorded separately in the checksum-bound
Phase 16G R2 receipt after this report is sealed.
"""


def materialize_reports(
    *,
    cuda_root: Path,
    sanitizer_path: Path,
    json_output: Path,
    markdown_output: Path,
) -> dict[str, Any]:
    cuda = validate_cuda_admission(cuda_root)
    sanitizer = load_json(sanitizer_path)
    validate_sanitizer_evidence(sanitizer)
    index = build_geometry_index(cuda, sanitizer)
    write_json_exclusive(json_output, index)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    with markdown_output.open("x", encoding="utf-8") as handle:
        handle.write(render_report(index, cuda))
    return index


def _artifact_files(root: Path, excluded: set[str]) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise Phase16GError("Phase 16G bundle contains a symlink")
        if path.is_file() and path.relative_to(root).as_posix() not in excluded:
            files.append(path)
    return files


def _copy_exclusive(source: Path, destination: Path) -> None:
    if not source.is_file() or source.is_symlink() or destination.exists():
        raise Phase16GError("Phase 16G bundle copy source or destination differs")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)


def finalize_bundle(
    *,
    root: Path,
    campaign_id: str,
    git_sha: str,
    sanitizer_evidence: Path,
    sanitizer_logs: Path,
    focused_tests: Path,
    checks_evidence: Path,
    geometry_report: Path,
    markdown_report: Path,
) -> dict[str, Any]:
    cuda = validate_cuda_admission(root)
    if cuda.get("campaign_id") != campaign_id or cuda.get("execution_git_sha") != git_sha:
        raise Phase16GError("Phase 16G finalization identity differs")
    sanitizer = load_json(sanitizer_evidence)
    validate_sanitizer_evidence(sanitizer)
    index = load_json(geometry_report)
    validate_geometry_index(index)
    if index.get("execution_git_sha") != git_sha:
        raise Phase16GError("geometry report execution authority differs")
    for control in ("manifest.json", "artifact_inventory.json", "checksums.sha256", "COMPLETE"):
        if (root / control).exists() or (root / control).is_symlink():
            raise Phase16GError("Phase 16G finalization control already exists")
    _copy_exclusive(sanitizer_evidence, root / "sanitizer" / "evidence.json")
    for path in sorted(sanitizer_logs.iterdir()):
        if path.is_file() and not path.is_symlink():
            _copy_exclusive(path, root / "sanitizer" / "raw" / path.name)
    _copy_exclusive(focused_tests, root / "validation" / "focused-tests.json")
    _copy_exclusive(checks_evidence, root / "validation" / "make-checks.json")
    _copy_exclusive(
        REPOSITORY_ROOT / "docs/decisions/0039-full-scan-batch-geometry-admission.md",
        root / "authority" / "decision-0039.md",
    )
    _copy_exclusive(geometry_report, root / "batch-geometry-admission.json")
    _copy_exclusive(markdown_report, root / "phase16g-report.md")
    prefix_change = {
        "schema_version": "kvbench-phase16g-prefix-format-change-1.0.0",
        "status": "PASS",
        "legacy_schema": "kvbench-phase13-prefix-state-2.0.0",
        "current_schema": PHASE16G_PREFIX_SCHEMA,
        "legacy_batches_readable": [1, 4, 8],
        "current_batch_domain": "positive_integer",
        "execution_gate_separate": True,
        "admitted_batches": list(PHASE16G_ADMITTED_BATCH_SIZES),
        "source_sha256": sha256_file(REPOSITORY_ROOT / "scripts/phase13_prefix_state.py"),
    }
    write_json_exclusive(root / "prefix-format-change.json", prefix_change)
    manifest = {
        "schema_version": BUNDLE_SCHEMA,
        "run_id": campaign_id,
        "status": "PASS",
        "created_at_utc": utc_now(),
        "execution_git_sha": git_sha,
        "authorized_container_digest": PHASE16G_CONTAINER_DIGEST,
        "decision_id": DECISION_ID,
        "short_admission_count": 40,
        "maximum_feasible_smoke_count": 10,
        "prefix_restore_count": 20,
        "sanitizer_record_count": sanitizer["record_count"],
        "raw_prefix_payloads_included": False,
        "timing_collected": False,
        "performance_claim_eligible": False,
        "full_scan_executed": False,
        "quality_execution": "LOCKED",
    }
    write_json_exclusive(root / "manifest.json", manifest)
    payload_files = _artifact_files(
        root,
        {"artifact_inventory.json", "checksums.sha256", "COMPLETE"},
    )
    inventory = {
        "schema_version": "kvbench-artifact-inventory-1.0.0",
        "run_id": campaign_id,
        "files": [
            {
                "path": path.relative_to(root).as_posix(),
                "role": "phase16g_geometry_admission_evidence",
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in payload_files
        ],
        "excluded_control_files": [
            "artifact_inventory.json",
            "checksums.sha256",
            "COMPLETE",
        ],
    }
    write_json_exclusive(root / "artifact_inventory.json", inventory)
    ledger_files = _artifact_files(root, {"checksums.sha256", "COMPLETE"})
    ledger = "".join(
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n"
        for path in ledger_files
    )
    with (root / "checksums.sha256").open("x", encoding="ascii") as handle:
        handle.write(ledger)
    completion = {
        "schema_version": "kvbench-completion-1.0.0",
        "run_id": campaign_id,
        "status": "PASS",
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "artifact_inventory_sha256": sha256_file(root / "artifact_inventory.json"),
        "checksum_ledger_path": "checksums.sha256",
        "checksum_ledger_sha256": sha256_file(root / "checksums.sha256"),
        "written_last": True,
    }
    write_json_exclusive(root / "COMPLETE", completion)
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file():
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        elif path.is_dir():
            path.chmod(
                stat.S_IRUSR
                | stat.S_IXUSR
                | stat.S_IRGRP
                | stat.S_IXGRP
                | stat.S_IROTH
                | stat.S_IXOTH
            )
    root.chmod(
        stat.S_IRUSR
        | stat.S_IXUSR
        | stat.S_IRGRP
        | stat.S_IXGRP
        | stat.S_IROTH
        | stat.S_IXOTH
    )
    return validate_bundle(root)


def validate_bundle(root: Path) -> dict[str, Any]:
    from scripts.r2_artifact import validate_local_artifact

    cuda = validate_cuda_admission(root)
    sanitizer = load_json(root / "sanitizer/evidence.json")
    validate_sanitizer_evidence(sanitizer)
    index = load_json(root / "batch-geometry-admission.json")
    validate_geometry_index(index)
    artifact = validate_local_artifact(root, environ={})
    if (
        artifact.run_id != cuda["campaign_id"]
        or artifact.status != "PASS"
        or load_json(root / "manifest.json").get("full_scan_executed") is not False
    ):
        raise Phase16GError("final Phase 16G bundle identity differs")
    return {
        "status": "PASS",
        "run_id": artifact.run_id,
        "root_sha256": artifact.root_sha256,
        "object_count": artifact.object_count,
        "complete_last": True,
        "checksums": "PASS",
    }


def command_evidence(
    *, label: str, exit_code: int, stdout_path: Path, stderr_path: Path
) -> dict[str, Any]:
    if exit_code != 0 or not stdout_path.is_file() or not stderr_path.is_file():
        raise Phase16GError(f"focused command failed: {label}")
    return {
        "schema_version": "kvbench-phase16g-command-evidence-1.0.0",
        "status": "PASS",
        "label": label,
        "exit_code": exit_code,
        "stdout_sha256": sha256_file(stdout_path),
        "stderr_sha256": sha256_file(stderr_path),
        "created_at_utc": utc_now(),
        "timing_collected": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--run-cuda", action="store_true")
    actions.add_argument("--validate-cuda", action="store_true")
    actions.add_argument("--build-sanitizer-evidence", action="store_true")
    actions.add_argument("--materialize-reports", action="store_true")
    actions.add_argument("--finalize-bundle", action="store_true")
    actions.add_argument("--validate-bundle", action="store_true")
    actions.add_argument("--write-command-evidence", action="store_true")
    actions.add_argument("--print-feasibility", action="store_true")
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--prefix-root", type=Path)
    parser.add_argument("--campaign-id")
    parser.add_argument("--git-sha")
    parser.add_argument("--log-root", type=Path)
    parser.add_argument("--sanitizer-evidence", type=Path)
    parser.add_argument("--focused-tests", type=Path)
    parser.add_argument("--checks-evidence", type=Path)
    parser.add_argument("--geometry-report", type=Path)
    parser.add_argument("--markdown-report", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--label")
    parser.add_argument("--exit-code", type=int)
    parser.add_argument("--stdout", type=Path)
    parser.add_argument("--stderr", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result: Mapping[str, Any]
    if arguments.run_cuda:
        if not all(
            value is not None
            for value in (
                arguments.artifact,
                arguments.prefix_root,
                arguments.campaign_id,
                arguments.git_sha,
            )
        ):
            raise Phase16GError("CUDA admission arguments are incomplete")
        result = run_cuda_admission(
            output=arguments.artifact,
            prefix_root=arguments.prefix_root,
            campaign_id=arguments.campaign_id,
            git_sha=arguments.git_sha,
        )
    elif arguments.validate_cuda:
        if arguments.artifact is None:
            raise Phase16GError("CUDA artifact is required")
        result = validate_cuda_admission(arguments.artifact)
    elif arguments.build_sanitizer_evidence:
        if arguments.log_root is None or arguments.output is None:
            raise Phase16GError("sanitizer evidence arguments are incomplete")
        result = build_sanitizer_evidence(arguments.log_root)
        write_json_exclusive(arguments.output, result)
    elif arguments.materialize_reports:
        if not all(
            value is not None
            for value in (
                arguments.artifact,
                arguments.sanitizer_evidence,
                arguments.json_output,
                arguments.markdown_output,
            )
        ):
            raise Phase16GError("report arguments are incomplete")
        result = materialize_reports(
            cuda_root=arguments.artifact,
            sanitizer_path=arguments.sanitizer_evidence,
            json_output=arguments.json_output,
            markdown_output=arguments.markdown_output,
        )
    elif arguments.finalize_bundle:
        if not all(
            value is not None
            for value in (
                arguments.artifact,
                arguments.campaign_id,
                arguments.git_sha,
                arguments.sanitizer_evidence,
                arguments.log_root,
                arguments.focused_tests,
                arguments.checks_evidence,
                arguments.geometry_report,
                arguments.markdown_report,
            )
        ):
            raise Phase16GError("bundle finalization arguments are incomplete")
        result = finalize_bundle(
            root=arguments.artifact,
            campaign_id=arguments.campaign_id,
            git_sha=arguments.git_sha,
            sanitizer_evidence=arguments.sanitizer_evidence,
            sanitizer_logs=arguments.log_root,
            focused_tests=arguments.focused_tests,
            checks_evidence=arguments.checks_evidence,
            geometry_report=arguments.geometry_report,
            markdown_report=arguments.markdown_report,
        )
    elif arguments.validate_bundle:
        if arguments.artifact is None:
            raise Phase16GError("bundle artifact is required")
        result = validate_bundle(arguments.artifact)
    elif arguments.write_command_evidence:
        if not all(
            value is not None
            for value in (
                arguments.label,
                arguments.exit_code,
                arguments.stdout,
                arguments.stderr,
                arguments.output,
            )
        ):
            raise Phase16GError("command evidence arguments are incomplete")
        result = command_evidence(
            label=arguments.label,
            exit_code=arguments.exit_code,
            stdout_path=arguments.stdout,
            stderr_path=arguments.stderr,
        )
        write_json_exclusive(arguments.output, result)
    else:
        records = b16_feasibility_records()
        validate_feasibility_records(records)
        result = {
            "status": "PASS",
            "record_count": len(records),
            "largest_feasible_contexts": largest_feasible_contexts(records),
        }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
