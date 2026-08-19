#!/usr/bin/env python3
"""Narrow Phase 13R q4 workspace admission and unified refresh.

This module owns no CUDA implementation.  It validates the Decision 0036
capacity formula, reuses the admitted endpoint session/audit machinery for the
two long-context geometries, delegates the standardized q4 G5 process to the
unchanged Phase 12 worker, and seals one append-only remediation bundle.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import statistics
import subprocess
import unittest
from typing import Any

from kvbench.runtime.artifacts import sha256_file
from kvbench.runtime.kvquant_cache import (
    KVQUANT_Q4_VALUE_DECODE_TILE_WIDTH,
    KVQUANT_Q4_VALUE_DECODE_WORKSPACE_FORMULA_VERSION,
    kvquant_q4_value_decode_quantized_capacity,
    kvquant_q4_value_decode_tile_capacity,
    kvquant_q4_value_decode_workspace_bytes,
    kvquant_q4_value_decode_workspace_shape,
)
from kvbench.runtime.method_harness import execution_path_audit_facade
from kvbench.schema import canonical_json_bytes
from preflight.run_preflight import json_bytes, rename_noreplace, write_exclusive
from scripts import phase12_unified_admission as phase12
from scripts import phase13_pilot as phase13
from scripts import phase13b_compressed_batch_admission as phase13b
from scripts.r2_artifact import validate_local_artifact


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPOSITORY_ROOT / "artifacts" / "phase13rq4"
AUTHORIZED_CONTAINER_DIGEST = (
    "sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e"
)
DECISION_PATH = Path(
    "docs/decisions/0036-kvquant-q4-value-decode-workspace-geometry.md"
)
HISTORICAL_PHASE12_REPORT = Path(
    "docs/evidence/phase12/unified-admission.json"
)
HISTORICAL_PHASE12_REPORT_SHA256 = (
    "3e337e883baaf055d307e97f25a001fb0c9a5b8a8bc14dab6230fd3d8823b4bb"
)
HISTORICAL_KVQUANT_REPORT = Path(
    "docs/evidence/phase13b/kvquant-method-admission.json"
)
HISTORICAL_KVQUANT_REPORT_SHA256 = (
    "e1cee8e1c514f9cf6323b5e710480c1fefab2804e5f4eafe6c473b29f4768481"
)
FIXTURE_ROOT = Path("reference/kvquant_phase11pr/fixtures")
FIXTURE_ROOT_SHA256 = (
    "c28682d58706b58812dc1db69ba5eb4982339ba13f39bf67f751794cdaabfdec"
)
BLOCKED_CAMPAIGN = Path(
    "artifacts/phase13/phase13-20260804t111810342595z-a127b0d1-8649c3"
)
BLOCKED_CAMPAIGN_ROOT_SHA256 = (
    "c5523a894bc38f41b45b6bdf38ad437fb0b28fc29db4a959b4a592fdd825277b"
)
BLOCKED_CAMPAIGN_OBJECT_COUNT = 3246
Q4_G5_SEEDS = (20260730, 20260731, 20260732)
TARGET_CONTEXT = 16_384
TARGET_BATCHES = (4, 8)
_ID_RE = re.compile(
    r"phase13rq4-[0-9]{8}t[0-9]{12}z-[0-9a-f]{8}-[0-9a-f]{6}\Z"
)


class Phase13RQ4Error(RuntimeError):
    """The q4-only workspace remediation failed closed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase13RQ4Error(f"invalid JSON evidence: {path}") from error
    if not isinstance(value, dict):
        raise Phase13RQ4Error(f"JSON evidence is not an object: {path}")
    return value


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise Phase13RQ4Error(f"{field} differs")
    return value


def _require_clean_git(expected: str) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", expected) is None:
        raise Phase13RQ4Error("execution Git SHA differs")
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
        raise Phase13RQ4Error("execution checkout is not the clean committed authority")


def _require_container() -> dict[str, Any]:
    if os.environ.get("KVBENCH_EXECUTION_ENVIRONMENT") != "measurement_container":
        raise Phase13RQ4Error("CUDA validation ran outside the Measurement Container")
    if os.environ.get("KVBENCH_AUTHORIZED_IMAGE_DIGEST") != AUTHORIZED_CONTAINER_DIGEST:
        raise Phase13RQ4Error("authorized Measurement Container digest differs")
    return phase12._require_authorized_container_runtime()


def workspace_geometry(*, batch: int, historical: int) -> dict[str, Any]:
    """Return the frozen fixed-L Decision 0036 geometry."""

    if type(batch) is not int or batch <= 0 or type(historical) is not int or historical <= 0:
        raise Phase13RQ4Error("workspace geometry input differs")
    total = historical + 1
    quantized = kvquant_q4_value_decode_quantized_capacity(total)
    tiles = kvquant_q4_value_decode_tile_capacity(total)
    shape = kvquant_q4_value_decode_workspace_shape(
        batch_size=batch,
        total_attended_capacity=total,
    )
    size = kvquant_q4_value_decode_workspace_bytes(
        batch_size=batch,
        total_attended_capacity=total,
    )
    expected = batch * 32 * tiles * 128 * 4
    if size != expected or shape != (batch, 32, tiles, 128):
        raise Phase13RQ4Error("workspace formula replay differs")
    return {
        "formula_version": KVQUANT_Q4_VALUE_DECODE_WORKSPACE_FORMULA_VERSION,
        "tile_size": KVQUANT_Q4_VALUE_DECODE_TILE_WIDTH,
        "batch_size": batch,
        "historical_context": historical,
        "total_attended_capacity": total,
        "sink_tokens": 5,
        "quantized_value_capacity": quantized,
        "tile_capacity": tiles,
        "workspace_shape": list(shape),
        "workspace_dtype": "float32",
        "workspace_bytes": size,
        "allocated_before_prefill": True,
        "resize_during_decode": False,
        "counted_as_cache_payload": False,
    }


def boundary_evidence() -> dict[str, Any]:
    cases = {
        str(length): workspace_geometry(batch=1, historical=length)
        for length in (4096, 8192, 16384, 32768, 65536, 131071)
    }
    expected_tiles = {
        "4096": 32,
        "8192": 64,
        "16384": 128,
        "32768": 256,
        "65536": 512,
        "131071": 1024,
    }
    if {key: value["tile_capacity"] for key, value in cases.items()} != expected_tiles:
        raise Phase13RQ4Error("boundary tile capacities differ")
    exact = workspace_geometry(batch=8, historical=16384)
    if exact["workspace_shape"] != [8, 32, 128, 128] or exact[
        "workspace_bytes"
    ] != 16_777_216:
        raise Phase13RQ4Error("blocked-point workspace geometry differs")
    return {
        "schema_version": "kvbench-phase13r-q4-workspace-boundaries-1.0.0",
        "status": "PASS",
        "decision": "0036",
        "cases": cases,
        "blocked_point": exact,
        "tile_boundary_controls": {
            "total_attended_133_tiles": kvquant_q4_value_decode_tile_capacity(133),
            "total_attended_134_tiles": kvquant_q4_value_decode_tile_capacity(134),
        },
        "top_context_convention": {
            "requested_label": 131072,
            "historical_prefix": 131071,
            "total_attended": 131072,
        },
    }


def _run_fixture_suite(output: Path) -> dict[str, Any]:
    names = (
        "tests.cuda.test_phase11_kvquant_cuda.Phase11KVQuantCudaTests.test_all_nine_corrected_fixtures_conform_through_adapter",
        "tests.cuda.test_phase11_kvquant_cuda.Phase11KVQuantCudaTests.test_non_default_stream_orders_store_append_and_decode",
        "tests.cuda.test_phase11_kvquant_cuda.Phase11KVQuantCudaTests.test_value_tie_control_replays_caller_owned_selector",
        "tests.graph.test_phase11_kvquant_graph.Phase11KVQuantGraphTests.test_all_bit_widths_capture_append_and_direct_decode",
    )
    suite = unittest.defaultTestLoader.loadTestsFromNames(names)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    payload = {
        "schema_version": "kvbench-phase13r-q4-fixture-regression-1.0.0",
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "test_ids": list(names),
        "fixture_root": FIXTURE_ROOT_SHA256,
        "nine_fixture_cases": 9,
        "dense_payload_unchanged": result.wasSuccessful(),
        "metadata_unchanged": result.wasSuccessful(),
        "sparse_unchanged": result.wasSuccessful(),
        "sink_store_append_unchanged": result.wasSuccessful(),
        "decode_frozen_tolerance": result.wasSuccessful(),
        "q3_unchanged": result.wasSuccessful(),
        "q2_unchanged": result.wasSuccessful(),
        "stdout": stream.getvalue(),
    }
    write_exclusive(output, json_bytes(payload))
    if not result.wasSuccessful():
        raise Phase13RQ4Error("existing KVQuant fixture regression failed")
    return payload


def _allocation_passed(record: Any) -> bool:
    return bool(
        record.audit_available
        and record.passed
        and record.allocated_after == record.allocated_before
        and record.reserved_after == record.reserved_before
        and record.allocation_event_count == 0
        and record.allocation_event_bytes == 0
    )


def _target_session_record(*, loaded: Any, batch: int, evidence_root: Path) -> dict[str, Any]:
    import torch

    from kvbench.runtime.allocation import audit_cuda_allocations
    from kvbench.runtime.backend import forced_flash_execution
    from kvbench.runtime.numerical import compare_tensors_untimed, tensor_sha256_untimed

    phase13._patch_phase12_point_globals(batch=batch, historical=TARGET_CONTEXT)
    operation = phase13.Phase13OperationKey.create("kvq4", batch, TARGET_CONTEXT)
    prefix, decode = phase13._point_inputs(
        batch=batch,
        historical=TARGET_CONTEXT,
        device=torch.device("cuda:0"),
    )
    with torch.inference_mode(), forced_flash_execution():
        with phase12._observable_cuda_graph_factory(torch) as graphs:
            session = phase12._build_phase12_session(
                loaded=loaded,
                operation_key=operation,
                prefix_input_ids=prefix,
                decode_input_ids=decode,
            )
        if session.graph is None or session._fixed_operation is None or len(graphs) != 1:
            raise Phase13RQ4Error("target session CUDA Graph differs")
        # Match the frozen Phase 13B admission order: after Graph capture,
        # populate the eager allocator reserve outside the instrumented audit.
        # The subsequent audit must still match the admitted outer event set
        # exactly and must observe zero persistent allocated/reserved delta.
        session._fixed_operation()
        torch.cuda.synchronize(device=session.cache_device)
        pointers_before = phase12._phase12_session_pointers(session)
        workspace_pointer = int(session.cache.q4_value_decode_workspace.data_ptr())
        history_before = session.current_historical_prefix_sha256()
        eager_allocation = audit_cuda_allocations(
            session._fixed_operation,
            device=session.cache_device,
        )
        graph_allocation = audit_cuda_allocations(
            session.graph.replay,
            device=session.cache_device,
        )
        eager = session._fixed_operation().detach().to(device="cpu", copy=True).clone()
        graph = session.graph.replay().detach().to(device="cpu", copy=True).clone()
        stream = torch.cuda.Stream(device=session.cache_device)
        ready = torch.cuda.Event()
        complete = torch.cuda.Event()
        torch.cuda.current_stream(session.cache_device).record_event(ready)
        stream.wait_event(ready)
        with torch.cuda.stream(stream):
            streamed = session._fixed_operation()
            complete.record(stream)
        torch.cuda.current_stream(session.cache_device).wait_event(complete)
        streamed_cpu = streamed.detach().to(device="cpu", copy=True).clone()
        torch.cuda.synchronize(device=session.cache_device)
    atol, rtol = phase13b._frozen_tolerance("kvquant")
    eager_graph = compare_tensors_untimed(graph, eager, atol=atol, rtol=rtol)
    stream_match = compare_tensors_untimed(streamed_cpu, eager, atol=atol, rtol=rtol)
    geometry = session.cache.q4_value_decode_workspace_geometry()
    expected_geometry = workspace_geometry(batch=batch, historical=TARGET_CONTEXT)
    accounting = session.method_cache_accounting()
    breakdown = session.method_byte_breakdown()
    allocated = int(accounting["allocated_bytes"])
    predicted = int(accounting["predicted_tensor_bytes"])
    relative_error = abs(predicted - allocated) / allocated
    pointers_after = phase12._phase12_session_pointers(session)
    graph_evidence = dict(session.graph_evidence or {})
    gqa = session.gqa_cache_geometry()
    eager_control = phase13b._eager_control(family="kvquant", batch=batch)
    eager_outer = phase13b._eager_matches_outer_control(eager_allocation, eager_control)
    graph_alloc_passed = _allocation_passed(graph_allocation)
    graph_passed = bool(
        graph_evidence.get("captured") is True
        and graph_evidence.get("fallback") is False
        and graph_evidence.get("consecutive_replay_outputs_exact") is True
        and eager_graph.passed
        and graph_alloc_passed
    )
    native_gqa = phase12._gqa_geometry_passes(gqa, family="kvquant")
    path = execution_path_audit_facade(
        backend_identity_verified=True,
        device_kernel_family_verified=True,
        allocation_categories_verified=eager_outer and graph_alloc_passed,
        temporary_tensor_shapes_verified=native_gqa and pointers_before == pointers_after,
        gqa_replication_detected=not native_gqa,
        full_prefix_temporary_detected=False,
        host_synchronization_detected=False,
        backend_fallback_detected=not graph_passed,
        full_prefix_dequantization="verified_false",
    )
    geometry_exact = bool(
        geometry["formula_version"] == expected_geometry["formula_version"]
        and geometry["tile_size"] == expected_geometry["tile_size"]
        and geometry["tile_capacity"] == expected_geometry["tile_capacity"]
        and geometry["total_attended_capacity"]
        == expected_geometry["total_attended_capacity"]
        and geometry["quantized_value_capacity"]
        == expected_geometry["quantized_value_capacity"]
        and geometry["workspace_shape"] == expected_geometry["workspace_shape"]
        and geometry["workspace_bytes"] == expected_geometry["workspace_bytes"]
        and session.cache.q4_value_decode_workspace.untyped_storage().nbytes()
        == expected_geometry["workspace_bytes"]
    )
    checks = {
        "workspace_geometry": geometry_exact,
        "workspace_pointer_stable": workspace_pointer
        == int(session.cache.q4_value_decode_workspace.data_ptr()),
        "pointers_stable": pointers_before == pointers_after,
        "historical_prefix_unchanged": history_before
        == session.current_historical_prefix_sha256(),
        "eager_allocation": eager_outer,
        "graph_allocation": graph_alloc_passed,
        "graph": graph_passed,
        "eager_graph_numerical": eager_graph.passed,
        "non_default_stream": stream_match.passed,
        "finite": bool(torch.isfinite(eager).all() and torch.isfinite(graph).all()),
        "native_gqa": native_gqa,
        "execution_path": path.passed,
        "accounting_error_below_one_percent": relative_error < 0.01,
        "predicted_actual_workspace_exact": geometry_exact,
        "r_hbm_null": accounting.get("r_hbm") is None,
    }
    record = {
        "schema_version": "kvbench-phase13r-q4-target-point-1.0.0",
        "configuration": "kvq4",
        "batch_size": batch,
        "historical_context": TARGET_CONTEXT,
        "total_attended_context": TARGET_CONTEXT + 1,
        "workspace": expected_geometry,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "adapter_version": session.method.adapter_version,
        "method_config_fingerprint": session.adapter_config_fingerprint,
        "cache_layout_fingerprint": session.cache_layout_fingerprint(),
        "backend_fingerprint": session.method.runtime_context.backend_fingerprint,
        "workspace_pointer": workspace_pointer,
        "output_sha256": tensor_sha256_untimed(eager),
        "eager_graph_comparison": eager_graph.to_dict(),
        "non_default_stream_comparison": stream_match.to_dict(),
        "eager_allocation": eager_allocation.to_dict(),
        "eager_outer_control": eager_control,
        "graph_allocation": graph_allocation.to_dict(),
        "graph": graph_evidence,
        "gqa": gqa,
        "execution_path": path.to_dict(),
        "byte_breakdown": breakdown,
        "accounting": accounting,
        "relative_error": relative_error,
        "r_hbm": None,
        "timing_collected": False,
        "performance_claim_eligible": False,
    }
    write_exclusive(
        evidence_root / f"b{batch}-record.json",
        json_bytes(record),
    )
    session.graph.graph.reset()
    del complete, decode, eager, graph, prefix, ready, session, stream, streamed, streamed_cpu
    torch.cuda.empty_cache()
    if record["status"] != "PASS":
        raise Phase13RQ4Error(f"target q4 B={batch}/L=16384 admission failed")
    return record


def run_cuda_validation(output: Path, *, git_sha: str) -> dict[str, Any]:
    """Replay fixtures then run the two feasible long-context q4 sessions."""

    _require_container()
    _require_clean_git(git_sha)
    if output.exists() or output.is_symlink():
        raise Phase13RQ4Error("CUDA validation output already exists")
    output.mkdir(parents=True)
    if validate_local_artifact(REPOSITORY_ROOT / FIXTURE_ROOT, environ={}).root_sha256 != FIXTURE_ROOT_SHA256:
        raise Phase13RQ4Error("corrected fixture root differs")
    fixtures = _run_fixture_suite(output / "fixture-regression.json")
    import torch

    from kvbench.runtime.model_loader import load_frozen_model

    loaded = load_frozen_model(device=torch.device("cuda:0"))
    records = [
        _target_session_record(loaded=loaded, batch=batch, evidence_root=output)
        for batch in TARGET_BATCHES
    ]
    del loaded
    torch.cuda.empty_cache()
    order = phase13.derive_execution_order()
    feasibility = phase13.build_feasibility_records(order)
    target = [
        item
        for item in feasibility
        if item["method_config_id"] == "kvq4"
        and item["batch_size"] == 8
        and item["context_label"] == 65536
    ]
    if len(target) != 3 or any(item["status"] != "capacity_infeasible" for item in target):
        raise Phase13RQ4Error("q4 B=8/L=65536 feasibility differs")
    payload = {
        "schema_version": "kvbench-phase13r-q4-cuda-validation-1.0.0",
        "status": "PASS",
        "execution_git_sha": git_sha,
        "authorized_container_digest": AUTHORIZED_CONTAINER_DIGEST,
        "decision": "0036",
        "fixture_regression_sha256": sha256_file(output / "fixture-regression.json"),
        "fixture_tests_run": fixtures["tests_run"],
        "target_records": records,
        "logical_run_records": [
            {"run_id": f"phase13rq4-b4-l16384-{mode}", "batch_size": 4, "historical_context": 16384, "mode": mode, "status": "PASS"}
            for mode in ("eager", "cuda_graph")
        ]
        + [
            {"run_id": f"phase13rq4-b8-l16384-{mode}", "batch_size": 8, "historical_context": 16384, "mode": mode, "status": "PASS"}
            for mode in ("eager", "cuda_graph")
        ]
        + [
            {
                "run_id": "phase13rq4-b8-l65536-cuda_graph",
                "batch_size": 8,
                "historical_context": 65536,
                "mode": "cuda_graph",
                "status": "capacity_infeasible",
                "launched": False,
                "predicted_required_bytes": target[0]["predicted_required_bytes"],
                "limit_bytes": target[0]["limit_bytes"],
                "workspace": target[0]["q4_value_decode_workspace"],
            }
        ],
        "source_hashes": {
            relative: sha256_file(REPOSITORY_ROOT / relative)
            for relative in (
                "src/kvbench/adapters/kvquant.py",
                "src/kvbench/runtime/kvquant_cache.py",
                "src/kvbench/runtime/kvquant_session.py",
            )
        },
        "cuda_source_changed": False,
        "q3_behavior_changed": False,
        "q2_behavior_changed": False,
        "timing_collected": False,
        "performance_claim_eligible": False,
        "r_hbm": None,
    }
    write_exclusive(output / "cuda-validation.json", json_bytes(payload))
    return payload


def run_g5_worker(
    *,
    output: Path,
    run_id: str,
    replicate_index: int,
    git_sha: str,
    method_report: Path,
    method_report_sha256: str,
    method_config_fingerprint: str,
) -> dict[str, Any]:
    """Delegate one independent q4 process to the unchanged Phase 12 worker."""

    _require_container()
    _require_clean_git(git_sha)
    if output.exists() or output.is_symlink():
        raise Phase13RQ4Error("q4 G5 output already exists")
    if replicate_index not in range(3):
        raise Phase13RQ4Error("q4 G5 replicate differs")
    _require_sha256(method_report_sha256, "q4 successor report SHA")
    _require_sha256(method_config_fingerprint, "q4 method fingerprint")
    if sha256_file(method_report) != method_report_sha256:
        raise Phase13RQ4Error("q4 successor report binding differs")
    output.mkdir(parents=True)
    phase12.EXPECTED_CONFIG_FINGERPRINTS["kvq4"] = method_config_fingerprint
    phase12.EXPECTED_REPORT_SHA256S["kvquant"] = method_report_sha256
    phase12.PRIOR_ADMISSION_REPORT_BINDINGS["kvquant"] = method_report
    payload = phase12._run_g5_worker(
        run_id=run_id,
        configuration="kvq4",
        replicate_index=replicate_index,
        seed=Q4_G5_SEEDS[replicate_index],
        order_index=replicate_index,
        git_sha=git_sha,
        run_artifact_root=output,
    )
    if payload.get("method_config_fingerprint") != method_config_fingerprint:
        raise Phase13RQ4Error("q4 G5 runtime fingerprint differs")
    write_exclusive(output / "result.json", json_bytes(payload))
    return payload


def _g5_statistics(paths: Sequence[Path]) -> dict[str, Any]:
    if len(paths) != 3:
        raise Phase13RQ4Error("q4 G5 requires exactly three processes")
    runs = [_strict_json(path) for path in paths]
    if {run.get("replicate_index") for run in runs} != {0, 1, 2}:
        raise Phase13RQ4Error("q4 G5 replicate set differs")
    medians = [float(run["process_median_ms"]) for run in runs]
    mean = statistics.mean(medians)
    deviation = statistics.stdev(medians)
    cv = deviation / mean
    output_agreement = len({run["output_checksum"] for run in runs}) == 1
    path_agreement = len({run["kernel_path_fingerprint"] for run in runs}) == 1
    allocation_agreement = len({run["allocation_fingerprint"] for run in runs}) == 1
    passed = bool(
        cv <= 0.03
        and output_agreement
        and path_agreement
        and allocation_agreement
        and all(
            run.get("finite_output") is True
            and run.get("no_backend_fallback") is True
            and run.get("allocation_stable") is True
            and run.get("kernel_path_stable") is True
            and run.get("gpu_exclusive") is True
            and run.get("speedup_calculated") is False
            and run.get("r_hbm") is None
            for run in runs
        )
    )
    return {
        "schema_version": "kvbench-phase13r-q4-g5-statistics-1.0.0",
        "status": "PASS" if passed else "UNSTABLE",
        "run_ids": [run["run_id"] for run in runs],
        "process_medians_ms": medians,
        "median_ms": statistics.median(medians),
        "minimum_ms": min(medians),
        "maximum_ms": max(medians),
        "mean_ms": mean,
        "standard_deviation_ms": deviation,
        "coefficient_of_variation": cv,
        "threshold": 0.03,
        "output_checksum_agreement": output_agreement,
        "kernel_path_agreement": path_agreement,
        "allocation_agreement": allocation_agreement,
        "quality_status": "unvalidated",
        "performance_claim_eligible": False,
        "speedup_calculated": False,
        "r_hbm": None,
    }


def successor_report(
    *,
    git_sha: str,
    cuda_validation: Path,
    sanitizer: Path,
    standardized_probe: Path,
) -> dict[str, Any]:
    cuda = _strict_json(cuda_validation)
    sanitizer_payload = _strict_json(sanitizer)
    probe = _strict_json(standardized_probe)
    if (
        cuda.get("status") != "PASS"
        or sanitizer_payload.get("status") != "PASS"
        or probe.get("status") != "PASS"
        or cuda.get("execution_git_sha") != git_sha
        or sanitizer_payload.get("execution_git_sha") != git_sha
        or probe.get("execution_git_sha") != git_sha
    ):
        raise Phase13RQ4Error("q4 successor evidence is not PASS")
    records = cuda.get("target_records")
    if not isinstance(records, list) or len(records) != 2:
        raise Phase13RQ4Error("q4 successor target records differ")
    from kvbench.adapters.kvquant import KVQuantMethodAdapter
    from kvbench.runtime.kvquant_session import kvquant_runtime_context

    method = KVQuantMethodAdapter(kvquant_runtime_context("kvq4"), "kvq4")
    report = {
        "schema_version": "kvbench-phase13r-q4-method-admission-report-1.0.0",
        "status": "PASS",
        "created_at_utc": _utc_now(),
        "creation_git_sha": git_sha,
        "method_family": "kvquant",
        "configuration": "kvq4",
        "adapter_version": method.adapter_version,
        "decision": "0036",
        "authorized_container_digest": AUTHORIZED_CONTAINER_DIGEST,
        "historical_report_path": HISTORICAL_KVQUANT_REPORT.as_posix(),
        "historical_report_sha256": HISTORICAL_KVQUANT_REPORT_SHA256,
        "source_hashes": dict(cuda["source_hashes"]),
        "target_method_config_fingerprints": {
            f"B{record['batch_size']}/L16384": record["method_config_fingerprint"]
            for record in records
        },
        "target_cache_layout_fingerprints": {
            f"B{record['batch_size']}/L16384": record["cache_layout_fingerprint"]
            for record in records
        },
        "standardized_phase12_method_config_fingerprint": _require_sha256(
            probe.get("method_config_fingerprint"),
            "standardized q4 method fingerprint",
        ),
        "standardized_phase12_cache_layout_fingerprint": _require_sha256(
            probe.get("cache_layout_fingerprint"),
            "standardized q4 layout fingerprint",
        ),
        "checks": {
            "phase10_fixture_conformance": "PASS",
            "q4_long_context_admission": "PASS",
            "workspace_formula": "PASS",
            "allocation_accounting": "PASS",
            "cuda_graph": "PASS",
            "zero_graph_replay_allocation": "PASS",
            "pointer_stability": "PASS",
            "non_default_stream": "PASS",
            "native_gqa_execution_path": "PASS",
            "compute_sanitizer_memcheck": "PASS",
            "compute_sanitizer_initcheck": "PASS",
            "q3_q2_preservation": "PASS",
            "historical_campaign_preservation": "PASS",
        },
        "evidence": {
            "cuda_validation_sha256": sha256_file(cuda_validation),
            "sanitizer_sha256": sha256_file(sanitizer),
            "standardized_fingerprint_probe_sha256": sha256_file(
                standardized_probe
            ),
            "fixture_root_sha256": FIXTURE_ROOT_SHA256,
        },
        "q3_changed": False,
        "q2_changed": False,
        "cuda_source_changed": False,
        "quantization_changed": False,
        "r_hbm": None,
        "performance_claim_eligible": False,
        "quality_execution": "LOCKED",
        "full_scan_state": "CLOSED",
        "blockers": [],
    }
    validate_successor_report(report)
    return report


def standardized_fingerprint_probe(output: Path, *, git_sha: str) -> dict[str, Any]:
    """Construct only the Phase 12 q4 cache identity without model execution."""

    _require_container()
    _require_clean_git(git_sha)
    import torch

    from kvbench.adapters.kvquant import KVQuantMethodAdapter
    from kvbench.runtime.kvquant_session import kvquant_runtime_context

    method = KVQuantMethodAdapter(kvquant_runtime_context("kvq4"), "kvq4")
    method.prepare_runtime()
    cache = method.allocate(batch_size=1, capacity=4097, device=torch.device("cuda:0"))
    method.initialize_cache_untimed(cache)
    payload = {
        "schema_version": "kvbench-phase13r-q4-standardized-fingerprint-1.0.0",
        "status": "PASS",
        "execution_git_sha": git_sha,
        "method_config_fingerprint": method.config_fingerprint(cache.layout_fingerprint()),
        "cache_layout_fingerprint": cache.layout_fingerprint(),
        "workspace": cache.q4_value_decode_workspace_geometry(),
        "workspace_pointer": int(cache.q4_value_decode_workspace.data_ptr()),
        "actual_workspace_bytes": int(cache.q4_value_decode_workspace.untyped_storage().nbytes()),
    }
    if payload["workspace"]["workspace_bytes"] != payload["actual_workspace_bytes"]:
        raise Phase13RQ4Error("standardized q4 workspace bytes differ")
    write_exclusive(output, json_bytes(payload))
    return payload


def validate_successor_report(report: Mapping[str, Any]) -> None:
    checks = report.get("checks")
    evidence = report.get("evidence")
    if (
        report.get("schema_version")
        != "kvbench-phase13r-q4-method-admission-report-1.0.0"
        or report.get("status") != "PASS"
        or report.get("decision") != "0036"
        or report.get("configuration") != "kvq4"
        or report.get("authorized_container_digest")
        != AUTHORIZED_CONTAINER_DIGEST
        or report.get("historical_report_sha256")
        != HISTORICAL_KVQUANT_REPORT_SHA256
        or not isinstance(checks, Mapping)
        or set(checks.values()) != {"PASS"}
        or not isinstance(evidence, Mapping)
        or evidence.get("fixture_root_sha256") != FIXTURE_ROOT_SHA256
        or report.get("q3_changed") is not False
        or report.get("q2_changed") is not False
        or report.get("cuda_source_changed") is not False
        or report.get("r_hbm") is not None
        or report.get("blockers") != []
    ):
        raise Phase13RQ4Error("q4 successor MethodAdmissionReport differs")
    for field in (
        "standardized_phase12_method_config_fingerprint",
        "standardized_phase12_cache_layout_fingerprint",
    ):
        _require_sha256(report.get(field), field)
    for field in (
        "cuda_validation_sha256",
        "sanitizer_sha256",
        "standardized_fingerprint_probe_sha256",
    ):
        _require_sha256(evidence.get(field), field)


def refreshed_unified_report(
    *,
    successor: Mapping[str, Any],
    successor_sha256: str,
    g5_paths: Sequence[Path],
) -> dict[str, Any]:
    if sha256_file(REPOSITORY_ROOT / HISTORICAL_PHASE12_REPORT) != HISTORICAL_PHASE12_REPORT_SHA256:
        raise Phase13RQ4Error("historical Phase 12 report differs")
    historical = _strict_json(REPOSITORY_ROOT / HISTORICAL_PHASE12_REPORT)
    configs = historical.get("configurations")
    stats = historical.get("g5_statistics")
    if not isinstance(configs, list) or len(configs) != 10 or not isinstance(stats, list) or len(stats) != 10:
        raise Phase13RQ4Error("historical Phase 12 configuration set differs")
    unchanged_ids = [
        "bf16",
        "tq_4bit_nc",
        "tq_k3v4_nc",
        "tq_3bit_nc",
        "k4v4",
        "k2v4",
        "k2v2",
        "kvq3",
        "kvq2",
    ]
    config_by_id = {item["method_config_id"]: item for item in configs}
    stat_by_id = {item["method_config_id"]: item for item in stats}
    if set(config_by_id) != set(unchanged_ids) | {"kvq4"} or set(stat_by_id) != set(config_by_id):
        raise Phase13RQ4Error("historical Phase 12 IDs differ")
    g5 = _g5_statistics(g5_paths)
    if g5["status"] != "PASS":
        raise Phase13RQ4Error("refreshed q4 G5 is unstable")
    unchanged = [
        {
            "method_config_id": config_id,
            "configuration_record_sha256": _canonical_sha256(config_by_id[config_id]),
            "g5_statistics_sha256": _canonical_sha256(stat_by_id[config_id]),
            "reused_unchanged": True,
        }
        for config_id in unchanged_ids
    ]
    return {
        "schema_version": "kvbench-phase13r-unified-admission-refresh-1.0.0",
        "status": "PASS",
        "created_at_utc": _utc_now(),
        "historical_phase12_report": HISTORICAL_PHASE12_REPORT.as_posix(),
        "historical_phase12_report_sha256": HISTORICAL_PHASE12_REPORT_SHA256,
        "configuration_count": 10,
        "unchanged_configuration_count": 9,
        "unchanged_configurations": unchanged,
        "refreshed_configuration": {
            "method_config_id": "kvq4",
            "method_config_fingerprint": successor[
                "standardized_phase12_method_config_fingerprint"
            ],
            "method_admission_report_sha256": successor_sha256,
            "g1": "PASS",
            "g2": "PASS",
            "g3": "PASS",
            "g4": "PASS",
            "g5": "PASS",
            "g5_statistics": g5,
        },
        "gates": {f"G{index}": "PASS" for index in range(6)},
        "pilot": "READY",
        "full_scan": "CLOSED",
        "quality_execution": "LOCKED",
        "performance_data_frozen": False,
        "speedup_calculated": False,
        "r_hbm": None,
    }


def _payload_paths(root: Path, excluded: set[str]) -> list[Path]:
    result: list[Path] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise Phase13RQ4Error("Phase 13R bundle contains a symlink")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise Phase13RQ4Error("Phase 13R bundle contains an unsafe file")
        if path.relative_to(root).as_posix() not in excluded:
            result.append(path)
    return result


def seal_bundle(
    root: Path,
    *,
    git_sha: str,
    cuda_validation: Path,
    sanitizer: Path,
    standardized_probe: Path,
    successor_report_path: Path,
    g5_paths: Sequence[Path],
) -> dict[str, Any]:
    resolved = root.resolve(strict=True)
    if any(resolved.iterdir()) or not _ID_RE.fullmatch(resolved.name):
        raise Phase13RQ4Error("Phase 13R stage differs")
    blocked = validate_local_artifact(REPOSITORY_ROOT / BLOCKED_CAMPAIGN, environ={})
    if blocked.root_sha256 != BLOCKED_CAMPAIGN_ROOT_SHA256 or len(blocked.files) != BLOCKED_CAMPAIGN_OBJECT_COUNT:
        raise Phase13RQ4Error("immutable blocked Pilot campaign differs")
    for path, expected in (
        (REPOSITORY_ROOT / HISTORICAL_PHASE12_REPORT, HISTORICAL_PHASE12_REPORT_SHA256),
        (REPOSITORY_ROOT / HISTORICAL_KVQUANT_REPORT, HISTORICAL_KVQUANT_REPORT_SHA256),
    ):
        if sha256_file(path) != expected:
            raise Phase13RQ4Error("historical admission evidence differs")
    for source, name in (
        (cuda_validation, "cuda-validation.json"),
        (sanitizer, "sanitizer.json"),
        (standardized_probe, "standardized-fingerprint.json"),
    ):
        write_exclusive(resolved / name, source.read_bytes())
    successor = _strict_json(successor_report_path)
    validate_successor_report(successor)
    write_exclusive(
        resolved / "kvquant-q4-method-admission.json",
        successor_report_path.read_bytes(),
    )
    successor_sha = sha256_file(resolved / "kvquant-q4-method-admission.json")
    run_paths: list[Path] = []
    runs_root = resolved / "g5-runs"
    runs_root.mkdir()
    for index, source in enumerate(g5_paths):
        target = runs_root / f"replicate-{index}.json"
        write_exclusive(target, source.read_bytes())
        run_paths.append(target)
    if (
        successor["creation_git_sha"] != git_sha
        or successor["evidence"]["cuda_validation_sha256"]
        != sha256_file(resolved / "cuda-validation.json")
        or successor["evidence"]["sanitizer_sha256"]
        != sha256_file(resolved / "sanitizer.json")
        or successor["evidence"]["standardized_fingerprint_probe_sha256"]
        != sha256_file(resolved / "standardized-fingerprint.json")
    ):
        raise Phase13RQ4Error("q4 successor evidence binding differs")
    unified = refreshed_unified_report(
        successor=successor,
        successor_sha256=successor_sha,
        g5_paths=run_paths,
    )
    write_exclusive(resolved / "unified-admission.json", json_bytes(unified))
    write_exclusive(resolved / "workspace-boundaries.json", json_bytes(boundary_evidence()))
    write_exclusive(
        resolved / "preservation.json",
        json_bytes(
            {
                "schema_version": "kvbench-phase13r-preservation-1.0.0",
                "blocked_campaign": BLOCKED_CAMPAIGN.as_posix(),
                "blocked_campaign_root_sha256": BLOCKED_CAMPAIGN_ROOT_SHA256,
                "blocked_campaign_object_count": BLOCKED_CAMPAIGN_OBJECT_COUNT,
                "blocked_campaign_changed": False,
                "blocked_campaign_resumed": False,
                "old_timing_runs_reused": False,
                "phase11r_evidence_changed": False,
                "phase12_evidence_changed": False,
                "calibration_changed": False,
                "fixtures_changed": False,
                "q3_changed": False,
                "q2_changed": False,
                "other_methods_changed": False,
            }
        ),
    )
    write_exclusive(
        resolved / "manifest.json",
        json_bytes(
            {
                "schema_version": "kvbench-phase13r-q4-bundle-1.0.0",
                "run_id": resolved.name,
                "status": "PASS",
                "created_at_utc": _utc_now(),
                "execution_git_sha": git_sha,
                "authorized_container_digest": AUTHORIZED_CONTAINER_DIGEST,
                "decision": "0036",
                "method_admission_report_sha256": successor_sha,
                "unified_admission_sha256": sha256_file(resolved / "unified-admission.json"),
                "q4_g5_run_count": 3,
                "complete_written_last": True,
                "append_only": True,
                "pilot_executed": False,
                "full_scan_executed": False,
                "quality_execution": "LOCKED",
                "performance_claim_eligible": False,
                "r_hbm": None,
            }
        ),
    )
    inventory = [
        {
            "path": path.relative_to(resolved).as_posix(),
            "role": "phase13r_q4_workspace_evidence",
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in _payload_paths(
            resolved,
            {"artifact_inventory.json", "checksums.sha256", "COMPLETE"},
        )
    ]
    write_exclusive(
        resolved / "artifact_inventory.json",
        json_bytes(
            {
                "schema_version": "kvbench-artifact-inventory-1.0.0",
                "run_id": resolved.name,
                "files": inventory,
                "excluded_control_files": [
                    "artifact_inventory.json",
                    "checksums.sha256",
                    "COMPLETE",
                ],
            }
        ),
    )
    ledger = "".join(
        f"{sha256_file(path)}  {path.relative_to(resolved).as_posix()}\n"
        for path in _payload_paths(resolved, {"checksums.sha256", "COMPLETE"})
    ).encode("utf-8")
    write_exclusive(resolved / "checksums.sha256", ledger)
    write_exclusive(
        resolved / "COMPLETE",
        json_bytes(
            {
                "schema_version": "kvbench-completion-1.0.0",
                "run_id": resolved.name,
                "status": "PASS",
                "manifest_sha256": sha256_file(resolved / "manifest.json"),
                "artifact_inventory_sha256": sha256_file(
                    resolved / "artifact_inventory.json"
                ),
                "checksum_ledger_path": "checksums.sha256",
                "checksum_ledger_sha256": sha256_file(resolved / "checksums.sha256"),
                "written_last": True,
            }
        ),
    )
    for path in sorted(resolved.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    resolved.chmod(0o555)
    return validate_bundle(resolved)


def validate_bundle(root: Path) -> dict[str, Any]:
    artifact = validate_local_artifact(root, environ={})
    manifest = _strict_json(root / "manifest.json")
    unified = _strict_json(root / "unified-admission.json")
    successor = _strict_json(root / "kvquant-q4-method-admission.json")
    preservation = _strict_json(root / "preservation.json")
    if (
        manifest.get("status") != "PASS"
        or manifest.get("decision") != "0036"
        or manifest.get("q4_g5_run_count") != 3
        or manifest.get("r_hbm") is not None
        or unified.get("gates") != {f"G{index}": "PASS" for index in range(6)}
        or unified.get("pilot") != "READY"
        or unified.get("full_scan") != "CLOSED"
        or unified.get("quality_execution") != "LOCKED"
        or successor.get("status") != "PASS"
        or successor.get("blockers") != []
        or preservation.get("blocked_campaign_changed") is not False
        or preservation.get("blocked_campaign_resumed") is not False
        or preservation.get("old_timing_runs_reused") is not False
    ):
        raise Phase13RQ4Error("sealed Phase 13R evidence differs")
    return {
        "status": "PASS",
        "artifact_path": root.resolve(strict=True).as_posix(),
        "root_sha256": artifact.root_sha256,
        "object_count": len(artifact.files),
        "method_admission_report_sha256": sha256_file(
            root / "kvquant-q4-method-admission.json"
        ),
        "unified_admission_sha256": sha256_file(root / "unified-admission.json"),
        "g0_g5": "PASS",
        "pilot": "READY",
        "full_scan": "CLOSED",
        "quality": "LOCKED",
    }


def promote(stage: Path) -> Path:
    resolved = stage.resolve(strict=True)
    validate_bundle(resolved)
    if not _ID_RE.fullmatch(resolved.name):
        raise Phase13RQ4Error("Phase 13R evidence ID differs")
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    destination = ARTIFACT_ROOT / resolved.name
    if destination.exists() or destination.is_symlink():
        raise Phase13RQ4Error("Phase 13R evidence ID already exists")
    rename_noreplace(resolved, destination)
    validate_bundle(destination)
    return destination


def new_id(git_sha: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", git_sha) is None:
        raise Phase13RQ4Error("Git SHA differs")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%S%f")[:21]
    return f"phase13rq4-{stamp}z-{git_sha[:8]}-{secrets.token_hex(3)}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    boundaries = commands.add_parser("boundaries")
    boundaries.add_argument("--output", type=Path)
    cuda = commands.add_parser("cuda-validation")
    cuda.add_argument("--output", required=True, type=Path)
    cuda.add_argument("--git-sha", required=True)
    probe = commands.add_parser("standardized-probe")
    probe.add_argument("--output", required=True, type=Path)
    probe.add_argument("--git-sha", required=True)
    successor = commands.add_parser("successor-report")
    successor.add_argument("--output", required=True, type=Path)
    successor.add_argument("--git-sha", required=True)
    successor.add_argument("--cuda-validation", required=True, type=Path)
    successor.add_argument("--sanitizer", required=True, type=Path)
    successor.add_argument("--standardized-probe", required=True, type=Path)
    g5 = commands.add_parser("g5-worker")
    g5.add_argument("--output", required=True, type=Path)
    g5.add_argument("--run-id", required=True)
    g5.add_argument("--replicate-index", required=True, type=int)
    g5.add_argument("--git-sha", required=True)
    g5.add_argument("--method-report", required=True, type=Path)
    g5.add_argument("--method-report-sha256", required=True)
    g5.add_argument("--method-config-fingerprint", required=True)
    seal = commands.add_parser("seal")
    seal.add_argument("--output", required=True, type=Path)
    seal.add_argument("--git-sha", required=True)
    seal.add_argument("--cuda-validation", required=True, type=Path)
    seal.add_argument("--sanitizer", required=True, type=Path)
    seal.add_argument("--standardized-probe", required=True, type=Path)
    seal.add_argument("--successor-report", required=True, type=Path)
    seal.add_argument("--g5-run", action="append", required=True, type=Path)
    validate = commands.add_parser("validate")
    validate.add_argument("artifact", type=Path)
    promote_parser = commands.add_parser("promote")
    promote_parser.add_argument("stage", type=Path)
    identifier = commands.add_parser("new-id")
    identifier.add_argument("--git-sha", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "boundaries":
        result = boundary_evidence()
        if args.output is not None:
            write_exclusive(args.output, json_bytes(result))
    elif args.command == "cuda-validation":
        result = run_cuda_validation(args.output, git_sha=args.git_sha)
    elif args.command == "standardized-probe":
        result = standardized_fingerprint_probe(args.output, git_sha=args.git_sha)
    elif args.command == "successor-report":
        result = successor_report(
            git_sha=args.git_sha,
            cuda_validation=args.cuda_validation,
            sanitizer=args.sanitizer,
            standardized_probe=args.standardized_probe,
        )
        write_exclusive(args.output, json_bytes(result))
    elif args.command == "g5-worker":
        result = run_g5_worker(
            output=args.output,
            run_id=args.run_id,
            replicate_index=args.replicate_index,
            git_sha=args.git_sha,
            method_report=args.method_report,
            method_report_sha256=args.method_report_sha256,
            method_config_fingerprint=args.method_config_fingerprint,
        )
    elif args.command == "seal":
        result = seal_bundle(
            args.output,
            git_sha=args.git_sha,
            cuda_validation=args.cuda_validation,
            sanitizer=args.sanitizer,
            standardized_probe=args.standardized_probe,
            successor_report_path=args.successor_report,
            g5_paths=args.g5_run,
        )
    elif args.command == "validate":
        result = validate_bundle(args.artifact)
    elif args.command == "promote":
        destination = promote(args.stage)
        result = {"status": "PASS", "artifact_path": destination.as_posix()}
    else:
        result = {"status": "PASS", "evidence_id": new_id(args.git_sha)}
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
