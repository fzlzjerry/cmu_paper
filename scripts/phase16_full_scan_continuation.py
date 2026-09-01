"""Append-only continuation for interrupted Phase 16 replicate segments.

This control-plane module never replaces the frozen Phase 16 worker.  It
preserves completed records, gives infrastructure/incomplete attempts a new
run ID, and resumes the original execution order with the original timing
execution SHA.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any

from kvbench.runtime.artifacts import sha256_file
from preflight.run_preflight import json_bytes, write_exclusive
from scripts import phase16_full_scan as phase16
from scripts.r2_artifact import validate_local_artifact


CONTROLLER_SCHEMA = "kvbench-phase16-segment-continuation-1.0.0"
ATTEMPT_SCHEMA = "kvbench-phase16-continuation-attempt-1.0.0"
REPLACEMENT_SCHEMA = "kvbench-phase16-infrastructure-replacement-1.0.0"
REPLACEMENT_SUFFIX = "-infra-replacement-"
REPLACEMENT_ELIGIBLE = frozenset(
    {"infrastructure_failed", "postflight_snapshot_unavailable"}
)
_SHA_RE = re.compile(r"[0-9a-f]{40}\Z")


class Phase16ContinuationError(RuntimeError):
    """The append-only segment-continuation contract failed closed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _git_blob(commit: str, relative: str) -> bytes:
    result = subprocess.run(
        ("/usr/bin/git", "show", f"{commit}:{relative}"),
        cwd=phase16.REPOSITORY_ROOT,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise Phase16ContinuationError(f"Git blob is unavailable: {relative}")
    return result.stdout


def timing_critical_equivalence(
    *, timing_git_sha: str, controller_git_sha: str
) -> dict[str, Any]:
    if _SHA_RE.fullmatch(timing_git_sha) is None or _SHA_RE.fullmatch(
        controller_git_sha
    ) is None:
        raise Phase16ContinuationError("execution SHA is invalid")
    files = []
    for relative in phase16.TIMING_CRITICAL_PATHS:
        before = _git_blob(timing_git_sha, relative)
        after = _git_blob(controller_git_sha, relative)
        files.append(
            {
                "path": relative,
                "timing_sha256": hashlib.sha256(before).hexdigest(),
                "controller_sha256": hashlib.sha256(after).hexdigest(),
                "unchanged": before == after,
            }
        )
    payload = {
        "schema_version": "kvbench-phase16-timing-equivalence-1.0.0",
        "timing_execution_git_sha": timing_git_sha,
        "controller_execution_git_sha": controller_git_sha,
        "authorized_container_digest": phase16.PHASE16G_CONTAINER_DIGEST,
        "files": files,
        "files_sha256": _canonical_sha256(files),
        "timing_critical_changed": any(not row["unchanged"] for row in files),
    }
    if payload["timing_critical_changed"]:
        raise Phase16ContinuationError("timing-critical code changed")
    return payload


def _segment_paths(
    family_root: Path, replicate: int
) -> tuple[str, Path, dict[str, Any]]:
    if replicate not in range(phase16.REPLICATES):
        raise Phase16ContinuationError("replicate index differs")
    family_id = family_root.name
    if phase16._FAMILY_RE.fullmatch(family_id) is None:
        raise Phase16ContinuationError("family ID differs")
    order = phase16._strict_json(
        family_root / "execution_orders" / f"replicate-{replicate}.json"
    )
    committed = phase16._strict_json(phase16.ORDER_PATH)["segments"][replicate]
    if order != committed or order.get("seed") != phase16.SEEDS[replicate]:
        raise Phase16ContinuationError("frozen execution order differs")
    return (
        phase16._segment_id(family_id, replicate),
        family_root / "segments" / f"replicate-{replicate}",
        order,
    )


def _initialize_segment(
    *,
    family_root: Path,
    replicate: int,
    timing_git_sha: str,
    segment_id: str,
    segment_root: Path,
    order: Mapping[str, Any],
) -> None:
    if any(segment_root.iterdir()):
        return
    reservation = phase16._strict_json(family_root / "family-reservation.json")
    logical_catalog = phase16.load_logical_prefix_catalog(
        family_root / "logical-prefixes", validate_artifacts=True
    )
    if (
        reservation.get("execution_git_sha") != timing_git_sha
        or reservation.get("timing_critical_hashes")
        != phase16.timing_critical_hashes()
        or reservation.get("logical_prefix_artifact_count")
        != len(logical_catalog)
        or reservation.get("materialized_cache_snapshots_created") != 0
    ):
        raise Phase16ContinuationError("family timing authority differs")
    (segment_root / "raw").mkdir()
    write_exclusive(segment_root / "execution_order.json", json_bytes(dict(order)))
    authority = phase16.load_phase16g_authority()
    write_exclusive(
        segment_root / "segment_manifest.json",
        json_bytes(
            {
                "schema_version": phase16.SEGMENT_SCHEMA,
                "family_id": family_root.name,
                "segment_id": segment_id,
                "replicate_index": replicate,
                "seed": phase16.SEEDS[replicate],
                "execution_git_sha": timing_git_sha,
                "authorized_container_digest": phase16.PHASE16G_CONTAINER_DIGEST,
                "decision": "0040",
                "geometry_decision": "0039",
                "logical_prefix_schema": phase16.logical_prefix.LOGICAL_PREFIX_SCHEMA,
                "logical_prefix_artifact_count": (
                    phase16.EXPECTED_LOGICAL_PREFIX_ARTIFACTS
                ),
                "materialized_cache_snapshots_created": 0,
                "cache_reconstruction_outside_timing": True,
                "phase16g_authority": authority,
                "logical_records": phase16.LOGICAL_POINTS,
                "append_only": True,
                "run_kind": "timing",
                "quality_status": "unvalidated",
                "performance_claim_eligible": False,
            }
        ),
    )


def _validate_initialized_segment(
    *,
    family_root: Path,
    replicate: int,
    timing_git_sha: str,
    segment_id: str,
    segment_root: Path,
    order: Mapping[str, Any],
) -> None:
    manifest = phase16._strict_json(segment_root / "segment_manifest.json")
    stored_order = phase16._strict_json(segment_root / "execution_order.json")
    reservation = phase16._strict_json(family_root / "family-reservation.json")
    if (
        manifest.get("family_id") != family_root.name
        or manifest.get("segment_id") != segment_id
        or manifest.get("replicate_index") != replicate
        or manifest.get("execution_git_sha") != timing_git_sha
        or manifest.get("authorized_container_digest")
        != phase16.PHASE16G_CONTAINER_DIGEST
        or manifest.get("logical_records") != phase16.LOGICAL_POINTS
        or stored_order != order
        or reservation.get("execution_git_sha") != timing_git_sha
        or reservation.get("timing_critical_hashes")
        != phase16.timing_critical_hashes()
    ):
        raise Phase16ContinuationError("initialized segment authority differs")


def _replacement_roots(raw_root: Path, logical_id: str) -> list[Path]:
    roots = []
    for root in raw_root.glob(f"{logical_id}{REPLACEMENT_SUFFIX}*"):
        suffix = root.name.removeprefix(f"{logical_id}{REPLACEMENT_SUFFIX}")
        if not root.is_dir() or not suffix.isdigit():
            raise Phase16ContinuationError("replacement run path differs")
        roots.append(root)
    return sorted(roots, key=lambda root: int(root.name.rsplit("-", 1)[1]))


def _snapshot_states(run_root: Path) -> list[str]:
    states = []
    for path in run_root.glob("**/*.classification.json"):
        payload = phase16._strict_json(path)
        state = payload.get("state")
        if state not in {"clean", "foreign_process_detected", "query_failed"}:
            raise Phase16ContinuationError("GPU snapshot classification differs")
        states.append(str(state))
    return states


def _is_telemetry_finalization_failure(run_root: Path) -> bool:
    manifest_path = run_root / "manifest.json"
    failure_path = run_root / "failure.json"
    stderr_path = run_root / "worker.stderr.txt"
    if not (
        manifest_path.is_file()
        and failure_path.is_file()
        and stderr_path.is_file()
    ):
        return False
    manifest = phase16._strict_json(manifest_path)
    failure = phase16._strict_json(failure_path)
    try:
        stderr = stderr_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return False
    stages = {path.name for path in (run_root / "stage-progress").glob("*.json")}
    postflight = [
        phase16._strict_json(path).get("state")
        for path in sorted(
            (run_root / "gpu-snapshots" / "postflight").glob(
                "*.classification.json"
            )
        )
    ]
    return bool(
        manifest.get("status") == "runtime_failed"
        and manifest.get("reason") == "supervised_worker_failed"
        and failure.get("returncode") == 1
        and "10-measurement-completed.json" in stages
        and "11-finalization-started.json" in stages
        and "12-finalization-completed.json" not in stages
        and "Phase12UnifiedAdmissionError: telemetry " in stderr
        and " is unavailable" in stderr
        and len(postflight) == 3
        and all(state == "query_failed" for state in postflight)
        and not (run_root / "result.json").exists()
    )


def _attempt_status(run_root: Path) -> str:
    manifest_path = run_root / "manifest.json"
    if not manifest_path.is_file():
        states = _snapshot_states(run_root)
        if "foreign_process_detected" in states:
            raise Phase16ContinuationError(
                "incomplete attempt contains a foreign GPU process"
            )
        return "incomplete"
    manifest = phase16._strict_json(manifest_path)
    status = manifest.get("status")
    if not isinstance(status, str):
        raise Phase16ContinuationError("run status differs")
    return status


def _replacement_eligible(run_root: Path) -> bool:
    status = _attempt_status(run_root)
    return bool(
        status == "incomplete"
        or status in REPLACEMENT_ELIGIBLE
        or _is_telemetry_finalization_failure(run_root)
    )


def _record_matches(
    *, manifest: Mapping[str, Any], record: Mapping[str, Any], logical_id: str
) -> None:
    expected = {
        "logical_record_id": logical_id,
        "method_config_id": record["method_config_id"],
        "method_config_fingerprint": record["method_config_fingerprint"],
        "batch_size": record["batch_size"],
        "context_label": record["context_label"],
        "historical_context": record["historical_context"],
        "total_attended_context": record["total_attended_context"],
        "replicate_index": record["replicate_index"],
        "seed": record["seed"],
        "order_index": record["order_index"],
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise Phase16ContinuationError("run identity differs from frozen order")


def _original_manifest_matches(
    *, manifest: Mapping[str, Any], record: Mapping[str, Any], logical_id: str
) -> None:
    _record_matches(manifest=manifest, record=record, logical_id=logical_id)
    if (
        manifest.get("run_id") != logical_id
        or manifest.get("replacement_of") is not None
    ):
        raise Phase16ContinuationError("original run identity differs")


def _replacement_manifest_matches(
    *, manifest: Mapping[str, Any], record: Mapping[str, Any], logical_id: str
) -> None:
    _record_matches(manifest=manifest, record=record, logical_id=logical_id)
    if (
        manifest.get("run_id") == logical_id
        or manifest.get("replacement_of") != logical_id
        or manifest.get("selective_rerun") is not False
    ):
        raise Phase16ContinuationError("replacement linkage differs")


def _effective_and_pending(
    *, segment_root: Path, order: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    raw_root = segment_root / "raw"
    segment_id = str(
        phase16._strict_json(segment_root / "segment_manifest.json")["segment_id"]
    )
    effective: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    replaced: list[dict[str, Any]] = []
    for raw_record in order["records"]:
        record = dict(raw_record)
        logical_id = phase16._logical_run_id(segment_id, record)
        original_root = raw_root / logical_id
        replacements = _replacement_roots(raw_root, logical_id)
        if original_root.exists():
            original_status = _attempt_status(original_root)
            if (original_root / "manifest.json").is_file():
                _original_manifest_matches(
                    manifest=phase16._strict_json(original_root / "manifest.json"),
                    record=record,
                    logical_id=logical_id,
                )
            original_eligible = _replacement_eligible(original_root)
        else:
            original_status = "absent"
            original_eligible = False
        replacement_manifests = []
        for root in replacements:
            if (root / "manifest.json").is_file():
                manifest = phase16._strict_json(root / "manifest.json")
                _replacement_manifest_matches(
                    manifest=manifest, record=record, logical_id=logical_id
                )
                replacement_manifests.append(manifest)
        if original_status in {"completed", "capacity_infeasible"}:
            if replacements:
                raise Phase16ContinuationError("valid original run was rerun")
            effective.append(phase16._strict_json(original_root / "manifest.json"))
            continue
        if original_root.exists() and not original_eligible:
            if replacements:
                raise Phase16ContinuationError("non-infrastructure failure was rerun")
            effective.append(phase16._strict_json(original_root / "manifest.json"))
            continue
        completed_replacements = [
            manifest
            for manifest in replacement_manifests
            if manifest.get("status") == "completed"
        ]
        if len(completed_replacements) > 1:
            raise Phase16ContinuationError("multiple completed replacements exist")
        if completed_replacements:
            if replacement_manifests[-1] is not completed_replacements[0]:
                raise Phase16ContinuationError("completed replacement is not final")
            effective.append(completed_replacements[0])
            replaced.append(
                {
                    "logical_record_id": logical_id,
                    "original_status": original_status,
                    "replacement_run_id": completed_replacements[0]["run_id"],
                    "replacement_status": "completed",
                }
            )
            continue
        if replacements:
            latest = replacements[-1]
            latest_status = _attempt_status(latest)
            if not _replacement_eligible(latest):
                manifest = phase16._strict_json(latest / "manifest.json")
                effective.append(manifest)
                replaced.append(
                    {
                        "logical_record_id": logical_id,
                        "original_status": original_status,
                        "replacement_run_id": manifest["run_id"],
                        "replacement_status": latest_status,
                    }
                )
                continue
        pending.append(
            {
                **record,
                "logical_record_id": logical_id,
                "replace": original_root.exists() or bool(replacements),
                "replacement_number": len(replacements) + 1,
                "original_status": original_status,
            }
        )
    return effective, pending, replaced


@contextmanager
def _replacement_identity(
    *,
    logical_id: str,
    replacement_run_id: str,
    controller_git_sha: str,
) -> Any:
    original_logical = phase16._logical_run_id
    original_manifest = phase16._run_manifest

    def replacement_logical(_segment_id: str, _record: Mapping[str, Any]) -> str:
        return replacement_run_id

    def replacement_manifest(**kwargs: Any) -> dict[str, Any]:
        kwargs["replacement_of"] = logical_id
        payload = original_manifest(**kwargs)
        payload["logical_record_id"] = logical_id
        payload["controller_execution_git_sha"] = controller_git_sha
        payload["timing_execution_git_sha"] = kwargs["record"].get(
            "timing_execution_git_sha"
        )
        return payload

    phase16._logical_run_id = replacement_logical
    phase16._run_manifest = replacement_manifest
    try:
        yield
    finally:
        phase16._logical_run_id = original_logical
        phase16._run_manifest = original_manifest


def _next_attempt_path(segment_root: Path, controller_git_sha: str) -> Path:
    prefix = f"continuation-attempt-{controller_git_sha[:8]}-"
    existing = sorted(segment_root.glob(f"{prefix}*.json"))
    return segment_root / f"{prefix}{len(existing) + 1:03d}.json"


def _write_controller_authority(
    *,
    segment_root: Path,
    timing_git_sha: str,
    controller_git_sha: str,
    order: Mapping[str, Any],
) -> dict[str, Any]:
    equivalence = timing_critical_equivalence(
        timing_git_sha=timing_git_sha, controller_git_sha=controller_git_sha
    )
    payload = {
        "schema_version": CONTROLLER_SCHEMA,
        "timing_execution_git_sha": timing_git_sha,
        "controller_execution_git_sha": controller_git_sha,
        "timing_critical_equivalence": equivalence,
        "execution_order_sha256": _canonical_sha256(order),
        "randomization_regenerated": False,
        "valid_completed_records_rerun": False,
        "capacity_infeasible_records_rerun": False,
        "replacement_statuses": sorted(REPLACEMENT_ELIGIBLE),
        "append_only": True,
    }
    path = segment_root / f"continuation-authority-{controller_git_sha}.json"
    if path.exists():
        if phase16._strict_json(path) != payload:
            raise Phase16ContinuationError("continuation authority differs")
    else:
        phase16._durable_write(path, payload)
    return payload


def _replacement_run(
    *,
    segment_root: Path,
    family_id: str,
    segment_id: str,
    record: Mapping[str, Any],
    timing_git_sha: str,
    controller_git_sha: str,
) -> dict[str, Any]:
    logical_id = str(record["logical_record_id"])
    replacement_run_id = (
        f"{logical_id}{REPLACEMENT_SUFFIX}"
        f"{int(record['replacement_number']):03d}"
    )
    if (segment_root / "raw" / replacement_run_id).exists():
        raise Phase16ContinuationError("replacement run ID already exists")
    worker_record = {**record, "timing_execution_git_sha": timing_git_sha}
    with _replacement_identity(
        logical_id=logical_id,
        replacement_run_id=replacement_run_id,
        controller_git_sha=controller_git_sha,
    ):
        manifest = phase16._run_one_process(
            segment_root=segment_root,
            family_id=family_id,
            segment_id=segment_id,
            record=worker_record,
            git_sha=timing_git_sha,
        )
    phase16._durable_write(
        segment_root / "raw" / replacement_run_id / "replacement-authority.json",
        {
            "schema_version": REPLACEMENT_SCHEMA,
            "run_id": replacement_run_id,
            "logical_record_id": logical_id,
            "replacement_of": logical_id,
            "original_status": record["original_status"],
            "replacement_number": record["replacement_number"],
            "timing_execution_git_sha": timing_git_sha,
            "controller_execution_git_sha": controller_git_sha,
            "timing_critical_changed": False,
            "selective_rerun": False,
        },
    )
    return manifest


def _should_refresh_container(manifest: Mapping[str, Any], run_root: Path) -> bool:
    return bool(
        manifest.get("status") in REPLACEMENT_ELIGIBLE
        or _is_telemetry_finalization_failure(run_root)
    )


def continue_segment(
    *,
    family_root: Path,
    replicate: int,
    timing_git_sha: str,
    controller_git_sha: str,
) -> dict[str, Any]:
    phase16.phase12._require_authorized_container_runtime()
    if any(name in os.environ for name in phase16.phase13._FORBIDDEN_ENVIRONMENT):
        raise Phase16ContinuationError("credentials entered Measurement Container")
    segment_id, segment_root, order = _segment_paths(family_root, replicate)
    _initialize_segment(
        family_root=family_root,
        replicate=replicate,
        timing_git_sha=timing_git_sha,
        segment_id=segment_id,
        segment_root=segment_root,
        order=order,
    )
    _validate_initialized_segment(
        family_root=family_root,
        replicate=replicate,
        timing_git_sha=timing_git_sha,
        segment_id=segment_id,
        segment_root=segment_root,
        order=order,
    )
    _write_controller_authority(
        segment_root=segment_root,
        timing_git_sha=timing_git_sha,
        controller_git_sha=controller_git_sha,
        order=order,
    )
    if (segment_root / "segment-result.json").exists():
        raise Phase16ContinuationError("completed segment cannot continue")
    _, pending, _ = _effective_and_pending(segment_root=segment_root, order=order)
    initial_pending = len(pending)
    invoked = 0
    completed = 0
    capacity = 0
    stopped_for_refresh = False
    stop_reason = None
    for record in pending:
        if record["status"] == "capacity_infeasible":
            if record["replace"]:
                raise Phase16ContinuationError("infeasible record replacement differs")
            manifest = phase16._write_capacity_record(
                segment_root=segment_root,
                family_id=family_root.name,
                segment_id=segment_id,
                record=record,
            )
            capacity += 1
        elif record["replace"]:
            manifest = _replacement_run(
                segment_root=segment_root,
                family_id=family_root.name,
                segment_id=segment_id,
                record=record,
                timing_git_sha=timing_git_sha,
                controller_git_sha=controller_git_sha,
            )
            invoked += 1
        else:
            manifest = phase16._run_one_process(
                segment_root=segment_root,
                family_id=family_root.name,
                segment_id=segment_id,
                record=record,
                git_sha=timing_git_sha,
            )
            invoked += 1
        if manifest["status"] == "completed":
            completed += 1
        run_root = segment_root / "raw" / str(manifest["run_id"])
        if _should_refresh_container(manifest, run_root):
            stopped_for_refresh = True
            stop_reason = str(manifest.get("reason"))
            break
    effective, remaining, replaced = _effective_and_pending(
        segment_root=segment_root, order=order
    )
    counts = Counter(str(record["status"]) for record in effective)
    local_complete = len(effective) == phase16.LOGICAL_POINTS and not remaining
    attempt = {
        "schema_version": ATTEMPT_SCHEMA,
        "family_id": family_root.name,
        "segment_id": segment_id,
        "replicate_index": replicate,
        "timing_execution_git_sha": timing_git_sha,
        "controller_execution_git_sha": controller_git_sha,
        "started_pending_records": initial_pending,
        "worker_processes_invoked": invoked,
        "completed_in_attempt": completed,
        "capacity_records_written": capacity,
        "effective_records_after_attempt": len(effective),
        "remaining_records": len(remaining),
        "replacement_records": len(replaced),
        "stopped_for_container_refresh": stopped_for_refresh,
        "stop_reason": stop_reason,
        "status": "LOCAL_COMPLETE" if local_complete else "CONTINUATION_REQUIRED",
        "written_at_utc": _utc_now(),
    }
    phase16._durable_write(
        _next_attempt_path(segment_root, controller_git_sha), attempt
    )
    if local_complete:
        result = {
            "schema_version": "kvbench-phase16-segment-result-1.1.0",
            "family_id": family_root.name,
            "segment_id": segment_id,
            "replicate_index": replicate,
            "status_counts": dict(sorted(counts.items())),
            "planned_records": phase16.LOGICAL_POINTS,
            "terminal_records": len(effective),
            "attempted_terminal_records": len(
                list((segment_root / "raw").glob("*/manifest.json"))
            ),
            "infrastructure_replacements": len(replaced),
            "timing_execution_git_sha": timing_git_sha,
            "controller_execution_git_sha": controller_git_sha,
            "timing_semantics_unchanged": True,
            "selective_reruns": 0,
            "status": "LOCAL_COMPLETE",
        }
        phase16._durable_write(segment_root / "segment-result.json", result)
        return result
    return attempt


def _effective_records(
    *, segment_root: Path, order: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    effective_manifests, pending, replaced = _effective_and_pending(
        segment_root=segment_root, order=order
    )
    if pending or len(effective_manifests) != phase16.LOGICAL_POINTS:
        raise Phase16ContinuationError("segment logical coverage is incomplete")
    all_records = phase16._segment_records(segment_root)
    by_run_id = {str(record["run_id"]): record for record in all_records}
    effective = []
    for manifest in effective_manifests:
        run_id = str(manifest["run_id"])
        if run_id not in by_run_id:
            raise Phase16ContinuationError("effective run evidence is absent")
        effective.append(by_run_id[run_id])
    if len({str(record["logical_record_id"]) for record in effective}) != len(
        effective
    ):
        raise Phase16ContinuationError("effective logical record is duplicated")
    return effective, all_records, replaced


def finalize_segment(*, family_root: Path, replicate: int) -> dict[str, Any]:
    segment_id, segment_root, order = _segment_paths(family_root, replicate)
    result = phase16._strict_json(segment_root / "segment-result.json")
    effective, attempts, replaced = _effective_records(
        segment_root=segment_root, order=order
    )
    if (
        result.get("terminal_records") != phase16.LOGICAL_POINTS
        or any(record.get("r_hbm") is not None for record in attempts)
        or (segment_root / "COMPLETE").exists()
    ):
        raise Phase16ContinuationError("segment finalization authority differs")
    phase16.phase13._parquet_rows(segment_root / "attempt_index.parquet", attempts)
    phase16.phase13._parquet_rows(segment_root / "run_index.parquet", effective)
    phase16.phase13._parquet_rows(segment_root / "point_records.parquet", effective)
    effective_ids = {str(record["run_id"]) for record in effective}
    exclusions = [
        {
            "run_id": record["run_id"],
            "logical_record_id": record["logical_record_id"],
            "status": record["status"],
            "reason": record["reason"],
            "effective_record": record["run_id"] in effective_ids,
            "replacement_excluded": record["run_id"] not in effective_ids,
            "fitting_excluded": (
                record["status"] != "completed"
                or record["run_id"] not in effective_ids
            ),
        }
        for record in attempts
        if record["status"] != "completed" or record["run_id"] not in effective_ids
    ]
    phase16.phase13._parquet_rows(segment_root / "exclusions.parquet", exclusions)
    phase16._durable_write(
        segment_root / "replacement-index.json",
        {
            "schema_version": "kvbench-phase16-replacement-index-1.0.0",
            "segment_id": segment_id,
            "replacement_records": replaced,
            "replacement_count": len(replaced),
            "valid_original_records_rerun": 0,
            "selective_reruns": 0,
        },
    )
    phase16._durable_write(
        segment_root / "inventory.json",
        {
            "schema_version": "kvbench-phase16-segment-inventory-1.1.0",
            "segment_id": segment_id,
            "terminal_records": len(effective),
            "attempted_terminal_records": len(attempts),
            "completed_records": sum(
                record["status"] == "completed" for record in effective
            ),
            "capacity_infeasible_records": sum(
                record["status"] == "capacity_infeasible" for record in effective
            ),
            "replacement_records": len(replaced),
            "exclusion_records": len(exclusions),
            "run_kind": "timing",
            "r_hbm": None,
        },
    )
    return phase16._seal_artifact(
        segment_root,
        run_id=segment_id,
        status="COMPLETE",
        role="phase16_full_scan_segment_evidence",
    )


def validate_segment(*, family_root: Path, replicate: int) -> dict[str, Any]:
    segment_id, segment_root, order = _segment_paths(family_root, replicate)
    artifact = validate_local_artifact(segment_root, environ={})
    effective, attempts, replaced = _effective_records(
        segment_root=segment_root, order=order
    )
    run_index = phase16._read_parquet(segment_root / "run_index.parquet")
    attempt_index = phase16._read_parquet(segment_root / "attempt_index.parquet")
    if (
        len(effective) != phase16.LOGICAL_POINTS
        or len(run_index) != phase16.LOGICAL_POINTS
        or len(attempt_index) != len(attempts)
        or any(record.get("r_hbm") is not None for record in run_index)
    ):
        raise Phase16ContinuationError("finalized segment index differs")
    return {
        "status": "PASS",
        "segment_id": segment_id,
        "root_sha256": artifact.root_sha256,
        "effective_records": len(effective),
        "attempted_terminal_records": len(attempts),
        "infrastructure_replacements": len(replaced),
    }


def promote_segment_remote(*, family_root: Path, replicate: int) -> dict[str, Any]:
    _segment_id, segment_root, order = _segment_paths(family_root, replicate)
    effective, _attempts, _replaced = _effective_records(
        segment_root=segment_root, order=order
    )
    original = phase16._segment_records

    def effective_records(root: Path) -> list[dict[str, Any]]:
        if root.resolve() == segment_root.resolve():
            return effective
        return original(root)

    phase16._segment_records = effective_records
    try:
        return phase16.promote_segment_remote(
            family_root=family_root, replicate=replicate
        )
    finally:
        phase16._segment_records = original


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--continue-segment", action="store_true")
    actions.add_argument("--finalize-segment", action="store_true")
    actions.add_argument("--validate-segment", action="store_true")
    actions.add_argument("--promote-segment-remote", action="store_true")
    parser.add_argument("--family-root", type=Path, required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--timing-git-sha")
    parser.add_argument("--controller-git-sha")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_arguments(argv)
    if args.continue_segment:
        if args.timing_git_sha is None or args.controller_git_sha is None:
            raise Phase16ContinuationError("continuation SHAs are required")
        payload = continue_segment(
            family_root=args.family_root,
            replicate=args.replicate,
            timing_git_sha=args.timing_git_sha,
            controller_git_sha=args.controller_git_sha,
        )
    elif args.finalize_segment:
        payload = finalize_segment(
            family_root=args.family_root, replicate=args.replicate
        )
    elif args.validate_segment:
        payload = validate_segment(
            family_root=args.family_root, replicate=args.replicate
        )
    else:
        payload = promote_segment_remote(
            family_root=args.family_root, replicate=args.replicate
        )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
