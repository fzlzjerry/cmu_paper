"""Phase 16 segmented Full Scan coordinator and QC.

This module owns campaign ordering, supervision, aggregation, and append-only
artifact controls.  It deliberately reuses the admitted Phase 13 fixed-L
worker, timing implementation, graph harness, and Phase 13F feasibility
formula.  It owns no adapter, kernel, cache layout, or timing boundary.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import secrets
import stat
import statistics
import subprocess
import sys
import types
from typing import Any

from kvbench.runtime.artifacts import sha256_file
from kvbench.schema.phase16g import (
    PHASE16G_ADMITTED_BATCH_SIZES,
    PHASE16G_CONTAINER_DIGEST,
    PHASE16G_GEOMETRY_REPORT_PATH,
    PHASE16G_GEOMETRY_REPORT_SHA256,
    PHASE16G_PREFIX_SCHEMA,
    require_admitted_geometry,
    validate_source_transition,
)
from preflight.run_preflight import json_bytes, write_exclusive
from scripts.r2_artifact import validate_local_artifact
import scripts.phase12_unified_admission as phase12
import scripts.phase13_pilot as phase13
import scripts.phase13d_continuation as phase13c
import scripts.phase16g_batch_geometry_admission as phase16g


class Phase16FullScanError(RuntimeError):
    """The frozen Full Scan contract or its evidence failed closed."""


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPOSITORY_ROOT / "artifacts" / "phase16"
PLAN_PATH = REPOSITORY_ROOT / "docs" / "plans" / "phase16-full-scan.md"
ORDER_PATH = (
    REPOSITORY_ROOT / "docs" / "plans" / "phase16-full-scan-execution-orders.json"
)
ADAPTIVE_AUTHORITY_PATH = (
    REPOSITORY_ROOT / "docs" / "plans" / "phase13d-candidate-table.json"
)
PHASE16G_REPORT_PATH = REPOSITORY_ROOT / PHASE16G_GEOMETRY_REPORT_PATH

CONFIGURATIONS = tuple(phase13.CONFIGURATIONS)
CONFIG_FINGERPRINTS = dict(phase13.CONFIG_FINGERPRINTS)
BATCH_SIZES = (1, 2, 4, 8, 16)
BASE_CONTEXT_LABELS = (
    4096,
    8192,
    16384,
    24576,
    32768,
    49152,
    65536,
    98304,
    131072,
)
SEEDS = (20260830, 20260831, 20260901, 20260902, 20260903)
REPLICATES = 5
WARMUP_STEPS = 64
MEASURED_STEPS = 256
MEASURED_BATCHES = phase13.MEASURED_BATCHES
CV_THRESHOLD = 0.03
BASE_LOGICAL_POINTS = 450
ADAPTIVE_LOGICAL_POINTS = 84
LOGICAL_POINTS = 534
PLANNED_PROCESS_RECORDS = 2670
EXPECTED_FEASIBLE_LOGICAL_POINTS = 441
EXPECTED_CAPACITY_INFEASIBLE_LOGICAL_POINTS = 93
EXPECTED_FEASIBLE_PROCESS_RECORDS = 2205
EXPECTED_CAPACITY_INFEASIBLE_PROCESS_RECORDS = 465
PHASE15_ROOT = "641fc02d8fa598097885b74a336b1b1f454d9844b90025cf0c4b427bee02d5e8"
PHASE15_ARTIFACT = (
    REPOSITORY_ROOT
    / "artifacts"
    / "phase15"
    / "phase15-20260828t144810363697z-446b334e-90460f"
)
PHASE13_BASE_PREFIX_CATALOG = (
    REPOSITORY_ROOT
    / "artifacts"
    / "phase13_prefix_catalogs"
    / "phase13-20260822t150835736582z-4ddd7b17-3a8fb3"
    / "catalog"
)
PHASE13D_PREFIX_CATALOG = Path(
    "/home/rockrock/phase13d_prefix_states/"
    "phase13d-20260825t030556684636z-a06837a3-83761a"
)
PHASE16G_PREFIX_ROOT = Path(
    "/home/rockrock/phase16g_prefix_states/"
    "phase16g-20260830t061918945302z-6c829eda-f16c0d"
)
CONTAINER_PREFIX_ROOTS = {
    "phase13_base": Path("/opt/kvbench-prefix-phase13"),
    "phase13d": Path("/opt/kvbench-prefix-phase13d"),
    "phase16g": Path("/opt/kvbench-prefix-phase16g"),
}
FAMILY_SCHEMA = "kvbench-phase16-full-scan-family-1.0.0"
ORDER_SCHEMA = "kvbench-phase16-execution-orders-1.0.0"
SEGMENT_SCHEMA = "kvbench-phase16-segment-1.0.0"
RUN_SCHEMA = "kvbench-phase16-process-run-1.0.0"
WORKER_PREFIX = "PHASE16_WORKER_RESULT="
_FAMILY_RE = re.compile(
    r"phase16-[0-9]{8}t[0-9]{12}z-[0-9a-f]{8}-[0-9a-f]{6}"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
TIMING_CRITICAL_PATHS = (
    "scripts/phase16_full_scan.py",
    "scripts/phase13_pilot.py",
    "scripts/phase12_unified_admission.py",
    "src/kvbench/runtime/fixed_l_runner.py",
    "src/kvbench/runtime/timing.py",
    "src/kvbench/runtime/cuda_graph.py",
    "src/kvbench/adapters/bf16.py",
    "src/kvbench/adapters/turboquant.py",
    "src/kvbench/adapters/kivi.py",
    "src/kvbench/adapters/kvquant.py",
    "src/kvbench/runtime/turboquant_cache.py",
    "src/kvbench/runtime/kivi_cache.py",
    "src/kvbench/runtime/kvquant_cache.py",
    "configs/models/primary_gqa_model.yaml",
    "configs/methods/bf16.yaml",
    "configs/methods/turboquant.yaml",
    "configs/methods/kivi.yaml",
    "configs/methods/kvquant.yaml",
    "docs/plans/phase16-full-scan-execution-orders.json",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase16FullScanError(f"invalid JSON evidence: {path}") from error
    if not isinstance(payload, dict):
        raise Phase16FullScanError(f"JSON evidence is not an object: {path}")
    return payload


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


def timing_critical_hashes() -> dict[str, Any]:
    files = {
        relative: sha256_file(REPOSITORY_ROOT / relative)
        for relative in TIMING_CRITICAL_PATHS
    }
    return {
        "schema_version": "kvbench-phase16-timing-critical-hashes-1.0.0",
        "files": files,
        "files_sha256": _canonical_sha256(files),
        "authorized_container_digest": PHASE16G_CONTAINER_DIGEST,
        "method_config_fingerprints": CONFIG_FINGERPRINTS,
    }


def actual_historical_context(label: int) -> int:
    return 131071 if label == 131072 else label


def _configure_reused_phase13() -> None:
    """Set only experiment constants before calling frozen reusable helpers."""

    phase13.BATCH_SIZES = BATCH_SIZES
    phase13.SEEDS = SEEDS
    phase13.REPLICATES = REPLICATES
    phase13.WARMUP_STEPS = WARMUP_STEPS
    phase13.MEASURED_STEPS = MEASURED_STEPS
    phase13.PLANNED_RECORD_COUNT = PLANNED_PROCESS_RECORDS


def load_phase16g_authority() -> dict[str, Any]:
    if sha256_file(PHASE16G_REPORT_PATH) != PHASE16G_GEOMETRY_REPORT_SHA256:
        raise Phase16FullScanError("Phase 16G geometry report checksum differs")
    report = _strict_json(PHASE16G_REPORT_PATH)
    transition = validate_source_transition(REPOSITORY_ROOT)
    if (
        report.get("status") != "PASS"
        or report.get("decision_id") != "0039"
        or report.get("admitted_full_scan_batches") != list(BATCH_SIZES)
        or report.get("authorized_container_digest") != PHASE16G_CONTAINER_DIGEST
        or report.get("prefix_format_version") != PHASE16G_PREFIX_SCHEMA
        or report.get("gates") != {f"G{i}": "PASS" for i in range(6)}
        or report.get("full_scan") != "READY"
        or report.get("quality_execution") != "LOCKED"
    ):
        raise Phase16FullScanError("Phase 16G execution authority differs")
    for configuration in CONFIGURATIONS:
        for batch in BATCH_SIZES:
            require_admitted_geometry(
                report, configuration=configuration, batch_size=batch
            )
    return {"report": report, "transition": transition}


def adaptive_points() -> list[dict[str, Any]]:
    authority = _strict_json(ADAPTIVE_AUTHORITY_PATH)
    proposals = authority.get("proposals")
    if not isinstance(proposals, list):
        raise Phase16FullScanError("Phase 13D candidate table differs")
    points: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for proposal in proposals:
        if not isinstance(proposal, Mapping) or proposal.get(
            "final_inclusion_status"
        ) is not True:
            continue
        configuration = str(proposal["method_config_id"])
        batch = int(proposal["batch_size"])
        context = int(proposal["historical_context"])
        key = (configuration, batch, context)
        if (
            key in seen
            or configuration not in CONFIGURATIONS
            or batch not in (1, 4, 8)
            or context in BASE_CONTEXT_LABELS
            or not 4096 <= context <= 131071
            or proposal.get("method_config_fingerprint")
            != CONFIG_FINGERPRINTS[configuration]
        ):
            raise Phase16FullScanError("Phase 13D adaptive point differs")
        seen.add(key)
        points.append(
            {
                "grid_source": "phase13d_adaptive",
                "target_id": str(proposal["target_id"]),
                "method_config_id": configuration,
                "method_config_fingerprint": CONFIG_FINGERPRINTS[configuration],
                "batch_size": batch,
                "context_label": context,
                "historical_context": context,
                "total_attended_context": context + 1,
            }
        )
    points.sort(
        key=lambda item: (
            CONFIGURATIONS.index(str(item["method_config_id"])),
            int(item["batch_size"]),
            int(item["context_label"]),
        )
    )
    if len(points) != ADAPTIVE_LOGICAL_POINTS:
        raise Phase16FullScanError("Phase 13D adaptive cardinality differs")
    return points


def logical_points() -> list[dict[str, Any]]:
    base = [
        {
            "grid_source": "base",
            "target_id": None,
            "method_config_id": configuration,
            "method_config_fingerprint": CONFIG_FINGERPRINTS[configuration],
            "batch_size": batch,
            "context_label": label,
            "historical_context": actual_historical_context(label),
            "total_attended_context": actual_historical_context(label) + 1,
        }
        for configuration in CONFIGURATIONS
        for batch in BATCH_SIZES
        for label in BASE_CONTEXT_LABELS
    ]
    if len(base) != BASE_LOGICAL_POINTS:
        raise AssertionError("base grid cardinality differs")
    points = base + adaptive_points()
    keys = {
        (
            str(item["method_config_id"]),
            int(item["batch_size"]),
            int(item["context_label"]),
        )
        for item in points
    }
    if len(points) != LOGICAL_POINTS or len(keys) != LOGICAL_POINTS:
        raise Phase16FullScanError("Full Scan logical grid differs")
    return points


def feasibility_records() -> list[dict[str, Any]]:
    _configure_reused_phase13()
    records = [phase13.feasibility_record(point) for point in logical_points()]
    feasible = sum(record["status"] == "feasible" for record in records)
    infeasible = sum(
        record["status"] == "capacity_infeasible" for record in records
    )
    if (
        len(records) != LOGICAL_POINTS
        or feasible != EXPECTED_FEASIBLE_LOGICAL_POINTS
        or infeasible != EXPECTED_CAPACITY_INFEASIBLE_LOGICAL_POINTS
        or any(record["adapter_geometry_supported_at_entry"] is not True for record in records)
    ):
        raise Phase16FullScanError("current Full Scan feasibility result differs")
    return records


def derive_execution_orders() -> dict[str, Any]:
    feasibility = feasibility_records()
    by_configuration: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in feasibility:
        by_configuration[str(item["method_config_id"])].append(dict(item))
    segments: list[dict[str, Any]] = []
    for replicate, seed in enumerate(SEEDS):
        rng = random.Random(seed)
        blocks = list(CONFIGURATIONS)
        rng.shuffle(blocks)
        records: list[dict[str, Any]] = []
        for configuration in blocks:
            block = [dict(item) for item in by_configuration[configuration]]
            rng.shuffle(block)
            for item in block:
                item["replicate_index"] = replicate
                item["seed"] = seed
                item["order_index"] = len(records)
                records.append(item)
        if len(records) != LOGICAL_POINTS:
            raise AssertionError("replicate order cardinality differs")
        segments.append(
            {
                "replicate_index": replicate,
                "seed": seed,
                "method_block_order": blocks,
                "records": records,
                "records_sha256": _canonical_sha256(records),
            }
        )
    payload = {
        "schema_version": ORDER_SCHEMA,
        "configurations": list(CONFIGURATIONS),
        "batches": list(BATCH_SIZES),
        "base_context_labels": list(BASE_CONTEXT_LABELS),
        "seeds": list(SEEDS),
        "base_logical_points": BASE_LOGICAL_POINTS,
        "adaptive_logical_points": ADAPTIVE_LOGICAL_POINTS,
        "logical_points": LOGICAL_POINTS,
        "planned_process_records": PLANNED_PROCESS_RECORDS,
        "segments": segments,
    }
    payload["orders_sha256"] = _canonical_sha256(segments)
    return payload


def validate_execution_orders(payload: Mapping[str, Any]) -> dict[str, Any]:
    expected = derive_execution_orders()
    if payload != expected:
        raise Phase16FullScanError("committed Full Scan execution orders differ")
    return dict(payload)


def new_family_id(git_sha: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", git_sha) is None:
        raise Phase16FullScanError("execution Git SHA is invalid")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%S%f")[:21]
    return f"phase16-{stamp}z-{git_sha[:8]}-{secrets.token_hex(3)}"


def reserve_family(*, family_id: str, git_sha: str) -> Path:
    if _FAMILY_RE.fullmatch(family_id) is None:
        raise Phase16FullScanError("Full Scan family ID is invalid")
    if not re.fullmatch(r"[0-9a-f]{40}", git_sha):
        raise Phase16FullScanError("Full Scan execution SHA is invalid")
    validate_execution_orders(_strict_json(ORDER_PATH))
    load_phase16g_authority()
    root = ARTIFACT_ROOT / family_id
    root.mkdir(parents=True, exist_ok=False)
    for relative in ("execution_orders", "segments", "publication", "outer-stage"):
        (root / relative).mkdir()
    order = _strict_json(ORDER_PATH)
    for segment in order["segments"]:
        replicate = int(segment["replicate_index"])
        write_exclusive(
            root / "execution_orders" / f"replicate-{replicate}.json",
            json_bytes(segment),
        )
        (root / "segments" / f"replicate-{replicate}").mkdir()
    write_exclusive(
        root / "family-reservation.json",
        json_bytes(
            {
                "schema_version": FAMILY_SCHEMA,
                "family_id": family_id,
                "execution_git_sha": git_sha,
                "reserved_at_utc": _utc_now(),
                "append_only": True,
                "segment_count": REPLICATES,
                "orders_sha256": order["orders_sha256"],
                "timing_critical_hashes": timing_critical_hashes(),
            }
        ),
    )
    return root.resolve(strict=True)


def _prefix_catalog_entries(path: Path) -> list[dict[str, Any]]:
    payload = _strict_json(path / "catalog.json")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not all(isinstance(item, dict) for item in entries):
        raise Phase16FullScanError(f"prefix catalog differs: {path}")
    if (
        payload.get("snapshot_count") != len(entries)
        or payload.get("restoration_outside_timing") is not True
        or payload.get("fresh_caller_owned_cache_per_timing_process") is not True
    ):
        raise Phase16FullScanError(f"prefix catalog authority differs: {path}")
    for item in entries:
        verified = item.get("state_bytes_verified_once_before_timing")
        if verified is None:
            verified = item.get("state_bytes_verified_before_timing")
        if verified is not True:
            raise Phase16FullScanError(f"prefix catalog verification differs: {path}")
    return [dict(item) for item in entries]


def prefix_index(*, container_paths: bool) -> dict[tuple[str, int, int], dict[str, Any]]:
    result: dict[tuple[str, int, int], dict[str, Any]] = {}
    sources = (
        ("phase13_base", PHASE13_BASE_PREFIX_CATALOG),
        ("phase13d", PHASE13D_PREFIX_CATALOG),
    )
    for source_name, host_root in sources:
        mounted_root = CONTAINER_PREFIX_ROOTS[source_name] if container_paths else host_root
        for entry in _prefix_catalog_entries(host_root):
            key = (
                str(entry["method_config_id"]),
                int(entry["batch_size"]),
                int(entry["context_label"]),
            )
            if key in result:
                raise Phase16FullScanError("prefix catalogs overlap")
            result[key] = {
                "kind": "snapshot",
                "schema_version": "kvbench-phase13-prefix-state-2.0.0",
                "snapshot_root": str(mounted_root / str(entry["snapshot_relative_path"])),
                "state_file_sha256": str(entry["state_file_sha256"]),
                "source": source_name,
            }
    report = _strict_json(PHASE16G_REPORT_PATH)
    geometry_root = (
        CONTAINER_PREFIX_ROOTS["phase16g"] if container_paths else PHASE16G_PREFIX_ROOT
    )
    for configuration in CONFIGURATIONS:
        for batch in (2, 16):
            record = report["new_geometry_records"][f"{configuration}/B{batch}"]
            matches = list(
                PHASE16G_PREFIX_ROOT.glob(
                    f"*-prefix-{configuration}-b{batch}-l4096"
                )
            )
            if len(matches) != 1:
                raise Phase16FullScanError("Phase 16G prefix path differs")
            result[(configuration, batch, 4096)] = {
                "kind": "snapshot",
                "schema_version": PHASE16G_PREFIX_SCHEMA,
                "snapshot_root": str(geometry_root / matches[0].name),
                "state_file_sha256": str(record["prefix_state_sha256"]),
                "source": "phase16g",
            }
    return result


def prefix_entry(record: Mapping[str, Any], *, container_paths: bool) -> dict[str, Any]:
    key = (
        str(record["method_config_id"]),
        int(record["batch_size"]),
        int(record["context_label"]),
    )
    existing = prefix_index(container_paths=container_paths).get(key)
    if existing is not None:
        return existing
    return {
        "kind": "direct_construct",
        "schema_version": None,
        "snapshot_root": None,
        "state_file_sha256": None,
        "source": "fresh_process_untimed_direct",
    }


def _geometry_authority_for_batch(
    batch: int, *, predecessor: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    authority = json.loads(
        json.dumps(
            predecessor
            if predecessor is not None
            else phase13._phase13b_successor_authority()
        )
    )
    if batch in (1, 4, 8):
        return authority
    report = _strict_json(PHASE16G_REPORT_PATH)
    for family, family_record in authority["families"].items():
        configurations = tuple(family_record["configurations"])
        family_record["batch_sizes"] = list(BATCH_SIZES)
        for configuration in configurations:
            record = report["new_geometry_records"][f"{configuration}/B{batch}"]
            family_record["adapter_config_fingerprints_l128"][
                f"{configuration}/B{batch}"
            ] = record["adapter_config_fingerprint"]
            family_record["cache_layout_fingerprints_l128"][
                f"{configuration}/B{batch}"
            ] = record["cache_layout_fingerprint"]
    return authority


def _direct_session(
    *, loaded: Any, operation: Any, prefix: Any, decode: Any
) -> tuple[Any, dict[str, Any]]:
    """Construct one admitted prefix directly, outside timing, without export."""

    import torch

    from kvbench.runtime.backend import forced_flash_execution

    witness = _canonical_sha256(
        {
            "schema_version": "kvbench-phase16-direct-prefix-witness-1.0.0",
            "configuration": operation.configuration,
            "method_config_fingerprint": CONFIG_FINGERPRINTS[operation.configuration],
            "batch_size": operation.batch_size,
            "historical_context": operation.historical_context,
            "input_recipe_sha256": phase12.PHASE12_INPUT_RECIPE_SHA256,
            "construction": "frozen_direct_prefill",
            "decision": "0039",
        }
    )
    family = phase12._method_family(operation.configuration)

    def callback(endpoint: Any, input_ids: Any, original: Any) -> Any:
        if tuple(input_ids.shape) != (
            operation.batch_size,
            operation.historical_context,
        ):
            raise Phase16FullScanError("direct prefix geometry differs")
        if family == "kvquant":
            with phase13._chunked_kvquant_prefix_store(endpoint):
                result = original(endpoint, input_ids)
        else:
            result = original(endpoint, input_ids)
        if hasattr(endpoint.cache, "history_sha256"):
            endpoint.cache.history_sha256 = types.MethodType(
                lambda self, historical_length: witness,
                endpoint.cache,
            )
        return result

    with (
        torch.inference_mode(),
        forced_flash_execution(),
        phase13._restored_prefix_hash_overrides(witness),
        phase13._patched_endpoint_prefill(callback),
    ):
        session = phase12._build_phase12_session(
            loaded=loaded,
            operation_key=operation,
            prefix_input_ids=prefix,
            decode_input_ids=decode,
        )
    phase13._bind_session_prefix_witness(session, witness)
    return session, {
        "schema_version": "kvbench-phase16-direct-prefix-receipt-1.0.0",
        "mode": "fresh_process_untimed_direct",
        "witness_sha256": witness,
        "fresh_target_allocation": True,
        "runtime_prefix_sharing": False,
        "snapshot_persisted": False,
        "cache_state_checksum": None,
        "logical_construction_witness": True,
        "bounded_kvquant_chunks": family == "kvquant",
    }


@contextmanager
def _worker_overrides(*, batch: int, entry: Mapping[str, Any]) -> Any:
    _configure_reused_phase13()
    original_builder = phase13._build_restored_session
    original_authority = phase13._phase13b_successor_authority

    def builder(**kwargs: Any) -> tuple[Any, dict[str, Any]]:
        if entry["kind"] == "direct_construct":
            return _direct_session(
                loaded=kwargs["loaded"],
                operation=kwargs["operation"],
                prefix=kwargs["prefix"],
                decode=kwargs["decode"],
            )
        if entry["schema_version"] == PHASE16G_PREFIX_SCHEMA:
            operation = kwargs["operation"]
            manifest = _strict_json(Path(str(entry["snapshot_root"])) / "manifest.json")
            session, receipt, _, _ = phase16g._build_restored_session(
                loaded=kwargs["loaded"],
                configuration=operation.configuration,
                batch=operation.batch_size,
                historical=operation.historical_context,
                snapshot_root=Path(str(entry["snapshot_root"])),
                manifest=manifest,
            )
            return session, receipt
        return original_builder(**kwargs)

    merged_authority = _geometry_authority_for_batch(
        batch, predecessor=original_authority()
    )
    phase13._build_restored_session = builder
    phase13._phase13b_successor_authority = lambda: merged_authority
    try:
        yield
    finally:
        phase13._build_restored_session = original_builder
        phase13._phase13b_successor_authority = original_authority


def run_worker(
    *,
    run_id: str,
    record: Mapping[str, Any],
    git_sha: str,
    run_root: Path,
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    batch = int(record["batch_size"])
    fallback_root = run_root / "no-prefix-snapshot"
    snapshot_root = (
        Path(str(entry["snapshot_root"]))
        if entry["snapshot_root"] is not None
        else fallback_root
    )
    expected_sha = (
        str(entry["state_file_sha256"])
        if entry["state_file_sha256"] is not None
        else "0" * 64
    )
    with _worker_overrides(batch=batch, entry=entry):
        payload = phase13._run_worker(
            run_id=run_id,
            configuration=str(record["method_config_id"]),
            batch=batch,
            context_label=int(record["context_label"]),
            replicate_index=int(record["replicate_index"]),
            order_index=int(record["order_index"]),
            git_sha=git_sha,
            run_artifact_root=run_root,
            prefix_state_root=snapshot_root,
            prefix_state_sha256=expected_sha,
        )
    report = _strict_json(PHASE16G_REPORT_PATH)
    geometry_key = require_admitted_geometry(
        report,
        configuration=str(record["method_config_id"]),
        batch_size=batch,
    )
    payload.update(
        {
            "schema_version": RUN_SCHEMA,
            "run_kind": "timing",
            "pilot_only": False,
            "claim_eligibility": "performance_only",
            "quality_status": "unvalidated",
            "performance_claim_eligible": False,
            "phase16g_geometry_binding": {
                "decision": "0039",
                "geometry_key": geometry_key,
                "report_path": PHASE16G_GEOMETRY_REPORT_PATH,
                "report_sha256": PHASE16G_GEOMETRY_REPORT_SHA256,
                "prefix_schema": PHASE16G_PREFIX_SCHEMA,
            },
            "prefix_source": dict(entry),
            "r_hbm": None,
        }
    )
    payload["kernel_path_fingerprint"] = _canonical_sha256(
        {
            "phase13_worker_kernel_path": payload["kernel_path_fingerprint"],
            "phase16g_geometry": payload["phase16g_geometry_binding"],
        }
    )
    return payload


def _durable_write(path: Path, payload: Mapping[str, Any]) -> None:
    write_exclusive(path, json_bytes(dict(payload)))
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _captured_worker_snapshot(
    *, run_root: Path, supervised_pid: int | None, supervised_start_ticks: int | None
) -> dict[str, Any]:
    phase = "preflight" if supervised_pid is None else "postflight"
    evidence = run_root / "internal-gpu-snapshots" / phase
    evidence.mkdir(parents=True, exist_ok=False)
    for attempt in range(3):
        command = [
            sys.executable,
            str(REPOSITORY_ROOT / "preflight" / "process_query.py"),
        ]
        if supervised_pid is not None:
            command.extend(
                (
                    "--supervised-root-pid",
                    str(supervised_pid),
                    "--supervised-root-start-ticks",
                    str(supervised_start_ticks),
                )
            )
        invocation_exception = None
        try:
            result = subprocess.run(
                command,
                cwd=REPOSITORY_ROOT,
                env={
                    "PATH": "/usr/local/cuda-13.0/bin:/usr/local/bin:/usr/bin:/bin",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONNOUSERSITE": "1",
                    "PYTHONPATH": f"{REPOSITORY_ROOT / 'src'}:{REPOSITORY_ROOT}",
                    "LANG": "C.UTF-8",
                    "LC_ALL": "C.UTF-8",
                    "TZ": "UTC",
                },
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            invocation = {
                "return_code": result.returncode,
                "raw_stdout": result.stdout,
                "raw_stderr": result.stderr,
                "invocation_exception": None,
            }
        except (OSError, subprocess.TimeoutExpired) as error:
            invocation_exception = f"{type(error).__name__}: {error}"
            invocation = {
                "return_code": None,
                "raw_stdout": "",
                "raw_stderr": "",
                "invocation_exception": invocation_exception,
            }
        raw = {
            "schema_version": "kvbench-phase16-gpu-snapshot-raw-1.0.0",
            "timestamp": _utc_now(),
            "phase": phase,
            "attempt": attempt,
            "command_api": "preflight/process_query.py",
            "command": command,
            **invocation,
        }
        raw_path = evidence / f"attempt-{attempt:02d}.raw.json"
        _durable_write(raw_path, raw)
        state, parsed, parser_exception, process_list = phase13c._classify_snapshot_attempt(
            invocation=invocation
        )
        classification = {
            **raw,
            "schema_version": "kvbench-phase16-gpu-snapshot-classification-1.0.0",
            "raw_record_path": raw_path.name,
            "parser_result": parsed,
            "parser_exception": parser_exception,
            "process_list": process_list,
            "state": state,
        }
        _durable_write(
            evidence / f"attempt-{attempt:02d}.classification.json",
            classification,
        )
        if state == "foreign_process_detected":
            raise Phase16FullScanError("actual foreign GPU process detected")
        if state == "clean":
            assert parsed is not None
            parsed["phase16_snapshot_state"] = "clean"
            parsed["phase16_snapshot_attempt"] = attempt
            return parsed
        if attempt < 2:
            continue
        if phase == "preflight":
            raise Phase16FullScanError("preflight GPU process query unavailable")
        return {
            "phase16_snapshot_state": "query_failed",
            "phase16_snapshot_attempt": attempt,
            "allowed_compute_processes": [],
            "foreign_compute_processes": [],
            "unknown_processes": [],
            "errors": ["query_failed_after_three_attempts"],
        }
    raise AssertionError("snapshot retry loop is unreachable")


@contextmanager
def _worker_snapshot_overrides(run_root: Path) -> Any:
    original_capture = phase12._capture_process_snapshot
    original_owned = phase12._require_owned_snapshot

    def capture(
        *,
        supervised_pid: int | None = None,
        supervised_start_ticks: int | None = None,
    ) -> dict[str, Any]:
        return _captured_worker_snapshot(
            run_root=run_root,
            supervised_pid=supervised_pid,
            supervised_start_ticks=supervised_start_ticks,
        )

    def require_owned(snapshot: Mapping[str, Any], *, pid: int, start_ticks: int) -> None:
        if snapshot.get("phase16_snapshot_state") == "query_failed":
            return
        original_owned(snapshot, pid=pid, start_ticks=start_ticks)

    phase12._capture_process_snapshot = capture
    phase12._require_owned_snapshot = require_owned
    try:
        yield
    finally:
        phase12._capture_process_snapshot = original_capture
        phase12._require_owned_snapshot = original_owned


# Replace the first definition with the same timing path plus snapshot-only
# wrappers.  The indirection keeps the reused Phase 13 worker byte-identical.
_run_worker_without_snapshot_override = run_worker


def run_worker(  # type: ignore[no-redef]
    *,
    run_id: str,
    record: Mapping[str, Any],
    git_sha: str,
    run_root: Path,
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    with _worker_snapshot_overrides(run_root):
        payload = _run_worker_without_snapshot_override(
            run_id=run_id,
            record=record,
            git_sha=git_sha,
            run_root=run_root,
            entry=entry,
        )
    snapshot = payload.get("gpu_process_owned_after_measurement")
    unavailable = bool(
        isinstance(snapshot, Mapping)
        and snapshot.get("phase16_snapshot_state") == "query_failed"
    )
    payload["postflight_snapshot_unavailable"] = unavailable
    payload["fitting_eligible"] = not unavailable
    if unavailable:
        payload["gpu_exclusive"] = None
    return payload


def _segment_id(family_id: str, replicate: int) -> str:
    return f"{family_id}-replicate-{replicate}"


def _logical_run_id(segment_id: str, record: Mapping[str, Any]) -> str:
    return (
        f"{segment_id}-o{int(record['order_index']):03d}-"
        f"{record['method_config_id']}-b{record['batch_size']}-"
        f"l{record['context_label']}"
    )


def _run_manifest(
    *,
    family_id: str,
    segment_id: str,
    run_id: str,
    record: Mapping[str, Any],
    status: str,
    reason: str | None,
    result_path: str | None,
    replacement_of: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "kvbench-phase16-run-manifest-1.0.0",
        "family_id": family_id,
        "segment_id": segment_id,
        "run_id": run_id,
        "logical_record_id": _logical_run_id(segment_id, record),
        "replacement_of": replacement_of,
        "status": status,
        "reason": reason,
        "method_config_id": record["method_config_id"],
        "method_config_fingerprint": record["method_config_fingerprint"],
        "grid_source": record["grid_source"],
        "batch_size": record["batch_size"],
        "context_label": record["context_label"],
        "historical_context": record["historical_context"],
        "total_attended_context": record["total_attended_context"],
        "replicate_index": record["replicate_index"],
        "seed": record["seed"],
        "order_index": record["order_index"],
        "runner_kind": "fixed_l",
        "graph_mode": "cuda_graph",
        "warmup_steps": WARMUP_STEPS,
        "measured_steps": MEASURED_STEPS,
        "measured_batches": MEASURED_BATCHES,
        "run_kind": "timing",
        "quality_status": "unvalidated",
        "claim_eligibility": "performance_only",
        "performance_claim_eligible": False,
        "selective_rerun": False,
        "result_path": result_path,
        "r_hbm": None,
    }


def _write_terminal_manifest(
    *, run_root: Path, payload: Mapping[str, Any]
) -> dict[str, Any]:
    path = run_root / "manifest.json"
    write_exclusive(path, json_bytes(dict(payload)))
    return dict(payload)


def _external_snapshot(run_root: Path, phase: str) -> Any:
    return phase13c.capture_persisted_snapshot(
        evidence_root=run_root / "gpu-snapshots" / phase,
        phase=phase,
    )


def _worker_failure_status(stderr: str, timeout_stage: str | None) -> tuple[str, str]:
    lowered = stderr.lower()
    if timeout_stage is not None:
        return "infrastructure_failed", f"supervisor_stage_timeout:{timeout_stage}"
    if "graph" in lowered and "capture" in lowered:
        return "graph_capture_failed", "worker_graph_capture_failed"
    if "out of memory" in lowered or "allocation" in lowered:
        return "allocation_failed", "worker_allocation_failed"
    if "output" in lowered and ("mismatch" in lowered or "drift" in lowered):
        return "output_mismatch", "worker_output_mismatch"
    if "fallback" in lowered:
        return "backend_fallback", "worker_backend_fallback"
    return "runtime_failed", "supervised_worker_failed"


def _run_one_process(
    *,
    segment_root: Path,
    family_id: str,
    segment_id: str,
    record: Mapping[str, Any],
    git_sha: str,
) -> dict[str, Any]:
    logical_id = _logical_run_id(segment_id, record)
    run_id = logical_id
    run_root = segment_root / "raw" / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "stage-progress").mkdir()
    write_exclusive(
        run_root / "started.json",
        json_bytes(
            {
                "schema_version": "kvbench-phase16-run-start-1.0.0",
                "run_id": run_id,
                "started_at_utc": _utc_now(),
                "record": dict(record),
            }
        ),
    )
    pre = _external_snapshot(run_root, "preflight")
    pre_policy = phase13c.snapshot_policy(phase="preflight", state=pre.state)
    if pre_policy == "abort_campaign":
        raise Phase16FullScanError("actual foreign GPU process detected before run")
    if pre_policy == "fail_current_before_measurement":
        return _write_terminal_manifest(
            run_root=run_root,
            payload=_run_manifest(
                family_id=family_id,
                segment_id=segment_id,
                run_id=run_id,
                record=record,
                status="infrastructure_failed",
                reason="preflight_snapshot_unavailable",
                result_path=None,
            ),
        )
    entry = prefix_entry(record, container_paths=True)
    write_exclusive(run_root / "prefix-entry.json", json_bytes(entry))
    command = (
        sys.executable,
        str(REPOSITORY_ROOT / "scripts" / "phase16_full_scan.py"),
        "--run-worker",
        "--run-id",
        run_id,
        "--record",
        str(run_root / "worker-record.json"),
        "--git-sha",
        git_sha,
        "--run-root",
        str(run_root),
        "--prefix-entry",
        str(run_root / "prefix-entry.json"),
    )
    write_exclusive(run_root / "worker-record.json", json_bytes(dict(record)))
    result = phase13.run_stage_supervised_command(
        command,
        working_directory=str(REPOSITORY_ROOT),
        environment=phase12._child_environment(),
        stage_timeouts=phase13.stage_timeout_contract(
            batch=int(record["batch_size"]),
            historical=int(record["historical_context"]),
        ),
        stage_observer=lambda: phase13._read_stage_observations(
            root=run_root / "stage-progress", run_id=run_id
        ),
        startup_stage="startup",
        transition_stage="transition",
        observer_poll_seconds=phase13.STAGE_OBSERVER_POLL_SECONDS,
    )
    post = _external_snapshot(run_root, "postflight")
    post_policy = phase13c.snapshot_policy(phase="postflight", state=post.state)
    write_exclusive(run_root / "worker.stdout.txt", result.stdout)
    write_exclusive(run_root / "worker.stderr.txt", result.stderr)
    write_exclusive(run_root / "worker.supervision.json", json_bytes(result.to_dict()))
    if post_policy == "invalidate_current_and_abort":
        raise Phase16FullScanError("actual foreign GPU process detected after run")
    if not phase12._supervision_passed(result):
        status, reason = _worker_failure_status(
            result.stderr.decode("utf-8", errors="replace"), result.timeout_stage
        )
        write_exclusive(
            run_root / "failure.json",
            json_bytes(
                {
                    "schema_version": "kvbench-phase16-run-failure-1.0.0",
                    "run_id": run_id,
                    "status": status,
                    "reason": reason,
                    "returncode": result.returncode,
                    "timeout_stage": result.timeout_stage,
                    "preserved": True,
                }
            ),
        )
        return _write_terminal_manifest(
            run_root=run_root,
            payload=_run_manifest(
                family_id=family_id,
                segment_id=segment_id,
                run_id=run_id,
                record=record,
                status=status,
                reason=reason,
                result_path=None,
            ),
        )
    matches = [
        line[len(WORKER_PREFIX) :]
        for line in result.stdout.decode("utf-8", errors="strict").splitlines()
        if line.startswith(WORKER_PREFIX)
    ]
    if len(matches) != 1:
        raise Phase16FullScanError("worker result channel differs")
    payload = json.loads(matches[0])
    if not isinstance(payload, dict) or payload.get("run_id") != run_id:
        raise Phase16FullScanError("worker result identity differs")
    write_exclusive(run_root / "result.json", json_bytes(payload))
    write_exclusive(
        run_root / "result-binding.json",
        json_bytes(
            {
                "schema_version": "kvbench-phase16-result-binding-1.0.0",
                "run_id": run_id,
                "result_sha256": sha256_file(run_root / "result.json"),
            }
        ),
    )
    post_unavailable = bool(
        post_policy == "exclude_current_and_continue"
        or payload.get("postflight_snapshot_unavailable") is True
    )
    status = "postflight_snapshot_unavailable" if post_unavailable else "completed"
    return _write_terminal_manifest(
        run_root=run_root,
        payload=_run_manifest(
            family_id=family_id,
            segment_id=segment_id,
            run_id=run_id,
            record=record,
            status=status,
            reason="postflight_snapshot_unavailable" if post_unavailable else None,
            result_path="result.json",
        ),
    )


def _write_capacity_record(
    *,
    segment_root: Path,
    family_id: str,
    segment_id: str,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    run_id = _logical_run_id(segment_id, record)
    run_root = segment_root / "raw" / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    write_exclusive(
        run_root / "disposition.json",
        json_bytes(
            {
                "schema_version": "kvbench-phase16-nonlaunch-1.0.0",
                "run_id": run_id,
                "status": "capacity_infeasible",
                "reason": record["reason"],
                "predicted_required_bytes": record["predicted_required_bytes"],
                "limit_bytes": record["limit_bytes"],
                "cuda_process_launched": False,
            }
        ),
    )
    return _write_terminal_manifest(
        run_root=run_root,
        payload=_run_manifest(
            family_id=family_id,
            segment_id=segment_id,
            run_id=run_id,
            record=record,
            status="capacity_infeasible",
            reason=str(record["reason"]),
            result_path=None,
        ),
    )


def run_segment(
    *, family_root: Path, replicate: int, git_sha: str
) -> dict[str, Any]:
    phase12._require_authorized_container_runtime()
    if any(name in os.environ for name in phase13._FORBIDDEN_ENVIRONMENT):
        raise Phase16FullScanError("credentials entered Measurement Container")
    if replicate not in range(REPLICATES):
        raise Phase16FullScanError("replicate index differs")
    observed_head = subprocess.run(
        ("/usr/bin/git", "rev-parse", "HEAD"),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    observed_status = subprocess.run(
        ("/usr/bin/git", "status", "--porcelain=v1", "--untracked-files=all"),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if observed_head != git_sha or observed_status:
        raise Phase16FullScanError("segment source authority differs")
    family_id = family_root.name
    reservation = _strict_json(family_root / "family-reservation.json")
    if (
        reservation.get("execution_git_sha") != git_sha
        or reservation.get("timing_critical_hashes") != timing_critical_hashes()
    ):
        raise Phase16FullScanError("timing-critical hash authority differs")
    segment_id = _segment_id(family_id, replicate)
    segment_root = family_root / "segments" / f"replicate-{replicate}"
    if {item.name for item in segment_root.iterdir()}:
        raise Phase16FullScanError("fresh segment root is not empty")
    (segment_root / "raw").mkdir()
    order = _strict_json(
        family_root / "execution_orders" / f"replicate-{replicate}.json"
    )
    committed = _strict_json(ORDER_PATH)["segments"][replicate]
    if order != committed or order["seed"] != SEEDS[replicate]:
        raise Phase16FullScanError("segment execution order differs")
    authority = load_phase16g_authority()
    write_exclusive(segment_root / "execution_order.json", json_bytes(order))
    write_exclusive(
        segment_root / "segment_manifest.json",
        json_bytes(
            {
                "schema_version": SEGMENT_SCHEMA,
                "family_id": family_id,
                "segment_id": segment_id,
                "replicate_index": replicate,
                "seed": SEEDS[replicate],
                "execution_git_sha": git_sha,
                "authorized_container_digest": PHASE16G_CONTAINER_DIGEST,
                "decision": "0039",
                "phase16g_authority": authority,
                "logical_records": LOGICAL_POINTS,
                "append_only": True,
                "run_kind": "timing",
                "quality_status": "unvalidated",
                "performance_claim_eligible": False,
            }
        ),
    )
    records: list[dict[str, Any]] = []
    for record in order["records"]:
        if record["status"] == "capacity_infeasible":
            manifest = _write_capacity_record(
                segment_root=segment_root,
                family_id=family_id,
                segment_id=segment_id,
                record=record,
            )
        else:
            manifest = _run_one_process(
                segment_root=segment_root,
                family_id=family_id,
                segment_id=segment_id,
                record=record,
                git_sha=git_sha,
            )
        records.append(manifest)
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[str(record["status"])] += 1
    payload = {
        "schema_version": "kvbench-phase16-segment-result-1.0.0",
        "family_id": family_id,
        "segment_id": segment_id,
        "replicate_index": replicate,
        "status_counts": dict(sorted(counts.items())),
        "planned_records": LOGICAL_POINTS,
        "terminal_records": len(records),
        "timing_semantics_unchanged": True,
        "selective_reruns": 0,
        "status": "LOCAL_COMPLETE",
    }
    write_exclusive(segment_root / "segment-result.json", json_bytes(payload))
    return payload


def _segment_records(segment_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for manifest_path in sorted((segment_root / "raw").glob("*/manifest.json")):
        manifest = _strict_json(manifest_path)
        result_path = manifest_path.parent / "result.json"
        result = _strict_json(result_path) if result_path.is_file() else None
        record = {
            **manifest,
            "manifest_path": manifest_path.relative_to(segment_root).as_posix(),
            "manifest_sha256": sha256_file(manifest_path),
            "result_path": (
                result_path.relative_to(segment_root).as_posix()
                if result is not None
                else None
            ),
            "result_sha256": sha256_file(result_path) if result is not None else None,
        }
        if result is not None:
            for field in (
                "process_median_ms",
                "host_wall_cuda_event_ratio",
                "kernel_count",
                "finite_output",
                "no_backend_fallback",
                "allocation_stable",
                "kernel_path_stable",
                "gpu_exclusive",
                "output_checksum",
                "kernel_path_fingerprint",
                "allocation_fingerprint",
                "temperature_min_c",
                "temperature_max_c",
                "sm_clock_min_mhz",
                "sm_clock_max_mhz",
                "memory_clock_min_mhz",
                "memory_clock_max_mhz",
                "power_min_w",
                "power_max_w",
                "runner",
                "postflight_snapshot_unavailable",
                "fitting_eligible",
            ):
                record[field] = result.get(field)
        records.append(record)
    return records


def _payload_files(root: Path, excluded: set[str]) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise Phase16FullScanError("artifact contains a symlink")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise Phase16FullScanError("artifact contains an unsafe file")
        if path.relative_to(root).as_posix() not in excluded:
            files.append(path)
    return files


def _seal_artifact(
    root: Path, *, run_id: str, status: str, role: str
) -> dict[str, Any]:
    write_exclusive(
        root / "manifest.json",
        json_bytes(
            {
                "schema_version": "kvbench-phase16-artifact-manifest-1.0.0",
                "run_id": run_id,
                "status": status,
                "created_at_utc": _utc_now(),
                "authorized_container_digest": PHASE16G_CONTAINER_DIGEST,
                "append_only": True,
                "complete_written_last": True,
                "quality_status": "unvalidated",
                "performance_claim_eligible": False,
                "r_hbm": None,
            }
        ),
    )
    inventory = [
        {
            "path": path.relative_to(root).as_posix(),
            "role": role,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in _payload_files(
            root, {"artifact_inventory.json", "checksums.sha256", "COMPLETE"}
        )
    ]
    write_exclusive(
        root / "artifact_inventory.json",
        json_bytes(
            {
                "schema_version": "kvbench-artifact-inventory-1.0.0",
                "run_id": run_id,
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
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n"
        for path in _payload_files(root, {"checksums.sha256", "COMPLETE"})
    ).encode("utf-8")
    write_exclusive(root / "checksums.sha256", ledger)
    write_exclusive(
        root / "COMPLETE",
        json_bytes(
            {
                "schema_version": "kvbench-completion-1.0.0",
                "run_id": run_id,
                "status": status,
                "manifest_sha256": sha256_file(root / "manifest.json"),
                "artifact_inventory_sha256": sha256_file(
                    root / "artifact_inventory.json"
                ),
                "checksum_ledger_path": "checksums.sha256",
                "checksum_ledger_sha256": sha256_file(root / "checksums.sha256"),
                "written_last": True,
            }
        ),
    )
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)
    artifact = validate_local_artifact(root, environ={})
    return {
        "run_id": run_id,
        "root_sha256": artifact.root_sha256,
        "object_count": len(artifact.files),
        "status": status,
    }


def finalize_segment(segment_root: Path) -> dict[str, Any]:
    result = _strict_json(segment_root / "segment-result.json")
    segment_id = str(result["segment_id"])
    records = _segment_records(segment_root)
    if (
        len(records) != LOGICAL_POINTS
        or result.get("terminal_records") != LOGICAL_POINTS
        or any(record.get("r_hbm") is not None for record in records)
    ):
        raise Phase16FullScanError("segment terminal record set differs")
    phase13._parquet_rows(segment_root / "run_index.parquet", records)
    phase13._parquet_rows(segment_root / "point_records.parquet", records)
    exclusions = [
        {
            "run_id": record["run_id"],
            "status": record["status"],
            "reason": record["reason"],
            "fitting_excluded": record["status"] != "completed",
        }
        for record in records
        if record["status"] != "completed"
    ]
    phase13._parquet_rows(segment_root / "exclusions.parquet", exclusions)
    write_exclusive(
        segment_root / "inventory.json",
        json_bytes(
            {
                "schema_version": "kvbench-phase16-segment-inventory-1.0.0",
                "segment_id": segment_id,
                "terminal_records": len(records),
                "completed_records": sum(
                    record["status"] == "completed" for record in records
                ),
                "capacity_infeasible_records": sum(
                    record["status"] == "capacity_infeasible" for record in records
                ),
                "exclusion_records": len(exclusions),
                "run_kind": "timing",
                "r_hbm": None,
            }
        ),
    )
    return _seal_artifact(
        segment_root,
        run_id=segment_id,
        status="COMPLETE",
        role="phase16_full_scan_segment_evidence",
    )


def _byte_features(completed: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    base = phase13._point_byte_features(completed)
    if not completed:
        return {
            **base,
            "cache_data_bytes": None,
            "scale_zero_bytes": None,
            "norm_bytes": None,
            "peak_memory_bytes": None,
        }
    runner = completed[0]["runner"]
    breakdown = runner["cache_byte_breakdown"]
    configuration = str(completed[0]["method_config_id"])
    family = phase12._method_family(configuration)
    if family == "turboquant":
        scale_zero = int(breakdown["value_scale_metadata_bytes"]) + int(
            breakdown["value_zero_point_metadata_bytes"]
        )
        norm = int(breakdown["key_norm_metadata_bytes"])
    elif family == "kivi":
        scale_zero = sum(
            int(breakdown[name])
            for name in (
                "key_scales",
                "key_zero_points",
                "value_scales",
                "value_zero_points",
            )
        )
        norm = 0
    elif family == "kvquant":
        scale_zero = int(breakdown["key_metadata"]) + int(
            breakdown["value_metadata"]
        )
        norm = 0
    else:
        scale_zero = int(breakdown.get("scale_bytes", 0)) + int(
            breakdown.get("zero_point_bytes", 0)
        )
        norm = 0
    peak_values = []
    for record in completed:
        memory = record["runner"].get("memory_evidence")
        if isinstance(memory, Mapping):
            for key in (
                "max_memory_allocated_bytes",
                "peak_memory_bytes",
                "allocated_peak_bytes",
            ):
                if isinstance(memory.get(key), int):
                    peak_values.append(int(memory[key]))
    return {
        **base,
        "cache_data_bytes": base["data_payload_bytes"],
        "scale_zero_bytes": scale_zero,
        "norm_bytes": norm,
        "peak_memory_bytes": max(peak_values) if peak_values else None,
    }


def point_summaries(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[
            (
                str(record["method_config_id"]),
                int(record["batch_size"]),
                int(record["context_label"]),
            )
        ].append(record)
    summaries: list[dict[str, Any]] = []
    for point in logical_points():
        key = (
            str(point["method_config_id"]),
            int(point["batch_size"]),
            int(point["context_label"]),
        )
        matching = grouped[key]
        eligible = [record for record in matching if record["status"] == "completed"]
        if len(matching) != REPLICATES:
            raise Phase16FullScanError("logical point replicate cardinality differs")
        stats = {
            "median_ms": None,
            "mean_ms": None,
            "standard_deviation_ms": None,
            "minimum_ms": None,
            "maximum_ms": None,
            "cv": None,
        }
        agreements = False
        output_agreement = path_agreement = allocation_agreement = False
        finite = no_fallback = allocation_stable = kernel_stable = False
        host_ratio = None
        telemetry = {
            "temperature_min_c": None,
            "temperature_max_c": None,
            "sm_clock_min_mhz": None,
            "sm_clock_max_mhz": None,
            "memory_clock_min_mhz": None,
            "memory_clock_max_mhz": None,
            "power_min_w": None,
            "power_max_w": None,
        }
        if len(eligible) >= 3:
            stats = phase13.point_statistics(
                [float(record["process_median_ms"]) for record in eligible]
            )
            output_agreement = len(
                {str(record["output_checksum"]) for record in eligible}
            ) == 1
            path_agreement = len(
                {str(record["kernel_path_fingerprint"]) for record in eligible}
            ) == 1
            allocation_agreement = len(
                {str(record["allocation_fingerprint"]) for record in eligible}
            ) == 1
            finite = all(record["finite_output"] is True for record in eligible)
            no_fallback = all(
                record["no_backend_fallback"] is True for record in eligible
            )
            allocation_stable = all(
                record["allocation_stable"] is True for record in eligible
            )
            kernel_stable = all(
                record["kernel_path_stable"] is True for record in eligible
            )
            agreements = bool(
                output_agreement
                and path_agreement
                and allocation_agreement
                and finite
                and no_fallback
                and allocation_stable
                and kernel_stable
            )
            host_ratio = statistics.median(
                float(record["host_wall_cuda_event_ratio"]) for record in eligible
            )
            telemetry = {
                "temperature_min_c": min(
                    float(record["temperature_min_c"]) for record in eligible
                ),
                "temperature_max_c": max(
                    float(record["temperature_max_c"]) for record in eligible
                ),
                "sm_clock_min_mhz": min(
                    int(record["sm_clock_min_mhz"]) for record in eligible
                ),
                "sm_clock_max_mhz": max(
                    int(record["sm_clock_max_mhz"]) for record in eligible
                ),
                "memory_clock_min_mhz": min(
                    int(record["memory_clock_min_mhz"]) for record in eligible
                ),
                "memory_clock_max_mhz": max(
                    int(record["memory_clock_max_mhz"]) for record in eligible
                ),
                "power_min_w": min(float(record["power_min_w"]) for record in eligible),
                "power_max_w": max(float(record["power_max_w"]) for record in eligible),
            }
        statuses = [str(record["status"]) for record in matching]
        if all(status == "capacity_infeasible" for status in statuses):
            disposition = "capacity_infeasible"
        elif len(eligible) < 3:
            disposition = "failed"
        elif not agreements:
            disposition = "failed"
        elif float(stats["cv"]) <= CV_THRESHOLD:
            disposition = "stable"
        else:
            disposition = "unstable"
        summaries.append(
            {
                **point,
                "replicate_count": len(matching),
                "completed_replicates": len(eligible),
                "terminal_statuses": statuses,
                **stats,
                "process_medians_ms": [
                    float(record["process_median_ms"])
                    for record in sorted(
                        eligible, key=lambda item: int(item["replicate_index"])
                    )
                ],
                "host_wall_cuda_event_ratio": host_ratio,
                **telemetry,
                **_byte_features(eligible),
                "output_checksum_agreement": output_agreement,
                "kernel_path_agreement": path_agreement,
                "allocation_agreement": allocation_agreement,
                "finite_outputs": finite,
                "no_backend_fallback": no_fallback,
                "allocation_stable": allocation_stable,
                "kernel_path_stable": kernel_stable,
                "disposition": disposition,
                "monotonicity_warning": False,
                "monotonicity_warning_only": True,
                "quality_status": "unvalidated",
                "claim_eligibility": "performance_only",
                "performance_claim_eligible": False,
                "r_hbm": None,
            }
        )
    by_block: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for item in summaries:
        if item["disposition"] == "stable":
            by_block[(str(item["method_config_id"]), int(item["batch_size"]))].append(item)
    for block in by_block.values():
        ordered = sorted(block, key=lambda item: int(item["historical_context"]))
        prior = None
        for item in ordered:
            current = float(item["median_ms"])
            if prior is not None and current < prior:
                item["monotonicity_warning"] = True
            prior = current
    return summaries


def same_work_ratios(summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (
            str(item["method_config_id"]),
            int(item["batch_size"]),
            int(item["context_label"]),
        ): item
        for item in summaries
    }
    rows: list[dict[str, Any]] = []
    for method in summaries:
        configuration = str(method["method_config_id"])
        if configuration == "bf16":
            continue
        key = (
            "bf16",
            int(method["batch_size"]),
            int(method["context_label"]),
        )
        baseline = indexed.get(key)
        same_work = bool(
            baseline is not None
            and baseline["historical_context"] == method["historical_context"]
            and baseline["total_attended_context"] == method["total_attended_context"]
        )
        eligible = bool(
            same_work
            and baseline["disposition"] == "stable"
            and method["disposition"] == "stable"
        )
        reason = None
        if not same_work:
            reason = "exact_same_work_bf16_observation_absent"
        elif baseline["disposition"] != "stable":
            reason = f"bf16_{baseline['disposition']}"
        elif method["disposition"] != "stable":
            reason = f"method_{method['disposition']}"
        rows.append(
            {
                "method_config_id": configuration,
                "method_config_fingerprint": method["method_config_fingerprint"],
                "batch_size": method["batch_size"],
                "context_label": method["context_label"],
                "historical_context": method["historical_context"],
                "total_attended_context": method["total_attended_context"],
                "runner_kind": "fixed_l",
                "graph_mode": "cuda_graph",
                "warmup_steps": WARMUP_STEPS,
                "measured_steps": MEASURED_STEPS,
                "process_replicates": REPLICATES,
                "performance_only_ratio": (
                    float(baseline["median_ms"]) / float(method["median_ms"])
                    if eligible
                    else None
                ),
                "calculated": eligible,
                "null_reason": reason,
                "quality_status": "unvalidated",
                "performance_claim_eligible": False,
                "claim_class": "performance_only",
                "not_quality_preserving_speedup": True,
            }
        )
    return rows


def capacity_amplification_records(
    summaries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    _configure_reused_phase13()
    rows: list[dict[str, Any]] = []
    for item in summaries:
        configuration = str(item["method_config_id"])
        if configuration == "bf16":
            continue
        method_feasibility = phase13.feasibility_record(dict(item))
        bf16_record = dict(item)
        bf16_record["method_config_id"] = "bf16"
        bf16_record["method_config_fingerprint"] = CONFIG_FINGERPRINTS["bf16"]
        bf16_feasibility = phase13.feasibility_record(bf16_record)
        if (
            method_feasibility["status"] == "feasible"
            and bf16_feasibility["status"] == "capacity_infeasible"
        ):
            rows.append(
                {
                    "method_config_id": configuration,
                    "method_config_fingerprint": item["method_config_fingerprint"],
                    "batch_size": item["batch_size"],
                    "context_label": item["context_label"],
                    "historical_context": item["historical_context"],
                    "bf16_predicted_bytes": bf16_feasibility[
                        "predicted_required_bytes"
                    ],
                    "method_predicted_bytes": method_feasibility[
                        "predicted_required_bytes"
                    ],
                    "memory_limit_bytes": method_feasibility["limit_bytes"],
                    "method_rho_alloc": item["rho_alloc"],
                    "method_r_alloc": item["r_alloc"],
                    "bf16_infeasible_reason": bf16_feasibility["reason"],
                    "latency_ratio": None,
                    "claim_class": "capacity_amplification",
                    "quality_status": "unvalidated",
                    "performance_claim_eligible": False,
                }
            )
    return rows


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as error:
        raise Phase16FullScanError("pinned analysis image is required") from error
    return [dict(item) for item in pq.read_table(path).to_pylist()]


def phase15_feature_join() -> list[dict[str, Any]]:
    if validate_local_artifact(PHASE15_ARTIFACT, environ={}).root_sha256 != PHASE15_ROOT:
        raise Phase16FullScanError("Phase 15 local profiler authority differs")
    source_rows: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for table in (
        "hbm_ratios.parquet",
        "traffic_amplification.parquet",
        "decode_traffic.parquet",
    ):
        for row in _read_parquet(PHASE15_ARTIFACT / table):
            configuration = row.get("method_config_id", row.get("configuration"))
            if configuration in CONFIGURATIONS:
                source_rows[str(configuration)][table].append(row)
    rows: list[dict[str, Any]] = []
    for configuration in CONFIGURATIONS:
        tables = source_rows.get(configuration, {})
        rows.append(
            {
                "method_config_id": configuration,
                "method_config_fingerprint": CONFIG_FINGERPRINTS[configuration],
                "batch_size": 1,
                "historical_context": 131071,
                "total_attended_context": 131072,
                "graph_mode": "cuda_graph",
                "phase15_r2_root": PHASE15_ROOT,
                "feature_scope": "phase15_common_point_only",
                "hbm_ratio_records": tables.get("hbm_ratios.parquet", []),
                "traffic_amplification_records": tables.get(
                    "traffic_amplification.parquet", []
                ),
                "decode_traffic_records": tables.get("decode_traffic.parquet", []),
                "extrapolation_permitted": False,
            }
        )
    return rows


def _publication_record(family_root: Path, replicate: int) -> dict[str, Any]:
    publish_path = family_root / "publication" / f"replicate-{replicate}.publish.json"
    verify_path = family_root / "publication" / f"replicate-{replicate}.verify.json"
    publish = _strict_json(publish_path)
    verify = _strict_json(verify_path)
    if publish.get("status") != "PASS" or verify.get("status") != "PASS":
        raise Phase16FullScanError("segment R2 publication or retrieval differs")
    publication = publish.get("publish")
    retrieval = verify.get("verify")
    if (
        not isinstance(publication, Mapping)
        or not isinstance(retrieval, Mapping)
        or publication.get("root_sha256") != retrieval.get("root_sha256")
        or publication.get("complete_last") is not True
    ):
        raise Phase16FullScanError("segment R2 receipt identity differs")
    return {
        "replicate_index": replicate,
        "segment_id": _segment_id(family_root.name, replicate),
        "local_path": f"segments/replicate-{replicate}",
        "root_sha256": publication["root_sha256"],
        "r2_uri": publication["uri"],
        "object_count": publication["object_count"],
        "complete_last": True,
        "clean_retrieval": True,
        "publish_receipt_sha256": sha256_file(publish_path),
        "verify_receipt_sha256": sha256_file(verify_path),
        "bucket_lock": publish.get("bucket_lock"),
    }


def _render_plots(
    root: Path,
    summaries: Sequence[Mapping[str, Any]],
    ratios: Sequence[Mapping[str, Any]],
    capacity: Sequence[Mapping[str, Any]],
    profiler: Sequence[Mapping[str, Any]],
) -> None:
    plots = root / "plots"
    plots.mkdir()
    stable = [item for item in summaries if item["disposition"] == "stable"]
    phase13._svg_line_plot(
        plots / "timing-by-configuration-and-batch.svg",
        title="Full Scan T(L) by configuration and batch",
        y_label="median CUDA ms",
        series={
            f"{configuration}/B{batch}": [
                (float(item["historical_context"]), float(item["median_ms"]))
                for item in stable
                if item["method_config_id"] == configuration
                and item["batch_size"] == batch
            ]
            for configuration in CONFIGURATIONS
            for batch in BATCH_SIZES
        },
        note="Performance-only; quality unvalidated; not claim eligible",
    )
    phase13._svg_line_plot(
        plots / "method-configurations.svg",
        title="Per-method configuration comparison at B=1",
        y_label="median CUDA ms",
        series={
            configuration: [
                (float(item["historical_context"]), float(item["median_ms"]))
                for item in stable
                if item["method_config_id"] == configuration and item["batch_size"] == 1
            ]
            for configuration in CONFIGURATIONS
        },
    )
    phase13._svg_line_plot(
        plots / "cv-versus-context.svg",
        title="CV versus context",
        y_label="CV",
        series={
            f"{configuration}/B{batch}": [
                (float(item["historical_context"]), float(item["cv"]))
                for item in summaries
                if item["method_config_id"] == configuration
                and item["batch_size"] == batch
                and item["cv"] is not None
            ]
            for configuration in CONFIGURATIONS
            for batch in BATCH_SIZES
        },
    )
    phase13._svg_line_plot(
        plots / "allocated-ratios.svg",
        title="rho_alloc and r_alloc versus context at B=1",
        y_label="ratio",
        series={
            f"{configuration}/{field}": [
                (float(item["historical_context"]), float(item[field]))
                for item in summaries
                if item["method_config_id"] == configuration
                and item["batch_size"] == 1
                and item[field] is not None
            ]
            for configuration in CONFIGURATIONS
            for field in ("rho_alloc", "r_alloc")
        },
    )
    phase13._svg_line_plot(
        plots / "performance-only-ratio.svg",
        title="Performance-only same-work ratio",
        y_label="BF16 / method median",
        series={
            configuration: [
                (float(item["historical_context"]), float(item["performance_only_ratio"]))
                for item in ratios
                if item["method_config_id"] == configuration
                and item["performance_only_ratio"] is not None
            ]
            for configuration in CONFIGURATIONS
            if configuration != "bf16"
        },
        note="Performance-only ratio; quality unvalidated; not a quality-preserving speedup",
    )
    phase13._svg_plot(
        plots / "capacity-feasible-region.svg",
        title="Capacity-feasible region",
        message=(
            f"{sum(item['disposition'] != 'capacity_infeasible' for item in summaries)} "
            f"feasible and {sum(item['disposition'] == 'capacity_infeasible' for item in summaries)} "
            f"capacity-infeasible logical points; {len(capacity)} capacity-amplification rows."
        ),
    )
    phase13._svg_line_plot(
        plots / "byte-components.svg",
        title="Metadata, residual, sink, and outlier bytes",
        y_label="bytes",
        series={
            field: [
                (float(item["historical_context"]), float(item[field]))
                for item in summaries
                if item["method_config_id"] == "kvq4"
                and item["batch_size"] == 1
                and item[field] is not None
            ]
            for field in (
                "metadata_bytes",
                "residual_bytes",
                "sink_bytes",
                "outlier_value_bytes",
                "outlier_index_bytes",
            )
        },
    )
    phase13._svg_plot(
        plots / "phase15-r-hbm-vs-r-alloc.svg",
        title="Phase 15 common-point r_hbm versus r_alloc",
        message=(
            f"{len(profiler)} configuration-scoped references at B=1, L=131071 only; "
            "no extrapolation over Full Scan B or L."
        ),
    )


def _coverage_status(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    sufficient = True
    for configuration in CONFIGURATIONS:
        for batch in BATCH_SIZES:
            matching = [
                item
                for item in summaries
                if item["method_config_id"] == configuration
                and item["batch_size"] == batch
            ]
            feasible = [
                item for item in matching if item["disposition"] != "capacity_infeasible"
            ]
            stable = [item for item in feasible if item["disposition"] == "stable"]
            required = min(3, len(feasible))
            passed = len(stable) >= required and required > 0
            sufficient = sufficient and passed
            rows.append(
                {
                    "method_config_id": configuration,
                    "batch_size": batch,
                    "feasible_logical_points": len(feasible),
                    "stable_logical_points": len(stable),
                    "required_stable_logical_points": required,
                    "phase17_coverage_sufficient": passed,
                }
            )
    return {"sufficient": sufficient, "rows": rows}


def materialize_outer(family_root: Path, *, git_sha: str) -> dict[str, Any]:
    if family_root.name == "outer" or _FAMILY_RE.fullmatch(family_root.name) is None:
        raise Phase16FullScanError("Full Scan family root differs")
    observed_head = subprocess.run(
        ("/usr/bin/git", "rev-parse", "HEAD"),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed_head != git_sha:
        raise Phase16FullScanError("outer analysis source differs")
    segment_index = [_publication_record(family_root, index) for index in range(5)]
    all_records: list[dict[str, Any]] = []
    for index, segment in enumerate(segment_index):
        root = family_root / "segments" / f"replicate-{index}"
        artifact = validate_local_artifact(root, environ={})
        if artifact.root_sha256 != segment["root_sha256"]:
            raise Phase16FullScanError("local segment root differs from R2 receipt")
        all_records.extend(_segment_records(root))
    if len(all_records) != PLANNED_PROCESS_RECORDS:
        raise Phase16FullScanError("Full Scan process record cardinality differs")
    summaries = point_summaries(all_records)
    ratios = same_work_ratios(summaries)
    capacity = capacity_amplification_records(summaries)
    profiler = phase15_feature_join()
    exclusions = [
        {
            "method_config_id": item["method_config_id"],
            "batch_size": item["batch_size"],
            "context_label": item["context_label"],
            "disposition": item["disposition"],
            "machine_readable_reason": (
                "capacity_infeasible"
                if item["disposition"] == "capacity_infeasible"
                else item["disposition"]
            ),
        }
        for item in summaries
        if item["disposition"] != "stable"
    ]
    coverage = _coverage_status(summaries)
    statuses: dict[str, int] = defaultdict(int)
    for record in all_records:
        statuses[str(record["status"])] += 1
    stable = sum(item["disposition"] == "stable" for item in summaries)
    unstable = sum(item["disposition"] == "unstable" for item in summaries)
    failed = sum(item["disposition"] == "failed" for item in summaries)
    capacity_points = sum(
        item["disposition"] == "capacity_infeasible" for item in summaries
    )
    mismatches = {
        "output": sum(
            item["completed_replicates"] >= 3
            and item["output_checksum_agreement"] is not True
            for item in summaries
            if item["disposition"] != "capacity_infeasible"
        ),
        "kernel_path": sum(
            item["completed_replicates"] >= 3
            and item["kernel_path_agreement"] is not True
            for item in summaries
            if item["disposition"] != "capacity_infeasible"
        ),
        "allocation": sum(
            item["completed_replicates"] >= 3
            and item["allocation_agreement"] is not True
            for item in summaries
            if item["disposition"] != "capacity_infeasible"
        ),
    }
    phase_status = "PASS" if coverage["sufficient"] else "PARTIAL"
    qc = {
        "schema_version": "kvbench-phase16-full-scan-qc-1.0.0",
        "family_id": family_root.name,
        "execution_git_sha": git_sha,
        "status": phase_status,
        "base_logical_points": BASE_LOGICAL_POINTS,
        "adaptive_logical_points": ADAPTIVE_LOGICAL_POINTS,
        "logical_points": LOGICAL_POINTS,
        "planned_process_records": PLANNED_PROCESS_RECORDS,
        "status_counts": dict(sorted(statuses.items())),
        "stable_logical_points": stable,
        "unstable_logical_points": unstable,
        "failed_logical_points": failed,
        "capacity_infeasible_logical_points": capacity_points,
        "maximum_cv": max(
            (float(item["cv"]) for item in summaries if item["cv"] is not None),
            default=None,
        ),
        "consistency_mismatches": mismatches,
        "phase17_coverage": coverage,
        "same_work_ratio_count": sum(item["calculated"] for item in ratios),
        "capacity_amplification_count": len(capacity),
        "phase15_feature_join_count": len(profiler),
        "phase15_feature_scope": "phase15_common_point_only",
        "timing_rows_r_hbm_null": all(item["r_hbm"] is None for item in all_records),
        "selective_reruns": 0,
        "quality_execution": "LOCKED",
        "performance_data_frozen": False,
        "phase17": "READY" if phase_status == "PASS" else "NOT_READY",
    }
    outer = family_root / "outer-stage"
    if any(outer.iterdir()):
        raise Phase16FullScanError("outer stage is not empty")
    phase13._parquet_rows(
        outer / "base_grid.parquet",
        [item for item in logical_points() if item["grid_source"] == "base"],
    )
    phase13._parquet_rows(outer / "adaptive_grid.parquet", adaptive_points())
    phase13._parquet_rows(outer / "feasibility.parquet", feasibility_records())
    phase13._parquet_rows(outer / "segment_index.parquet", segment_index)
    phase13._parquet_rows(outer / "raw_run_index.parquet", all_records)
    phase13._parquet_rows(outer / "point_summary.parquet", summaries)
    phase13._parquet_rows(outer / "same_work_ratios.parquet", ratios)
    phase13._parquet_rows(outer / "capacity_amplification.parquet", capacity)
    phase13._parquet_rows(outer / "profiler_feature_join.parquet", profiler)
    phase13._parquet_rows(outer / "exclusions.parquet", exclusions)
    write_exclusive(outer / "full_scan_qc.json", json_bytes(qc))
    write_exclusive(
        outer / "family_manifest.json",
        json_bytes(
            {
                "schema_version": FAMILY_SCHEMA,
                "family_id": family_root.name,
                "execution_git_sha": git_sha,
                "authorized_container_digest": PHASE16G_CONTAINER_DIGEST,
                "decision": "0039",
                "prefix_schema": PHASE16G_PREFIX_SCHEMA,
                "configurations": list(CONFIGURATIONS),
                "method_fingerprints": CONFIG_FINGERPRINTS,
                "batches": list(BATCH_SIZES),
                "base_context_labels": list(BASE_CONTEXT_LABELS),
                "warmup_steps": WARMUP_STEPS,
                "measured_steps": MEASURED_STEPS,
                "measured_batches": MEASURED_BATCHES,
                "replicates": REPLICATES,
                "seeds": list(SEEDS),
                "orders_sha256": _strict_json(ORDER_PATH)["orders_sha256"],
                "timing_critical_hashes": timing_critical_hashes(),
                "segment_roots": segment_index,
                "raw_segment_objects_duplicated": False,
                "run_kind": "timing",
                "quality_status": "unvalidated",
                "performance_claim_eligible": False,
                "r_hbm": None,
            }
        ),
    )
    report = (
        "# Phase 16 Full Scan\n\n"
        f"Status: {phase_status}\n\n"
        f"Campaign family: `{family_root.name}`\n\n"
        f"The frozen dataset contains {PLANNED_PROCESS_RECORDS} terminal process "
        f"records across {LOGICAL_POINTS} logical points: {stable} stable, "
        f"{unstable} unstable, {failed} failed, and {capacity_points} "
        "capacity-infeasible. Timing remains performance-only and quality "
        "unvalidated; ordinary timing rows keep `r_hbm=null`.\n\n"
        f"Phase 17: {'READY' if phase_status == 'PASS' else 'NOT READY'}  \n"
        "Quality: LOCKED  \n"
        "PERFORMANCE_DATA_FROZEN: absent\n"
    )
    write_exclusive(outer / "full_scan_report.md", report.encode("utf-8"))
    _render_plots(outer, summaries, ratios, capacity, profiler)
    write_exclusive(
        outer / "inventory.json",
        json_bytes(
            {
                "schema_version": "kvbench-phase16-outer-inventory-1.0.0",
                "family_id": family_root.name,
                "segment_count": REPLICATES,
                "segment_objects_duplicated": False,
                "logical_points": LOGICAL_POINTS,
                "process_records": PLANNED_PROCESS_RECORDS,
                "plot_count": 8,
                "quality_status": "unvalidated",
                "r_hbm": None,
            }
        ),
    )
    sealed = _seal_artifact(
        outer,
        run_id=f"{family_root.name}-outer",
        status=phase_status,
        role="phase16_full_scan_outer_evidence",
    )
    final = family_root / "outer"
    if final.exists() or final.is_symlink():
        raise Phase16FullScanError("final outer bundle already exists")
    os.rename(outer, final)
    sealed["path"] = str(final)
    sealed["qc"] = qc
    return sealed


def validate_full_scan(outer: Path) -> dict[str, Any]:
    resolved = outer.resolve(strict=True)
    if resolved.name != "outer" or _FAMILY_RE.fullmatch(resolved.parent.name) is None:
        raise Phase16FullScanError("Full Scan outer path differs")
    artifact = validate_local_artifact(resolved, environ={})
    family = _strict_json(resolved / "family_manifest.json")
    qc = _strict_json(resolved / "full_scan_qc.json")
    if (
        family.get("family_id") != resolved.parent.name
        or family.get("authorized_container_digest") != PHASE16G_CONTAINER_DIGEST
        or family.get("decision") != "0039"
        or family.get("prefix_schema") != PHASE16G_PREFIX_SCHEMA
        or family.get("configurations") != list(CONFIGURATIONS)
        or family.get("batches") != list(BATCH_SIZES)
        or family.get("seeds") != list(SEEDS)
        or family.get("timing_critical_hashes") != timing_critical_hashes()
        or family.get("raw_segment_objects_duplicated") is not False
        or qc.get("planned_process_records") != PLANNED_PROCESS_RECORDS
        or qc.get("timing_rows_r_hbm_null") is not True
        or qc.get("quality_execution") != "LOCKED"
        or qc.get("performance_data_frozen") is not False
    ):
        raise Phase16FullScanError("Full Scan outer authority differs")
    family_root = resolved.parent
    segment_roots = family.get("segment_roots")
    if not isinstance(segment_roots, list) or len(segment_roots) != REPLICATES:
        raise Phase16FullScanError("Full Scan segment references differ")
    terminal = 0
    for index, record in enumerate(segment_roots):
        if not isinstance(record, Mapping) or record.get("replicate_index") != index:
            raise Phase16FullScanError("Full Scan segment index differs")
        segment = family_root / "segments" / f"replicate-{index}"
        segment_artifact = validate_local_artifact(segment, environ={})
        if segment_artifact.root_sha256 != record.get("root_sha256"):
            raise Phase16FullScanError("Full Scan segment root differs")
        rows = _segment_records(segment)
        if len(rows) != LOGICAL_POINTS:
            raise Phase16FullScanError("Full Scan segment terminal count differs")
        terminal += len(rows)
    if terminal != PLANNED_PROCESS_RECORDS:
        raise Phase16FullScanError("Full Scan terminal process count differs")
    validate_execution_orders(_strict_json(ORDER_PATH))
    return {
        "status": "PASS",
        "phase16_status": qc["status"],
        "family_id": family["family_id"],
        "outer_root_sha256": artifact.root_sha256,
        "segment_count": REPLICATES,
        "terminal_process_records": terminal,
        "logical_points": LOGICAL_POINTS,
        "phase17": qc["phase17"],
        "quality_execution": "LOCKED",
        "performance_data_frozen": False,
    }


def validate_plan() -> dict[str, Any]:
    if not PLAN_PATH.is_file() or PLAN_PATH.is_symlink():
        raise Phase16FullScanError("Phase 16 plan is absent")
    orders = validate_execution_orders(_strict_json(ORDER_PATH))
    authority = load_phase16g_authority()
    feasibility = feasibility_records()
    prefixes = prefix_index(container_paths=False)
    return {
        "status": "PASS",
        "configurations": len(CONFIGURATIONS),
        "batches": list(BATCH_SIZES),
        "base_logical_points": BASE_LOGICAL_POINTS,
        "adaptive_logical_points": ADAPTIVE_LOGICAL_POINTS,
        "logical_points": LOGICAL_POINTS,
        "planned_process_records": PLANNED_PROCESS_RECORDS,
        "feasible_logical_points": sum(
            item["status"] == "feasible" for item in feasibility
        ),
        "capacity_infeasible_logical_points": sum(
            item["status"] == "capacity_infeasible" for item in feasibility
        ),
        "prefix_snapshots_available": len(prefixes),
        "direct_prefix_logical_points": sum(
            item["status"] == "feasible"
            and (
                str(item["method_config_id"]),
                int(item["batch_size"]),
                int(item["context_label"]),
            )
            not in prefixes
            for item in feasibility
        ),
        "orders_sha256": orders["orders_sha256"],
        "decision": authority["transition"]["decision"],
        "container": PHASE16G_CONTAINER_DIGEST,
    }


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    operations = parser.add_mutually_exclusive_group(required=True)
    operations.add_argument("--write-execution-orders", action="store_true")
    operations.add_argument("--validate-plan", action="store_true")
    operations.add_argument("--new-family-id", action="store_true")
    operations.add_argument("--reserve-family", action="store_true")
    operations.add_argument("--run-segment", action="store_true")
    operations.add_argument("--run-worker", action="store_true")
    operations.add_argument("--finalize-segment", action="store_true")
    operations.add_argument("--materialize-outer", action="store_true")
    operations.add_argument("--validate-full-scan", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--git-sha")
    parser.add_argument("--family-id")
    parser.add_argument("--family-root", type=Path)
    parser.add_argument("--replicate", type=int)
    parser.add_argument("--segment-root", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--record", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--prefix-entry", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_arguments(argv)
    if args.write_execution_orders:
        if args.output is None:
            raise Phase16FullScanError("execution-order output is required")
        write_exclusive(args.output, json_bytes(derive_execution_orders()))
        print(args.output)
        return 0
    if args.validate_plan:
        print(json.dumps(validate_plan(), sort_keys=True))
        return 0
    if args.new_family_id:
        if args.git_sha is None:
            raise Phase16FullScanError("Git SHA is required")
        print(new_family_id(args.git_sha))
        return 0
    if args.reserve_family:
        if args.family_id is None or args.git_sha is None:
            raise Phase16FullScanError("family ID and Git SHA are required")
        print(reserve_family(family_id=args.family_id, git_sha=args.git_sha))
        return 0
    if args.run_segment:
        if args.family_root is None or args.replicate is None or args.git_sha is None:
            raise Phase16FullScanError("segment arguments are required")
        print(
            json.dumps(
                run_segment(
                    family_root=args.family_root,
                    replicate=args.replicate,
                    git_sha=args.git_sha,
                ),
                sort_keys=True,
            )
        )
        return 0
    if args.run_worker:
        if any(
            value is None
            for value in (
                args.run_id,
                args.record,
                args.git_sha,
                args.run_root,
                args.prefix_entry,
            )
        ):
            raise Phase16FullScanError("worker arguments are required")
        payload = run_worker(
            run_id=str(args.run_id),
            record=_strict_json(args.record),
            git_sha=str(args.git_sha),
            run_root=args.run_root,
            entry=_strict_json(args.prefix_entry),
        )
        print(WORKER_PREFIX + json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return 0
    if args.finalize_segment:
        if args.segment_root is None:
            raise Phase16FullScanError("segment root is required")
        print(json.dumps(finalize_segment(args.segment_root), sort_keys=True))
        return 0
    if args.materialize_outer:
        if args.family_root is None or args.git_sha is None:
            raise Phase16FullScanError("outer arguments are required")
        print(
            json.dumps(
                materialize_outer(args.family_root, git_sha=args.git_sha),
                sort_keys=True,
            )
        )
        return 0
    if args.validate_full_scan:
        if args.artifact is None:
            raise Phase16FullScanError("outer artifact is required")
        print(json.dumps(validate_full_scan(args.artifact), sort_keys=True))
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
