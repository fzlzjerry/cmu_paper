"""Preregistered Phase 13D knee densification coordinator and analysis.

This module reuses the admitted Phase 13 fixed-L CUDA-Graph worker, allocation
formula, prefix-state format, timing implementation, and provisional fitting
code.  It owns only the deterministic densification candidate set, an
append-only campaign, QC, and the combined source-index/refit.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import secrets
import shutil
import stat
import statistics
import subprocess
import sys
import time
from typing import Any, NoReturn

from preflight.run_preflight import json_bytes, rename_noreplace, write_exclusive
from kvbench.runtime.artifacts import sha256_file
from kvbench.runtime.process_supervision import run_stage_supervised_command
from scripts.r2_artifact import validate_local_artifact
from scripts import phase12_unified_admission as phase12
from scripts import phase13_pilot as pilot
from scripts.phase13_prefix_state import validate_prefix_state


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPOSITORY_ROOT / "artifacts" / "phase13d"
STAGING_ROOT = ARTIFACT_ROOT / ".kvbench-staging"
SOURCE_CAMPAIGN_ID = "phase13-20260822t150835736582z-4ddd7b17-3a8fb3"
SOURCE_CAMPAIGN_ROOT = REPOSITORY_ROOT / "artifacts" / "phase13" / SOURCE_CAMPAIGN_ID
SOURCE_ROOT_SHA256 = (
    "feb2e5a8ebba8b729c182fc8170107c9acf8128edd3e5618c8f1b90530557531"
)
SOURCE_R2_URI = (
    "r2://kvbench-artifacts/kvbench/sha256/"
    f"{SOURCE_ROOT_SHA256}/"
)
SOURCE_RECEIPT = REPOSITORY_ROOT / "docs/evidence/phase13-successor/r2-publication.json"
BLOCKED_CAMPAIGN_ID = "phase13-20260804t111810342595z-a127b0d1-8649c3"
BLOCKED_CAMPAIGN_ROOT = REPOSITORY_ROOT / "artifacts" / "phase13" / BLOCKED_CAMPAIGN_ID
BLOCKED_CAMPAIGN_ROOT_SHA256 = (
    "c5523a894bc38f41b45b6bdf38ad437fb0b28fc29db4a959b4a592fdd825277b"
)
BLOCKED_CAMPAIGN_OBJECT_COUNT = 3_246
PLAN_PATH = REPOSITORY_ROOT / "docs/plans/phase13d-knee-densification.md"
CANDIDATE_PATH = REPOSITORY_ROOT / "docs/plans/phase13d-candidate-table.json"
ORDER_PATH = REPOSITORY_ROOT / "docs/plans/phase13d-execution-order.json"

CONFIGURATIONS = pilot.CONFIGURATIONS
CONFIG_FINGERPRINTS = pilot.CONFIG_FINGERPRINTS
TARGET_COUNT = 25
MULTIPLIERS = tuple(
    Decimal(value)
    for value in ("0.60", "0.75", "0.90", "1.00", "1.10", "1.25", "1.50")
)
MINIMUM_HISTORICAL_CONTEXT = 4096
MAXIMUM_HISTORICAL_CONTEXT = 131071
ROUNDING_MULTIPLE = 128
SEEDS = (20260823, 20260824, 20260825)
REPLICATES = 3
WARMUP_STEPS = pilot.WARMUP_STEPS
MEASURED_STEPS = pilot.MEASURED_STEPS
MEASURED_BATCHES = pilot.MEASURED_BATCHES
CV_THRESHOLD = pilot.CV_THRESHOLD
AUTHORIZED_CONTAINER_DIGEST = pilot.AUTHORIZED_CONTAINER_DIGEST
GPU_UUID = pilot.GPU_UUID

CANDIDATE_SCHEMA = "kvbench-phase13d-candidate-table-1.0.0"
ORDER_SCHEMA = "kvbench-phase13d-execution-order-1.0.0"
CAMPAIGN_SCHEMA = "kvbench-phase13d-campaign-1.0.0"
RUN_SCHEMA = "kvbench-phase13d-run-1.0.0"
PREFIX_BUILDER_PREFIX = "PHASE13D_PREFIX_BUILDER_RESULT="
WORKER_PREFIX = "PHASE13D_WORKER_RESULT="
_CAMPAIGN_RE = re.compile(
    r"phase13d-[0-9]{8}t[0-9]{12}z-[0-9a-f]{8}-[0-9a-f]{6}\Z"
)
_RUN_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,159}\Z")


class Phase13DError(RuntimeError):
    """The preregistered densification contract failed closed."""


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


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase13DError(f"invalid JSON evidence: {path}") from error
    if not isinstance(value, dict):
        raise Phase13DError(f"JSON evidence is not an object: {path}")
    return value


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ModuleNotFoundError as error:
        raise Phase13DError("pinned pyarrow analysis environment is required") from error
    return [dict(row) for row in parquet.read_table(path).to_pylist()]


def _write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    pilot._parquet_rows(path, rows)


def _validate_source_receipt() -> dict[str, Any]:
    receipt = _strict_json(SOURCE_RECEIPT)
    publication = receipt.get("publication")
    retrieval = receipt.get("clean_retrieval")
    preservation = receipt.get("historical_campaign_preservation")
    if (
        receipt.get("status") != "PASS"
        or receipt.get("schema_version")
        != "kvbench-phase13-successor-r2-publication-1.0.0"
        or receipt.get("artifact_id") != SOURCE_CAMPAIGN_ID
        or receipt.get("clean_retrieval_count") != 1
        or receipt.get("complete_last") is not True
        or not isinstance(publication, Mapping)
        or publication.get("root_sha256") != SOURCE_ROOT_SHA256
        or publication.get("uri") != SOURCE_R2_URI
        or publication.get("status") != "PASS"
        or publication.get("complete_last") is not True
        or publication.get("conditional_writes") is not True
        or publication.get("content_addressed") is not True
        or publication.get("object_count") != 17_384
        or not isinstance(retrieval, Mapping)
        or retrieval.get("root_sha256") != SOURCE_ROOT_SHA256
        or retrieval.get("result") != "PASS"
        or retrieval.get("verification_result") != "PASS"
        or retrieval.get("uri") != SOURCE_R2_URI
        or retrieval.get("object_count") != 17_384
        or retrieval.get("destination_initially_empty") is not True
        or retrieval.get("checksum_ledger_valid") is not True
        or retrieval.get("inventory_valid") is not True
        or retrieval.get("complete_marker_valid") is not True
        or retrieval.get("unexpected_objects") is not False
        or not isinstance(preservation, Mapping)
        or preservation.get("campaign_id") != BLOCKED_CAMPAIGN_ID
        or preservation.get("object_count") != BLOCKED_CAMPAIGN_OBJECT_COUNT
        or preservation.get("preserved") is not True
        or preservation.get("timing_reused") is not False
    ):
        raise Phase13DError("source Phase 13 durable receipt differs")
    return receipt


def validate_source_campaign() -> dict[str, Any]:
    local = pilot.validate_campaign(
        SOURCE_CAMPAIGN_ROOT, expected_campaign_id=SOURCE_CAMPAIGN_ID
    )
    if (
        local.get("status") != "PASS"
        or local.get("root_sha256") != SOURCE_ROOT_SHA256
        or local.get("object_count") != 17_384
    ):
        raise Phase13DError("source Phase 13 campaign differs")
    receipt = _validate_source_receipt()
    blocked = validate_local_artifact(BLOCKED_CAMPAIGN_ROOT, environ={})
    if (
        blocked.root_sha256 != BLOCKED_CAMPAIGN_ROOT_SHA256
        or len(blocked.files) != BLOCKED_CAMPAIGN_OBJECT_COUNT
    ):
        raise Phase13DError("immutable blocked Phase 13 campaign differs")
    qc = _strict_json(SOURCE_CAMPAIGN_ROOT / "pilot_qc.json")
    if (
        qc.get("planned_point_records") != 810
        or qc.get("feasible_gpu_run_records") != 684
        or qc.get("capacity_infeasible_run_records") != 126
        or qc.get("status_counts")
        != {"capacity_infeasible": 126, "completed": 684}
        or qc.get("stable_points") != 228
        or qc.get("unstable_points") != 0
        or qc.get("failed_points") != 0
        or qc.get("knee_density_sufficient") != 5
        or qc.get("knee_density_insufficient") != TARGET_COUNT
        or qc.get("full_scan") != "CLOSED"
        or qc.get("quality_execution") != "LOCKED"
        or qc.get("performance_data_frozen") is not False
    ):
        raise Phase13DError("source Phase 13 QC differs")
    return {
        "status": "PASS",
        "campaign_id": SOURCE_CAMPAIGN_ID,
        "root_sha256": SOURCE_ROOT_SHA256,
        "object_count": local["object_count"],
        "clean_retrieval": receipt["clean_retrieval"]["result"],
        "blocked_campaign_id": BLOCKED_CAMPAIGN_ID,
        "blocked_campaign_root_sha256": BLOCKED_CAMPAIGN_ROOT_SHA256,
        "blocked_campaign_object_count": BLOCKED_CAMPAIGN_OBJECT_COUNT,
    }


def _parse_json_field(value: Any, *, name: str) -> Any:
    if not isinstance(value, str):
        raise Phase13DError(f"source parquet {name} field differs")
    try:
        return json.loads(value)
    except json.JSONDecodeError as error:
        raise Phase13DError(f"source parquet {name} field is invalid") from error


def derive_targets() -> list[dict[str, Any]]:
    rows = _read_parquet(SOURCE_CAMPAIGN_ROOT / "provisional_knees.parquet")
    targets: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        density = _parse_json_field(row.get("density_json"), name="density_json")
        if density.get("assessed") is not True or density.get("sufficient") is True:
            continue
        knee = row.get("L_star")
        if (
            row.get("fit_status") != "knee_observed"
            or not isinstance(knee, (int, float))
            or not math.isfinite(float(knee))
            or float(knee) <= 0
        ):
            raise Phase13DError("densification target lacks an observed finite knee")
        configuration = str(row["method_config_id"])
        batch = int(row["batch_size"])
        if configuration not in CONFIGURATIONS or batch not in pilot.BATCH_SIZES:
            raise Phase13DError("densification target identity differs")
        target_id = f"{configuration}-b{batch}"
        targets.append(
            {
                "target_id": target_id,
                "source_row_index": row_index,
                "method_config_id": configuration,
                "method_config_fingerprint": CONFIG_FINGERPRINTS[configuration],
                "batch_size": batch,
                "original_fit_status": str(row["fit_status"]),
                "original_L_star": float(knee),
                "original_bootstrap_estimable": bool(row["bootstrap_estimable"]),
                "original_bootstrap_knee_lower_95": row[
                    "bootstrap_knee_lower_95"
                ],
                "original_bootstrap_knee_upper_95": row[
                    "bootstrap_knee_upper_95"
                ],
                "original_below_count": int(density["below_count"]),
                "original_near_count": int(density["near_count"]),
                "original_above_count": int(density["above_count"]),
                "original_density_sufficient": False,
                "source_campaign_id": SOURCE_CAMPAIGN_ID,
                "source_root_sha256": SOURCE_ROOT_SHA256,
            }
        )
    targets.sort(
        key=lambda item: (
            CONFIGURATIONS.index(str(item["method_config_id"])),
            int(item["batch_size"]),
        )
    )
    if len(targets) != TARGET_COUNT or len({item["target_id"] for item in targets}) != TARGET_COUNT:
        raise Phase13DError(
            f"source target count differs: expected {TARGET_COUNT}, observed {len(targets)}"
        )
    return targets


def round_context_half_up(raw_context: Decimal) -> int:
    if not raw_context.is_finite() or raw_context <= 0:
        raise Phase13DError("raw densification context is invalid")
    units = (raw_context / Decimal(ROUNDING_MULTIPLE)).quantize(
        Decimal("1"), rounding=ROUND_HALF_UP
    )
    return int(units) * ROUNDING_MULTIPLE


def derive_candidate_generation(
    targets: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    source_summaries = _read_parquet(SOURCE_CAMPAIGN_ROOT / "point_summary.parquet")
    existing: dict[tuple[str, int], set[int]] = defaultdict(set)
    for row in source_summaries:
        existing[(str(row["method_config_id"]), int(row["batch_size"]))].add(
            int(row["context_label"])
        )
    proposals: list[dict[str, Any]] = []
    for target in targets:
        target_id = str(target["target_id"])
        configuration = str(target["method_config_id"])
        batch = int(target["batch_size"])
        knee = Decimal(str(target["original_L_star"]))
        seen: set[int] = set()
        for proposal_index, multiplier in enumerate(MULTIPLIERS):
            raw = knee * multiplier
            rounded = round_context_half_up(raw)
            clamped = min(
                MAXIMUM_HISTORICAL_CONTEXT,
                max(MINIMUM_HISTORICAL_CONTEXT, rounded),
            )
            if clamped != rounded:
                clamp_action = "clamped_to_minimum" if rounded < clamped else "clamped_to_maximum"
            else:
                clamp_action = "none"
            duplicate = clamped in seen
            seen.add(clamped)
            already_existing = clamped in existing[(configuration, batch)]
            outside = not (
                MINIMUM_HISTORICAL_CONTEXT
                <= clamped
                <= MAXIMUM_HISTORICAL_CONTEXT
            )
            included = not duplicate and not already_existing and not outside
            proposals.append(
                {
                    "target_id": target_id,
                    "proposal_index": proposal_index,
                    "method_config_id": configuration,
                    "method_config_fingerprint": CONFIG_FINGERPRINTS[configuration],
                    "batch_size": batch,
                    "provisional_knee": float(knee),
                    "multiplier": format(multiplier, ".2f"),
                    "raw_context": float(raw),
                    "rounded_context": rounded,
                    "historical_context": clamped,
                    "total_attended_context": clamped + 1,
                    "rounding_rule": "nearest_multiple_128_half_up",
                    "clamp_action": clamp_action,
                    "duplicate_status": duplicate,
                    "existing_point_status": already_existing,
                    "outside_valid_range": outside,
                    "final_inclusion_status": included,
                    "exclusion_reason": (
                        None
                        if included
                        else "duplicate_within_target"
                        if duplicate
                        else "existing_source_point"
                        if already_existing
                        else "outside_model_range"
                    ),
                }
            )
    if len(proposals) != TARGET_COUNT * len(MULTIPLIERS):
        raise Phase13DError("candidate proposal cardinality differs")
    included_keys = {
        (
            str(row["target_id"]),
            int(row["historical_context"]),
        )
        for row in proposals
        if row["final_inclusion_status"] is True
    }
    if len(included_keys) != sum(
        row["final_inclusion_status"] is True for row in proposals
    ):
        raise Phase13DError("included candidate identity is duplicated")
    return proposals


def derive_candidate_table() -> dict[str, Any]:
    targets = derive_targets()
    proposals = derive_candidate_generation(targets)
    included = [row for row in proposals if row["final_inclusion_status"] is True]
    payload = {
        "schema_version": CANDIDATE_SCHEMA,
        "source_campaign_id": SOURCE_CAMPAIGN_ID,
        "source_root_sha256": SOURCE_ROOT_SHA256,
        "source_r2_uri": SOURCE_R2_URI,
        "target_count": len(targets),
        "multipliers": [format(value, ".2f") for value in MULTIPLIERS],
        "rounding_rule": "nearest_multiple_128_half_up",
        "rounding_multiple": ROUNDING_MULTIPLE,
        "minimum_historical_context": MINIMUM_HISTORICAL_CONTEXT,
        "maximum_historical_context": MAXIMUM_HISTORICAL_CONTEXT,
        "density_rule": {
            "source": "scripts.phase13_pilot.knee_density",
            "below": "context < 0.75 * L_star",
            "near": "0.75 * L_star <= context <= 1.25 * L_star",
            "above": "context > 1.25 * L_star",
            "sufficient": "at_least_one_stable_context_in_each_region",
        },
        "targets": targets,
        "proposals": proposals,
        "proposal_count": len(proposals),
        "included_candidate_count": len(included),
    }
    payload["targets_sha256"] = _canonical_sha256(targets)
    payload["proposals_sha256"] = _canonical_sha256(proposals)
    return payload


def _validate_candidate_table_structure(candidate_table: Mapping[str, Any]) -> None:
    targets = candidate_table.get("targets")
    proposals = candidate_table.get("proposals")
    if (
        candidate_table.get("schema_version") != CANDIDATE_SCHEMA
        or candidate_table.get("source_campaign_id") != SOURCE_CAMPAIGN_ID
        or candidate_table.get("source_root_sha256") != SOURCE_ROOT_SHA256
        or candidate_table.get("source_r2_uri") != SOURCE_R2_URI
        or candidate_table.get("target_count") != TARGET_COUNT
        or candidate_table.get("multipliers")
        != [format(value, ".2f") for value in MULTIPLIERS]
        or candidate_table.get("rounding_rule")
        != "nearest_multiple_128_half_up"
        or candidate_table.get("rounding_multiple") != ROUNDING_MULTIPLE
        or candidate_table.get("minimum_historical_context")
        != MINIMUM_HISTORICAL_CONTEXT
        or candidate_table.get("maximum_historical_context")
        != MAXIMUM_HISTORICAL_CONTEXT
        or not isinstance(targets, list)
        or len(targets) != TARGET_COUNT
        or not isinstance(proposals, list)
        or len(proposals) != TARGET_COUNT * len(MULTIPLIERS)
        or candidate_table.get("targets_sha256") != _canonical_sha256(targets)
        or candidate_table.get("proposals_sha256") != _canonical_sha256(proposals)
        or candidate_table.get("proposal_count") != len(proposals)
        or candidate_table.get("included_candidate_count")
        != sum(row.get("final_inclusion_status") is True for row in proposals)
    ):
        raise Phase13DError("candidate table structure or digest differs")


def _block_seed(seed: int, target_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{target_id}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


def derive_execution_order(candidate_table: Mapping[str, Any]) -> dict[str, Any]:
    _validate_candidate_table_structure(candidate_table)
    targets = [dict(value) for value in candidate_table["targets"]]
    contexts: dict[str, list[int]] = defaultdict(list)
    for proposal in candidate_table["proposals"]:
        if proposal["final_inclusion_status"] is True:
            contexts[str(proposal["target_id"])].append(
                int(proposal["historical_context"])
            )
    records: list[dict[str, Any]] = []
    block_orders: list[dict[str, Any]] = []
    for replicate_index, seed in enumerate(SEEDS):
        blocks = [str(target["target_id"]) for target in targets]
        random.Random(seed).shuffle(blocks)
        block_orders.append(
            {
                "replicate_index": replicate_index,
                "seed": seed,
                "target_block_order": blocks,
            }
        )
        global_index = 0
        target_by_id = {str(item["target_id"]): item for item in targets}
        for block_index, target_id in enumerate(blocks):
            target = target_by_id[target_id]
            ordered_contexts = sorted(set(contexts[target_id]))
            random.Random(_block_seed(seed, target_id)).shuffle(ordered_contexts)
            for within_block_index, historical in enumerate(ordered_contexts):
                records.append(
                    {
                        "replicate_index": replicate_index,
                        "seed": seed,
                        "block_index": block_index,
                        "within_block_index": within_block_index,
                        "order_index": global_index,
                        "target_id": target_id,
                        "method_config_id": target["method_config_id"],
                        "method_config_fingerprint": target[
                            "method_config_fingerprint"
                        ],
                        "batch_size": target["batch_size"],
                        "context_label": historical,
                        "historical_context": historical,
                        "total_attended_context": historical + 1,
                        "runner_kind": "fixed_l",
                        "graph_mode": "cuda_graph",
                    }
                )
                global_index += 1
    included_count = int(candidate_table["included_candidate_count"])
    if len(records) != included_count * REPLICATES:
        raise Phase13DError("execution-order cardinality differs")
    identity = {
        (
            row["replicate_index"],
            row["target_id"],
            row["context_label"],
        )
        for row in records
    }
    if len(identity) != len(records):
        raise Phase13DError("execution order contains duplicate run identities")
    payload = {
        "schema_version": ORDER_SCHEMA,
        "source_campaign_id": SOURCE_CAMPAIGN_ID,
        "candidate_table_sha256": _canonical_sha256(candidate_table),
        "seeds": list(SEEDS),
        "blocked_randomization": True,
        "target_config_batch_blocks": True,
        "context_order_within_block_randomized": True,
        "block_orders": block_orders,
        "planned_run_records": len(records),
        "records": records,
    }
    payload["records_sha256"] = _canonical_sha256(records)
    return payload


def validate_preregistration(*, replay_source: bool = True) -> dict[str, Any]:
    candidate = _strict_json(CANDIDATE_PATH)
    _validate_candidate_table_structure(candidate)
    if replay_source and candidate != derive_candidate_table():
        raise Phase13DError("committed Phase 13D candidate table differs")
    order = _strict_json(ORDER_PATH)
    if order != derive_execution_order(candidate):
        raise Phase13DError("committed Phase 13D execution order differs")
    return {
        "status": "PASS",
        "targets": candidate["target_count"],
        "proposals": candidate["proposal_count"],
        "included_candidates": candidate["included_candidate_count"],
        "planned_run_records": order["planned_run_records"],
        "seeds": order["seeds"],
    }


def write_preregistration(*, candidate_output: Path, order_output: Path) -> dict[str, Any]:
    if candidate_output.exists() or order_output.exists():
        raise Phase13DError("Phase 13D preregistration output already exists")
    candidate = derive_candidate_table()
    order = derive_execution_order(candidate)
    write_exclusive(candidate_output, json_bytes(candidate))
    write_exclusive(order_output, json_bytes(order))
    return {
        "status": "PASS",
        "target_count": candidate["target_count"],
        "proposal_count": candidate["proposal_count"],
        "included_candidate_count": candidate["included_candidate_count"],
        "planned_run_records": order["planned_run_records"],
    }


def _candidate_contexts() -> tuple[int, ...]:
    table = _strict_json(CANDIDATE_PATH)
    values = sorted(
        {
            int(row["historical_context"])
            for row in table["proposals"]
            if row["final_inclusion_status"] is True
        }
    )
    return tuple(values)


def _configure_pilot_contexts() -> None:
    pilot.CONTEXT_LABELS = _candidate_contexts()


def feasibility_records(order: Mapping[str, Any]) -> list[dict[str, Any]]:
    expected = derive_execution_order(_strict_json(CANDIDATE_PATH))
    if dict(order) != expected:
        raise Phase13DError("Phase 13D feasibility order differs")
    records = [pilot.feasibility_record(record) for record in order["records"]]
    if len(records) != int(order["planned_run_records"]):
        raise Phase13DError("Phase 13D feasibility cardinality differs")
    return records


def new_campaign_id(git_sha: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", git_sha) is None:
        raise Phase13DError("execution Git SHA is invalid")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%S%f")[:21]
    return f"phase13d-{stamp}z-{git_sha[:8]}-{secrets.token_hex(3)}"


def _validate_campaign_id(value: str) -> str:
    if _CAMPAIGN_RE.fullmatch(value) is None:
        raise Phase13DError("Phase 13D campaign ID is invalid")
    return value


def reserve_campaign(*, campaign_id: str, git_sha: str) -> Path:
    identifier = _validate_campaign_id(campaign_id)
    STAGING_ROOT.mkdir(parents=True, exist_ok=True)
    stage = STAGING_ROOT / f"{identifier}.{secrets.token_hex(4)}.staging"
    stage.mkdir()
    for relative in ("runs", "unified", "plots"):
        (stage / relative).mkdir()
    write_exclusive(
        stage / "campaign-reservation.json",
        json_bytes(
            {
                "schema_version": "kvbench-phase13d-reservation-1.0.0",
                "campaign_id": identifier,
                "execution_git_sha": git_sha,
                "reserved_at_utc": _utc_now(),
                "append_only": True,
                "overwrite": False,
            }
        ),
    )
    return stage.resolve(strict=True)


def _require_clean_execution_sha(git_sha: str) -> None:
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
        raise Phase13DError("execution source is not the clean frozen Git SHA")


def _prefix_entry(record: Mapping[str, Any]) -> dict[str, Any]:
    configuration = str(record["method_config_id"])
    batch = int(record["batch_size"])
    historical = int(record["historical_context"])
    return {
        "snapshot_id": f"phase13d-prefix-{configuration}-b{batch}-l{historical}",
        "target_id": str(record["target_id"]),
        "method_config_id": configuration,
        "method_config_fingerprint": CONFIG_FINGERPRINTS[configuration],
        "method_family": phase12._method_family(configuration),
        "context_label": historical,
        "historical_context": historical,
        "capacity": historical + 1,
        "batch_size": batch,
        "source_batch": batch,
        "target_batch": batch,
        "batch_reuse_policy": "exact_target_batch_only",
        "cross_batch_reuse": False,
    }


def prefix_catalog_plan(
    feasibility: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    unique: dict[tuple[str, int, int], dict[str, Any]] = {}
    for record in feasibility:
        if record["status"] != "feasible":
            continue
        key = (
            str(record["method_config_id"]),
            int(record["batch_size"]),
            int(record["historical_context"]),
        )
        entry = _prefix_entry(record)
        previous = unique.setdefault(key, entry)
        if previous != entry:
            raise Phase13DError("prefix catalog identity drifted across replicates")
    entries = sorted(
        unique.values(),
        key=lambda item: (
            CONFIGURATIONS.index(str(item["method_config_id"])),
            int(item["batch_size"]),
            int(item["historical_context"]),
        ),
    )
    candidate = _strict_json(CANDIDATE_PATH)
    feasible_unique = {
        (
            str(row["method_config_id"]),
            int(row["batch_size"]),
            int(row["historical_context"]),
        )
        for row in feasibility
        if row["status"] == "feasible"
    }
    included_unique = {
        (
            str(row["method_config_id"]),
            int(row["batch_size"]),
            int(row["historical_context"]),
        )
        for row in candidate["proposals"]
        if row["final_inclusion_status"] is True
    }
    if not feasible_unique.issubset(included_unique) or len(entries) != len(feasible_unique):
        raise Phase13DError("prefix catalog coverage differs")
    return entries


def _direct_prefix_worker(
    *,
    snapshot_id: str,
    configuration: str,
    source_batch: int,
    context_label: int,
    git_sha: str,
    build_root: Path,
    output: Path,
) -> dict[str, Any]:
    attestation = phase12._require_authorized_container_runtime()
    _configure_pilot_contexts()
    _require_clean_execution_sha(git_sha)

    import torch

    from kvbench.runtime.model_loader import load_frozen_model

    historical = context_label
    recorder = pilot._WorkerStageRecorder(
        root=build_root / "stage-progress",
        run_id=snapshot_id,
        stage_sequence=pilot.PREFIX_BUILD_STAGE_SEQUENCE,
    )
    device = torch.device("cuda:0")
    recorder.record("model_load", "started")
    loaded = load_frozen_model(device=device)
    recorder.record("model_load", "completed")
    recorder.record("prefix_construction", "started")
    torch.cuda.synchronize(device=device)
    torch.cuda.reset_peak_memory_stats(device=device)
    started = time.monotonic()
    chunk_metrics: dict[str, Any] = {}
    session = pilot._direct_session_with_snapshot(
        loaded=loaded,
        configuration=configuration,
        batch=source_batch,
        historical=historical,
        snapshot_root=output,
        abort_after_snapshot=True,
        chunked_kvquant_prefix=configuration in {"kvq4", "kvq3", "kvq2"},
        construction_metrics_output=chunk_metrics,
    )
    if session is not None:
        raise Phase13DError("prefix builder retained a direct session")
    torch.cuda.synchronize(device=device)
    elapsed = time.monotonic() - started
    feasibility = pilot.feasibility_record(
        {
            "method_config_id": configuration,
            "method_config_fingerprint": CONFIG_FINGERPRINTS[configuration],
            "batch_size": source_batch,
            "context_label": historical,
            "historical_context": historical,
            "total_attended_context": historical + 1,
            "target_id": f"{configuration}-b{source_batch}",
        }
    )
    peak_allocated = int(torch.cuda.max_memory_allocated(device=device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device=device))
    limit = int(feasibility["limit_bytes"])
    if (
        feasibility["status"] != "feasible"
        or peak_allocated > limit
        or peak_reserved > limit
    ):
        raise Phase13DError("prefix construction exceeded the frozen 0.88 limit")
    if configuration in {"kvq4", "kvq3", "kvq2"}:
        construction = dict(chunk_metrics)
        if construction.get("mode") != "fixed_bounded_kvquant_chunks":
            raise Phase13DError("KVQuant prefix construction mode differs")
    else:
        construction = {
            "formula_version": "phase13d-frozen-direct-prefix-v1",
            "mode": "frozen_direct_prefill",
            "formal_timing_allocator_changed": False,
            "full_context_fp32_copy": False,
            "elapsed_seconds": elapsed,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "predicted_required_bytes": int(feasibility["predicted_required_bytes"]),
            "memory_limit_bytes": limit,
            "within_frozen_memory_limit": True,
        }
    construction["observed_outer_elapsed_seconds"] = elapsed
    construction["peak_allocated_bytes"] = peak_allocated
    construction["peak_reserved_bytes"] = peak_reserved
    construction["within_frozen_memory_limit"] = True
    recorder.record("prefix_construction", "completed")
    recorder.record("finalization", "started")
    manifest = validate_prefix_state(
        output,
        configuration=configuration,
        family=phase12._method_family(configuration),
        batch=source_batch,
        historical=historical,
        method_config_fingerprint=CONFIG_FINGERPRINTS[configuration],
        verify_state_bytes=True,
    )
    pilot._fsync_prefix_builder_snapshot(output)
    recorder.record("finalization", "completed")
    return {
        "schema_version": "kvbench-phase13d-prefix-builder-result-1.0.0",
        "snapshot_id": snapshot_id,
        "configuration": configuration,
        "context_label": context_label,
        "historical_context": historical,
        "source_batch": source_batch,
        "state_file_sha256": manifest["state_file_sha256"],
        "state_file_bytes": manifest["state_file_bytes"],
        "prefix_construction": construction,
        "container_runtime_attestation": attestation,
    }


def _emit_and_exit(prefix: str, payload: Mapping[str, Any]) -> NoReturn:
    sys.stdout.write(
        prefix
        + json.dumps(dict(payload), sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def _run_prefix_builder_process(
    *, prefix_root: Path, entry: Mapping[str, Any], git_sha: str
) -> dict[str, Any]:
    snapshot_id = str(entry["snapshot_id"])
    build_root = prefix_root / "builds" / snapshot_id
    snapshot_root = prefix_root / "states" / snapshot_id
    if build_root.exists() or snapshot_root.exists() or snapshot_root.is_symlink():
        raise Phase13DError("prefix snapshot ID already exists")
    build_root.mkdir()
    (build_root / "stage-progress").mkdir()
    child_output = build_root / "snapshot"
    pre = phase12._capture_process_snapshot()
    phase12._require_idle_snapshot(pre)
    command = (
        sys.executable,
        str(REPOSITORY_ROOT / "scripts/phase13d_knee_densification.py"),
        "--build-prefix-state",
        "--snapshot-id",
        snapshot_id,
        "--configuration",
        str(entry["method_config_id"]),
        "--source-batch",
        str(entry["source_batch"]),
        "--context-label",
        str(entry["context_label"]),
        "--git-sha",
        git_sha,
        "--build-root",
        str(build_root),
        "--output",
        str(child_output),
    )
    timeouts = pilot.stage_timeout_contract(
        batch=int(entry["source_batch"]),
        historical=int(entry["historical_context"]),
    )
    result = run_stage_supervised_command(
        command,
        working_directory=str(REPOSITORY_ROOT),
        environment=pilot._prefix_builder_child_environment(),
        stage_timeouts=timeouts,
        stage_observer=lambda: pilot._read_stage_observations(
            root=build_root / "stage-progress",
            run_id=snapshot_id,
            stage_sequence=pilot.PREFIX_BUILD_STAGE_SEQUENCE,
        ),
        startup_stage="startup",
        transition_stage="transition",
        observer_poll_seconds=pilot.STAGE_OBSERVER_POLL_SECONDS,
    )
    post = phase12._capture_process_snapshot()
    phase12._require_idle_snapshot(post)
    phase12._write_supervised_command_evidence(
        root=build_root,
        prefix="builder",
        result=result,
        pre_snapshot=pre,
        post_snapshot=post,
    )
    if not pilot._supervision_passed(result):
        raise Phase13DError(f"prefix builder failed: {snapshot_id}")
    matches = [
        line[len(PREFIX_BUILDER_PREFIX) :]
        for line in result.stdout.decode("utf-8", errors="strict").splitlines()
        if line.startswith(PREFIX_BUILDER_PREFIX)
    ]
    if len(matches) != 1:
        raise Phase13DError("prefix builder result channel differs")
    payload = json.loads(matches[0])
    if not isinstance(payload, dict) or payload.get("snapshot_id") != snapshot_id:
        raise Phase13DError("prefix builder result identity differs")
    construction = payload.get("prefix_construction")
    if (
        not isinstance(construction, dict)
        or construction.get("within_frozen_memory_limit") is not True
        or construction.get("full_context_fp32_copy") is not False
        or float(construction.get("observed_outer_elapsed_seconds", 0.0)) <= 0.0
    ):
        raise Phase13DError("prefix construction evidence differs")
    manifest = validate_prefix_state(
        child_output,
        configuration=str(entry["method_config_id"]),
        family=str(entry["method_family"]),
        batch=int(entry["source_batch"]),
        historical=int(entry["historical_context"]),
        method_config_fingerprint=str(entry["method_config_fingerprint"]),
        verify_state_bytes=True,
    )
    if (
        manifest.get("state_file_sha256") != payload.get("state_file_sha256")
        or manifest.get("state_file_bytes") != payload.get("state_file_bytes")
    ):
        raise Phase13DError("prefix snapshot bytes differ after child finalization")
    rename_noreplace(child_output, snapshot_root)
    for path in snapshot_root.iterdir():
        path.chmod(0o444)
    snapshot_root.chmod(0o555)
    return {
        **dict(entry),
        "snapshot_relative_path": f"states/{snapshot_id}",
        "state_file_sha256": manifest["state_file_sha256"],
        "state_file_bytes": manifest["state_file_bytes"],
        "source_layout_fingerprint": manifest["source_layout_fingerprint"],
        "builder_supervision_sha256": sha256_file(
            build_root / "builder.supervision.json"
        ),
        "prefix_construction": construction,
        "state_bytes_verified_before_timing": True,
        "runtime_prefix_sharing": False,
    }


def prepare_prefix_catalog(
    *, prefix_root: Path, feasibility: Sequence[Mapping[str, Any]], git_sha: str
) -> dict[str, Any]:
    if prefix_root.exists() or prefix_root.is_symlink():
        raise Phase13DError("Phase 13D prefix root must be new")
    prefix_root.mkdir(parents=True)
    (prefix_root / "builds").mkdir()
    (prefix_root / "states").mkdir()
    entries = []
    for entry in prefix_catalog_plan(feasibility):
        entries.append(
            _run_prefix_builder_process(
                prefix_root=prefix_root,
                entry=entry,
                git_sha=git_sha,
            )
        )
    payload = {
        "schema_version": "kvbench-phase13d-prefix-catalog-1.0.0",
        "execution_git_sha": git_sha,
        "authorized_container_digest": AUTHORIZED_CONTAINER_DIGEST,
        "snapshot_count": len(entries),
        "formal_process_replicates": REPLICATES,
        "exact_configuration_batch_context_only": True,
        "fresh_caller_owned_cache_per_timing_process": True,
        "restoration_outside_timing": True,
        "runtime_prefix_sharing": False,
        "old_timing_samples_reused": False,
        "entries": entries,
    }
    write_exclusive(prefix_root / "catalog.json", json_bytes(payload))
    return payload


def _prefix_catalog_index(
    catalog: Mapping[str, Any], prefix_root: Path
) -> dict[tuple[str, int, int], dict[str, Any]]:
    values = catalog.get("entries")
    if not isinstance(values, list):
        raise Phase13DError("prefix catalog entries are absent")
    index: dict[tuple[str, int, int], dict[str, Any]] = {}
    for value in values:
        if not isinstance(value, dict):
            raise Phase13DError("prefix catalog entry differs")
        key = (
            str(value["method_config_id"]),
            int(value["batch_size"]),
            int(value["historical_context"]),
        )
        if key in index:
            raise Phase13DError("prefix catalog entry is duplicated")
        state_root = prefix_root / str(value["snapshot_relative_path"])
        if not state_root.is_dir() or state_root.is_symlink():
            raise Phase13DError("prefix catalog state path differs")
        index[key] = {**value, "snapshot_root": state_root}
    return index


def _run_id(campaign_id: str, record: Mapping[str, Any]) -> str:
    value = (
        f"{campaign_id}-r{record['replicate_index']}-"
        f"o{int(record['order_index']):03d}-{record['method_config_id']}-"
        f"b{record['batch_size']}-l{record['context_label']}"
    )
    if _RUN_RE.fullmatch(value) is None:
        raise Phase13DError("Phase 13D run ID is invalid")
    return value


def _status_manifest(
    *,
    run_root: Path,
    campaign_id: str,
    run_id: str,
    record: Mapping[str, Any],
    status: str,
    reason: str | None,
) -> dict[str, Any]:
    manifest = {
        "schema_version": "kvbench-phase13d-run-manifest-1.0.0",
        "campaign_id": campaign_id,
        "run_id": run_id,
        "status": status,
        "reason": reason,
        "target_id": record["target_id"],
        "method_config_id": record["method_config_id"],
        "method_config_fingerprint": record["method_config_fingerprint"],
        "replicate_index": record["replicate_index"],
        "seed": record["seed"],
        "order_index": record["order_index"],
        "batch_size": record["batch_size"],
        "context_label": record["context_label"],
        "historical_context": record["historical_context"],
        "total_attended_context": record["total_attended_context"],
        "runner_kind": "fixed_l",
        "graph_mode": "cuda_graph",
        "warmup_steps": WARMUP_STEPS,
        "measured_steps": MEASURED_STEPS,
        "measured_batches": MEASURED_BATCHES,
        "authorized_container_digest": AUTHORIZED_CONTAINER_DIGEST,
        "selective_rerun": False,
        "quality_status": "unvalidated",
        "performance_claim_eligible": False,
        "r_hbm": None,
    }
    workspace = record.get("q4_value_decode_workspace")
    if record["method_config_id"] == "kvq4":
        if not isinstance(workspace, Mapping):
            raise Phase13DError("q4 densification record lacks workspace geometry")
        manifest["q4_value_decode_workspace"] = dict(workspace)
    write_exclusive(run_root / "manifest.json", json_bytes(manifest))
    return manifest


def _write_nonlaunched(
    *,
    stage: Path,
    campaign_id: str,
    record: Mapping[str, Any],
    status: str,
    reason: str,
) -> dict[str, Any]:
    run_id = _run_id(campaign_id, record)
    run_root = stage / "runs" / run_id
    run_root.mkdir()
    write_exclusive(
        run_root / "disposition.json",
        json_bytes(
            {
                "schema_version": "kvbench-phase13d-nonlaunch-1.0.0",
                "run_id": run_id,
                "status": status,
                "reason": reason,
                "cuda_process_launched": False,
                "record_preserved": True,
            }
        ),
    )
    return _status_manifest(
        run_root=run_root,
        campaign_id=campaign_id,
        run_id=run_id,
        record=record,
        status=status,
        reason=reason,
    )


def _run_worker(
    *,
    run_id: str,
    configuration: str,
    batch: int,
    context_label: int,
    replicate_index: int,
    order_index: int,
    git_sha: str,
    run_artifact_root: Path,
    prefix_state_root: Path,
    prefix_state_sha256: str,
) -> dict[str, Any]:
    _configure_pilot_contexts()
    payload = pilot._run_worker(
        run_id=run_id,
        configuration=configuration,
        batch=batch,
        context_label=context_label,
        replicate_index=replicate_index,
        order_index=order_index,
        git_sha=git_sha,
        run_artifact_root=run_artifact_root,
        prefix_state_root=prefix_state_root,
        prefix_state_sha256=prefix_state_sha256,
    )
    payload["schema_version"] = RUN_SCHEMA
    payload["phase13d_source_campaign_id"] = SOURCE_CAMPAIGN_ID
    return payload


def _run_one_process(
    *,
    stage: Path,
    campaign_id: str,
    record: Mapping[str, Any],
    git_sha: str,
    prefix_entry: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        prefix_entry.get("source_batch") != int(record["batch_size"])
        or prefix_entry.get("target_batch") != int(record["batch_size"])
        or prefix_entry.get("historical_context")
        != int(record["historical_context"])
        or prefix_entry.get("batch_reuse_policy") != "exact_target_batch_only"
        or prefix_entry.get("cross_batch_reuse") is not False
    ):
        raise Phase13DError("formal run prefix geometry differs")
    run_id = _run_id(campaign_id, record)
    run_root = stage / "runs" / run_id
    run_root.mkdir()
    stage_progress = run_root / "stage-progress"
    stage_progress.mkdir()
    write_exclusive(
        run_root / "started.json",
        json_bytes(
            {
                "schema_version": "kvbench-phase13d-run-start-1.0.0",
                "run_id": run_id,
                "started_at_utc": _utc_now(),
                "record": dict(record),
            }
        ),
    )
    pre = phase12._capture_process_snapshot()
    phase12._require_idle_snapshot(pre)
    command = (
        sys.executable,
        str(REPOSITORY_ROOT / "scripts/phase13d_knee_densification.py"),
        "--run-worker",
        "--run-id",
        run_id,
        "--configuration",
        str(record["method_config_id"]),
        "--batch-size",
        str(record["batch_size"]),
        "--context-label",
        str(record["context_label"]),
        "--replicate-index",
        str(record["replicate_index"]),
        "--order-index",
        str(record["order_index"]),
        "--git-sha",
        git_sha,
        "--run-artifact-root",
        str(run_root),
        "--prefix-state-root",
        str(prefix_entry["snapshot_root"]),
        "--prefix-state-sha256",
        str(prefix_entry["state_file_sha256"]),
    )
    timeouts = pilot.stage_timeout_contract(
        batch=int(record["batch_size"]),
        historical=int(record["historical_context"]),
    )
    result = run_stage_supervised_command(
        command,
        working_directory=str(REPOSITORY_ROOT),
        environment=phase12._child_environment(),
        stage_timeouts=timeouts,
        stage_observer=lambda: pilot._read_stage_observations(
            root=stage_progress,
            run_id=run_id,
        ),
        startup_stage="startup",
        transition_stage="transition",
        observer_poll_seconds=pilot.STAGE_OBSERVER_POLL_SECONDS,
    )
    post = phase12._capture_process_snapshot()
    phase12._require_idle_snapshot(post)
    phase12._write_supervised_command_evidence(
        root=run_root,
        prefix="worker",
        result=result,
        pre_snapshot=pre,
        post_snapshot=post,
    )
    if not pilot._supervision_passed(result):
        reason = (
            f"supervisor_stage_timeout:{result.timeout_stage}"
            if result.timeout_stage is not None
            else "supervised_worker_failed"
        )
        write_exclusive(
            run_root / "failure.json",
            json_bytes(
                {
                    "schema_version": "kvbench-phase13d-run-failure-1.0.0",
                    "run_id": run_id,
                    "failure_reason": reason,
                    "worker_returncode": result.returncode,
                    "timeout_stage": result.timeout_stage,
                    "selective_retry_permitted": False,
                    "campaign_preserved": True,
                }
            ),
        )
        return _status_manifest(
            run_root=run_root,
            campaign_id=campaign_id,
            run_id=run_id,
            record=record,
            status="runtime_failed",
            reason=reason,
        )
    matches = [
        line[len(WORKER_PREFIX) :]
        for line in result.stdout.decode("utf-8", errors="strict").splitlines()
        if line.startswith(WORKER_PREFIX)
    ]
    if len(matches) != 1:
        raise Phase13DError("worker result channel differs")
    payload = json.loads(matches[0])
    if not isinstance(payload, dict) or payload.get("run_id") != run_id:
        raise Phase13DError("worker result identity differs")
    write_exclusive(run_root / "result.json", json_bytes(payload))
    manifest = _status_manifest(
        run_root=run_root,
        campaign_id=campaign_id,
        run_id=run_id,
        record=record,
        status="completed",
        reason=None,
    )
    write_exclusive(
        run_root / "result-binding.json",
        json_bytes(
            {
                "schema_version": "kvbench-phase13d-run-result-binding-1.0.0",
                "run_id": run_id,
                "result_sha256": sha256_file(run_root / "result.json"),
            }
        ),
    )
    return manifest


def run_campaign(
    *,
    stage: Path,
    campaign_id: str,
    git_sha: str,
    prefix_root: Path,
) -> dict[str, Any]:
    identifier = _validate_campaign_id(campaign_id)
    phase12._require_authorized_container_runtime()
    if any(name in os.environ for name in pilot._FORBIDDEN_ENVIRONMENT):
        raise Phase13DError("credentials entered the Measurement Container")
    _configure_pilot_contexts()
    _require_clean_execution_sha(git_sha)
    validate_preregistration(replay_source=False)
    resolved = stage.resolve(strict=True)
    if not (resolved / "campaign-reservation.json").is_file():
        raise Phase13DError("campaign reservation is absent")
    order = _strict_json(ORDER_PATH)
    feasibility = feasibility_records(order)
    catalog = prepare_prefix_catalog(
        prefix_root=prefix_root,
        feasibility=feasibility,
        git_sha=git_sha,
    )
    catalog_index = _prefix_catalog_index(catalog, prefix_root)
    published_catalog = {
        **{key: value for key, value in catalog.items() if key != "entries"},
        "entries": [
            {
                key: value
                for key, value in entry.items()
                if key not in {"snapshot_relative_path"}
            }
            for entry in catalog["entries"]
        ],
        "prefix_state_payloads_published": False,
    }
    write_exclusive(
        resolved / "unified" / "prefix-catalog.json",
        json_bytes(published_catalog),
    )
    write_exclusive(resolved / "execution_order.json", json_bytes(order))
    write_exclusive(
        resolved / "unified" / "feasibility.json",
        json_bytes({"records": feasibility}),
    )
    candidate = _strict_json(CANDIDATE_PATH)
    write_exclusive(
        resolved / "campaign_manifest.json",
        json_bytes(
            {
                "schema_version": CAMPAIGN_SCHEMA,
                "campaign_id": identifier,
                "execution_git_sha": git_sha,
                "authorized_container_digest": AUTHORIZED_CONTAINER_DIGEST,
                "source_campaign_id": SOURCE_CAMPAIGN_ID,
                "source_root_sha256": SOURCE_ROOT_SHA256,
                "source_r2_uri": SOURCE_R2_URI,
                "blocked_campaign_id": BLOCKED_CAMPAIGN_ID,
                "blocked_campaign_root_sha256": BLOCKED_CAMPAIGN_ROOT_SHA256,
                "blocked_campaign_object_count": BLOCKED_CAMPAIGN_OBJECT_COUNT,
                "target_count": TARGET_COUNT,
                "included_candidate_count": candidate["included_candidate_count"],
                "planned_run_records": order["planned_run_records"],
                "warmup_steps": WARMUP_STEPS,
                "measured_steps": MEASURED_STEPS,
                "measured_batches": MEASURED_BATCHES,
                "replicates": REPLICATES,
                "seeds": list(SEEDS),
                "runner_kind": "fixed_l",
                "graph_mode": "cuda_graph",
                "candidate_table_sha256": sha256_file(CANDIDATE_PATH),
                "execution_order_sha256": sha256_file(ORDER_PATH),
                "prefix_snapshot_count": len(catalog["entries"]),
                "prefix_restore_outside_timing": True,
                "fresh_caller_owned_cache_per_process": True,
                "runtime_prefix_sharing": False,
                "source_timing_samples_reused": False,
                "selective_reruns": 0,
                "full_scan": "CLOSED",
                "quality_execution": "LOCKED",
                "performance_data_frozen": False,
            }
        ),
    )
    manifests: list[dict[str, Any]] = []
    stop_reason: str | None = None
    for record in feasibility:
        if record["status"] == "capacity_infeasible":
            manifests.append(
                _write_nonlaunched(
                    stage=resolved,
                    campaign_id=identifier,
                    record=record,
                    status="capacity_infeasible",
                    reason=str(record["reason"]),
                )
            )
            continue
        if stop_reason is not None:
            manifests.append(
                _write_nonlaunched(
                    stage=resolved,
                    campaign_id=identifier,
                    record=record,
                    status="aborted",
                    reason=stop_reason,
                )
            )
            continue
        key = (
            str(record["method_config_id"]),
            int(record["batch_size"]),
            int(record["historical_context"]),
        )
        manifest = _run_one_process(
            stage=resolved,
            campaign_id=identifier,
            record=record,
            git_sha=git_sha,
            prefix_entry=catalog_index[key],
        )
        manifests.append(manifest)
        if manifest["status"] == "runtime_failed":
            stop_reason = str(manifest["reason"])
    counts = Counter(str(item["status"]) for item in manifests)
    result = {
        "schema_version": "kvbench-phase13d-local-campaign-1.0.0",
        "campaign_id": identifier,
        "execution_git_sha": git_sha,
        "planned_run_records": order["planned_run_records"],
        "status_counts": dict(sorted(counts.items())),
        "stop_reason": stop_reason,
        "selective_reruns": 0,
        "phase13d_status": "BLOCKED" if stop_reason else "LOCAL_COMPLETE",
        "durable_publication": "PENDING_HOST_SIDE",
        "full_scan": "CLOSED",
        "quality_execution": "LOCKED",
        "performance_data_frozen": False,
        "r_hbm": None,
    }
    write_exclusive(resolved / "unified" / "local-campaign.json", json_bytes(result))
    return result


def _run_records(root: Path, *, expected_count: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for manifest_path in sorted((root / "runs").glob("*/manifest.json")):
        manifest = _strict_json(manifest_path)
        run_root = manifest_path.parent
        result_path = run_root / "result.json"
        result = _strict_json(result_path) if result_path.is_file() else None
        records.append(
            {
                **manifest,
                "manifest_path": manifest_path.relative_to(root).as_posix(),
                "manifest_sha256": sha256_file(manifest_path),
                "result_path": (
                    result_path.relative_to(root).as_posix()
                    if result is not None
                    else None
                ),
                "result_sha256": (
                    sha256_file(result_path) if result is not None else None
                ),
                "process_median_ms": (
                    float(result["process_median_ms"])
                    if result is not None
                    else None
                ),
                "host_wall_cuda_event_ratio": (
                    float(result["host_wall_cuda_event_ratio"])
                    if result is not None
                    else None
                ),
                "kernel_count": result.get("kernel_count") if result else None,
                "finite_output": result.get("finite_output") if result else None,
                "no_backend_fallback": (
                    result.get("no_backend_fallback") if result else None
                ),
                "allocation_stable": (
                    result.get("allocation_stable") if result else None
                ),
                "kernel_path_stable": (
                    result.get("kernel_path_stable") if result else None
                ),
                "gpu_exclusive": result.get("gpu_exclusive") if result else None,
                "output_checksum": result.get("output_checksum") if result else None,
                "kernel_path_fingerprint": (
                    result.get("kernel_path_fingerprint") if result else None
                ),
                "allocation_fingerprint": (
                    result.get("allocation_fingerprint") if result else None
                ),
                "temperature_min_c": (
                    result.get("temperature_min_c") if result else None
                ),
                "temperature_max_c": (
                    result.get("temperature_max_c") if result else None
                ),
                "sm_clock_min_mhz": result.get("sm_clock_min_mhz") if result else None,
                "sm_clock_max_mhz": result.get("sm_clock_max_mhz") if result else None,
                "memory_clock_min_mhz": (
                    result.get("memory_clock_min_mhz") if result else None
                ),
                "memory_clock_max_mhz": (
                    result.get("memory_clock_max_mhz") if result else None
                ),
                "power_min_w": result.get("power_min_w") if result else None,
                "power_max_w": result.get("power_max_w") if result else None,
                "runner": result.get("runner") if result else None,
            }
        )
    if len(records) != expected_count:
        raise Phase13DError("Phase 13D run-record cardinality differs")
    return records


def _new_point_summaries(
    records: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[
            (
                str(record["method_config_id"]),
                int(record["batch_size"]),
                int(record["historical_context"]),
            )
        ].append(record)
    summaries: list[dict[str, Any]] = []
    included = [
        row
        for row in candidate["proposals"]
        if row["final_inclusion_status"] is True
    ]
    for proposal in included:
        configuration = str(proposal["method_config_id"])
        batch = int(proposal["batch_size"])
        historical = int(proposal["historical_context"])
        matching = grouped[(configuration, batch, historical)]
        completed = [item for item in matching if item["status"] == "completed"]
        byte_features = pilot._point_byte_features(completed)
        if len(completed) == REPLICATES:
            values = [float(item["process_median_ms"]) for item in completed]
            stats = pilot.point_statistics(values)
            output_agreement = len(
                {str(item["output_checksum"]) for item in completed}
            ) == 1
            path_agreement = len(
                {str(item["kernel_path_fingerprint"]) for item in completed}
            ) == 1
            allocation_agreement = len(
                {str(item["allocation_fingerprint"]) for item in completed}
            ) == 1
            finite = all(item["finite_output"] is True for item in completed)
            no_fallback = all(
                item["no_backend_fallback"] is True for item in completed
            )
            gpu_exclusive = all(item["gpu_exclusive"] is True for item in completed)
            within_process_stability = all(
                item["allocation_stable"] is True
                and item["kernel_path_stable"] is True
                for item in completed
            )
            agreements = (
                output_agreement
                and path_agreement
                and allocation_agreement
                and finite
                and no_fallback
                and gpu_exclusive
                and within_process_stability
            )
            disposition = pilot.classify_point(
                statistics_record=stats,
                agreements=agreements,
            )
            process_medians = [
                float(item["process_median_ms"])
                for item in sorted(
                    completed, key=lambda value: int(value["replicate_index"])
                )
            ]
            host_ratio = statistics.median(
                float(item["host_wall_cuda_event_ratio"]) for item in completed
            )
            telemetry = {
                "temperature_min_c": min(
                    float(item["temperature_min_c"]) for item in completed
                ),
                "temperature_max_c": max(
                    float(item["temperature_max_c"]) for item in completed
                ),
                "sm_clock_min_mhz": min(
                    int(item["sm_clock_min_mhz"]) for item in completed
                ),
                "sm_clock_max_mhz": max(
                    int(item["sm_clock_max_mhz"]) for item in completed
                ),
                "memory_clock_min_mhz": min(
                    int(item["memory_clock_min_mhz"]) for item in completed
                ),
                "memory_clock_max_mhz": max(
                    int(item["memory_clock_max_mhz"]) for item in completed
                ),
                "power_min_w": min(float(item["power_min_w"]) for item in completed),
                "power_max_w": max(float(item["power_max_w"]) for item in completed),
            }
        else:
            stats = {
                "median_ms": None,
                "mean_ms": None,
                "standard_deviation_ms": None,
                "minimum_ms": None,
                "maximum_ms": None,
                "cv": None,
            }
            output_agreement = path_agreement = allocation_agreement = False
            finite = no_fallback = gpu_exclusive = within_process_stability = False
            process_medians = []
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
            statuses = {str(item["status"]) for item in matching}
            disposition = (
                "capacity_infeasible"
                if statuses == {"capacity_infeasible"}
                else "runtime_failed"
                if "runtime_failed" in statuses
                else "aborted"
            )
        summaries.append(
            {
                "target_id": proposal["target_id"],
                "method_config_id": configuration,
                "method_config_fingerprint": CONFIG_FINGERPRINTS[configuration],
                "batch_size": batch,
                "context_label": historical,
                "historical_context": historical,
                "source_campaign_id": None,
                "densification_campaign": True,
                "replicate_count": len(matching),
                "completed_replicates": len(completed),
                **stats,
                "process_medians_ms": process_medians,
                "host_wall_cuda_event_ratio": host_ratio,
                **telemetry,
                **byte_features,
                "output_checksum_agreement": output_agreement,
                "kernel_path_agreement": path_agreement,
                "allocation_agreement": allocation_agreement,
                "finite_outputs": finite,
                "no_backend_fallback": no_fallback,
                "gpu_exclusive": gpu_exclusive,
                "within_process_stability": within_process_stability,
                "disposition": disposition,
                "monotonicity_warning": False,
                "monotonicity_warning_only": True,
                "quality_status": "unvalidated",
                "performance_claim_eligible": False,
                "r_hbm": None,
            }
        )
    return summaries


def _mark_monotonicity_warnings(
    *,
    source: Sequence[Mapping[str, Any]],
    densified: list[dict[str, Any]],
) -> None:
    """Apply the frozen warning-only monotonicity check to combined contexts."""

    for configuration in CONFIGURATIONS:
        for batch in pilot.BATCH_SIZES:
            available: list[tuple[int, float, dict[str, Any] | None]] = []
            available.extend(
                (
                    int(row["context_label"]),
                    float(row["median_ms"]),
                    None,
                )
                for row in source
                if row["method_config_id"] == configuration
                and int(row["batch_size"]) == batch
                and row["disposition"] in {"stable", "unstable"}
            )
            available.extend(
                (
                    int(row["context_label"]),
                    float(row["median_ms"]),
                    row,
                )
                for row in densified
                if row["method_config_id"] == configuration
                and int(row["batch_size"]) == batch
                and row["disposition"] in {"stable", "unstable"}
            )
            available.sort(key=lambda item: item[0])
            previous: float | None = None
            for _, current, new_row in available:
                if previous is not None and current < previous and new_row is not None:
                    new_row["monotonicity_warning"] = True
                previous = current


def _source_summaries() -> list[dict[str, Any]]:
    rows = _read_parquet(SOURCE_CAMPAIGN_ROOT / "point_summary.parquet")
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        process_medians = _parse_json_field(
            row["process_medians_ms"], name="process_medians_ms"
        )
        normalized.append(
            {
                **row,
                "process_medians_ms": process_medians,
                "source_campaign_id": SOURCE_CAMPAIGN_ID,
                "source_row_index": index,
                "densification_campaign": False,
            }
        )
    if len(normalized) != 270:
        raise Phase13DError("source point-summary cardinality differs")
    return normalized


def _session_bootstrap(
    summaries: Sequence[Mapping[str, Any]], *, seed: int
) -> dict[str, Any]:
    return pilot._session_bootstrap_knee(summaries, seed=seed)


def _target_resolution(
    *,
    target: Mapping[str, Any],
    fit_status: str,
    density: Mapping[str, Any],
    new_summaries: Sequence[Mapping[str, Any]],
    proposals: Sequence[Mapping[str, Any]],
) -> tuple[str, dict[str, Any]]:
    target_id = str(target["target_id"])
    target_points = [
        row for row in new_summaries if str(row["target_id"]) == target_id
    ]
    target_proposals = [
        row for row in proposals if str(row["target_id"]) == target_id
    ]
    included = [
        row for row in target_proposals if row["final_inclusion_status"] is True
    ]
    completed_stable = [
        row for row in target_points if row["disposition"] == "stable"
    ]
    all_feasible_exhausted = len(completed_stable) == len(included)
    limitation = {
        "preregistered_included_candidates": len(included),
        "stable_included_candidates": len(completed_stable),
        "capacity_infeasible_candidates": sum(
            row["disposition"] == "capacity_infeasible" for row in target_points
        ),
        "outside_model_range_proposals": sum(
            row["outside_valid_range"] is True for row in target_proposals
        ),
        "existing_source_proposals": sum(
            row["existing_point_status"] is True for row in target_proposals
        ),
        "duplicate_proposals": sum(
            row["duplicate_status"] is True for row in target_proposals
        ),
        "all_preregistered_feasible_candidates_exhausted": all_feasible_exhausted,
    }
    if fit_status == "unstable_data" or any(
        row["disposition"] == "unstable" for row in target_points
    ):
        return "unstable_data", limitation
    if fit_status in {"knee_below_range", "knee_above_range", "no_positive_slope"}:
        return fit_status, limitation
    if fit_status == "knee_observed" and density.get("sufficient") is True:
        return "density_sufficient", limitation
    if fit_status == "knee_observed" and all_feasible_exhausted:
        limitation["explicit_limitation"] = (
            "all preregistered feasible candidates were exhausted; remaining "
            "proposals are existing, duplicate, clamped, capacity-infeasible, "
            "or outside the frozen model range"
        )
        return "insufficient_feasible_span", limitation
    if fit_status == "fit_failed":
        return "fit_failed", limitation
    return "densification_required", limitation


def _combined_fit_records(
    *,
    source: Sequence[Mapping[str, Any]],
    densified: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
    campaign_id: str,
) -> list[dict[str, Any]]:
    targets = {str(row["target_id"]): row for row in candidate["targets"]}
    proposals = [dict(row) for row in candidate["proposals"]]
    records: list[dict[str, Any]] = []
    for configuration in CONFIGURATIONS:
        for batch in pilot.BATCH_SIZES:
            original = [
                row
                for row in source
                if row["method_config_id"] == configuration
                and int(row["batch_size"]) == batch
                and row["disposition"] == "stable"
            ]
            added = [
                row
                for row in densified
                if row["method_config_id"] == configuration
                and int(row["batch_size"]) == batch
                and row["disposition"] == "stable"
            ]
            all_rows = sorted(
                original + added, key=lambda row: int(row["context_label"])
            )
            has_unstable = any(
                row["method_config_id"] == configuration
                and int(row["batch_size"]) == batch
                and row["disposition"] == "unstable"
                for row in densified
            )
            fit = (
                {"fit_status": "unstable_data"}
                if has_unstable
                else pilot.provisional_knee_fit(
                    [
                        (float(row["context_label"]), float(value))
                        for row in all_rows
                        for value in row["process_medians_ms"]
                    ]
                )
            )
            knee_model = fit.get("knee_model")
            knee = (
                knee_model.get("L_star")
                if isinstance(knee_model, Mapping)
                else None
            )
            density = pilot.knee_density(
                [int(row["context_label"]) for row in all_rows],
                float(knee) if isinstance(knee, (int, float)) else None,
            )
            bootstrap = _session_bootstrap(
                all_rows,
                seed=20260823 + 100 * CONFIGURATIONS.index(configuration) + batch,
            )
            target_id = f"{configuration}-b{batch}"
            target = targets.get(target_id)
            if target is None:
                resolution = (
                    "density_sufficient"
                    if density.get("sufficient") is True
                    else str(fit["fit_status"])
                )
                limitation: dict[str, Any] = {}
            else:
                resolution, limitation = _target_resolution(
                    target=target,
                    fit_status=str(fit["fit_status"]),
                    density=density,
                    new_summaries=densified,
                    proposals=proposals,
                )
            constant = fit.get("constant_floor", {})
            linear = fit.get("linear", {})
            knee_fields = fit.get("knee_model", {})
            records.append(
                {
                    "target_id": target_id,
                    "method_config_id": configuration,
                    "batch_size": batch,
                    "was_original_densification_target": target is not None,
                    "original_L_star": (
                        target.get("original_L_star") if target is not None else None
                    ),
                    "original_fit_status": (
                        target.get("original_fit_status") if target is not None else None
                    ),
                    "source_stable_point_count": len(original),
                    "densification_stable_point_count": len(added),
                    "combined_stable_point_count": len(all_rows),
                    "session_observation_count": sum(
                        len(row["process_medians_ms"]) for row in all_rows
                    ),
                    "fit_status": fit["fit_status"],
                    "resolution_status": resolution,
                    "constant_tau": constant.get("tau"),
                    "linear_a": linear.get("a"),
                    "linear_s": linear.get("s"),
                    "knee_tau": knee_fields.get("tau"),
                    "knee_a": knee_fields.get("a"),
                    "knee_s": knee_fields.get("s"),
                    "refined_L_star": knee_fields.get("L_star"),
                    "r_squared": knee_fields.get("r_squared"),
                    "residual_summary": (
                        {
                            "minimum": min(knee_fields.get("residuals", [])),
                            "maximum": max(knee_fields.get("residuals", [])),
                            "mean": statistics.mean(knee_fields.get("residuals", [])),
                        }
                        if knee_fields.get("residuals")
                        else None
                    ),
                    "bootstrap_estimable": bootstrap["estimable"],
                    "bootstrap_knee_lower_95": bootstrap["lower_95"],
                    "bootstrap_knee_upper_95": bootstrap["upper_95"],
                    "bootstrap_valid_draws": bootstrap["valid_draws"],
                    "below_count": density.get("below_count"),
                    "near_count": density.get("near_count"),
                    "above_count": density.get("above_count"),
                    "density_sufficient": density.get("sufficient"),
                    "density_rule": "source_phase13_exact_below_near_above",
                    "limitation": limitation,
                    "source_campaign_ids": [SOURCE_CAMPAIGN_ID, campaign_id],
                    "fit_json": fit,
                    "density_json": density,
                    "pilot_only": True,
                    "quality_status": "unvalidated",
                    "performance_claim_eligible": False,
                }
            )
    if len(records) != 30:
        raise Phase13DError("combined fit cardinality differs")
    return records


def _combined_source_index(
    source: Sequence[Mapping[str, Any]],
    densified: Sequence[Mapping[str, Any]],
    campaign_id: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in source:
        if row["disposition"] != "stable":
            continue
        rows.append(
            {
                "campaign_id": SOURCE_CAMPAIGN_ID,
                "campaign_root_sha256": SOURCE_ROOT_SHA256,
                "point_summary_path": "point_summary.parquet",
                "point_summary_row_index": row["source_row_index"],
                "method_config_id": row["method_config_id"],
                "batch_size": row["batch_size"],
                "historical_context": row["historical_context"],
                "process_medians_ms": row["process_medians_ms"],
                "raw_samples_copied": False,
            }
        )
    for row_index, row in enumerate(densified):
        if row["disposition"] != "stable":
            continue
        rows.append(
            {
                "campaign_id": campaign_id,
                "campaign_root_sha256": "content_address_assigned_after_sealing",
                "point_summary_path": "point_summary.parquet",
                "point_summary_row_index": row_index,
                "method_config_id": row["method_config_id"],
                "batch_size": row["batch_size"],
                "historical_context": row["historical_context"],
                "process_medians_ms": row["process_medians_ms"],
                "raw_samples_copied": False,
            }
        )
    return rows


def _ratio_records(
    source: Sequence[Mapping[str, Any]],
    densified: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    stable = {
        (
            str(row["method_config_id"]),
            int(row["batch_size"]),
            int(row["historical_context"]),
        ): row
        for row in (*source, *densified)
        if row["disposition"] == "stable"
    }
    ratios: list[dict[str, Any]] = []
    for row in densified:
        if row["method_config_id"] == "bf16":
            continue
        key = (
            "bf16",
            int(row["batch_size"]),
            int(row["historical_context"]),
        )
        baseline = stable.get(key)
        calculated = bool(row["disposition"] == "stable" and baseline is not None)
        ratios.append(
            {
                "method_config_id": row["method_config_id"],
                "batch_size": row["batch_size"],
                "historical_context": row["historical_context"],
                "pilot_ratio": (
                    float(baseline["median_ms"]) / float(row["median_ms"])
                    if calculated and baseline is not None
                    else None
                ),
                "calculated": calculated,
                "pilot_only": True,
                "quality_status": "unvalidated",
                "performance_claim_eligible": False,
                "not_quality_preserving_speedup": True,
            }
        )
    return ratios


def _render_plots(
    root: Path,
    *,
    source: Sequence[Mapping[str, Any]],
    densified: Sequence[Mapping[str, Any]],
    fits: Sequence[Mapping[str, Any]],
) -> None:
    plots = root / "plots"
    by_fit = {str(row["target_id"]): row for row in fits}
    target_ids = sorted(
        {str(row["target_id"]) for row in densified},
        key=lambda value: (
            CONFIGURATIONS.index(value.rsplit("-b", 1)[0]),
            int(value.rsplit("-b", 1)[1]),
        ),
    )
    for target_id in target_ids:
        configuration, batch_text = target_id.rsplit("-b", 1)
        batch = int(batch_text)
        original = [
            row
            for row in source
            if row["method_config_id"] == configuration
            and int(row["batch_size"]) == batch
            and row["disposition"] == "stable"
        ]
        added = [
            row
            for row in densified
            if row["method_config_id"] == configuration
            and int(row["batch_size"]) == batch
            and row["disposition"] in {"stable", "unstable"}
        ]
        fit = by_fit[target_id]
        pilot._svg_line_plot(
            plots / f"{target_id}.svg",
            title=(
                f"Phase 13D {target_id}: refined knee "
                f"{fit.get('refined_L_star')} [{fit.get('bootstrap_knee_lower_95')}, "
                f"{fit.get('bootstrap_knee_upper_95')}]"
            ),
            y_label="median CUDA ms per operation",
            series={
                "original": [
                    (float(row["historical_context"]), float(row["median_ms"]))
                    for row in original
                ],
                "densified": [
                    (float(row["historical_context"]), float(row["median_ms"]))
                    for row in added
                ],
            },
        )
    cv_series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in densified:
        if isinstance(row.get("cv"), (int, float)):
            cv_series[str(row["target_id"])].append(
                (float(row["historical_context"]), float(row["cv"]))
            )
    for values in cv_series.values():
        values.sort(key=lambda item: item[0])
    pilot._svg_line_plot(
        plots / "new-point-cv.svg",
        title="Phase 13D new-point CV",
        y_label="process-median CV",
        series=dict(cv_series),
    )
    unresolved = [
        row
        for row in fits
        if row["was_original_densification_target"] is True
        and row["resolution_status"]
        in {"densification_required", "unstable_data", "fit_failed"}
    ]
    pilot._svg_plot(
        plots / "unresolved-targets.svg",
        title="Phase 13D unresolved targets",
        message=(
            "none"
            if not unresolved
            else ", ".join(str(row["target_id"]) for row in unresolved)
        ),
    )


def materialize_analysis(root: Path) -> dict[str, Any]:
    campaign = _strict_json(root / "unified" / "local-campaign.json")
    candidate = _strict_json(CANDIDATE_PATH)
    order = _strict_json(root / "execution_order.json")
    expected_count = int(order["planned_run_records"])
    runs = _run_records(root, expected_count=expected_count)
    feasibility = _strict_json(root / "unified" / "feasibility.json")["records"]
    summaries = _new_point_summaries(runs, candidate)
    source = _source_summaries()
    _mark_monotonicity_warnings(source=source, densified=summaries)
    fits = _combined_fit_records(
        source=source,
        densified=summaries,
        candidate=candidate,
        campaign_id=str(campaign["campaign_id"]),
    )
    combined_index = _combined_source_index(
        source, summaries, str(campaign["campaign_id"])
    )
    ratios = _ratio_records(source, summaries)
    target_rows = [dict(row) for row in candidate["targets"]]
    proposal_rows = [dict(row) for row in candidate["proposals"]]
    exclusions = [
        {
            "kind": "candidate",
            "target_id": row["target_id"],
            "historical_context": row["historical_context"],
            "status": "excluded",
            "reason": row["exclusion_reason"],
        }
        for row in proposal_rows
        if row["final_inclusion_status"] is not True
    ] + [
        {
            "kind": "run",
            "target_id": row["target_id"],
            "historical_context": row["historical_context"],
            "status": row["status"],
            "reason": row["reason"],
        }
        for row in runs
        if row["status"] != "completed"
    ]
    _write_parquet(root / "target_table.parquet", target_rows)
    _write_parquet(root / "candidate_generation.parquet", proposal_rows)
    _write_parquet(root / "feasibility.parquet", feasibility)
    _write_parquet(root / "raw_run_index.parquet", runs)
    _write_parquet(root / "point_summary.parquet", summaries)
    _write_parquet(root / "combined_source_index.parquet", combined_index)
    _write_parquet(root / "refined_knees.parquet", fits)
    _write_parquet(root / "exclusions.parquet", exclusions)
    stable = [row for row in summaries if row["disposition"] == "stable"]
    unstable = [row for row in summaries if row["disposition"] == "unstable"]
    failed = [
        row
        for row in summaries
        if row["disposition"] in {"failed", "runtime_failed", "aborted"}
    ]
    target_fits = [
        row for row in fits if row["was_original_densification_target"] is True
    ]
    unresolved = [
        row
        for row in target_fits
        if row["resolution_status"]
        in {"densification_required", "unstable_data", "fit_failed"}
    ]
    status_counts = Counter(str(row["status"]) for row in runs)
    feasible_count = sum(row["status"] == "feasible" for row in feasibility)
    infeasible_count = sum(
        row["status"] == "capacity_infeasible" for row in feasibility
    )
    all_feasible_completed = status_counts.get("completed", 0) == feasible_count
    local_pass = bool(
        campaign["phase13d_status"] == "LOCAL_COMPLETE"
        and all_feasible_completed
        and status_counts.get("capacity_infeasible", 0) == infeasible_count
        and sum(status_counts.values()) == expected_count
        and not unstable
        and not failed
        and not unresolved
    )
    original_fit_status_counts = Counter(
        str(row["original_fit_status"]) for row in target_rows
    )
    refined_fit_status_counts = Counter(str(row["fit_status"]) for row in fits)
    resolution_counts = Counter(
        str(row["resolution_status"]) for row in target_fits
    )
    qc = {
        "schema_version": "kvbench-phase13d-qc-1.0.0",
        "campaign_id": campaign["campaign_id"],
        "phase13d_status": (
            "LOCAL_PASS_PENDING_PUBLICATION" if local_pass else "PARTIAL"
        ),
        "source_campaign_id": SOURCE_CAMPAIGN_ID,
        "source_root_sha256": SOURCE_ROOT_SHA256,
        "target_count": len(target_rows),
        "proposal_count": len(proposal_rows),
        "included_candidate_count": candidate["included_candidate_count"],
        "feasible_run_records": feasible_count,
        "capacity_infeasible_run_records": infeasible_count,
        "status_counts": dict(sorted(status_counts.items())),
        "stable_new_points": len(stable),
        "unstable_new_points": len(unstable),
        "failed_new_points": len(failed),
        "maximum_cv": max(
            (float(row["cv"]) for row in summaries if row["cv"] is not None),
            default=None,
        ),
        "output_mismatches": sum(
            row["output_checksum_agreement"] is not True
            for row in summaries
            if row["disposition"] in {"stable", "unstable"}
        ),
        "kernel_path_drift": sum(
            row["kernel_path_agreement"] is not True
            for row in summaries
            if row["disposition"] in {"stable", "unstable"}
        ),
        "allocation_drift": sum(
            row["allocation_agreement"] is not True
            for row in summaries
            if row["disposition"] in {"stable", "unstable"}
        ),
        "original_fit_status_counts": dict(sorted(original_fit_status_counts.items())),
        "refined_fit_status_counts": dict(sorted(refined_fit_status_counts.items())),
        "resolution_status_counts": dict(sorted(resolution_counts.items())),
        "density_sufficient_count": resolution_counts.get("density_sufficient", 0),
        "knee_below_range_count": resolution_counts.get("knee_below_range", 0),
        "knee_above_range_count": resolution_counts.get("knee_above_range", 0),
        "no_positive_slope_count": resolution_counts.get("no_positive_slope", 0),
        "insufficient_feasible_span_count": resolution_counts.get(
            "insufficient_feasible_span", 0
        ),
        "remaining_unresolved_targets": [row["target_id"] for row in unresolved],
        "pilot_only_ratio_records": len(ratios),
        "pilot_only_ratios_calculated": sum(row["calculated"] for row in ratios),
        "source_raw_samples_copied": False,
        "selective_reruns": 0,
        "r_hbm": None,
        "full_scan": "CLOSED",
        "quality_execution": "LOCKED",
        "performance_data_frozen": False,
        "performance_claim_eligible": False,
        "phase14_readiness": "READY" if local_pass else "NOT_READY",
    }
    write_exclusive(root / "densification_qc.json", json_bytes(qc))
    report_lines = [
        "# Phase 13D Knee Densification",
        "",
        f"- Campaign: `{campaign['campaign_id']}`",
        f"- Local status: `{qc['phase13d_status']}`",
        f"- Source campaign/root: `{SOURCE_CAMPAIGN_ID}` / `{SOURCE_ROOT_SHA256}`",
        f"- Targets/proposals/included candidates: {len(target_rows)}/{len(proposal_rows)}/{candidate['included_candidate_count']}",
        f"- Feasible/infeasible run records: {feasible_count}/{infeasible_count}",
        f"- Stable/unstable/failed new points: {len(stable)}/{len(unstable)}/{len(failed)}",
        f"- Maximum CV: `{qc['maximum_cv']}`",
        f"- Resolution statuses: `{json.dumps(dict(sorted(resolution_counts.items())), sort_keys=True)}`",
        f"- Remaining unresolved targets: `{json.dumps(qc['remaining_unresolved_targets'])}`",
        f"- Phase 14 readiness: `{qc['phase14_readiness']}`",
        "- Full Scan: `CLOSED`",
        "- Quality execution: `LOCKED`",
        "- Pilot-only ratios remain quality-unvalidated and performance-claim ineligible.",
        "- No final speedup, HBM, capacity, knee, or quality claim is made.",
        "",
    ]
    write_exclusive(
        root / "densification_report.md",
        "\n".join(report_lines).encode("utf-8"),
    )
    _render_plots(root, source=source, densified=summaries, fits=fits)
    inventory = {
        "schema_version": "kvbench-phase13d-scientific-inventory-1.0.0",
        "campaign_id": campaign["campaign_id"],
        "target_rows": len(target_rows),
        "candidate_proposals": len(proposal_rows),
        "raw_run_records": len(runs),
        "new_point_summaries": len(summaries),
        "combined_source_rows": len(combined_index),
        "refined_fit_records": len(fits),
        "plot_count": len(list((root / "plots").glob("*.svg"))),
        "source_bundle_copied": False,
        "r_hbm": None,
    }
    write_exclusive(root / "inventory.json", json_bytes(inventory))
    return qc


def _payload_paths(root: Path, excluded: set[str]) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(root.rglob("*"), key=lambda value: value.relative_to(root).as_posix()):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise Phase13DError("Phase 13D campaign contains a symlink")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise Phase13DError("Phase 13D campaign contains an unsafe file")
        if path.relative_to(root).as_posix() not in excluded:
            paths.append(path)
    return paths


def seal_campaign(stage: Path, *, campaign_id: str) -> Path:
    identifier = _validate_campaign_id(campaign_id)
    root = stage.resolve(strict=True)
    qc = _strict_json(root / "densification_qc.json")
    if qc.get("campaign_id") != identifier:
        raise Phase13DError("Phase 13D QC campaign identity differs")
    manifest = {
        "schema_version": "kvbench-phase13d-artifact-manifest-1.0.0",
        "run_id": identifier,
        "campaign_id": identifier,
        "status": qc["phase13d_status"],
        "created_at_utc": _utc_now(),
        "execution_git_sha": _strict_json(root / "campaign_manifest.json")[
            "execution_git_sha"
        ],
        "authorized_container_digest": AUTHORIZED_CONTAINER_DIGEST,
        "source_campaign_id": SOURCE_CAMPAIGN_ID,
        "source_root_sha256": SOURCE_ROOT_SHA256,
        "append_only": True,
        "complete_written_last": True,
        "quality_status": "unvalidated",
        "performance_claim_eligible": False,
        "r_hbm": None,
    }
    write_exclusive(root / "manifest.json", json_bytes(manifest))
    excluded = {"artifact_inventory.json", "checksums.sha256", "COMPLETE"}
    inventory_items = [
        {
            "path": path.relative_to(root).as_posix(),
            "role": "phase13d_densification_evidence",
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in _payload_paths(root, excluded)
    ]
    write_exclusive(
        root / "artifact_inventory.json",
        json_bytes(
            {
                "schema_version": "kvbench-artifact-inventory-1.0.0",
                "run_id": identifier,
                "files": inventory_items,
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
        for path in _payload_paths(root, {"checksums.sha256", "COMPLETE"})
    ).encode("utf-8")
    write_exclusive(root / "checksums.sha256", ledger)
    write_exclusive(
        root / "COMPLETE",
        json_bytes(
            {
                "schema_version": "kvbench-completion-1.0.0",
                "run_id": identifier,
                "status": qc["phase13d_status"],
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
    final = ARTIFACT_ROOT / identifier
    if final.exists() or final.is_symlink():
        raise Phase13DError("finalized Phase 13D campaign already exists")
    rename_noreplace(root, final)
    for path in sorted(final.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    final.chmod(0o555)
    return final


def validate_campaign(
    root: Path, *, expected_campaign_id: str | None = None
) -> dict[str, Any]:
    artifact = validate_local_artifact(root, environ={})
    manifest = _strict_json(root / "manifest.json")
    campaign_id = _validate_campaign_id(str(manifest.get("campaign_id")))
    if expected_campaign_id is not None and campaign_id != expected_campaign_id:
        raise Phase13DError("Phase 13D campaign identity differs")
    required = {
        "campaign_manifest.json",
        "execution_order.json",
        "target_table.parquet",
        "candidate_generation.parquet",
        "feasibility.parquet",
        "raw_run_index.parquet",
        "point_summary.parquet",
        "combined_source_index.parquet",
        "refined_knees.parquet",
        "exclusions.parquet",
        "densification_qc.json",
        "densification_report.md",
        "inventory.json",
        "artifact_inventory.json",
        "checksums.sha256",
        "COMPLETE",
    }
    if any(not (root / name).is_file() for name in required):
        raise Phase13DError("Phase 13D required output is absent")
    candidate = _strict_json(CANDIDATE_PATH)
    order = _strict_json(root / "execution_order.json")
    if order != derive_execution_order(candidate):
        raise Phase13DError("Phase 13D finalized execution order differs")
    campaign = _strict_json(root / "campaign_manifest.json")
    qc = _strict_json(root / "densification_qc.json")
    runs = _run_records(root, expected_count=int(order["planned_run_records"]))
    target_rows = _read_parquet(root / "target_table.parquet")
    proposals = _read_parquet(root / "candidate_generation.parquet")
    feasibility = _read_parquet(root / "feasibility.parquet")
    summaries = _read_parquet(root / "point_summary.parquet")
    combined = _read_parquet(root / "combined_source_index.parquet")
    fits = _read_parquet(root / "refined_knees.parquet")
    if (
        campaign.get("campaign_id") != campaign_id
        or re.fullmatch(
            r"[0-9a-f]{40}", str(campaign.get("execution_git_sha"))
        )
        is None
        or manifest.get("execution_git_sha") != campaign.get("execution_git_sha")
        or campaign.get("authorized_container_digest") != AUTHORIZED_CONTAINER_DIGEST
        or campaign.get("source_campaign_id") != SOURCE_CAMPAIGN_ID
        or campaign.get("source_root_sha256") != SOURCE_ROOT_SHA256
        or campaign.get("blocked_campaign_id") != BLOCKED_CAMPAIGN_ID
        or campaign.get("blocked_campaign_root_sha256")
        != BLOCKED_CAMPAIGN_ROOT_SHA256
        or campaign.get("blocked_campaign_object_count")
        != BLOCKED_CAMPAIGN_OBJECT_COUNT
        or campaign.get("target_count") != TARGET_COUNT
        or campaign.get("source_timing_samples_reused") is not False
        or campaign.get("selective_reruns") != 0
        or qc.get("campaign_id") != campaign_id
        or qc.get("target_count") != TARGET_COUNT
        or qc.get("full_scan") != "CLOSED"
        or qc.get("quality_execution") != "LOCKED"
        or qc.get("performance_data_frozen") is not False
        or qc.get("r_hbm") is not None
        or len(target_rows) != TARGET_COUNT
        or len(proposals) != TARGET_COUNT * len(MULTIPLIERS)
        or len(feasibility) != int(order["planned_run_records"])
        or len(summaries) != int(candidate["included_candidate_count"])
        or len(fits) != 30
        or not combined
    ):
        raise Phase13DError("Phase 13D campaign semantics differ")
    status_counts = Counter(str(row["status"]) for row in runs)
    if dict(sorted(status_counts.items())) != qc.get("status_counts"):
        raise Phase13DError("Phase 13D run status counts differ")
    target_fits = [
        row for row in fits if row["was_original_densification_target"] is True
    ]
    allowed_resolution = {
        "density_sufficient",
        "knee_below_range",
        "knee_above_range",
        "no_positive_slope",
        "insufficient_feasible_span",
        "densification_required",
        "unstable_data",
        "fit_failed",
    }
    if (
        len(target_fits) != TARGET_COUNT
        or any(row["resolution_status"] not in allowed_resolution for row in target_fits)
        or any(row.get("r_hbm") is not None for row in summaries)
    ):
        raise Phase13DError("Phase 13D refined target resolution differs")
    local_pass = qc.get("phase13d_status") == "LOCAL_PASS_PENDING_PUBLICATION"
    unresolved = qc.get("remaining_unresolved_targets")
    if local_pass and unresolved != []:
        raise Phase13DError("Phase 13D local PASS retains unresolved targets")
    return {
        "status": "PASS",
        "campaign_id": campaign_id,
        "phase13d_status": qc["phase13d_status"],
        "root_sha256": artifact.root_sha256,
        "object_count": len(artifact.files),
        "targets": TARGET_COUNT,
        "included_candidates": candidate["included_candidate_count"],
        "planned_run_records": order["planned_run_records"],
    }


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--validate-source", action="store_true")
    actions.add_argument("--write-preregistration", action="store_true")
    actions.add_argument("--validate-preregistration", action="store_true")
    actions.add_argument("--print-feasibility", action="store_true")
    actions.add_argument("--new-campaign-id", action="store_true")
    actions.add_argument("--reserve-campaign", action="store_true")
    actions.add_argument("--run-campaign", action="store_true")
    actions.add_argument("--run-worker", action="store_true")
    actions.add_argument("--build-prefix-state", action="store_true")
    actions.add_argument("--materialize-analysis", action="store_true")
    actions.add_argument("--finalize-staged-campaign", action="store_true")
    actions.add_argument("--validate-campaign", action="store_true")
    parser.add_argument("--candidate-output", type=Path)
    parser.add_argument("--order-output", type=Path)
    parser.add_argument("--git-sha")
    parser.add_argument("--campaign-id")
    parser.add_argument("--stage", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--prefix-root", type=Path)
    parser.add_argument("--snapshot-id")
    parser.add_argument("--configuration", choices=CONFIGURATIONS)
    parser.add_argument("--source-batch", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--context-label", type=int)
    parser.add_argument("--build-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--replicate-index", type=int)
    parser.add_argument("--order-index", type=int)
    parser.add_argument("--run-artifact-root", type=Path)
    parser.add_argument("--prefix-state-root", type=Path)
    parser.add_argument("--prefix-state-sha256")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_arguments(argv)
    if args.validate_source:
        print(json.dumps(validate_source_campaign(), sort_keys=True))
        return 0
    if args.write_preregistration:
        if args.candidate_output is None or args.order_output is None:
            raise Phase13DError("preregistration output paths are required")
        print(
            json.dumps(
                write_preregistration(
                    candidate_output=args.candidate_output,
                    order_output=args.order_output,
                ),
                sort_keys=True,
            )
        )
        return 0
    if args.validate_preregistration:
        print(json.dumps(validate_preregistration(), sort_keys=True))
        return 0
    if args.print_feasibility:
        validate_preregistration(replay_source=False)
        rows = feasibility_records(_strict_json(ORDER_PATH))
        print(
            json.dumps(
                {
                    "planned_run_records": len(rows),
                    "feasible_run_records": sum(
                        row["status"] == "feasible" for row in rows
                    ),
                    "capacity_infeasible_run_records": sum(
                        row["status"] == "capacity_infeasible" for row in rows
                    ),
                    "unique_candidates": len(
                        {
                            (
                                row["method_config_id"],
                                row["batch_size"],
                                row["historical_context"],
                            )
                            for row in rows
                        }
                    ),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.new_campaign_id:
        if args.git_sha is None:
            raise Phase13DError("--git-sha is required")
        print(new_campaign_id(args.git_sha))
        return 0
    if args.reserve_campaign:
        if args.campaign_id is None or args.git_sha is None:
            raise Phase13DError("campaign ID and Git SHA are required")
        print(reserve_campaign(campaign_id=args.campaign_id, git_sha=args.git_sha))
        return 0
    if args.run_campaign:
        if any(
            value is None
            for value in (args.stage, args.campaign_id, args.git_sha, args.prefix_root)
        ):
            raise Phase13DError("campaign execution paths and identities are required")
        print(
            json.dumps(
                run_campaign(
                    stage=Path(args.stage),
                    campaign_id=str(args.campaign_id),
                    git_sha=str(args.git_sha),
                    prefix_root=Path(args.prefix_root),
                ),
                sort_keys=True,
            )
        )
        return 0
    if args.build_prefix_state:
        required = (
            args.snapshot_id,
            args.configuration,
            args.source_batch,
            args.context_label,
            args.git_sha,
            args.build_root,
            args.output,
        )
        if any(value is None for value in required):
            raise Phase13DError("prefix builder arguments are required")
        payload = _direct_prefix_worker(
            snapshot_id=str(args.snapshot_id),
            configuration=str(args.configuration),
            source_batch=int(args.source_batch),
            context_label=int(args.context_label),
            git_sha=str(args.git_sha),
            build_root=Path(args.build_root),
            output=Path(args.output),
        )
        _emit_and_exit(PREFIX_BUILDER_PREFIX, payload)
    if args.run_worker:
        required = (
            args.run_id,
            args.configuration,
            args.batch_size,
            args.context_label,
            args.replicate_index,
            args.order_index,
            args.git_sha,
            args.run_artifact_root,
            args.prefix_state_root,
            args.prefix_state_sha256,
        )
        if any(value is None for value in required):
            raise Phase13DError("worker arguments are required")
        payload = _run_worker(
            run_id=str(args.run_id),
            configuration=str(args.configuration),
            batch=int(args.batch_size),
            context_label=int(args.context_label),
            replicate_index=int(args.replicate_index),
            order_index=int(args.order_index),
            git_sha=str(args.git_sha),
            run_artifact_root=Path(args.run_artifact_root),
            prefix_state_root=Path(args.prefix_state_root),
            prefix_state_sha256=str(args.prefix_state_sha256),
        )
        print(WORKER_PREFIX + json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return 0
    if args.materialize_analysis:
        if args.stage is None:
            raise Phase13DError("--stage is required")
        print(json.dumps(materialize_analysis(args.stage), sort_keys=True))
        return 0
    if args.finalize_staged_campaign:
        if args.stage is None or args.campaign_id is None:
            raise Phase13DError("stage and campaign ID are required")
        print(seal_campaign(args.stage, campaign_id=args.campaign_id))
        return 0
    if args.validate_campaign:
        if args.artifact is None:
            raise Phase13DError("--artifact is required")
        print(json.dumps(validate_campaign(args.artifact), sort_keys=True))
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
