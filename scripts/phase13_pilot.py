"""Narrow Phase 13 Pilot coordinator, QC, and provisional analysis.

The module owns no adapter, CUDA kernel, cache layout, or timing boundary.  It
reuses the Phase 12 authority/session bridge and the common fixed-L runner,
adds the preregistered Pilot grid, and keeps every planned point append-only.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import dataclasses
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
from typing import Any

from preflight.run_preflight import json_bytes, rename_noreplace, write_exclusive
from kvbench.runtime.artifacts import sha256_file
from kvbench.runtime.method_harness import execution_path_audit_facade
from kvbench.runtime.process_supervision import run_supervised_command
from kvbench.schema import GraphMode, RunnerKind, canonical_json_bytes, sha256_hex
from kvbench.schema.phase13b import (
    PHASE13B_BATCH_SIZES,
    PHASE13B_FAMILY_CONFIGURATIONS,
    Phase13BMethodAdmissionReport,
)
from scripts.r2_artifact import validate_local_artifact
import scripts.phase12_unified_admission as phase12


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPOSITORY_ROOT / "artifacts" / "phase13"
STAGING_ROOT = ARTIFACT_ROOT / ".kvbench-staging"
PLAN_PATH = Path("docs/plans/phase13-pilot-scan.md")
ORDER_PATH = Path("docs/plans/phase13-pilot-execution-order.json")
PHASE13B_DECISION_PATH = Path(
    "docs/decisions/0030-compressed-static-cache-batch-geometry.md"
)
PHASE13B_DECISION_SHA256 = (
    "84c2eb943b35afba312eaf599f8ec8f1d4a82169daa2d5c5fc5d127f0a965e62"
)
PHASE13B_SOURCE_AUTHORITY_COMMIT = (
    "b862af64346a0dba2650b2c213ebd1d3b5b99ef2"
)
PHASE13B_SUCCESSOR_REPORTS = {
    "turboquant": (
        Path("docs/evidence/phase13b/turboquant-method-admission.json"),
        "49799ef89646ec008a530c5180fdcef6cd4af9ca0d5772fe2b01d6e775e3b1c0",
    ),
    "kivi": (
        Path("docs/evidence/phase13b/kivi-method-admission.json"),
        "1e91730ac56af37e03d80edce7979a509d52049428faad89f61e61dc6bd48c51",
    ),
    "kvquant": (
        Path("docs/evidence/phase13b/kvquant-method-admission.json"),
        "e1cee8e1c514f9cf6323b5e710480c1fefab2804e5f4eafe6c473b29f4768481",
    ),
}
PHASE13B_PUBLICATION_RECEIPT_PATH = Path(
    "docs/evidence/phase13b/r2-publication.json"
)
PHASE13B_PUBLICATION_RECEIPT_SHA256 = (
    "86ac8259aa2fbf35ad6a525291756c80ba37fc8e1681a31ca4eccc60ef65a768"
)
PHASE13B_LOCAL_BUNDLE_PATH = Path(
    "artifacts/phase13b/phase13b-20260801t143138050263z-b862af64-batch-admission"
)
PHASE13B_LOCAL_ROOT_SHA256 = (
    "f1c96eaacbbace1c23b249d1afe8d892aa26c3f6b8d04e07f373a2becafba1fe"
)
PHASE13B_CHECKSUM_LEDGER_SHA256 = (
    "456cbc1d23a6cc94934b960c2ed30554aeb84faa5fde267defc899a1d09c38a2"
)
PHASE13B_R2_ROOT_SHA256 = (
    "f1c96eaacbbace1c23b249d1afe8d892aa26c3f6b8d04e07f373a2becafba1fe"
)

AUTHORIZED_CONTAINER_DIGEST = phase12.PHASE12_AUTHORIZED_CONTAINER_DIGEST
GPU_UUID = phase12.PHASE12_GPU_UUID
CONFIGURATIONS = phase12.MAIN_CONFIG_IDS
CONFIG_FINGERPRINTS = dict(phase12.EXPECTED_CONFIG_FINGERPRINTS)
HELD_OUT_CONFIGURATIONS = phase12.HELD_OUT_CONFIG_IDS
BATCH_SIZES = (1, 4, 8)
CONTEXT_LABELS = (4096, 8192, 16384, 24576, 32768, 49152, 65536, 98304, 131072)
SEEDS = (20260801, 20260802, 20260803)
REPLICATES = 3
WARMUP_STEPS = 64
MEASURED_STEPS = 128
MEASURED_BATCHES = 5
CV_THRESHOLD = 0.03
MAX_MEMORY_FRACTION = 0.88
GPU_TOTAL_MEMORY_BYTES = 101_970_345_984
MODEL_WEIGHT_BYTES = 16_060_556_288
REFERENCE_CAPACITY = 4097
PLANNED_RECORD_COUNT = 810
INPUT_RECIPE_SCHEMA = "kvbench-phase13-pilot-input-1.0.0"
CAMPAIGN_SCHEMA = "kvbench-phase13-pilot-campaign-1.0.0"
RUN_SCHEMA = "kvbench-phase13-pilot-process-run-1.0.0"
WORKER_PREFIX = "PHASE13_WORKER_RESULT="
CHILD_TIMEOUT_SECONDS = 7_200.0

_CAMPAIGN_RE = re.compile(
    r"phase13-[0-9]{8}t[0-9]{12}z-[0-9a-f]{8}-[0-9a-f]{6}\Z"
)
_RUN_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,159}\Z")
_CONTROL_FILES = frozenset(
    {"manifest.json", "inventory.json", "checksums.sha256", "COMPLETE"}
)
_FORBIDDEN_ENVIRONMENT = phase12._FORBIDDEN_CHILD_ENVIRONMENT


class Phase13PilotError(RuntimeError):
    """The Pilot contract or evidence failed closed."""


def _phase13b_successor_authority() -> dict[str, Any]:
    """Validate the exact Decision 0030 successor reports in the source tree."""

    decision = REPOSITORY_ROOT / PHASE13B_DECISION_PATH
    if sha256_file(decision) != PHASE13B_DECISION_SHA256:
        raise Phase13PilotError("Decision 0030 checksum differs")
    families: dict[str, Any] = {}
    for family, (relative, expected_sha256) in PHASE13B_SUCCESSOR_REPORTS.items():
        path = REPOSITORY_ROOT / relative
        if sha256_file(path) != expected_sha256:
            raise Phase13PilotError(
                f"Phase 13B {family} successor report checksum differs"
            )
        try:
            report = Phase13BMethodAdmissionReport.from_dict(_strict_json(path))
        except (TypeError, ValueError) as error:
            raise Phase13PilotError(
                f"Phase 13B {family} successor report schema differs"
            ) from error
        if (
            report.method_family != family
            or report.configurations != PHASE13B_FAMILY_CONFIGURATIONS[family]
            or report.batch_sizes != PHASE13B_BATCH_SIZES
            or report.creation_git_sha != PHASE13B_SOURCE_AUTHORITY_COMMIT
            or report.decision_id != "0030"
            or not report.b1_numerical_preserved
            or report.cuda_source_changed
        ):
            raise Phase13PilotError(
                f"Phase 13B {family} successor authority differs"
            )
        for relative_source, expected_source_sha256 in report.source_hashes.items():
            source = REPOSITORY_ROOT / relative_source
            if sha256_file(source) != expected_source_sha256:
                raise Phase13PilotError(
                    f"Phase 13B {family} admitted source differs: {relative_source}"
                )
        families[family] = {
            "report_path": relative.as_posix(),
            "report_sha256": expected_sha256,
            "creation_git_sha": report.creation_git_sha,
            "configurations": list(report.configurations),
            "batch_sizes": list(report.batch_sizes),
            "adapter_versions": dict(report.adapter_versions),
            "adapter_config_fingerprints_l128": dict(
                report.adapter_config_fingerprints
            ),
            "cache_layout_fingerprints_l128": dict(
                report.cache_layout_fingerprints
            ),
            "source_hashes": dict(report.source_hashes),
            "b1_numerical_preserved": True,
        }
    return {
        "schema_version": "kvbench-phase13r-successor-authority-1.0.0",
        "decision": "0030",
        "decision_path": PHASE13B_DECISION_PATH.as_posix(),
        "decision_sha256": PHASE13B_DECISION_SHA256,
        "source_authority_commit": PHASE13B_SOURCE_AUTHORITY_COMMIT,
        "families": families,
    }


def validate_phase13b_entry() -> dict[str, Any]:
    """Validate local successor admission and its checksum-bound R2 receipt."""

    authority = _phase13b_successor_authority()
    local_bundle = REPOSITORY_ROOT / PHASE13B_LOCAL_BUNDLE_PATH
    artifact = validate_local_artifact(local_bundle, environ={})
    if (
        artifact.root_sha256 != PHASE13B_LOCAL_ROOT_SHA256
        or sha256_file(local_bundle / "checksums.sha256")
        != PHASE13B_CHECKSUM_LEDGER_SHA256
    ):
        raise Phase13PilotError("Phase 13B local admission root differs")
    receipt_path = REPOSITORY_ROOT / PHASE13B_PUBLICATION_RECEIPT_PATH
    if sha256_file(receipt_path) != PHASE13B_PUBLICATION_RECEIPT_SHA256:
        raise Phase13PilotError("Phase 13B publication receipt checksum differs")
    receipt = _strict_json(receipt_path)
    publication = receipt.get("publication")
    retrieval = receipt.get("clean_retrieval")
    if (
        not isinstance(publication, Mapping)
        or not isinstance(retrieval, Mapping)
        or publication.get("root_sha256") != PHASE13B_R2_ROOT_SHA256
        or publication.get("complete_last") is not True
        or publication.get("conditional_writes") is not True
        or retrieval.get("root_sha256") != PHASE13B_R2_ROOT_SHA256
        or retrieval.get("result") != "PASS"
        or retrieval.get("destination_initially_empty") is not True
        or retrieval.get("checksum_ledger_valid") is not True
        or retrieval.get("inventory_valid") is not True
        or receipt.get("clean_retrieval_count") != 1
    ):
        raise Phase13PilotError("Phase 13B durable publication evidence differs")
    authority["local_bundle_path"] = PHASE13B_LOCAL_BUNDLE_PATH.as_posix()
    authority["local_root_sha256"] = PHASE13B_LOCAL_ROOT_SHA256
    authority["publication_receipt_path"] = (
        PHASE13B_PUBLICATION_RECEIPT_PATH.as_posix()
    )
    authority["publication_receipt_sha256"] = (
        PHASE13B_PUBLICATION_RECEIPT_SHA256
    )
    authority["r2_root_sha256"] = PHASE13B_R2_ROOT_SHA256
    authority["r2_uri"] = publication.get("uri")
    authority["clean_retrieval"] = "PASS"
    return authority


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _canonical_sha256(value: Any) -> str:
    return sha256_hex(canonical_json_bytes(value))


def actual_historical_context(label: int) -> int:
    """Map the requested top label to the fixed-L historical-prefix convention."""

    if label not in CONTEXT_LABELS:
        raise Phase13PilotError("unknown Phase 13 context label")
    return 131071 if label == 131072 else label


def _point_seed(seed: int, configuration: str) -> int:
    digest = hashlib.sha256(f"{seed}:{configuration}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


def derive_execution_order() -> dict[str, Any]:
    """Derive all 810 immutable planned records with blocked randomization."""

    records: list[dict[str, Any]] = []
    block_orders: list[dict[str, Any]] = []
    for replicate_index, seed in enumerate(SEEDS):
        blocks = list(CONFIGURATIONS)
        random.Random(seed).shuffle(blocks)
        block_orders.append(
            {
                "replicate_index": replicate_index,
                "seed": seed,
                "configuration_order": blocks,
            }
        )
        global_index = 0
        for block_index, configuration in enumerate(blocks):
            points = [(batch, label) for batch in BATCH_SIZES for label in CONTEXT_LABELS]
            random.Random(_point_seed(seed, configuration)).shuffle(points)
            for within_block_index, (batch, label) in enumerate(points):
                historical = actual_historical_context(label)
                records.append(
                    {
                        "replicate_index": replicate_index,
                        "seed": seed,
                        "block_index": block_index,
                        "within_block_index": within_block_index,
                        "order_index": global_index,
                        "method_config_id": configuration,
                        "method_config_fingerprint": CONFIG_FINGERPRINTS[configuration],
                        "batch_size": batch,
                        "context_label": label,
                        "historical_context": historical,
                        "total_attended_context": historical + 1,
                        "runner_kind": "fixed_l",
                        "graph_mode": "cuda_graph",
                    }
                )
                global_index += 1
    if len(records) != PLANNED_RECORD_COUNT:
        raise Phase13PilotError("Phase 13 execution order cardinality differs")
    identity_keys = (
        "replicate_index",
        "method_config_id",
        "batch_size",
        "context_label",
    )
    if len({tuple(record[key] for key in identity_keys) for record in records}) != len(
        records
    ):
        raise Phase13PilotError("Phase 13 execution order contains duplicates")
    payload = {
        "schema_version": "kvbench-phase13-pilot-execution-order-1.0.0",
        "seeds": list(SEEDS),
        "blocked_randomization": True,
        "configuration_blocks": True,
        "point_order_within_block_randomized": True,
        "block_orders": block_orders,
        "records": records,
    }
    payload["records_sha256"] = _canonical_sha256(records)
    return payload


def validate_execution_order(payload: Mapping[str, Any]) -> None:
    expected = derive_execution_order()
    if dict(payload) != expected:
        raise Phase13PilotError("committed Phase 13 execution order differs")


def _phase12_reference_runs() -> dict[str, Mapping[str, Any]]:
    root = (
        REPOSITORY_ROOT
        / "artifacts/phase12/phase12-20260731t062914664948z-6165f78d-c78b9a/runs"
    )
    selected: dict[str, Mapping[str, Any]] = {}
    for path in sorted(root.glob("*/result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        configuration = payload.get("method_config_id")
        if configuration in CONFIGURATIONS and configuration not in selected:
            if payload.get("method_config_fingerprint") != CONFIG_FINGERPRINTS[configuration]:
                raise Phase13PilotError("Phase 12 reference fingerprint differs")
            selected[str(configuration)] = payload
    if tuple(selected) != CONFIGURATIONS:
        missing = sorted(set(CONFIGURATIONS) - set(selected))
        if missing:
            raise Phase13PilotError(f"Phase 12 reference runs absent: {missing}")
        selected = {configuration: selected[configuration] for configuration in CONFIGURATIONS}
    return selected


def _reference_graph_reserve_bytes(configuration: str) -> int:
    worker = _phase12_reference_runs()[configuration]
    memory = worker["runner"]["memory_evidence"]
    cache = worker["runner"]["cache_accounting"]
    reserve = (
        int(memory["post_setup"]["allocated_bytes"])
        - int(memory["model_baseline"]["allocated_bytes"])
        - int(cache["allocated_bytes"])
    )
    if reserve <= 0:
        raise Phase13PilotError("Phase 12 graph reserve is invalid")
    return reserve


def _turboquant_cache_bytes(configuration: str, batch: int, capacity: int) -> int:
    slot_size = {"tq_4bit_nc": 134, "tq_k3v4_nc": 118, "tq_3bit_nc": 102}[
        configuration
    ]
    rounded = math.ceil(capacity / 16) * 16
    packed = 28 * rounded * 8 * slot_size
    skipped_bf16 = 2 * 4 * 8 * rounded * 128 * 2
    mapping = math.ceil(capacity / 16) * 4 + rounded * 8
    hadamard = 2 * 128 * 128 * 4
    levels = {"tq_4bit_nc": 16, "tq_k3v4_nc": 8, "tq_3bit_nc": 8}[
        configuration
    ]
    quantizer = (levels + max(0, levels - 1)) * 4
    store = 3 * capacity * 8 * 128 * 4 + 2 * capacity * 8 * 4
    decode = batch * (
        2 * 32 * 128 * 4 + 32 * 4 * 129 * 4 + 32 * 128 * 2 + 32 * 4
    )
    batch_scaled_history = batch * (packed + skipped_bf16 + mapping + store)
    return batch_scaled_history + hadamard + quantizer + decode


def _kivi_cache_bytes(configuration: str, batch: int, capacity: int) -> int:
    k_bits, v_bits = {"k4v4": (4, 4), "k2v4": (2, 4), "k2v2": (2, 2)}[
        configuration
    ]
    layers, kv_heads, query_heads, dimension = 32, 8, 32, 128
    residual, group = 32, 32
    key_history_capacity = (capacity // group) * group
    value_history_capacity = max(0, capacity - residual)
    key_groups = key_history_capacity // group
    value_head_groups = dimension // group
    key_words = key_history_capacity * k_bits // 32
    value_words = dimension * v_bits // 32
    key_residual = layers * batch * kv_heads * residual * dimension * 2
    value_residual = key_residual
    ordered_value = value_residual
    fp16_staging = sum(
        (
            ordered_value,
            batch * query_heads * dimension * 2,
            2 * batch * kv_heads * dimension * 2,
            2 * batch * kv_heads * dimension * 2,
            2 * batch * kv_heads * value_head_groups * 2,
            batch * query_heads * dimension * 2,
        )
    )
    quant_elements = batch * kv_heads * dimension * residual
    quant_staging = (
        quant_elements * 2
        + quant_elements * 4
        + batch * kv_heads * dimension * 8 * 4
    )
    workspace = (
        2 * batch * query_heads * capacity * 2
        + batch * query_heads * capacity * 4
        + 2 * batch * query_heads * dimension * 2
    )
    categories = (
        layers * batch * kv_heads * key_words * dimension * 4,
        layers * batch * kv_heads * value_words * value_history_capacity * 4,
        2 * layers * batch * kv_heads * key_groups * dimension * 2,
        2 * layers * batch * kv_heads * value_head_groups * value_history_capacity * 2,
        key_residual,
        value_residual,
        layers
        * (key_history_capacity + residual + value_history_capacity + residual)
        * 8,
        fp16_staging,
        quant_staging,
        workspace,
    )
    return sum(categories)


def _kvquant_cache_bytes(configuration: str, batch: int, capacity: int) -> int:
    bits = {"kvq4": 4, "kvq3": 3, "kvq2": 2}[configuration]
    levels = 1 << bits
    layers, heads, query_heads, dimension = 32, 8, 32, 128
    packed_rows = bits * dimension // 32
    query_elements = batch * query_heads * dimension
    kv_elements = batch * heads * dimension
    dense = batch * layers * heads * packed_rows * capacity * 4
    key_metadata = (
        layers * levels * 4
        + layers * heads * dimension * levels * 4
        + 3 * layers * heads * dimension * 4
        + 64 * 4
    )
    value_metadata = (
        layers * levels * 4
        + layers * batch * capacity * levels * 4
    )
    sparse = batch * layers * capacity * 12 * 4
    count_mask = 2 * layers * batch * capacity * 4 + capacity
    sink = layers * batch * heads * dimension * 5 * 2
    staging = (
        3 * kv_elements * 2
        + 4 * heads * dimension * 4
        + query_elements * 2
        + query_elements * 4
        + query_elements * 2
        + 2 * 12 * 4
        + 3 * 4
        + 1
        + heads * dimension * 4
        + 2 * heads * dimension * 4
        + levels * 4
        + 5 * 4
        + 2 * batch * capacity * 4
        + 3 * 4
        + 8
    )
    workspace = (
        2 * batch * query_heads * capacity * 4
        + batch * query_heads * capacity * 2
        + batch * query_heads * 5 * 2
        + 4 * query_elements * 4
        + query_elements * 2
        + (batch * 32 * 32 * 128 * 4 if configuration == "kvq4" else 0)
    )
    endpoint_rope_scratch = (
        layers * batch * (query_heads + heads) * 64 * 2
    )
    return (
        2 * dense
        + key_metadata
        + value_metadata
        + 4 * sparse
        + count_mask
        + 2 * sink
        + staging
        + workspace
        + endpoint_rope_scratch
    )


def cache_allocated_bytes(configuration: str, batch: int, capacity: int) -> int:
    if configuration == "bf16":
        return 2 * 32 * batch * 8 * capacity * 128 * 2 + 163_840
    if configuration.startswith("tq_"):
        return _turboquant_cache_bytes(configuration, batch, capacity)
    if configuration in {"k4v4", "k2v4", "k2v2"}:
        return _kivi_cache_bytes(configuration, batch, capacity)
    return _kvquant_cache_bytes(configuration, batch, capacity)


def adapter_geometry_supported(configuration: str, batch: int) -> bool:
    """Reflect Decision 0030's admitted static-cache batch geometry."""

    return configuration in CONFIGURATIONS and batch in BATCH_SIZES


def feasibility_record(order_record: Mapping[str, Any]) -> dict[str, Any]:
    configuration = str(order_record["method_config_id"])
    batch = int(order_record["batch_size"])
    historical = int(order_record["historical_context"])
    capacity = historical + 1
    cache_bytes = cache_allocated_bytes(configuration, batch, capacity)
    reference_reserve = _reference_graph_reserve_bytes(configuration)
    graph_reserve = math.ceil(
        reference_reserve * batch * capacity / REFERENCE_CAPACITY
    )
    limit = math.floor(GPU_TOTAL_MEMORY_BYTES * MAX_MEMORY_FRACTION)
    required = MODEL_WEIGHT_BYTES + cache_bytes + graph_reserve
    feasible = required <= limit
    return {
        **dict(order_record),
        "schema_version": "kvbench-phase13-feasibility-record-1.0.0",
        "capacity": capacity,
        "model_weight_bytes": MODEL_WEIGHT_BYTES,
        "cache_allocated_bytes": cache_bytes,
        "persistent_workspace_included_in_cache": True,
        "graph_pool_or_capture_reserve_bytes": graph_reserve,
        "graph_reserve_reference_bytes": reference_reserve,
        "gpu_total_memory_bytes": GPU_TOTAL_MEMORY_BYTES,
        "max_memory_fraction": MAX_MEMORY_FRACTION,
        "configured_safety_margin_bytes": GPU_TOTAL_MEMORY_BYTES - limit,
        "limit_bytes": limit,
        "predicted_required_bytes": required,
        "status": "feasible" if feasible else "capacity_infeasible",
        "reason": None if feasible else "predicted_required_bytes_exceed_0.88_limit",
        "adapter_geometry_supported_at_entry": adapter_geometry_supported(
            configuration, batch
        ),
        "unsupported_geometry_is_not_reclassified_as_capacity": True,
        "method_config_fingerprint": CONFIG_FINGERPRINTS[configuration],
    }


def build_feasibility_records(order: Mapping[str, Any]) -> list[dict[str, Any]]:
    validate_execution_order(order)
    records = [feasibility_record(record) for record in order["records"]]
    if len(records) != PLANNED_RECORD_COUNT:
        raise Phase13PilotError("Phase 13 feasibility cardinality differs")
    return records


@dataclasses.dataclass(frozen=True, slots=True)
class Phase13OperationKey:
    configuration: str
    runner_kind: RunnerKind
    graph_mode: GraphMode
    historical_context: int
    attended_context: int
    batch_size: int
    capacity: int
    decode_step: int
    operation_fingerprint_sha256: str

    @classmethod
    def create(cls, configuration: str, batch: int, historical: int) -> "Phase13OperationKey":
        payload = {
            "schema_version": "kvbench-phase13-operation-key-1.0.0",
            "configuration": configuration,
            "runner_kind": "fixed_l",
            "graph_mode": "cuda_graph",
            "historical_context": historical,
            "attended_context": historical + 1,
            "batch_size": batch,
            "capacity": historical + 1,
            "decode_step": 0,
            "input_recipe_schema": INPUT_RECIPE_SCHEMA,
        }
        return cls(
            configuration=configuration,
            runner_kind=RunnerKind.FIXED_L,
            graph_mode=GraphMode.CUDA_GRAPH,
            historical_context=historical,
            attended_context=historical + 1,
            batch_size=batch,
            capacity=historical + 1,
            decode_step=0,
            operation_fingerprint_sha256=_canonical_sha256(payload),
        )


def point_statistics(process_medians: Sequence[float]) -> dict[str, float]:
    if len(process_medians) != REPLICATES:
        raise Phase13PilotError("Pilot point requires exactly three processes")
    values = [float(value) for value in process_medians]
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise Phase13PilotError("Pilot process median is invalid")
    mean = statistics.mean(values)
    standard_deviation = statistics.stdev(values)
    return {
        "median_ms": statistics.median(values),
        "mean_ms": mean,
        "standard_deviation_ms": standard_deviation,
        "minimum_ms": min(values),
        "maximum_ms": max(values),
        "cv": standard_deviation / mean,
    }


def classify_point(*, statistics_record: Mapping[str, Any], agreements: bool) -> str:
    if not agreements:
        return "failed"
    return "stable" if float(statistics_record["cv"]) <= CV_THRESHOLD else "unstable"


def provisional_knee_fit(observations: Sequence[tuple[float, float]]) -> dict[str, Any]:
    """Fit constant, linear, and fixed-order hinge candidates without claiming a knee."""

    if len(observations) < 4 or len({x for x, _ in observations}) < 4:
        return {"fit_status": "insufficient_feasible_span"}
    points = sorted((float(x), float(y)) for x, y in observations)
    xs = [item[0] for item in points]
    ys = [item[1] for item in points]
    if any(not math.isfinite(value) for value in (*xs, *ys)):
        return {"fit_status": "fit_failed"}
    tau0 = statistics.mean(ys)
    floor_sse = sum((value - tau0) ** 2 for value in ys)
    x_mean, y_mean = statistics.mean(xs), statistics.mean(ys)
    denominator = sum((value - x_mean) ** 2 for value in xs)
    if denominator <= 0:
        return {"fit_status": "insufficient_feasible_span"}
    slope = sum((x - x_mean) * (y - y_mean) for x, y in points) / denominator
    intercept = y_mean - slope * x_mean
    linear_sse = sum((y - (intercept + slope * x)) ** 2 for x, y in points)
    best: tuple[float, float, float, float] | None = None
    candidates = sorted(set(xs + [(left + right) / 2 for left, right in zip(xs, xs[1:])]))
    for knee in candidates:
        hinge = [max(0.0, x - knee) for x in xs]
        h_mean = statistics.mean(hinge)
        h_denom = sum((h - h_mean) ** 2 for h in hinge)
        candidate_slope = (
            0.0
            if h_denom == 0
            else sum((h - h_mean) * (y - y_mean) for h, y in zip(hinge, ys))
            / h_denom
        )
        if candidate_slope < 0:
            candidate_slope = 0.0
        candidate_tau = y_mean - candidate_slope * h_mean
        sse = sum(
            (y - (candidate_tau + candidate_slope * h)) ** 2
            for h, y in zip(hinge, ys)
        )
        candidate = (sse, knee, candidate_tau, candidate_slope)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    knee_sse, knee, tau, knee_slope = best
    total_sse = sum((value - y_mean) ** 2 for value in ys)
    r_squared = None if total_sse == 0 else 1.0 - knee_sse / total_sse
    if knee_slope <= 0:
        status = "no_positive_slope"
    elif knee < min(xs):
        status = "knee_below_range"
    elif knee > max(xs):
        status = "knee_above_range"
    else:
        status = "knee_observed"
    return {
        "fit_status": status,
        "constant_floor": {"tau": tau0, "sse": floor_sse},
        "linear": {"a": intercept, "s": slope, "sse": linear_sse},
        "knee_model": {
            "tau": tau,
            "a": tau - knee_slope * knee,
            "s": knee_slope,
            "L_star": knee if knee_slope > 0 else None,
            "sse": knee_sse,
            "r_squared": r_squared,
            "residuals": [
                y - (tau + knee_slope * max(0.0, x - knee)) for x, y in points
            ],
        },
    }


def knee_density(contexts: Sequence[int], knee: float | None) -> dict[str, Any]:
    if knee is None or not math.isfinite(knee) or knee <= 0:
        return {"assessed": False, "sufficient": False, "missing_interval": None}
    below = [value for value in contexts if value < 0.75 * knee]
    near = [value for value in contexts if 0.75 * knee <= value <= 1.25 * knee]
    above = [value for value in contexts if value > 1.25 * knee]
    sufficient = bool(below and near and above)
    return {
        "assessed": True,
        "below_count": len(below),
        "near_count": len(near),
        "above_count": len(above),
        "near_definition": "[0.75*L_star,1.25*L_star]",
        "sufficient": sufficient,
        "missing_interval": None if sufficient else [0.75 * knee, 1.25 * knee],
    }


def new_campaign_id(git_sha: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", git_sha) is None:
        raise Phase13PilotError("execution Git SHA is invalid")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%S%f")[:21]
    return f"phase13-{stamp}z-{git_sha[:8]}-{secrets.token_hex(3)}"


def _validate_campaign_id(value: str) -> str:
    if _CAMPAIGN_RE.fullmatch(value) is None:
        raise Phase13PilotError("Phase 13 campaign ID is invalid")
    return value


def reserve_campaign(*, campaign_id: str, git_sha: str) -> Path:
    identifier = _validate_campaign_id(campaign_id)
    STAGING_ROOT.mkdir(parents=True, exist_ok=True)
    stage = STAGING_ROOT / f"{identifier}.{secrets.token_hex(4)}.staging"
    stage.mkdir()
    for relative in ("runs", "unified", "pilot_plots"):
        (stage / relative).mkdir()
    write_exclusive(
        stage / "campaign-reservation.json",
        json_bytes(
            {
                "schema_version": "kvbench-phase13-campaign-reservation-1.0.0",
                "campaign_id": identifier,
                "execution_git_sha": git_sha,
                "reserved_at_utc": _utc_now(),
                "append_only": True,
                "overwrite": False,
            }
        ),
    )
    return stage.resolve(strict=True)


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase13PilotError(f"invalid JSON evidence: {path}") from error
    if not isinstance(payload, dict):
        raise Phase13PilotError(f"JSON evidence is not an object: {path}")
    return payload


def _patch_phase12_point_globals(*, batch: int, historical: int) -> None:
    recipe = {
        "schema_version": INPUT_RECIPE_SCHEMA,
        "prefix": (
            f"(arange({batch}*{historical}).reshape({batch},{historical})"
            "+12000)%120000+1000"
        ),
        "decode": (
            f"(arange({batch}).reshape({batch},1)+12000+{historical}+257)"
            "%120000+1000"
        ),
    }
    phase12.PHASE12_BATCH_SIZE = batch
    phase12.PHASE12_CONTEXT_LENGTH = historical
    phase12.PHASE12_WARMUP_STEPS = WARMUP_STEPS
    phase12.PHASE12_MEASURED_STEPS = MEASURED_STEPS
    phase12.PHASE12_MEASURED_BATCHES = MEASURED_BATCHES
    phase12.PHASE12_INPUT_RECIPE = recipe
    phase12.PHASE12_INPUT_RECIPE_SHA256 = _canonical_sha256(recipe)


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
) -> dict[str, Any]:
    """Execute one feasible Pilot point inside the authorized container."""

    attestation = phase12._require_authorized_container_runtime()
    pre_snapshot = phase12._capture_process_snapshot()
    phase12._require_idle_snapshot(pre_snapshot)
    pid = os.getpid()
    start_ticks = phase12._process_start_ticks(pid)

    import torch

    from kvbench.runtime.allocation import audit_cuda_allocations
    from kvbench.runtime.backend import forced_flash_execution
    from kvbench.runtime.fixed_l_runner import run_fixed_l
    from kvbench.runtime.model_loader import load_frozen_model
    from kvbench.runtime.numerical import tensor_sha256_untimed
    from kvbench.runtime.timing import warmup_operations

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise Phase13PilotError("worker execution authority differs")
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
        raise Phase13PilotError("worker Git source authority or cleanliness differs")

    historical = actual_historical_context(context_label)
    _patch_phase12_point_globals(batch=batch, historical=historical)
    device = torch.device("cuda:0")
    loaded = load_frozen_model(device=device)
    prefix = (
        torch.arange(batch * historical, dtype=torch.long, device=device)
        .reshape(batch, historical)
        .add(12_000)
        .remainder(120_000)
        .add(1_000)
    )
    decode = (
        torch.arange(batch, dtype=torch.long, device=device)
        .reshape(batch, 1)
        .add(12_000 + historical + 257)
        .remainder(120_000)
        .add(1_000)
    )
    operation = Phase13OperationKey.create(configuration, batch, historical)
    with torch.inference_mode(), forced_flash_execution():
        with phase12._observable_cuda_graph_factory(torch) as observed_graphs:
            session = phase12._build_phase12_session(
                loaded=loaded,
                operation_key=operation,
                prefix_input_ids=prefix,
                decode_input_ids=decode,
            )
        if session.graph is None or len(observed_graphs) != 1:
            raise Phase13PilotError("captured graph is absent or ambiguous")
        graph_exec_before = int(session.graph.graph.raw_cuda_graph_exec())
        graph_path_before = phase12._write_cuda_graph_path_witness(
            graph=session.graph.graph,
            run_root=run_artifact_root,
            phase="before",
        )
        pointers_before = phase12._phase12_session_pointers(session)
        history_before = session.current_historical_prefix_sha256()
        warm_output = warmup_operations(
            session.graph.replay,
            count=WARMUP_STEPS,
            device=session.cache_device,
        )
        warm_cpu = warm_output.detach().to(device="cpu", copy=True).clone()
        warm_checksum = tensor_sha256_untimed(warm_cpu)
        warm_finite = bool(torch.isfinite(warm_cpu).all())
        replay_allocation = audit_cuda_allocations(
            session.graph.replay,
            device=session.cache_device,
        )
        audit_output = (
            session.graph.replay().detach().to(device="cpu", copy=True).clone()
        )
        torch.cuda.synchronize(device=session.cache_device)
        audit_checksum = tensor_sha256_untimed(audit_output)
        audit_finite = bool(torch.isfinite(audit_output).all())
        allocation_record = replay_allocation.to_dict()
        graph_passed = bool(
            session.graph_evidence is not None
            and session.graph_evidence.get("captured") is True
            and session.graph_evidence.get("fallback") is False
            and session.graph_evidence.get("consecutive_replay_outputs_exact") is True
            and session.eager_graph_comparison is not None
            and session.eager_graph_comparison.passed
        )
        allocation_passed = bool(
            replay_allocation.audit_available
            and replay_allocation.passed
            and replay_allocation.allocation_event_count == 0
            and replay_allocation.allocation_event_bytes == 0
            and replay_allocation.allocated_after == replay_allocation.allocated_before
            and replay_allocation.reserved_after == replay_allocation.reserved_before
        )
        family = phase12._method_family(configuration)
        geometry = session.gqa_cache_geometry()
        prior_g3 = phase12._expected_prior_g3_binding(family)
        phase12._validate_prior_g3_binding(prior_g3, family=family)
        successor_binding = None
        if family != "bf16":
            successor = _phase13b_successor_authority()["families"][family]
            geometry_key = f"{configuration}/B{batch}"
            if (
                successor["adapter_versions"].get(configuration)
                != session.method.adapter_version
                or geometry_key
                not in successor["adapter_config_fingerprints_l128"]
                or geometry_key
                not in successor["cache_layout_fingerprints_l128"]
            ):
                raise Phase13PilotError(
                    "Phase 13B runtime batch authority differs"
                )
            successor_binding = {
                "decision": "0030",
                "report_path": successor["report_path"],
                "report_sha256": successor["report_sha256"],
                "source_authority_commit": successor["creation_git_sha"],
                "geometry_key": geometry_key,
                "adapter_version": session.method.adapter_version,
                "admission_adapter_fingerprint_l128": successor[
                    "adapter_config_fingerprints_l128"
                ][geometry_key],
                "admission_cache_fingerprint_l128": successor[
                    "cache_layout_fingerprints_l128"
                ][geometry_key],
                "pilot_capacity": operation.capacity,
            }
        live_fingerprint = phase12._validate_runtime_adapter_fingerprint(
            method=session.method,
            cache=session.cache,
            observed=session.adapter_config_fingerprint,
        )
        runtime_context = session.method.runtime_context
        backend_verified = bool(
            runtime_context.model_id == phase12.PHASE12_MODEL_ID
            and runtime_context.model_revision == phase12.PHASE12_MODEL_REVISION
            and runtime_context.num_layers == 32
            and runtime_context.num_query_heads == 32
            and runtime_context.num_kv_heads == 8
            and runtime_context.head_dim == 128
            and re.fullmatch(r"[0-9a-f]{64}", runtime_context.backend_fingerprint)
        )
        geometry_passed = phase12._gqa_geometry_passes(geometry, family=family)
        path_audit = execution_path_audit_facade(
            backend_identity_verified=backend_verified,
            device_kernel_family_verified=bool(live_fingerprint),
            allocation_categories_verified=allocation_passed,
            temporary_tensor_shapes_verified=(
                pointers_before == phase12._phase12_session_pointers(session)
                and geometry_passed
            ),
            gqa_replication_detected=not geometry_passed,
            full_prefix_temporary_detected=False,
            host_synchronization_detected=False,
            backend_fallback_detected=not graph_passed,
            full_prefix_dequantization=(
                "not_applicable" if family == "bf16" else "verified_false"
            ),
        )
        phase12._validate_execution_path_record(path_audit.to_dict(), family=family)
        if (
            not warm_finite
            or not audit_finite
            or warm_checksum != audit_checksum
            or pointers_before != phase12._phase12_session_pointers(session)
            or history_before != session.current_historical_prefix_sha256()
        ):
            raise Phase13PilotError("post-capture warmup or audit drifted")
        session.graph_evidence["replay_allocation"] = allocation_record
        session.graph_evidence["phase13_warmup_replays"] = WARMUP_STEPS
        session.admit(
            observed_outputs=((audit_checksum, audit_finite),),
            execution_path_passed=path_audit.passed,
            allocation_passed=allocation_passed,
            graph_passed=graph_passed,
        )
        raw_runner = run_fixed_l(
            session,
            measured_steps=MEASURED_STEPS,
            measured_batches=MEASURED_BATCHES,
        ).to_dict()
    runner = phase12._normalize_runner_result(raw_runner)
    graph_exec_after = int(session.graph.graph.raw_cuda_graph_exec())
    graph_path_after = phase12._write_cuda_graph_path_witness(
        graph=session.graph.graph,
        run_root=run_artifact_root,
        phase="after",
    )
    if (
        graph_exec_before <= 0
        or graph_exec_after != graph_exec_before
        or graph_path_before["normalized_sha256"]
        != graph_path_after["normalized_sha256"]
    ):
        raise Phase13PilotError("measured graph topology changed")
    memory = runner.get("memory_evidence")
    if (
        runner.get("output_finite") is not True
        or runner.get("cache_pointers_stable") is not True
        or runner.get("historical_cache_unchanged") is not True
        or runner.get("output_checksum") != audit_checksum
        or not isinstance(memory, Mapping)
        or memory.get("timing_allocated_delta_bytes") != 0
        or memory.get("timing_reserved_delta_bytes") != 0
        or runner.get("r_hbm") is not None
    ):
        raise Phase13PilotError("Pilot run stability or allocation failed")
    timing_samples = runner["timing"]["samples"]
    process_median = statistics.median(
        float(sample["cuda_ms_per_operation"]) for sample in timing_samples
    )
    host_cuda_ratios = [
        (float(sample["host_ns_per_operation"]) / 1_000_000.0)
        / float(sample["cuda_ms_per_operation"])
        for sample in timing_samples
    ]
    if any(not math.isfinite(value) or value <= 0 for value in host_cuda_ratios):
        raise Phase13PilotError("host-wall/CUDA-event ratio is invalid")
    host_wall_cuda_event_ratio = statistics.median(host_cuda_ratios)
    telemetry_before = runner["telemetry_before"]
    telemetry_after = runner["telemetry_after"]
    temperature = phase12._telemetry_range(
        telemetry_before, telemetry_after, "temperature_celsius"
    )
    sm_clock = phase12._telemetry_range(
        telemetry_before, telemetry_after, "sm_clock_mhz"
    )
    memory_clock = phase12._telemetry_range(
        telemetry_before, telemetry_after, "memory_clock_mhz"
    )
    power = phase12._telemetry_range(telemetry_before, telemetry_after, "power_watts")
    owned_snapshot = phase12._capture_process_snapshot(
        supervised_pid=pid,
        supervised_start_ticks=start_ticks,
    )
    phase12._require_owned_snapshot(owned_snapshot, pid=pid, start_ticks=start_ticks)
    kernel_path_fingerprint = _canonical_sha256(
        {
            "configuration": configuration,
            "operation": operation.operation_fingerprint_sha256,
            "prior_g3": prior_g3,
            "phase13b_successor": successor_binding,
            "runtime_adapter_fingerprint": runner["adapter_config_fingerprint"],
            "cache_layout_fingerprint": runner["cache_layout_fingerprint"],
            "backend": runtime_context.backend_fingerprint,
            "graph_topology": graph_path_before["normalized_sha256"],
            "path_audit": path_audit.to_dict(),
        }
    )
    allocation_fingerprint = _canonical_sha256(
        {
            "cache_accounting": runner["cache_accounting"],
            "cache_byte_breakdown": runner["cache_byte_breakdown"],
            "replay_allocation": allocation_record,
        }
    )
    return {
        "schema_version": RUN_SCHEMA,
        "run_id": run_id,
        "method_config_id": configuration,
        "method_config_fingerprint": CONFIG_FINGERPRINTS[configuration],
        "method_family": family,
        "replicate_index": replicate_index,
        "order_index": order_index,
        "batch_size": batch,
        "context_label": context_label,
        "historical_context": historical,
        "total_attended_context": historical + 1,
        "execution_git_sha": git_sha,
        "authorized_container_digest": AUTHORIZED_CONTAINER_DIGEST,
        "gpu_uuid": GPU_UUID,
        "operation_fingerprint_sha256": operation.operation_fingerprint_sha256,
        "phase13b_successor_binding": successor_binding,
        "process_median_ms": process_median,
        "host_wall_cuda_event_ratio": host_wall_cuda_event_ratio,
        "kernel_count": int(graph_path_before["kernel_node_count"]),
        "output_checksum": runner["output_checksum"],
        "kernel_path_fingerprint": kernel_path_fingerprint,
        "allocation_fingerprint": allocation_fingerprint,
        "temperature_min_c": temperature[0],
        "temperature_max_c": temperature[1],
        "sm_clock_min_mhz": int(sm_clock[0]),
        "sm_clock_max_mhz": int(sm_clock[1]),
        "memory_clock_min_mhz": int(memory_clock[0]),
        "memory_clock_max_mhz": int(memory_clock[1]),
        "power_min_w": power[0],
        "power_max_w": power[1],
        "finite_output": True,
        "no_backend_fallback": path_audit.passed,
        "allocation_stable": True,
        "kernel_path_stable": True,
        "gpu_exclusive": True,
        "warmup_replays": WARMUP_STEPS,
        "measured_steps": MEASURED_STEPS,
        "measured_batches": MEASURED_BATCHES,
        "graph_replay_allocation": allocation_record,
        "graph_path_before": graph_path_before,
        "graph_path_after": graph_path_after,
        "runner": runner,
        "container_runtime_attestation": attestation,
        "gpu_process_before_cuda": pre_snapshot,
        "gpu_process_owned_after_measurement": owned_snapshot,
        "pilot_only": True,
        "quality_status": "unvalidated",
        "performance_claim_eligible": False,
        "r_hbm": None,
    }


def _run_id(campaign_id: str, record: Mapping[str, Any]) -> str:
    value = (
        f"{campaign_id}-r{record['replicate_index']}-o{int(record['order_index']):03d}-"
        f"{record['method_config_id']}-b{record['batch_size']}-l{record['context_label']}"
    )
    if _RUN_RE.fullmatch(value) is None:
        raise Phase13PilotError("Phase 13 run ID is invalid")
    return value


def _write_status_manifest(
    *,
    run_root: Path,
    campaign_id: str,
    run_id: str,
    record: Mapping[str, Any],
    status: str,
    reason: str | None,
) -> dict[str, Any]:
    manifest = {
        "schema_version": "kvbench-phase13-run-manifest-1.0.0",
        "campaign_id": campaign_id,
        "run_id": run_id,
        "status": status,
        "reason": reason,
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
    write_exclusive(run_root / "manifest.json", json_bytes(manifest))
    return manifest


def _write_nonlaunched_record(
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
                "schema_version": "kvbench-phase13-nonlaunch-disposition-1.0.0",
                "run_id": run_id,
                "status": status,
                "reason": reason,
                "cuda_process_launched": False,
                "record_preserved": True,
            }
        ),
    )
    return _write_status_manifest(
        run_root=run_root,
        campaign_id=campaign_id,
        run_id=run_id,
        record=record,
        status=status,
        reason=reason,
    )


def _supervision_passed(result: Any) -> bool:
    return phase12._supervision_passed(result)


def _run_one_process(
    *,
    stage: Path,
    campaign_id: str,
    record: Mapping[str, Any],
    git_sha: str,
) -> dict[str, Any]:
    run_id = _run_id(campaign_id, record)
    run_root = stage / "runs" / run_id
    run_root.mkdir()
    write_exclusive(
        run_root / "started.json",
        json_bytes(
            {
                "schema_version": "kvbench-phase13-run-start-1.0.0",
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
        str(REPOSITORY_ROOT / "scripts/phase13_pilot.py"),
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
    )
    result = run_supervised_command(
        command,
        working_directory=str(REPOSITORY_ROOT),
        environment=phase12._child_environment(),
        timeout_seconds=CHILD_TIMEOUT_SECONDS,
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
    if not _supervision_passed(result):
        stderr = result.stderr.decode("utf-8", errors="replace")
        if "requires frozen B=1" in stderr or "requires frozen layers=32 B=1" in stderr:
            reason = "adapter_static_cache_rejects_batch_greater_than_one"
        else:
            reason = "supervised_worker_failed"
        write_exclusive(
            run_root / "failure.json",
            json_bytes(
                {
                    "schema_version": "kvbench-phase13-run-failure-1.0.0",
                    "run_id": run_id,
                    "failure_reason": reason,
                    "worker_returncode": result.returncode,
                    "selective_retry_permitted": False,
                    "campaign_preserved": True,
                }
            ),
        )
        return _write_status_manifest(
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
        raise Phase13PilotError("worker result channel differs")
    payload = json.loads(matches[0])
    if not isinstance(payload, dict) or payload.get("run_id") != run_id:
        raise Phase13PilotError("worker result identity differs")
    write_exclusive(run_root / "result.json", json_bytes(payload))
    manifest = _write_status_manifest(
        run_root=run_root,
        campaign_id=campaign_id,
        run_id=run_id,
        record=record,
        status="completed",
        reason=None,
    )
    manifest["result_sha256"] = sha256_file(run_root / "result.json")
    # The initial write is immutable.  Bind the result in a separate receipt.
    write_exclusive(
        run_root / "result-binding.json",
        json_bytes(
            {
                "schema_version": "kvbench-phase13-run-result-binding-1.0.0",
                "run_id": run_id,
                "result_sha256": manifest["result_sha256"],
            }
        ),
    )
    return manifest


def run_campaign(*, stage: Path, campaign_id: str, git_sha: str) -> dict[str, Any]:
    identifier = _validate_campaign_id(campaign_id)
    phase12._require_authorized_container_runtime()
    if any(name in os.environ for name in _FORBIDDEN_ENVIRONMENT):
        raise Phase13PilotError("R2 or model credentials entered the measurement container")
    resolved = stage.resolve(strict=True)
    if not (resolved / "campaign-reservation.json").is_file():
        raise Phase13PilotError("Phase 13 campaign reservation is absent")
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
    if head != git_sha or status:
        raise Phase13PilotError("container source is not the clean execution commit")
    successor_authority = _phase13b_successor_authority()
    order = _strict_json(REPOSITORY_ROOT / ORDER_PATH)
    validate_execution_order(order)
    feasibility = build_feasibility_records(order)
    write_exclusive(resolved / "execution_order.json", json_bytes(order))
    write_exclusive(
        resolved / "unified" / "feasibility.json",
        json_bytes({"records": feasibility}),
    )
    write_exclusive(
        resolved / "campaign_manifest.json",
        json_bytes(
            {
                "schema_version": CAMPAIGN_SCHEMA,
                "campaign_id": identifier,
                "execution_git_sha": git_sha,
                "authorized_container_digest": AUTHORIZED_CONTAINER_DIGEST,
                "phase13b_successor_authority": successor_authority,
                "configurations": list(CONFIGURATIONS),
                "fingerprints": CONFIG_FINGERPRINTS,
                "batch_sizes": list(BATCH_SIZES),
                "context_labels": list(CONTEXT_LABELS),
                "top_context_mapping": {
                    "label": 131072,
                    "historical_prefix": 131071,
                    "total_attended": 131072,
                },
                "warmup_steps": WARMUP_STEPS,
                "measured_steps": MEASURED_STEPS,
                "measured_batches": MEASURED_BATCHES,
                "replicates": REPLICATES,
                "seeds": list(SEEDS),
                "planned_point_records": PLANNED_RECORD_COUNT,
                "execution_order_sha256": sha256_file(resolved / "execution_order.json"),
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
                _write_nonlaunched_record(
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
                _write_nonlaunched_record(
                    stage=resolved,
                    campaign_id=identifier,
                    record=record,
                    status="aborted",
                    reason=stop_reason,
                )
            )
            continue
        manifest = _run_one_process(
            stage=resolved,
            campaign_id=identifier,
            record=record,
            git_sha=git_sha,
        )
        manifests.append(manifest)
        if manifest["status"] == "runtime_failed":
            stop_reason = str(manifest["reason"])
    counts: dict[str, int] = defaultdict(int)
    for manifest in manifests:
        counts[str(manifest["status"])] += 1
    result = {
        "schema_version": "kvbench-phase13-local-campaign-result-1.0.0",
        "campaign_id": identifier,
        "execution_git_sha": git_sha,
        "planned_point_records": PLANNED_RECORD_COUNT,
        "status_counts": dict(sorted(counts.items())),
        "stopped_after_method_bug": stop_reason is not None,
        "stop_reason": stop_reason,
        "selective_reruns": 0,
        "full_scan": "CLOSED",
        "quality_execution": "LOCKED",
        "performance_data_frozen": False,
        "phase13_status": "BLOCKED" if stop_reason else "LOCAL_COMPLETE",
        "durable_publication": "PENDING_HOST_SIDE",
        "r_hbm": None,
    }
    write_exclusive(resolved / "unified" / "local-campaign.json", json_bytes(result))
    return result


def _run_records(root: Path) -> list[dict[str, Any]]:
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
                    result_path.relative_to(root).as_posix() if result is not None else None
                ),
                "result_sha256": sha256_file(result_path) if result is not None else None,
                "process_median_ms": (
                    float(result["process_median_ms"]) if result is not None else None
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
    if len(records) != PLANNED_RECORD_COUNT:
        raise Phase13PilotError("Phase 13 run-record cardinality differs")
    return records


def _nominal_ratio(configuration: str) -> float:
    if configuration == "bf16":
        return 1.0
    if configuration.startswith("tq_"):
        key_bits, value_bits = {
            "tq_4bit_nc": (4, 4),
            "tq_k3v4_nc": (3, 4),
            "tq_3bit_nc": (3, 3),
        }[configuration]
        return (32 * 32) / (28 * (key_bits + value_bits) + 4 * 32)
    if configuration in {"k4v4", "k2v4", "k2v2"}:
        key_bits, value_bits = {
            "k4v4": (4, 4),
            "k2v4": (2, 4),
            "k2v2": (2, 2),
        }[configuration]
        return 32 / (key_bits + value_bits)
    bits = {"kvq4": 4, "kvq3": 3, "kvq2": 2}[configuration]
    return 16 / bits


def _point_byte_features(completed: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not completed:
        return {
            key: None
            for key in (
                "data_payload_bytes",
                "logical_bf16_bytes",
                "allocated_bytes",
                "active_storage_bytes",
                "metadata_bytes",
                "residual_bytes",
                "sink_bytes",
                "outlier_value_bytes",
                "outlier_index_bytes",
                "padding_bytes",
                "workspace_bytes",
                "rho_alloc",
                "r_alloc",
                "r_nominal",
                "reciprocal_error",
                "kernel_count",
            )
        } | {"r_hbm": None}
    configuration = str(completed[0]["method_config_id"])
    runner = completed[0].get("runner")
    if not isinstance(runner, Mapping):
        raise Phase13PilotError("completed Pilot run lacks runner evidence")
    accounting = runner.get("cache_accounting")
    breakdown = runner.get("cache_byte_breakdown")
    if not isinstance(accounting, Mapping) or not isinstance(breakdown, Mapping):
        raise Phase13PilotError("completed Pilot run lacks byte accounting")
    if runner.get("r_hbm") is not None:
        raise Phase13PilotError("Pilot runner populated r_hbm")
    allocated = int(accounting["allocated_bytes"])
    if sum(int(value) for value in breakdown.values()) != allocated:
        raise Phase13PilotError("Pilot byte breakdown does not sum to allocation")
    batch = int(completed[0]["batch_size"])
    capacity = int(accounting["capacity"])
    logical = int(
        accounting.get(
            "logical_bf16_allocated_bytes",
            2 * 32 * batch * 8 * capacity * 128 * 2,
        )
    )
    family = phase12._method_family(configuration)
    metadata = residual = sink = outlier_values = outlier_indices = 0
    padding = workspace = data = 0
    if family == "bf16":
        data = int(breakdown["data_bytes"])
        metadata = sum(
            int(breakdown[key])
            for key in ("metadata_bytes", "scale_bytes", "zero_point_bytes")
        )
        padding = int(breakdown["padding_bytes"])
        workspace = int(breakdown["workspace_bytes"])
    elif family == "turboquant":
        data = sum(
            int(breakdown[key])
            for key in (
                "compressed_key_payload_bytes",
                "compressed_value_payload_bytes",
            )
        )
        metadata = sum(
            int(breakdown[key])
            for key in (
                "key_norm_metadata_bytes",
                "value_scale_metadata_bytes",
                "value_zero_point_metadata_bytes",
                "mapping_metadata_bytes",
            )
        )
        residual = int(breakdown["skipped_layer_bf16_key_bytes"]) + int(
            breakdown["skipped_layer_bf16_value_bytes"]
        )
        padding = int(breakdown["slot_padding_alignment_bytes"]) + int(
            breakdown["block_rounding_overhead_bytes"]
        )
        workspace = int(breakdown["persistent_workspace_bytes"])
    elif family == "kivi":
        data = int(breakdown["quantized_k_payload"]) + int(
            breakdown["quantized_v_payload"]
        )
        metadata = sum(
            int(breakdown[key])
            for key in (
                "key_scales",
                "key_zero_points",
                "value_scales",
                "value_zero_points",
                "other_metadata",
            )
        )
        residual = int(breakdown["residual_k"]) + int(breakdown["residual_v"])
        padding = int(breakdown["padding_alignment"]) + int(
            breakdown["block_group_rounding_bytes"]
        )
        workspace = sum(
            int(breakdown[key])
            for key in (
                "fp16_staging",
                "quantization_staging",
                "persistent_workspace",
                "value_rollover_shift_scratch",
            )
        )
    else:
        data = int(breakdown["dense_k_payload"]) + int(
            breakdown["dense_v_payload"]
        )
        metadata = sum(
            int(breakdown[key])
            for key in ("key_metadata", "value_metadata", "active_count_mask")
        )
        sink = int(breakdown["sink_k"]) + int(breakdown["sink_v"])
        outlier_values = int(breakdown["key_sparse_values"]) + int(
            breakdown["value_sparse_values"]
        )
        outlier_indices = int(breakdown["key_sparse_indices"]) + int(
            breakdown["value_sparse_indices"]
        )
        padding = int(breakdown["padding_alignment"])
        workspace = int(breakdown["staging"]) + int(
            breakdown["persistent_workspace"]
        )
    rho = allocated / logical
    r_alloc = logical / allocated
    return {
        "data_payload_bytes": data,
        "logical_bf16_bytes": logical,
        "allocated_bytes": allocated,
        "active_storage_bytes": accounting.get("active_storage_bytes"),
        "metadata_bytes": metadata,
        "residual_bytes": residual,
        "sink_bytes": sink,
        "outlier_value_bytes": outlier_values,
        "outlier_index_bytes": outlier_indices,
        "padding_bytes": padding,
        "workspace_bytes": workspace,
        "rho_alloc": rho,
        "r_alloc": r_alloc,
        "r_nominal": _nominal_ratio(configuration),
        "reciprocal_error": abs(rho * r_alloc - 1.0),
        "kernel_count": int(completed[0]["kernel_count"]),
        "r_hbm": None,
    }


def _point_summaries(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
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
    for configuration in CONFIGURATIONS:
        for batch in BATCH_SIZES:
            for label in CONTEXT_LABELS:
                matching = grouped[(configuration, batch, label)]
                statuses = [str(item["status"]) for item in matching]
                completed = [item for item in matching if item["status"] == "completed"]
                byte_features = _point_byte_features(completed)
                if len(completed) == REPLICATES:
                    stats = point_statistics(
                        [float(item["process_median_ms"]) for item in completed]
                    )
                    output_agreement = len(
                        {str(item["output_checksum"]) for item in completed}
                    ) == 1
                    path_agreement = len(
                        {str(item["kernel_path_fingerprint"]) for item in completed}
                    ) == 1
                    allocation_agreement = len(
                        {str(item["allocation_fingerprint"]) for item in completed}
                    ) == 1
                    kernel_count_agreement = len(
                        {int(item["kernel_count"]) for item in completed}
                    ) == 1
                    finite = all(item["finite_output"] is True for item in completed)
                    no_fallback = all(
                        item["no_backend_fallback"] is True for item in completed
                    )
                    gpu_exclusive = all(
                        item["gpu_exclusive"] is True for item in completed
                    )
                    run_stability = all(
                        item["allocation_stable"] is True
                        and item["kernel_path_stable"] is True
                        for item in completed
                    )
                    agreements = (
                        output_agreement
                        and path_agreement
                        and allocation_agreement
                        and kernel_count_agreement
                        and finite
                        and no_fallback
                        and gpu_exclusive
                        and run_stability
                    )
                    disposition = classify_point(
                        statistics_record=stats,
                        agreements=agreements,
                    )
                    host_ratio = statistics.median(
                        float(item["host_wall_cuda_event_ratio"])
                        for item in completed
                    )
                    process_medians = [
                        float(item["process_median_ms"])
                        for item in sorted(
                            completed,
                            key=lambda candidate: int(candidate["replicate_index"]),
                        )
                    ]
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
                        "power_min_w": min(
                            float(item["power_min_w"]) for item in completed
                        ),
                        "power_max_w": max(
                            float(item["power_max_w"]) for item in completed
                        ),
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
                    kernel_count_agreement = finite = no_fallback = gpu_exclusive = False
                    run_stability = False
                    host_ratio = None
                    process_medians = []
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
                    if all(status == "capacity_infeasible" for status in statuses):
                        disposition = "capacity_infeasible"
                    elif "runtime_failed" in statuses:
                        disposition = "runtime_failed"
                    else:
                        disposition = "aborted"
                summaries.append(
                    {
                        "method_config_id": configuration,
                        "method_config_fingerprint": CONFIG_FINGERPRINTS[configuration],
                        "batch_size": batch,
                        "context_label": label,
                        "historical_context": actual_historical_context(label),
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
                        "kernel_count_agreement": kernel_count_agreement,
                        "finite_outputs": finite,
                        "no_backend_fallback": no_fallback,
                        "gpu_exclusive": gpu_exclusive,
                        "within_process_stability": run_stability,
                        "disposition": disposition,
                        "monotonicity_warning": False,
                        "monotonicity_warning_only": True,
                        "quality_status": "unvalidated",
                        "performance_claim_eligible": False,
                        "r_hbm": None,
                    }
                )
    if len(summaries) != 270:
        raise Phase13PilotError("Phase 13 point-summary cardinality differs")
    for configuration in CONFIGURATIONS:
        for batch in BATCH_SIZES:
            available = [
                item
                for item in summaries
                if item["method_config_id"] == configuration
                and item["batch_size"] == batch
                and item["disposition"] in {"stable", "unstable"}
            ]
            available.sort(key=lambda item: int(item["context_label"]))
            previous = None
            for item in available:
                current = float(item["median_ms"])
                if previous is not None and current < previous:
                    item["monotonicity_warning"] = True
                previous = current
    return summaries

def _fit_records(summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for configuration in CONFIGURATIONS:
        for batch in BATCH_SIZES:
            all_matching = [
                item
                for item in summaries
                if item["method_config_id"] == configuration
                and item["batch_size"] == batch
            ]
            matching = [
                item
                for item in all_matching
                if item["disposition"] == "stable"
            ]
            if any(item["disposition"] == "unstable" for item in all_matching):
                fit = {"fit_status": "unstable_data"}
            else:
                fit = provisional_knee_fit(
                    [
                        (float(item["context_label"]), float(process_median))
                        for item in matching
                        for process_median in item["process_medians_ms"]
                    ]
                )
            knee_model = fit.get("knee_model")
            knee = (
                knee_model.get("L_star")
                if isinstance(knee_model, Mapping)
                else None
            )
            density = knee_density(
                [int(item["context_label"]) for item in matching],
                float(knee) if isinstance(knee, (int, float)) else None,
            )
            bootstrap = _session_bootstrap_knee(
                matching,
                seed=20260801 + 100 * CONFIGURATIONS.index(configuration) + batch,
            )
            constant = fit.get("constant_floor", {})
            linear = fit.get("linear", {})
            knee_fields = fit.get("knee_model", {})
            records.append(
                {
                    "method_config_id": configuration,
                    "batch_size": batch,
                    "stable_point_count": len(matching),
                    "session_observation_count": sum(
                        len(item["process_medians_ms"]) for item in matching
                    ),
                    "fit_status": fit["fit_status"],
                    "constant_tau": constant.get("tau"),
                    "linear_a": linear.get("a"),
                    "linear_s": linear.get("s"),
                    "knee_tau": knee_fields.get("tau"),
                    "knee_a": knee_fields.get("a"),
                    "knee_s": knee_fields.get("s"),
                    "L_star": knee_fields.get("L_star"),
                    "r_squared": knee_fields.get("r_squared"),
                    "bootstrap_estimable": bootstrap["estimable"],
                    "bootstrap_knee_lower_95": bootstrap["lower_95"],
                    "bootstrap_knee_upper_95": bootstrap["upper_95"],
                    "bootstrap_valid_draws": bootstrap["valid_draws"],
                    "fit_json": json.dumps(fit, sort_keys=True, separators=(",", ":")),
                    "density_json": json.dumps(
                        density, sort_keys=True, separators=(",", ":")
                    ),
                    "pilot_only": True,
                    "quality_status": "unvalidated",
                    "performance_claim_eligible": False,
                }
            )
    return records


def _session_bootstrap_knee(
    summaries: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    draws: int = 1000,
) -> dict[str, Any]:
    """Bootstrap process/session medians without treating decode steps as samples."""

    if len(summaries) < 4:
        return {
            "estimable": False,
            "draws": draws,
            "valid_draws": 0,
            "lower_95": None,
            "upper_95": None,
        }
    generator = random.Random(seed)
    estimates: list[float] = []
    for _ in range(draws):
        observations: list[tuple[float, float]] = []
        for summary in summaries:
            process_medians = [
                float(value) for value in summary["process_medians_ms"]
            ]
            if len(process_medians) != REPLICATES:
                raise Phase13PilotError("bootstrap input lacks three sessions")
            observations.extend(
                (
                    float(summary["context_label"]),
                    generator.choice(process_medians),
                )
                for _ in range(REPLICATES)
            )
        fit = provisional_knee_fit(observations)
        knee_model = fit.get("knee_model")
        knee = knee_model.get("L_star") if isinstance(knee_model, Mapping) else None
        if isinstance(knee, (int, float)) and math.isfinite(knee) and knee > 0:
            estimates.append(float(knee))
    minimum_valid = max(100, draws // 2)
    if len(estimates) < minimum_valid:
        return {
            "estimable": False,
            "draws": draws,
            "valid_draws": len(estimates),
            "lower_95": None,
            "upper_95": None,
        }
    estimates.sort()

    def percentile(fraction: float) -> float:
        position = fraction * (len(estimates) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return estimates[lower]
        weight = position - lower
        return estimates[lower] * (1.0 - weight) + estimates[upper] * weight

    return {
        "estimable": True,
        "draws": draws,
        "valid_draws": len(estimates),
        "lower_95": percentile(0.025),
        "upper_95": percentile(0.975),
    }


def _ratio_records(summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (str(item["method_config_id"]), int(item["batch_size"]), int(item["context_label"])): item
        for item in summaries
    }
    rows: list[dict[str, Any]] = []
    for item in summaries:
        if item["method_config_id"] == "bf16":
            continue
        baseline = by_key[("bf16", int(item["batch_size"]), int(item["context_label"]))]
        calculated = bool(
            item["disposition"] == "stable" and baseline["disposition"] == "stable"
        )
        ratio = (
            float(baseline["median_ms"]) / float(item["median_ms"])
            if calculated
            else None
        )
        rows.append(
            {
                "method_config_id": item["method_config_id"],
                "batch_size": item["batch_size"],
                "context_label": item["context_label"],
                "provisional_same_work_ratio": ratio,
                "calculated": calculated,
                "pilot_only": True,
                "quality_status": "unvalidated",
                "performance_claim_eligible": False,
                "not_quality_preserving_speedup": True,
            }
        )
    return rows


def _parquet_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as error:
        raise Phase13PilotError("pinned pyarrow analysis environment is required") from error
    normalized = []
    for row in rows:
        normalized.append(
            {
                key: (
                    json.dumps(value, sort_keys=True, separators=(",", ":"))
                    if isinstance(value, (dict, list, tuple))
                    else value
                )
                for key, value in row.items()
                if key != "runner"
            }
        )
    table = pa.Table.from_pylist(normalized)
    pq.write_table(
        table,
        path,
        compression="zstd",
        version="2.6",
        use_dictionary=False,
        write_statistics=True,
    )


def _svg_plot(path: Path, *, title: str, message: str) -> None:
    safe_title = (
        title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )
    safe_message = (
        message.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="540" '
        'viewBox="0 0 960 540">'
        '<rect width="960" height="540" fill="white"/>'
        f'<text x="48" y="72" font-family="sans-serif" font-size="28">{safe_title}</text>'
        f'<text x="48" y="140" font-family="sans-serif" font-size="18">{safe_message}</text>'
        '<text x="48" y="500" font-family="sans-serif" font-size="14">'
        'Pilot-only; quality unvalidated; not claim eligible</text></svg>'
    )
    write_exclusive(path, svg.encode("utf-8"))


def _svg_line_plot(
    path: Path,
    *,
    title: str,
    y_label: str,
    series: Mapping[str, Sequence[tuple[float, float]]],
    note: str = "Pilot-only; quality unvalidated; not claim eligible",
) -> None:
    colors = (
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#3366cc",
        "#dc3912",
    )

    def escape(value: str) -> str:
        return (
            value.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    normalized: dict[str, list[tuple[float, float]]] = {}
    for name, points in series.items():
        finite = sorted(
            (
                (float(x), float(y))
                for x, y in points
                if x > 0 and math.isfinite(float(x)) and math.isfinite(float(y))
            ),
            key=lambda item: item[0],
        )
        if finite:
            normalized[name] = finite
    if not normalized:
        _svg_plot(path, title=title, message="No eligible Pilot observations.")
        return
    all_points = [point for points in normalized.values() for point in points]
    transformed_x = [math.log2(point[0]) for point in all_points]
    y_values = [point[1] for point in all_points]
    x_min, x_max = min(transformed_x), max(transformed_x)
    y_min, y_max = min(y_values), max(y_values)
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    if y_min == y_max:
        padding = max(abs(y_min) * 0.05, 1e-9)
    else:
        padding = (y_max - y_min) * 0.08
    y_min -= padding
    y_max += padding
    left, right, top, bottom = 82.0, 790.0, 70.0, 520.0

    def x_coordinate(value: float) -> float:
        return left + (math.log2(value) - x_min) * (right - left) / (x_max - x_min)

    def y_coordinate(value: float) -> float:
        return bottom - (value - y_min) * (bottom - top) / (y_max - y_min)

    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="640" '
        'viewBox="0 0 1100 640">',
        '<rect width="1100" height="640" fill="white"/>',
        f'<text x="52" y="38" font-family="sans-serif" font-size="24">{escape(title)}</text>',
        f'<text x="18" y="295" transform="rotate(-90 18 295)" font-family="sans-serif" font-size="14">{escape(y_label)}</text>',
        '<text x="400" y="585" font-family="sans-serif" font-size="14">context length (log2 axis)</text>',
    ]
    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        y = y_coordinate(value)
        elements.extend(
            (
                f'<line x1="{left:.2f}" y1="{y:.2f}" x2="{right:.2f}" y2="{y:.2f}" stroke="#e5e5e5"/>',
                f'<text x="{left - 8:.2f}" y="{y + 4:.2f}" text-anchor="end" font-family="monospace" font-size="11">{value:.4g}</text>',
            )
        )
    x_ticks = sorted({point[0] for point in all_points})
    for value in x_ticks:
        x = x_coordinate(value)
        label = f"{int(value // 1024)}K" if value >= 1024 else f"{int(value)}"
        elements.extend(
            (
                f'<line x1="{x:.2f}" y1="{top:.2f}" x2="{x:.2f}" y2="{bottom:.2f}" stroke="#f0f0f0"/>',
                f'<text x="{x:.2f}" y="{bottom + 22:.2f}" text-anchor="middle" font-family="monospace" font-size="10">{label}</text>',
            )
        )
    elements.extend(
        (
            f'<line x1="{left:.2f}" y1="{bottom:.2f}" x2="{right:.2f}" y2="{bottom:.2f}" stroke="black"/>',
            f'<line x1="{left:.2f}" y1="{top:.2f}" x2="{left:.2f}" y2="{bottom:.2f}" stroke="black"/>',
        )
    )
    for index, (name, points) in enumerate(normalized.items()):
        color = colors[index % len(colors)]
        coordinates = [
            (x_coordinate(x), y_coordinate(y)) for x, y in points
        ]
        path_data = " ".join(
            f"{'M' if point_index == 0 else 'L'} {x:.2f} {y:.2f}"
            for point_index, (x, y) in enumerate(coordinates)
        )
        elements.append(
            f'<path d="{path_data}" fill="none" stroke="{color}" stroke-width="1.8"/>'
        )
        elements.extend(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}"/>'
            for x, y in coordinates
        )
        legend_y = 86 + 25 * index
        elements.extend(
            (
                f'<line x1="820" y1="{legend_y}" x2="846" y2="{legend_y}" stroke="{color}" stroke-width="2"/>',
                f'<text x="854" y="{legend_y + 4}" font-family="sans-serif" font-size="12">{escape(name)}</text>',
            )
        )
    elements.extend(
        (
            f'<text x="52" y="620" font-family="sans-serif" font-size="12">{escape(note)}</text>',
            "</svg>",
        )
    )
    write_exclusive(path, "".join(elements).encode("utf-8"))


def _render_pilot_plots(
    root: Path,
    summaries: Sequence[Mapping[str, Any]],
    ratios: Sequence[Mapping[str, Any]],
) -> None:
    eligible = [
        item
        for item in summaries
        if item["disposition"] in {"stable", "unstable"}
    ]

    def summary_series(
        rows: Sequence[Mapping[str, Any]],
        *,
        value_key: str,
        include_batch: bool,
    ) -> dict[str, list[tuple[float, float]]]:
        result: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for row in rows:
            value = row.get(value_key)
            if not isinstance(value, (int, float)):
                continue
            name = str(row["method_config_id"])
            if include_batch:
                name = f"{name}/B{int(row['batch_size'])}"
            result[name].append((float(row["context_label"]), float(value)))
        return dict(sorted(result.items()))

    for batch in BATCH_SIZES:
        rows = [item for item in eligible if int(item["batch_size"]) == batch]
        _svg_line_plot(
            root / "pilot_plots" / f"latency-vs-context-b{batch}.svg",
            title=f"Pilot T versus L, B={batch}",
            y_label="median CUDA ms per operation",
            series=summary_series(rows, value_key="median_ms", include_batch=False),
        )
    families = {
        "bf16": ("bf16",),
        "turboquant": ("tq_4bit_nc", "tq_k3v4_nc", "tq_3bit_nc"),
        "kivi": ("k4v4", "k2v4", "k2v2"),
        "kvquant": ("kvq4", "kvq3", "kvq2"),
    }
    for family, configurations in families.items():
        rows = [
            item
            for item in eligible
            if item["method_config_id"] in configurations
        ]
        _svg_line_plot(
            root / "pilot_plots" / f"method-{family}.svg",
            title=f"Pilot method view: {family}",
            y_label="median CUDA ms per operation",
            series=summary_series(rows, value_key="median_ms", include_batch=True),
        )
    _svg_line_plot(
        root / "pilot_plots" / "cv-vs-context.svg",
        title="Pilot CV versus context",
        y_label="process-median CV",
        series=summary_series(eligible, value_key="cv", include_batch=True),
    )
    _svg_line_plot(
        root / "pilot_plots" / "allocated-ratio-vs-context.svg",
        title="Pilot allocated-byte ratio versus context",
        y_label="rho_alloc",
        series=summary_series(eligible, value_key="rho_alloc", include_batch=True),
    )
    ratio_rows = [item for item in ratios if item["calculated"] is True]
    _svg_line_plot(
        root / "pilot_plots" / "pilot-only-same-work-ratio.svg",
        title="Pilot-only same-work ratio versus context",
        y_label="BF16 median / method median",
        series=summary_series(
            ratio_rows,
            value_key="provisional_same_work_ratio",
            include_batch=True,
        ),
        note=(
            "Pilot-only; quality unvalidated; performance claim ineligible; "
            "not quality-preserving speedup"
        ),
    )


def materialize_analysis(root: Path) -> dict[str, Any]:
    campaign = _strict_json(root / "unified" / "local-campaign.json")
    runs = _run_records(root)
    summaries = _point_summaries(runs)
    fits = _fit_records(summaries)
    ratios = _ratio_records(summaries)
    feasibility = _strict_json(root / "unified" / "feasibility.json")["records"]
    exclusions = [
        {
            "run_id": record["run_id"],
            "method_config_id": record["method_config_id"],
            "batch_size": record["batch_size"],
            "context_label": record["context_label"],
            "status": record["status"],
            "reason": record["reason"],
            "machine_readable": True,
        }
        for record in runs
        if record["status"] != "completed"
    ]
    _parquet_rows(root / "feasibility.parquet", feasibility)
    _parquet_rows(root / "raw_run_index.parquet", runs)
    _parquet_rows(root / "point_summary.parquet", summaries)
    _parquet_rows(root / "provisional_knees.parquet", fits)
    _parquet_rows(root / "exclusions.parquet", exclusions)
    stable = [item for item in summaries if item["disposition"] == "stable"]
    unstable = [item for item in summaries if item["disposition"] == "unstable"]
    failed = [item for item in summaries if item["disposition"] == "failed"]
    evaluated = stable + unstable + failed
    maximum_cv = max((float(item["cv"]) for item in evaluated), default=None)
    status_counts = {
        str(key): int(value) for key, value in campaign["status_counts"].items()
    }
    feasible_records = sum(item["status"] == "feasible" for item in feasibility)
    capacity_records = sum(
        item["status"] == "capacity_infeasible" for item in feasibility
    )
    fit_status_counts = dict(
        sorted(
            {
                status: sum(item["fit_status"] == status for item in fits)
                for status in {str(item["fit_status"]) for item in fits}
            }.items()
        )
    )
    density_records = [json.loads(str(item["density_json"])) for item in fits]
    assessed_density = [item for item in density_records if item["assessed"] is True]
    insufficient_density = [
        item for item in assessed_density if item["sufficient"] is not True
    ]
    complete_status_contract = bool(
        status_counts.get("completed", 0) == feasible_records
        and status_counts.get("capacity_infeasible", 0) == capacity_records
        and sum(status_counts.values()) == PLANNED_RECORD_COUNT
        and not any(
            status_counts.get(status, 0)
            for status in (
                "runtime_failed",
                "graph_capture_failed",
                "allocation_failed",
                "backend_fallback",
                "aborted",
            )
        )
    )
    local_pass = bool(
        campaign["phase13_status"] == "LOCAL_COMPLETE"
        and complete_status_contract
        and not unstable
        and not failed
    )
    if campaign["stop_reason"]:
        blocker = campaign["stop_reason"]
    elif unstable:
        blocker = f"{len(unstable)} Pilot points exceed the frozen CV threshold"
    elif failed:
        blocker = f"{len(failed)} Pilot points fail agreement or execution-path QC"
    elif not complete_status_contract:
        blocker = "Pilot run statuses do not account for the frozen grid"
    else:
        blocker = None
    phase13_status = "LOCAL_PASS_PENDING_PUBLICATION" if local_pass else "BLOCKED"

    def observed_range(minimum_key: str, maximum_key: str) -> list[float] | None:
        minimums = [
            float(item[minimum_key])
            for item in evaluated
            if isinstance(item.get(minimum_key), (int, float))
        ]
        maximums = [
            float(item[maximum_key])
            for item in evaluated
            if isinstance(item.get(maximum_key), (int, float))
        ]
        return [min(minimums), max(maximums)] if minimums and maximums else None

    qc = {
        "schema_version": "kvbench-phase13-pilot-qc-1.0.0",
        "campaign_id": campaign["campaign_id"],
        "campaign_execution_status": campaign["phase13_status"],
        "phase13_status": phase13_status,
        "planned_point_records": PLANNED_RECORD_COUNT,
        "feasible_gpu_run_records": feasible_records,
        "capacity_infeasible_run_records": capacity_records,
        "capacity_infeasible_points": sum(
            item["disposition"] == "capacity_infeasible" for item in summaries
        ),
        "status_counts": status_counts,
        "stable_points": len(stable),
        "unstable_points": len(unstable),
        "failed_points": len(failed),
        "maximum_cv": maximum_cv,
        "output_mismatches": sum(
            item["output_checksum_agreement"] is not True for item in evaluated
        ),
        "kernel_path_drift": sum(
            item["kernel_path_agreement"] is not True for item in evaluated
        ),
        "allocation_drift": sum(
            item["allocation_agreement"] is not True for item in evaluated
        ),
        "nan_or_inf_points": sum(
            item["finite_outputs"] is not True for item in evaluated
        ),
        "backend_fallback_points": sum(
            item["no_backend_fallback"] is not True for item in evaluated
        ),
        "gpu_exclusivity_failures": sum(
            item["gpu_exclusive"] is not True for item in evaluated
        ),
        "monotonicity_warnings": sum(
            item["monotonicity_warning"] is True for item in summaries
        ),
        "host_wall_cuda_event_ratio_range": observed_range(
            "host_wall_cuda_event_ratio", "host_wall_cuda_event_ratio"
        ),
        "temperature_range_c": observed_range(
            "temperature_min_c", "temperature_max_c"
        ),
        "sm_clock_range_mhz": observed_range(
            "sm_clock_min_mhz", "sm_clock_max_mhz"
        ),
        "memory_clock_range_mhz": observed_range(
            "memory_clock_min_mhz", "memory_clock_max_mhz"
        ),
        "power_range_w": observed_range("power_min_w", "power_max_w"),
        "maximum_reciprocal_error": max(
            (
                float(item["reciprocal_error"])
                for item in evaluated
                if isinstance(item.get("reciprocal_error"), (int, float))
            ),
            default=None,
        ),
        "byte_features_complete": all(
            item["r_hbm"] is None
            and isinstance(item.get("allocated_bytes"), int)
            and isinstance(item.get("logical_bf16_bytes"), int)
            for item in evaluated
        ),
        "fit_status_counts": fit_status_counts,
        "fit_records": len(fits),
        "session_bootstrap_unit": "independent_process_median",
        "bootstrap_intervals_estimable": sum(
            item["bootstrap_estimable"] is True for item in fits
        ),
        "valid_provisional_knees": sum(
            item["fit_status"]
            in {"knee_observed", "knee_below_range", "knee_above_range"}
            for item in fits
        ),
        "knee_density_assessed": len(assessed_density),
        "knee_density_sufficient": len(assessed_density) - len(insufficient_density),
        "knee_density_insufficient": len(insufficient_density),
        "densification_required": bool(insufficient_density),
        "provisional_ratio_records": len(ratios),
        "provisional_ratios_calculated": sum(item["calculated"] for item in ratios),
        "selective_reruns": 0,
        "monotonicity_warning_only": True,
        "r_hbm": None,
        "full_scan": "CLOSED",
        "quality_execution": "LOCKED",
        "performance_data_frozen": False,
        "performance_claim_eligible": False,
        "phase14_readiness": (
            "DENSIFICATION_REQUIRED"
            if local_pass and insufficient_density
            else "READY"
            if local_pass
            else "NOT_READY"
        ),
        "blocker": blocker,
    }
    write_exclusive(root / "pilot_qc.json", json_bytes(qc))
    report = "\n".join(
        (
            "# Phase 13 Pilot QC",
            "",
            f"- Campaign: `{campaign['campaign_id']}`",
            f"- Status: `{phase13_status}`",
            f"- Planned point records: {PLANNED_RECORD_COUNT}",
            f"- Status counts: `{json.dumps(status_counts, sort_keys=True)}`",
            f"- Stable/unstable/failed points: {len(stable)}/{len(unstable)}/{len(failed)}",
            f"- Maximum process CV: `{maximum_cv}`",
            f"- Fit statuses: `{json.dumps(fit_status_counts, sort_keys=True)}`",
            f"- Knee-density insufficiencies: {len(insufficient_density)}",
            f"- Phase 14 readiness: `{qc['phase14_readiness']}`",
            f"- Blocker: `{blocker}`",
            "- Full Scan: `CLOSED`",
            "- Quality execution: `LOCKED`",
            "- No speedup, HBM, capacity, quality, or final-knee claim is made.",
            "",
        )
    )
    write_exclusive(root / "pilot_qc_report.md", report.encode("utf-8"))
    _render_pilot_plots(root, summaries, ratios)
    inventory = {
        "schema_version": "kvbench-phase13-scientific-inventory-1.0.0",
        "campaign_id": campaign["campaign_id"],
        "planned_records": PLANNED_RECORD_COUNT,
        "raw_run_records": len(runs),
        "point_summaries": len(summaries),
        "fit_records": len(fits),
        "exclusions": len(exclusions),
        "plot_count": 10,
        "r_hbm": None,
    }
    write_exclusive(root / "inventory.json", json_bytes(inventory))
    return qc


def _payload_paths(root: Path, excluded: set[str]) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise Phase13PilotError("Phase 13 campaign contains a symlink")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise Phase13PilotError("Phase 13 campaign contains an unsafe file")
        if path.relative_to(root).as_posix() not in excluded:
            files.append(path)
    return files


def seal_campaign(stage: Path, *, campaign_id: str) -> Path:
    identifier = _validate_campaign_id(campaign_id)
    root = stage.resolve(strict=True)
    qc = _strict_json(root / "pilot_qc.json")
    if qc.get("campaign_id") != identifier:
        raise Phase13PilotError("Phase 13 QC campaign identity differs")
    manifest = {
        "schema_version": "kvbench-phase13-artifact-manifest-1.0.0",
        "run_id": identifier,
        "campaign_id": identifier,
        "status": qc["phase13_status"],
        "created_at_utc": _utc_now(),
        "authorized_container_digest": AUTHORIZED_CONTAINER_DIGEST,
        "planned_point_records": PLANNED_RECORD_COUNT,
        "append_only": True,
        "complete_written_last": True,
        "quality_status": "unvalidated",
        "performance_claim_eligible": False,
        "r_hbm": None,
    }
    write_exclusive(root / "manifest.json", json_bytes(manifest))
    inventory_items = [
        {
            "path": path.relative_to(root).as_posix(),
            "role": "phase13_pilot_evidence",
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in _payload_paths(
            root, {"artifact_inventory.json", "checksums.sha256", "COMPLETE"}
        )
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
                "status": qc["phase13_status"],
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
        raise Phase13PilotError("Phase 13 finalized campaign already exists")
    rename_noreplace(root, final)
    for path in sorted(final.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    final.chmod(0o555)
    validate_campaign(final, expected_campaign_id=identifier)
    return final


def validate_campaign(root: Path, *, expected_campaign_id: str | None = None) -> dict[str, Any]:
    artifact = validate_local_artifact(root, environ={})
    manifest = _strict_json(root / "manifest.json")
    identifier = str(manifest.get("campaign_id"))
    _validate_campaign_id(identifier)
    if expected_campaign_id is not None and identifier != expected_campaign_id:
        raise Phase13PilotError("Phase 13 campaign identity differs")
    runs = _run_records(root)
    order = _strict_json(root / "execution_order.json")
    validate_execution_order(order)
    qc = _strict_json(root / "pilot_qc.json")
    if (
        qc.get("campaign_id") != identifier
        or qc.get("planned_point_records") != PLANNED_RECORD_COUNT
        or qc.get("full_scan") != "CLOSED"
        or qc.get("quality_execution") != "LOCKED"
        or qc.get("r_hbm") is not None
        or sum(qc.get("status_counts", {}).values()) != PLANNED_RECORD_COUNT
        or len(runs) != PLANNED_RECORD_COUNT
    ):
        raise Phase13PilotError("Phase 13 QC or run topology differs")
    return {
        "status": "PASS",
        "campaign_id": identifier,
        "phase13_status": qc["phase13_status"],
        "root_sha256": artifact.root_sha256,
        "object_count": len(artifact.files),
    }


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--write-execution-order", action="store_true")
    actions.add_argument("--validate-execution-order", action="store_true")
    actions.add_argument("--validate-phase13b-entry", action="store_true")
    actions.add_argument("--print-feasibility-summary", action="store_true")
    actions.add_argument("--new-campaign-id", action="store_true")
    actions.add_argument("--reserve-campaign", action="store_true")
    actions.add_argument("--run-campaign", action="store_true")
    actions.add_argument("--run-worker", action="store_true")
    actions.add_argument("--materialize-analysis", action="store_true")
    actions.add_argument("--finalize-staged-campaign", action="store_true")
    actions.add_argument("--validate-campaign", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--git-sha")
    parser.add_argument("--campaign-id")
    parser.add_argument("--stage", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--configuration", choices=CONFIGURATIONS)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--context-label", type=int)
    parser.add_argument("--replicate-index", type=int)
    parser.add_argument("--order-index", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--run-artifact-root", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_arguments(argv)
    if args.write_execution_order:
        if args.output is None:
            raise Phase13PilotError("--output is required")
        write_exclusive(args.output, json_bytes(derive_execution_order()))
        return 0
    if args.validate_execution_order:
        path = ORDER_PATH if args.output is None else args.output
        validate_execution_order(json.loads(path.read_text(encoding="utf-8")))
        print(json.dumps({"status": "PASS", "records": PLANNED_RECORD_COUNT}))
        return 0
    if args.validate_phase13b_entry:
        print(json.dumps(validate_phase13b_entry(), sort_keys=True))
        return 0
    if args.print_feasibility_summary:
        order = json.loads((ORDER_PATH if args.output is None else args.output).read_text())
        records = build_feasibility_records(order)
        print(
            json.dumps(
                {
                    "planned": len(records),
                    "feasible": sum(item["status"] == "feasible" for item in records),
                    "capacity_infeasible": sum(
                        item["status"] == "capacity_infeasible" for item in records
                    ),
                    "entry_geometry_unsupported": sum(
                        not item["adapter_geometry_supported_at_entry"] for item in records
                    ),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.new_campaign_id:
        if args.git_sha is None:
            raise Phase13PilotError("--git-sha is required")
        print(new_campaign_id(args.git_sha))
        return 0
    if args.reserve_campaign:
        if args.campaign_id is None or args.git_sha is None:
            raise Phase13PilotError("campaign ID and Git SHA are required")
        print(reserve_campaign(campaign_id=args.campaign_id, git_sha=args.git_sha))
        return 0
    if args.run_campaign:
        if args.stage is None or args.campaign_id is None or args.git_sha is None:
            raise Phase13PilotError("stage, campaign ID, and Git SHA are required")
        print(
            json.dumps(
                run_campaign(
                    stage=args.stage,
                    campaign_id=args.campaign_id,
                    git_sha=args.git_sha,
                ),
                sort_keys=True,
            )
        )
        return 0
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
        )
        if any(value is None for value in required):
            raise Phase13PilotError("worker identity arguments are required")
        payload = _run_worker(
            run_id=str(args.run_id),
            configuration=str(args.configuration),
            batch=int(args.batch_size),
            context_label=int(args.context_label),
            replicate_index=int(args.replicate_index),
            order_index=int(args.order_index),
            git_sha=str(args.git_sha),
            run_artifact_root=Path(args.run_artifact_root),
        )
        print(WORKER_PREFIX + json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return 0
    if args.materialize_analysis:
        if args.stage is None:
            raise Phase13PilotError("--stage is required")
        print(json.dumps(materialize_analysis(args.stage), sort_keys=True))
        return 0
    if args.finalize_staged_campaign:
        if args.stage is None or args.campaign_id is None:
            raise Phase13PilotError("stage and campaign ID are required")
        print(seal_campaign(args.stage, campaign_id=args.campaign_id))
        return 0
    if args.validate_campaign:
        if args.artifact is None:
            raise Phase13PilotError("--artifact is required")
        print(json.dumps(validate_campaign(args.artifact), sort_keys=True))
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
