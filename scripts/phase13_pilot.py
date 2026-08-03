"""Narrow Phase 13 Pilot coordinator, QC, and provisional analysis.

The module owns no adapter, CUDA kernel, cache layout, or timing boundary.  It
reuses the Phase 12 authority/session bridge and the common fixed-L runner,
adds the preregistered Pilot grid, and keeps every planned point append-only.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
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
import time
import types
from typing import Any

from preflight.run_preflight import json_bytes, rename_noreplace, write_exclusive
from kvbench.runtime.artifacts import sha256_file
from kvbench.runtime.process_supervision import run_stage_supervised_command
from kvbench.schema import GraphMode, RunnerKind, canonical_json_bytes, sha256_hex
from kvbench.schema.phase13b import (
    PHASE13B_BATCH_SIZES,
    PHASE13B_FAMILY_CONFIGURATIONS,
    Phase13BMethodAdmissionReport,
)
from scripts.r2_artifact import validate_local_artifact
from scripts.phase13_prefix_state import (
    Phase13PrefixStateError,
    restore_prefix_state,
    save_prefix_state,
    validate_prefix_state,
)
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
PREFIX_HIDDEN_WIDTH = 4096
PREFIX_INTERMEDIATE_WIDTH = 14_336
PREFIX_QUERY_HEADS = 32
PREFIX_KV_HEADS = 8
PREFIX_HEAD_DIM = 128
PREFIX_DTYPE_BYTES = 2
PREFIX_INDEX_BYTES = 8
INPUT_RECIPE_SCHEMA = "kvbench-phase13-pilot-input-1.0.0"
CAMPAIGN_SCHEMA = "kvbench-phase13-pilot-campaign-1.0.0"
RUN_SCHEMA = "kvbench-phase13-pilot-process-run-1.0.0"
WORKER_PREFIX = "PHASE13_WORKER_RESULT="
PREFIX_BUILDER_PREFIX = "PHASE13_PREFIX_BUILDER_RESULT="
STAGE_EVENT_SCHEMA = "kvbench-phase13-stage-event-1.0.0"
STAGE_OBSERVER_POLL_SECONDS = 5.0
STAGE_SEQUENCE = (
    ("model_load", "started"),
    ("model_load", "completed"),
    ("prefix_construction", "started"),
    ("prefix_construction", "completed"),
    ("graph_capture", "started"),
    ("graph_capture", "completed"),
    ("warmup_and_audit", "started"),
    ("warmup_and_audit", "completed"),
    ("measurement", "started"),
    ("measurement", "completed"),
    ("finalization", "started"),
    ("finalization", "completed"),
)
PREFIX_BUILD_STAGE_SEQUENCE = (
    ("model_load", "started"),
    ("model_load", "completed"),
    ("prefix_construction", "started"),
    ("prefix_construction", "completed"),
    ("finalization", "started"),
    ("finalization", "completed"),
)
PREFIX_EQUIVALENCE_CONTEXT = 17
PREFIX_EQUIVALENCE_SOURCE_BATCH = 8
FIXED_STAGE_TIMEOUTS_SECONDS = {
    "startup": 600.0,
    "transition": 600.0,
    "model_load": 900.0,
    "measurement": 7_200.0,
    "finalization": 1_800.0,
}

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


def _validate_timeout_geometry(*, batch: int, historical: int) -> None:
    if (
        not isinstance(batch, int)
        or isinstance(batch, bool)
        or batch not in BATCH_SIZES
        or not isinstance(historical, int)
        or isinstance(historical, bool)
        or historical
        not in {
            actual_historical_context(label) for label in CONTEXT_LABELS
        }
    ):
        raise Phase13PilotError("prefix timeout geometry is invalid")


def prefix_construction_timeout_seconds(*, batch: int, historical: int) -> float:
    """Return the frozen point-scaled, finite prefix-construction deadline."""

    _validate_timeout_geometry(batch=batch, historical=historical)
    return float(max(3_600, 1_800 + math.ceil(batch * historical / 4)))


def graph_capture_timeout_seconds(*, batch: int, historical: int) -> float:
    """Bound capture plus its one full-history stability checksum."""

    _validate_timeout_geometry(batch=batch, historical=historical)
    return float(max(7_200, 1_800 + math.ceil(batch * historical / 20)))


def warmup_audit_timeout_seconds(*, batch: int, historical: int) -> float:
    """Bound warmup/audit plus its two full-history stability checksums."""

    _validate_timeout_geometry(batch=batch, historical=historical)
    return float(max(10_800, 3_600 + math.ceil(batch * historical / 10)))


def stage_timeout_contract(*, batch: int, historical: int) -> dict[str, float]:
    """Bind every worker stage without allowing heartbeat-based extensions."""

    return {
        **FIXED_STAGE_TIMEOUTS_SECONDS,
        "prefix_construction": prefix_construction_timeout_seconds(
            batch=batch,
            historical=historical,
        ),
        "graph_capture": graph_capture_timeout_seconds(
            batch=batch,
            historical=historical,
        ),
        "warmup_and_audit": warmup_audit_timeout_seconds(
            batch=batch,
            historical=historical,
        ),
    }


class _WorkerStageRecorder:
    """Write one exclusive, ordered file for each worker stage transition."""

    def __init__(
        self,
        *,
        root: Path,
        run_id: str,
        stage_sequence: Sequence[tuple[str, str]] = STAGE_SEQUENCE,
    ) -> None:
        self.root = root.resolve(strict=True)
        if (
            self.root.name != "stage-progress"
            or self.root.is_symlink()
            or any(self.root.iterdir())
            or _RUN_RE.fullmatch(run_id) is None
        ):
            raise Phase13PilotError("worker stage-progress root differs")
        self.run_id = run_id
        self.stage_sequence = tuple(stage_sequence)
        if not self.stage_sequence:
            raise Phase13PilotError("worker stage sequence is empty")
        self.next_sequence = 1

    def record(self, stage: str, state: str) -> None:
        if self.next_sequence > len(self.stage_sequence):
            raise Phase13PilotError("worker stage sequence overflowed")
        if (stage, state) != self.stage_sequence[self.next_sequence - 1]:
            raise Phase13PilotError("worker stage transition differs")
        payload = {
            "schema_version": STAGE_EVENT_SCHEMA,
            "sequence": self.next_sequence,
            "run_id": self.run_id,
            "stage": stage,
            "state": state,
            "recorded_at_utc": _utc_now(),
            "monotonic_ns": time.monotonic_ns(),
        }
        write_exclusive(
            self.root
            / f"{self.next_sequence:02d}-{stage}-{state}.json",
            json_bytes(payload),
        )
        self.next_sequence += 1


def _read_stage_observations(
    *,
    root: Path,
    run_id: str,
    stage_sequence: Sequence[tuple[str, str]] = STAGE_SEQUENCE,
) -> list[dict[str, object]]:
    """Read exact append-only stage events for the direct-child supervisor."""

    if (
        not root.is_dir()
        or root.is_symlink()
        or root.name != "stage-progress"
        or _RUN_RE.fullmatch(run_id) is None
    ):
        raise Phase13PilotError("stage observation root differs")
    paths = sorted(root.iterdir())
    sequence = tuple(stage_sequence)
    if not sequence or len(paths) > len(sequence):
        raise Phase13PilotError("stage observation count overflowed")
    observations: list[dict[str, object]] = []
    for index, path in enumerate(paths, start=1):
        stage, state = sequence[index - 1]
        expected_name = f"{index:02d}-{stage}-{state}.json"
        try:
            mode = path.lstat().st_mode
        except OSError as error:
            raise Phase13PilotError("stage observation stat failed") from error
        if (
            path.name != expected_name
            or path.is_symlink()
            or not stat.S_ISREG(mode)
        ):
            raise Phase13PilotError("stage observation path differs")
        payload = _strict_json(path)
        if (
            set(payload)
            != {
                "schema_version",
                "sequence",
                "run_id",
                "stage",
                "state",
                "recorded_at_utc",
                "monotonic_ns",
            }
            or payload.get("schema_version") != STAGE_EVENT_SCHEMA
            or payload.get("sequence") != index
            or payload.get("run_id") != run_id
            or payload.get("stage") != stage
            or payload.get("state") != state
            or not isinstance(payload.get("recorded_at_utc"), str)
            or not isinstance(payload.get("monotonic_ns"), int)
            or isinstance(payload.get("monotonic_ns"), bool)
        ):
            raise Phase13PilotError("stage observation payload differs")
        observations.append(
            {
                "sequence": index,
                "stage": stage,
                "state": state,
                "monotonic_ns": int(payload["monotonic_ns"]),
                "event_sha256": sha256_file(path),
            }
        )
    return observations


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


def prefix_construction_memory(
    *,
    batch: int,
    historical_context: int,
) -> dict[str, int | str]:
    """Return the source-derived full-prefix construction high-water mark.

    ``BF16DecodeEndpoint.prefill`` keeps the prefix token IDs, position tensor,
    RoPE tables, residual, and normalized hidden state live while a layer is
    evaluated.  The attention projection peak includes Q/K/V, the temporary
    RoPE halves, the Flash output and the output-projection result.  The larger
    Llama MLP peak is the two live hidden tensors plus the two intermediate
    operands and their out-of-place product.  These are setup-only bytes; they
    are never benchmark timing or adapter-owned cache storage.
    """

    if batch not in BATCH_SIZES or historical_context <= 0:
        raise Phase13PilotError("prefix construction geometry differs")
    token_rows = batch * historical_context
    hidden = token_rows * PREFIX_HIDDEN_WIDTH * PREFIX_DTYPE_BYTES
    intermediate = (
        token_rows * PREFIX_INTERMEDIATE_WIDTH * PREFIX_DTYPE_BYTES
    )
    query = hidden
    key_or_value = (
        token_rows * PREFIX_KV_HEADS * PREFIX_HEAD_DIM * PREFIX_DTYPE_BYTES
    )
    query_rope_half = query // 2
    key_rope_half = key_or_value // 2
    attention_peak = (
        2 * hidden
        + query
        + 2 * key_or_value
        + query_rope_half
        + key_rope_half
        + 2 * hidden
    )
    mlp_peak = 2 * hidden + 3 * intermediate
    token_ids = token_rows * PREFIX_INDEX_BYTES
    cache_positions = historical_context * PREFIX_INDEX_BYTES
    rope_tables = (
        2 * historical_context * PREFIX_HEAD_DIM * PREFIX_DTYPE_BYTES
    )
    fixed_decode_inputs = batch * PREFIX_INDEX_BYTES + PREFIX_INDEX_BYTES + (
        2 * PREFIX_HEAD_DIM * PREFIX_DTYPE_BYTES
    )
    control_tensors = (
        token_ids + cache_positions + rope_tables + fixed_decode_inputs
    )
    return {
        "formula_id": "phase13f-source-prefix-peak-v1",
        "prefix_token_id_bytes": token_ids,
        "prefix_cache_position_bytes": cache_positions,
        "prefix_rope_table_bytes": rope_tables,
        "fixed_decode_input_bytes": fixed_decode_inputs,
        "prefix_control_tensor_bytes": control_tensors,
        "hidden_bf16_bytes": hidden,
        "intermediate_bf16_bytes": intermediate,
        "attention_output_projection_peak_bytes": attention_peak,
        "mlp_peak_bytes": mlp_peak,
        "prefix_compute_peak_bytes": max(attention_peak, mlp_peak),
    }


def endpoint_workspace_bytes(batch: int) -> int:
    """Caller-owned query/Key RoPE scratch retained by the endpoint."""

    if batch not in BATCH_SIZES:
        raise Phase13PilotError("endpoint workspace batch differs")
    return (
        32
        * batch
        * (PREFIX_QUERY_HEADS + PREFIX_KV_HEADS)
        * (PREFIX_HEAD_DIM // 2)
        * PREFIX_DTYPE_BYTES
    )


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
    reference_endpoint_workspace = endpoint_workspace_bytes(1)
    if reference_reserve <= reference_endpoint_workspace:
        raise Phase13PilotError("Phase 12 graph reserve lacks a net graph pool")
    graph_reserve = math.ceil(
        (reference_reserve - reference_endpoint_workspace)
        * batch
        * capacity
        / REFERENCE_CAPACITY
    )
    endpoint_workspace = endpoint_workspace_bytes(batch)
    prefix = prefix_construction_memory(
        batch=batch,
        historical_context=historical,
    )
    limit = math.floor(GPU_TOTAL_MEMORY_BYTES * MAX_MEMORY_FRACTION)
    required = (
        MODEL_WEIGHT_BYTES
        + cache_bytes
        + endpoint_workspace
        + int(prefix["prefix_control_tensor_bytes"])
        + int(prefix["prefix_compute_peak_bytes"])
        + graph_reserve
    )
    feasible = required <= limit
    return {
        **dict(order_record),
        "schema_version": "kvbench-phase13-feasibility-record-2.0.0",
        "capacity": capacity,
        "model_weight_bytes": MODEL_WEIGHT_BYTES,
        "cache_allocated_bytes": cache_bytes,
        "persistent_workspace_included_in_cache": True,
        "endpoint_workspace_bytes": endpoint_workspace,
        **prefix,
        "graph_pool_or_capture_reserve_bytes": graph_reserve,
        "graph_reserve_reference_bytes": reference_reserve,
        "graph_reserve_reference_endpoint_workspace_bytes": (
            reference_endpoint_workspace
        ),
        "graph_reserve_scaling": "net_reference_x_batch_x_capacity",
        "gpu_total_memory_bytes": GPU_TOTAL_MEMORY_BYTES,
        "max_memory_fraction": MAX_MEMORY_FRACTION,
        "configured_safety_margin_bytes": GPU_TOTAL_MEMORY_BYTES - limit,
        "limit_bytes": limit,
        "predicted_required_bytes": required,
        "status": "feasible" if feasible else "capacity_infeasible",
        "reason": (
            None
            if feasible
            else "end_to_end_prefix_and_graph_peak_exceeds_0.88_limit"
        ),
        "prefix_allocator_blocks_retained_until_graph_capture": True,
        "formula_components_additive": True,
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


def _point_inputs(*, batch: int, historical: int, device: Any) -> tuple[Any, Any]:
    import torch

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
    return prefix, decode


def derive_prefix_catalog_plan(
    feasibility: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Use one maximum-feasible-batch state per config/context pair."""

    grouped: dict[tuple[str, int], set[int]] = defaultdict(set)
    for record in feasibility:
        if record.get("status") == "feasible":
            grouped[
                (str(record["method_config_id"]), int(record["context_label"]))
            ].add(int(record["batch_size"]))
    entries: list[dict[str, Any]] = []
    for configuration in CONFIGURATIONS:
        for context_label in CONTEXT_LABELS:
            batches = sorted(grouped.get((configuration, context_label), set()))
            if not batches:
                raise Phase13PilotError("prefix catalog lacks a feasible batch")
            if any(batch not in BATCH_SIZES for batch in batches):
                raise Phase13PilotError("prefix catalog batch geometry differs")
            historical = actual_historical_context(context_label)
            entries.append(
                {
                    "snapshot_id": f"prefix-{configuration}-l{context_label}",
                    "method_config_id": configuration,
                    "method_config_fingerprint": CONFIG_FINGERPRINTS[configuration],
                    "method_family": phase12._method_family(configuration),
                    "context_label": context_label,
                    "historical_context": historical,
                    "capacity": historical + 1,
                    "source_batch": max(batches),
                    "target_batches": batches,
                    "leading_rows_are_source_faithful": True,
                }
            )
    unique_targets = sum(len(entry["target_batches"]) for entry in entries)
    if len(entries) != len(CONFIGURATIONS) * len(CONTEXT_LABELS) or unique_targets != 228:
        raise Phase13PilotError("prefix catalog cardinality differs")
    return entries


class _PrefixStateCaptured(Exception):
    def __init__(self, manifest: Mapping[str, Any]) -> None:
        super().__init__("prefix state captured")
        self.manifest = dict(manifest)


@contextmanager
def _patched_endpoint_prefill(callback: Any) -> Any:
    from kvbench.runtime.bf16_endpoint import BF16DecodeEndpoint

    original = BF16DecodeEndpoint.prefill

    def patched(endpoint: Any, prefix_input_ids: Any) -> Any:
        return callback(endpoint, prefix_input_ids, original)

    BF16DecodeEndpoint.prefill = patched
    try:
        yield
    finally:
        BF16DecodeEndpoint.prefill = original


@contextmanager
def _restored_prefix_hash_overrides(witness: str) -> Any:
    import kvbench.runtime.kivi_session as kivi_session
    import kvbench.runtime.kvquant_session as kvquant_session
    import kvbench.runtime.phase3_endpoint_audit as bf16_audit

    originals = (
        bf16_audit._cache_pair_sha256,
        kivi_session._historical_prefix_sha256,
        kvquant_session._historical_prefix_sha256,
    )
    bf16_audit._cache_pair_sha256 = lambda *args, **kwargs: witness
    kivi_session._historical_prefix_sha256 = lambda *args, **kwargs: witness
    kvquant_session._historical_prefix_sha256 = lambda *args, **kwargs: witness
    try:
        yield
    finally:
        (
            bf16_audit._cache_pair_sha256,
            kivi_session._historical_prefix_sha256,
            kvquant_session._historical_prefix_sha256,
        ) = originals


def _bind_session_prefix_witness(session: Any, witness: str) -> None:
    session.current_historical_prefix_sha256 = types.MethodType(
        lambda self: witness,
        session,
    )
    if hasattr(session.cache, "history_sha256"):
        session.cache.history_sha256 = types.MethodType(
            lambda self, historical_length: witness,
            session.cache,
        )
    if session.current_historical_prefix_sha256() != witness:
        raise Phase13PilotError("restored prefix witness binding failed")


def _build_restored_session(
    *,
    loaded: Any,
    operation: Phase13OperationKey,
    prefix: Any,
    decode: Any,
    snapshot_root: Path,
    expected_state_sha256: str,
    equivalence_export_root: Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    family = phase12._method_family(operation.configuration)
    restore_receipt: dict[str, Any] | None = None

    def restore_callback(endpoint: Any, input_ids: Any, original: Any) -> None:
        del original
        nonlocal restore_receipt
        if tuple(input_ids.shape) != (operation.batch_size, operation.historical_context):
            raise Phase13PilotError("restored prefix input geometry differs")
        restore_receipt = restore_prefix_state(
            cache=endpoint.cache,
            family=family,
            configuration=operation.configuration,
            historical=operation.historical_context,
            root=snapshot_root,
            expected_state_sha256=expected_state_sha256,
        )
        if equivalence_export_root is not None:
            save_prefix_state(
                cache=endpoint.cache,
                family=family,
                configuration=operation.configuration,
                historical=operation.historical_context,
                source_batch=operation.batch_size,
                output=equivalence_export_root,
                authority={
                    "equivalence_export": True,
                    "source_state_sha256": expected_state_sha256,
                    "target_batch": operation.batch_size,
                },
            )
        witness = str(restore_receipt["witness_sha256"])
        if hasattr(endpoint.cache, "history_sha256"):
            endpoint.cache.history_sha256 = types.MethodType(
                lambda self, historical_length: witness,
                endpoint.cache,
            )
        return None

    manifest = validate_prefix_state(
        snapshot_root,
        configuration=operation.configuration,
        historical=operation.historical_context,
        verify_state_bytes=False,
    )
    if manifest.get("state_file_sha256") != expected_state_sha256:
        raise Phase13PilotError("prefix snapshot catalog digest differs")
    from scripts.phase13_prefix_state import restored_prefix_witness

    witness = restored_prefix_witness(manifest, target_batch=operation.batch_size)
    with _restored_prefix_hash_overrides(witness), _patched_endpoint_prefill(
        restore_callback
    ):
        session = phase12._build_phase12_session(
            loaded=loaded,
            operation_key=operation,
            prefix_input_ids=prefix,
            decode_input_ids=decode,
        )
    if restore_receipt is None or restore_receipt.get("witness_sha256") != witness:
        raise Phase13PilotError("prefix restore did not execute exactly once")
    _bind_session_prefix_witness(session, witness)
    return session, restore_receipt


def _build_prefix_state_worker(
    *,
    snapshot_id: str,
    configuration: str,
    source_batch: int,
    context_label: int,
    git_sha: str,
    build_root: Path,
    output: Path,
) -> dict[str, Any]:
    """Directly construct one untimed canonical state inside the container."""

    attestation = phase12._require_authorized_container_runtime()
    import torch

    from kvbench.runtime.backend import forced_flash_execution
    from kvbench.runtime.model_loader import load_frozen_model

    historical = actual_historical_context(context_label)
    recorder = _WorkerStageRecorder(
        root=build_root / "stage-progress",
        run_id=snapshot_id,
        stage_sequence=PREFIX_BUILD_STAGE_SEQUENCE,
    )
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
        raise Phase13PilotError("prefix builder source authority differs")
    _patch_phase12_point_globals(batch=source_batch, historical=historical)
    device = torch.device("cuda:0")
    recorder.record("model_load", "started")
    loaded = load_frozen_model(device=device)
    recorder.record("model_load", "completed")
    prefix, decode = _point_inputs(
        batch=source_batch,
        historical=historical,
        device=device,
    )
    operation = Phase13OperationKey.create(configuration, source_batch, historical)
    recorder.record("prefix_construction", "started")

    def capture_callback(endpoint: Any, input_ids: Any, original: Any) -> None:
        original(endpoint, input_ids)
        recorder.record("prefix_construction", "completed")
        recorder.record("finalization", "started")
        manifest = save_prefix_state(
            cache=endpoint.cache,
            family=phase12._method_family(configuration),
            configuration=configuration,
            historical=historical,
            source_batch=source_batch,
            output=output,
            authority={
                "execution_git_sha": git_sha,
                "authorized_container_digest": AUTHORIZED_CONTAINER_DIGEST,
                "method_config_fingerprint": CONFIG_FINGERPRINTS[configuration],
                "operation_fingerprint_sha256": operation.operation_fingerprint_sha256,
                "input_recipe_schema": INPUT_RECIPE_SCHEMA,
                "input_recipe_sha256": phase12.PHASE12_INPUT_RECIPE_SHA256,
                "snapshot_id": snapshot_id,
            },
        )
        recorder.record("finalization", "completed")
        raise _PrefixStateCaptured(manifest)

    captured: dict[str, Any] | None = None
    with torch.inference_mode(), forced_flash_execution(), _patched_endpoint_prefill(
        capture_callback
    ):
        try:
            phase12._build_phase12_session(
                loaded=loaded,
                operation_key=operation,
                prefix_input_ids=prefix,
                decode_input_ids=decode,
            )
        except _PrefixStateCaptured as signal:
            captured = signal.manifest
    if captured is None:
        raise Phase13PilotError("direct prefix snapshot was not captured")
    return {
        "schema_version": "kvbench-phase13-prefix-builder-result-1.0.0",
        "snapshot_id": snapshot_id,
        "configuration": configuration,
        "context_label": context_label,
        "historical_context": historical,
        "source_batch": source_batch,
        "state_file_sha256": captured["state_file_sha256"],
        "state_file_bytes": captured["state_file_bytes"],
        "container_runtime_attestation": attestation,
    }


def _run_prefix_builder_process(
    *,
    prefix_root: Path,
    entry: Mapping[str, Any],
    git_sha: str,
) -> dict[str, Any]:
    snapshot_id = str(entry["snapshot_id"])
    builds_root = prefix_root / "builds"
    states_root = prefix_root / "states"
    build_root = builds_root / snapshot_id
    snapshot_root = states_root / snapshot_id
    if build_root.exists() or snapshot_root.exists() or snapshot_root.is_symlink():
        raise Phase13PilotError("prefix snapshot ID already exists")
    build_root.mkdir()
    (build_root / "stage-progress").mkdir()
    child_output = build_root / "snapshot"
    pre = phase12._capture_process_snapshot()
    phase12._require_idle_snapshot(pre)
    command = (
        sys.executable,
        str(REPOSITORY_ROOT / "scripts/phase13_pilot.py"),
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
    historical = int(entry["historical_context"])
    timeouts = stage_timeout_contract(
        batch=int(entry["source_batch"]),
        historical=historical,
    )
    timeouts["finalization"] = max(
        7_200.0,
        float(1_800 + math.ceil(int(entry["source_batch"]) * historical / 20)),
    )
    result = run_stage_supervised_command(
        command,
        working_directory=str(REPOSITORY_ROOT),
        environment=phase12._child_environment(),
        stage_timeouts=timeouts,
        stage_observer=lambda: _read_stage_observations(
            root=build_root / "stage-progress",
            run_id=snapshot_id,
            stage_sequence=PREFIX_BUILD_STAGE_SEQUENCE,
        ),
        startup_stage="startup",
        transition_stage="transition",
        observer_poll_seconds=STAGE_OBSERVER_POLL_SECONDS,
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
    if not _supervision_passed(result):
        raise Phase13PilotError(f"prefix builder failed: {snapshot_id}")
    matches = [
        line[len(PREFIX_BUILDER_PREFIX) :]
        for line in result.stdout.decode("utf-8", errors="strict").splitlines()
        if line.startswith(PREFIX_BUILDER_PREFIX)
    ]
    if len(matches) != 1:
        raise Phase13PilotError("prefix builder result channel differs")
    payload = json.loads(matches[0])
    if not isinstance(payload, dict) or payload.get("snapshot_id") != snapshot_id:
        raise Phase13PilotError("prefix builder result identity differs")
    manifest = validate_prefix_state(
        child_output,
        configuration=str(entry["method_config_id"]),
        historical=historical,
        verify_state_bytes=True,
    )
    if (
        manifest.get("source_batch") != entry["source_batch"]
        or manifest.get("state_file_sha256") != payload.get("state_file_sha256")
        or manifest.get("authority", {}).get("execution_git_sha") != git_sha
        or manifest.get("authority", {}).get("authorized_container_digest")
        != AUTHORIZED_CONTAINER_DIGEST
    ):
        raise Phase13PilotError("prefix builder authority differs")
    rename_noreplace(child_output, snapshot_root)
    for path in sorted(snapshot_root.iterdir()):
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
        "state_bytes_verified_once_before_timing": True,
    }


def prepare_prefix_catalog(
    *,
    prefix_root: Path,
    feasibility: Sequence[Mapping[str, Any]],
    git_sha: str,
) -> dict[str, Any]:
    """Build and validate every reusable state before any formal timing run."""

    if not prefix_root.is_dir() or prefix_root.is_symlink() or any(prefix_root.iterdir()):
        raise Phase13PilotError("prefix catalog root must be new and empty")
    (prefix_root / "builds").mkdir()
    (prefix_root / "states").mkdir()
    plan = derive_prefix_catalog_plan(feasibility)
    completed = [
        _run_prefix_builder_process(
            prefix_root=prefix_root,
            entry=entry,
            git_sha=git_sha,
        )
        for entry in plan
    ]
    payload = {
        "schema_version": "kvbench-phase13-prefix-catalog-1.0.0",
        "execution_git_sha": git_sha,
        "authorized_container_digest": AUTHORIZED_CONTAINER_DIGEST,
        "snapshot_count": len(completed),
        "unique_feasible_points": sum(
            len(entry["target_batches"]) for entry in completed
        ),
        "formal_process_replicates": REPLICATES,
        "direct_constructions_per_snapshot": 1,
        "runtime_prefix_cache_sharing": False,
        "fresh_caller_owned_cache_per_timing_process": True,
        "restoration_outside_timing": True,
        "state_bytes_verified_before_timing": True,
        "entries": completed,
    }
    if payload["snapshot_count"] != 90 or payload["unique_feasible_points"] != 228:
        raise Phase13PilotError("completed prefix catalog cardinality differs")
    write_exclusive(prefix_root / "catalog.json", json_bytes(payload))
    return payload


def _prefix_catalog_index(
    catalog: Mapping[str, Any], prefix_root: Path
) -> dict[tuple[str, int], dict[str, Any]]:
    entries = catalog.get("entries")
    if not isinstance(entries, list) or len(entries) != 90:
        raise Phase13PilotError("prefix catalog entry set differs")
    index: dict[tuple[str, int], dict[str, Any]] = {}
    for value in entries:
        if not isinstance(value, dict):
            raise Phase13PilotError("prefix catalog entry differs")
        key = (str(value["method_config_id"]), int(value["context_label"]))
        if key in index:
            raise Phase13PilotError("prefix catalog entry is duplicated")
        state_root = prefix_root / str(value["snapshot_relative_path"])
        if not state_root.is_dir() or state_root.is_symlink():
            raise Phase13PilotError("prefix catalog state path differs")
        index[key] = {**value, "snapshot_root": state_root}
    return index


def _direct_session_with_snapshot(
    *,
    loaded: Any,
    configuration: str,
    batch: int,
    historical: int,
    snapshot_root: Path,
    abort_after_snapshot: bool,
) -> Any | None:
    import torch

    from kvbench.runtime.backend import forced_flash_execution

    _patch_phase12_point_globals(batch=batch, historical=historical)
    prefix, decode = _point_inputs(
        batch=batch,
        historical=historical,
        device=torch.device("cuda:0"),
    )
    operation = Phase13OperationKey.create(configuration, batch, historical)
    captured: dict[str, Any] | None = None

    def callback(endpoint: Any, input_ids: Any, original: Any) -> Any:
        nonlocal captured
        result = original(endpoint, input_ids)
        captured = save_prefix_state(
            cache=endpoint.cache,
            family=phase12._method_family(configuration),
            configuration=configuration,
            historical=historical,
            source_batch=batch,
            output=snapshot_root,
            authority={
                "equivalence_direct": True,
                "method_config_fingerprint": CONFIG_FINGERPRINTS[configuration],
                "operation_fingerprint_sha256": operation.operation_fingerprint_sha256,
                "input_recipe_sha256": phase12.PHASE12_INPUT_RECIPE_SHA256,
            },
        )
        if abort_after_snapshot:
            raise _PrefixStateCaptured(captured)
        return result

    with torch.inference_mode(), forced_flash_execution(), _patched_endpoint_prefill(
        callback
    ):
        try:
            session = phase12._build_phase12_session(
                loaded=loaded,
                operation_key=operation,
                prefix_input_ids=prefix,
                decode_input_ids=decode,
            )
        except _PrefixStateCaptured:
            session = None
    if captured is None:
        raise Phase13PilotError("equivalence direct snapshot was not captured")
    if abort_after_snapshot and session is not None:
        raise Phase13PilotError("equivalence source session was not aborted")
    if not abort_after_snapshot and session is None:
        raise Phase13PilotError("equivalence direct session is absent")
    return session


def _equivalence_session_record(session: Any, *, evidence_root: Path) -> dict[str, Any]:
    pointers_first = phase12._phase12_session_pointers(session)
    pointers_second = phase12._phase12_session_pointers(session)
    graph_path = phase12._write_cuda_graph_path_witness(
        graph=session.graph.graph,
        run_root=evidence_root,
        phase="before",
    )
    graph = session.graph_evidence
    record = {
        "cache_layout_fingerprint": session.cache_layout_fingerprint(),
        "cache_accounting": session.method_cache_accounting(),
        "cache_byte_breakdown": session.method_byte_breakdown(),
        "pointer_labels": sorted(pointers_first),
        "pointer_values": sorted(pointers_first.values()),
        "pointers_stable": pointers_first == pointers_second,
        "pointer_count": len(pointers_first),
        "pointers_unique": len(set(pointers_first.values())) == len(pointers_first),
        "output_checksum": graph["second_replay_checksum"],
        "kernel_path_fingerprint": graph_path["normalized_sha256"],
        "kernel_count": graph_path["kernel_node_count"],
        "graph_capture": graph.get("captured"),
        "graph_fallback": graph.get("fallback"),
        "graph_replay_exact": graph.get("consecutive_replay_outputs_exact"),
        "eager_graph_agreement": bool(
            session.eager_graph_comparison is not None
            and session.eager_graph_comparison.passed
        ),
    }
    if (
        record["pointers_stable"] is not True
        or record["pointers_unique"] is not True
        or record["graph_capture"] is not True
        or record["graph_fallback"] is not False
        or record["graph_replay_exact"] is not True
        or record["eager_graph_agreement"] is not True
    ):
        raise Phase13PilotError("equivalence session Graph or pointers failed")
    return record


def run_prefix_equivalence(
    *, output: Path, scratch_root: Path, git_sha: str
) -> dict[str, Any]:
    """One focused matrix proving source-batch slicing and fresh restoration."""

    phase12._require_authorized_container_runtime()
    if output.exists() or output.is_symlink():
        raise Phase13PilotError("prefix equivalence output already exists")
    if not scratch_root.is_dir() or scratch_root.is_symlink() or any(scratch_root.iterdir()):
        raise Phase13PilotError("prefix equivalence scratch must be new and empty")
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
        raise Phase13PilotError("equivalence source authority differs")
    import torch

    from kvbench.runtime.backend import forced_flash_execution
    from kvbench.runtime.model_loader import load_frozen_model

    loaded = load_frozen_model(device=torch.device("cuda:0"))
    records: list[dict[str, Any]] = []
    for configuration in CONFIGURATIONS:
        configuration_root = scratch_root / configuration
        configuration_root.mkdir()
        source_root = configuration_root / "source-b8"
        direct_root = configuration_root / "direct-b1"
        restored_export_root = configuration_root / "restored-b1"
        source_session = _direct_session_with_snapshot(
            loaded=loaded,
            configuration=configuration,
            batch=PREFIX_EQUIVALENCE_SOURCE_BATCH,
            historical=PREFIX_EQUIVALENCE_CONTEXT,
            snapshot_root=source_root,
            abort_after_snapshot=True,
        )
        if source_session is not None:
            raise Phase13PilotError("equivalence source unexpectedly retained a session")
        torch.cuda.empty_cache()
        with phase12._observable_cuda_graph_factory(torch) as direct_graphs:
            direct_session = _direct_session_with_snapshot(
                loaded=loaded,
                configuration=configuration,
                batch=1,
                historical=PREFIX_EQUIVALENCE_CONTEXT,
                snapshot_root=direct_root,
                abort_after_snapshot=False,
            )
        assert direct_session is not None
        if (
            len(direct_graphs) != 1
            or direct_session.graph is None
            or direct_session.graph.graph is not direct_graphs[0]
        ):
            raise Phase13PilotError(
                "equivalence direct CUDA Graph is absent or ambiguous"
            )
        _patch_phase12_point_globals(batch=1, historical=PREFIX_EQUIVALENCE_CONTEXT)
        prefix, decode = _point_inputs(
            batch=1,
            historical=PREFIX_EQUIVALENCE_CONTEXT,
            device=torch.device("cuda:0"),
        )
        operation = Phase13OperationKey.create(
            configuration, 1, PREFIX_EQUIVALENCE_CONTEXT
        )
        source_manifest = validate_prefix_state(
            source_root,
            configuration=configuration,
            historical=PREFIX_EQUIVALENCE_CONTEXT,
            verify_state_bytes=True,
        )
        with torch.inference_mode(), forced_flash_execution():
            with phase12._observable_cuda_graph_factory(torch) as restored_graphs:
                restored_session, restore_receipt = _build_restored_session(
                    loaded=loaded,
                    operation=operation,
                    prefix=prefix,
                    decode=decode,
                    snapshot_root=source_root,
                    expected_state_sha256=str(source_manifest["state_file_sha256"]),
                    equivalence_export_root=restored_export_root,
                )
        if (
            len(restored_graphs) != 1
            or restored_session.graph is None
            or restored_session.graph.graph is not restored_graphs[0]
        ):
            raise Phase13PilotError(
                "equivalence restored CUDA Graph is absent or ambiguous"
            )
        direct_state = validate_prefix_state(
            direct_root,
            configuration=configuration,
            historical=PREFIX_EQUIVALENCE_CONTEXT,
            verify_state_bytes=True,
        )
        restored_state = validate_prefix_state(
            restored_export_root,
            configuration=configuration,
            historical=PREFIX_EQUIVALENCE_CONTEXT,
            verify_state_bytes=True,
        )
        direct_evidence_root = configuration_root / "direct-evidence"
        restored_evidence_root = configuration_root / "restored-evidence"
        direct_evidence_root.mkdir()
        restored_evidence_root.mkdir()
        direct = _equivalence_session_record(
            direct_session, evidence_root=direct_evidence_root
        )
        restored = _equivalence_session_record(
            restored_session, evidence_root=restored_evidence_root
        )
        exact_fields = (
            "cache_layout_fingerprint",
            "cache_accounting",
            "cache_byte_breakdown",
            "pointer_labels",
            "pointer_count",
            "output_checksum",
            "kernel_path_fingerprint",
            "kernel_count",
            "graph_capture",
            "graph_fallback",
            "graph_replay_exact",
            "eager_graph_agreement",
        )
        passed = bool(
            direct_state["state_file_sha256"]
            == restored_state["state_file_sha256"]
            and all(direct[field] == restored[field] for field in exact_fields)
            and set(direct["pointer_values"]).isdisjoint(
                restored["pointer_values"]
            )
            and restore_receipt["fresh_target_allocation"] is True
            and restore_receipt["runtime_prefix_sharing"] is False
        )
        if not passed:
            raise Phase13PilotError(
                f"direct/restored prefix equivalence failed: {configuration}"
            )
        records.append(
            {
                "configuration": configuration,
                "source_batch": PREFIX_EQUIVALENCE_SOURCE_BATCH,
                "target_batch": 1,
                "historical_context": PREFIX_EQUIVALENCE_CONTEXT,
                "source_state_sha256": source_manifest["state_file_sha256"],
                "target_state_sha256": direct_state["state_file_sha256"],
                "restored_state_sha256": restored_state["state_file_sha256"],
                "direct": direct,
                "restored": restored,
                "fresh_target_allocation": True,
                "direct_and_restored_pointers_disjoint": True,
                "restore_outside_timing": True,
                "runtime_prefix_cache_sharing": False,
                "passed": True,
            }
        )
        del direct_session, restored_session
        torch.cuda.empty_cache()
    payload = {
        "schema_version": "kvbench-phase13-prefix-equivalence-1.0.0",
        "status": "PASS",
        "execution_git_sha": git_sha,
        "authorized_container_digest": AUTHORIZED_CONTAINER_DIGEST,
        "configuration_count": len(records),
        "all_configurations_passed": all(record["passed"] for record in records),
        "fresh_target_allocation": True,
        "restore_outside_timing": True,
        "runtime_prefix_cache_sharing": False,
        "timing_collected": False,
        "records": records,
    }
    if len(records) != len(CONFIGURATIONS) or not payload["all_configurations_passed"]:
        raise Phase13PilotError("prefix equivalence matrix differs")
    write_exclusive(output, json_bytes(payload))
    return payload


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
    stage_recorder = _WorkerStageRecorder(
        root=run_artifact_root / "stage-progress",
        run_id=run_id,
    )
    _patch_phase12_point_globals(batch=batch, historical=historical)
    device = torch.device("cuda:0")
    stage_recorder.record("model_load", "started")
    loaded = load_frozen_model(device=device)
    stage_recorder.record("model_load", "completed")
    prefix, decode = _point_inputs(
        batch=batch,
        historical=historical,
        device=device,
    )
    operation = Phase13OperationKey.create(configuration, batch, historical)
    stage_recorder.record("prefix_construction", "started")
    import kvbench.runtime.cuda_graph as cuda_graph_module

    original_capture_fixed_graph = cuda_graph_module.capture_fixed_graph
    graph_capture_started = False

    def observed_capture_fixed_graph(*args: Any, **kwargs: Any) -> Any:
        nonlocal graph_capture_started
        if graph_capture_started:
            raise Phase13PilotError("graph capture began more than once")
        stage_recorder.record("prefix_construction", "completed")
        stage_recorder.record("graph_capture", "started")
        graph_capture_started = True
        return original_capture_fixed_graph(*args, **kwargs)

    cuda_graph_module.capture_fixed_graph = observed_capture_fixed_graph
    with torch.inference_mode(), forced_flash_execution():
        try:
            with phase12._observable_cuda_graph_factory(torch) as observed_graphs:
                session, prefix_restore = _build_restored_session(
                    loaded=loaded,
                    operation=operation,
                    prefix=prefix,
                    decode=decode,
                    snapshot_root=prefix_state_root,
                    expected_state_sha256=prefix_state_sha256,
                )
        finally:
            cuda_graph_module.capture_fixed_graph = original_capture_fixed_graph
        if not graph_capture_started:
            raise Phase13PilotError("graph capture stage was not observed")
        stage_recorder.record("graph_capture", "completed")
        stage_recorder.record("warmup_and_audit", "started")
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
        path_passed = bool(
            backend_verified
            and live_fingerprint
            and geometry_passed
            and graph_passed
            and pointers_before == phase12._phase12_session_pointers(session)
        )
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
            execution_path_passed=path_passed,
            allocation_passed=allocation_passed,
            graph_passed=graph_passed,
        )
        stage_recorder.record("warmup_and_audit", "completed")
        stage_recorder.record("measurement", "started")
        raw_runner = run_fixed_l(
            session,
            measured_steps=MEASURED_STEPS,
            measured_batches=MEASURED_BATCHES,
        ).to_dict()
        stage_recorder.record("measurement", "completed")
        stage_recorder.record("finalization", "started")
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
            "phase13b_successor": successor_binding,
            "runtime_adapter_fingerprint": runner["adapter_config_fingerprint"],
            "cache_layout_fingerprint": runner["cache_layout_fingerprint"],
            "backend": runtime_context.backend_fingerprint,
            "graph_topology": graph_path_before["normalized_sha256"],
            "prefix_state_witness": prefix_restore["witness_sha256"],
            "path_checks": {
                "backend_identity": backend_verified,
                "runtime_adapter_fingerprint": bool(live_fingerprint),
                "native_gqa_geometry": geometry_passed,
                "graph_no_fallback": graph_passed,
            },
        }
    )
    allocation_fingerprint = _canonical_sha256(
        {
            "cache_accounting": runner["cache_accounting"],
            "cache_byte_breakdown": runner["cache_byte_breakdown"],
            "replay_allocation": allocation_record,
        }
    )
    payload = {
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
        "no_backend_fallback": path_passed,
        "allocation_stable": True,
        "kernel_path_stable": True,
        "gpu_exclusive": True,
        "warmup_replays": WARMUP_STEPS,
        "measured_steps": MEASURED_STEPS,
        "measured_batches": MEASURED_BATCHES,
        "graph_replay_allocation": allocation_record,
        "prefix_state_restore": prefix_restore,
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
    stage_recorder.record("finalization", "completed")
    return payload


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
    prefix_entry: Mapping[str, Any],
) -> dict[str, Any]:
    if int(record["batch_size"]) not in prefix_entry.get("target_batches", []):
        raise Phase13PilotError("formal run is not covered by its prefix snapshot")
    run_id = _run_id(campaign_id, record)
    run_root = stage / "runs" / run_id
    run_root.mkdir()
    stage_progress = run_root / "stage-progress"
    stage_progress.mkdir()
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
        "--prefix-state-root",
        str(prefix_entry["snapshot_root"]),
        "--prefix-state-sha256",
        str(prefix_entry["state_file_sha256"]),
    )
    timeouts = stage_timeout_contract(
        batch=int(record["batch_size"]),
        historical=int(record["historical_context"]),
    )
    result = run_stage_supervised_command(
        command,
        working_directory=str(REPOSITORY_ROOT),
        environment=phase12._child_environment(),
        stage_timeouts=timeouts,
        stage_observer=lambda: _read_stage_observations(
            root=stage_progress,
            run_id=run_id,
        ),
        startup_stage="startup",
        transition_stage="transition",
        observer_poll_seconds=STAGE_OBSERVER_POLL_SECONDS,
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
        if result.timeout_stage is not None:
            reason = f"supervisor_stage_timeout:{result.timeout_stage}"
        elif "requires frozen B=1" in stderr or "requires frozen layers=32 B=1" in stderr:
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
                    "timeout_stage": result.timeout_stage,
                    "timeout_stage_elapsed_seconds": (
                        result.timeout_stage_elapsed_seconds
                    ),
                    "stage_timeout_contract_seconds": timeouts,
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


def run_campaign(
    *,
    stage: Path,
    campaign_id: str,
    git_sha: str,
    prefix_root: Path,
) -> dict[str, Any]:
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
    if not prefix_root.is_dir() or prefix_root.is_symlink() or any(prefix_root.iterdir()):
        raise Phase13PilotError("mounted prefix root must be new and empty")
    equivalence_scratch = prefix_root / "equivalence"
    catalog_root = prefix_root / "catalog"
    equivalence_scratch.mkdir()
    catalog_root.mkdir()
    equivalence_path = resolved / "unified" / "prefix-equivalence.json"
    equivalence = run_prefix_equivalence(
        output=equivalence_path,
        scratch_root=equivalence_scratch,
        git_sha=git_sha,
    )
    if (
        equivalence.get("status") != "PASS"
        or equivalence.get("all_configurations_passed") is not True
        or equivalence.get("runtime_prefix_cache_sharing") is not False
    ):
        raise Phase13PilotError("prefix equivalence did not pass before timing")
    catalog = prepare_prefix_catalog(
        prefix_root=catalog_root,
        feasibility=feasibility,
        git_sha=git_sha,
    )
    catalog_index = _prefix_catalog_index(catalog, catalog_root)
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
        "temporary_state_files_published": False,
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
                "prefix_equivalence_sha256": sha256_file(equivalence_path),
                "prefix_catalog_sha256": sha256_file(
                    resolved / "unified" / "prefix-catalog.json"
                ),
                "prefix_restore_outside_timing": True,
                "fresh_caller_owned_cache_per_process": True,
                "runtime_prefix_cache_sharing": False,
                "per_point_fixture_replay": False,
                "per_point_sanitizer": False,
                "per_point_full_g1_g4_audit": False,
                "per_point_large_history_scan": False,
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
            prefix_entry=catalog_index[
                (str(record["method_config_id"]), int(record["context_label"]))
            ],
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
    campaign = _strict_json(root / "campaign_manifest.json")
    equivalence_path = root / "unified" / "prefix-equivalence.json"
    catalog_path = root / "unified" / "prefix-catalog.json"
    equivalence = _strict_json(equivalence_path)
    catalog = _strict_json(catalog_path)
    feasibility_payload = _strict_json(root / "unified" / "feasibility.json")
    feasibility = feasibility_payload.get("records")
    if (
        campaign.get("prefix_equivalence_sha256")
        != sha256_file(equivalence_path)
        or campaign.get("prefix_catalog_sha256") != sha256_file(catalog_path)
        or campaign.get("prefix_restore_outside_timing") is not True
        or campaign.get("fresh_caller_owned_cache_per_process") is not True
        or campaign.get("runtime_prefix_cache_sharing") is not False
        or campaign.get("per_point_fixture_replay") is not False
        or campaign.get("per_point_sanitizer") is not False
        or campaign.get("per_point_full_g1_g4_audit") is not False
        or campaign.get("per_point_large_history_scan") is not False
        or equivalence.get("status") != "PASS"
        or equivalence.get("configuration_count") != len(CONFIGURATIONS)
        or equivalence.get("all_configurations_passed") is not True
        or equivalence.get("fresh_target_allocation") is not True
        or equivalence.get("restore_outside_timing") is not True
        or equivalence.get("runtime_prefix_cache_sharing") is not False
        or equivalence.get("timing_collected") is not False
        or not isinstance(feasibility, list)
    ):
        raise Phase13PilotError("Phase 13 prefix contract evidence differs")
    equivalence_records = equivalence.get("records")
    if (
        not isinstance(equivalence_records, list)
        or any(not isinstance(item, dict) for item in equivalence_records)
        or {item.get("configuration") for item in equivalence_records}
        != set(CONFIGURATIONS)
        or any(
            item.get("passed") is not True
            or item.get("fresh_target_allocation") is not True
            or item.get("direct_and_restored_pointers_disjoint") is not True
            or item.get("restore_outside_timing") is not True
            or item.get("runtime_prefix_cache_sharing") is not False
            for item in equivalence_records
        )
    ):
        raise Phase13PilotError("Phase 13 prefix equivalence matrix differs")
    expected_catalog = {
        (item["method_config_id"], item["context_label"]): item
        for item in derive_prefix_catalog_plan(feasibility)
    }
    catalog_entries = catalog.get("entries")
    if (
        catalog.get("snapshot_count") != 90
        or catalog.get("unique_feasible_points") != 228
        or catalog.get("runtime_prefix_cache_sharing") is not False
        or catalog.get("fresh_caller_owned_cache_per_timing_process") is not True
        or catalog.get("restoration_outside_timing") is not True
        or catalog.get("temporary_state_files_published") is not False
        or not isinstance(catalog_entries, list)
        or len(catalog_entries) != 90
    ):
        raise Phase13PilotError("Phase 13 prefix catalog differs")
    catalog_index: dict[tuple[str, int], Mapping[str, Any]] = {}
    for entry in catalog_entries:
        if not isinstance(entry, dict):
            raise Phase13PilotError("Phase 13 prefix catalog entry differs")
        key = (str(entry.get("method_config_id")), int(entry.get("context_label", -1)))
        expected = expected_catalog.get(key)
        if (
            expected is None
            or key in catalog_index
            or any(entry.get(name) != value for name, value in expected.items())
            or re.fullmatch(r"[0-9a-f]{64}", str(entry.get("state_file_sha256")))
            is None
            or re.fullmatch(
                r"[0-9a-f]{64}", str(entry.get("builder_supervision_sha256"))
            )
            is None
            or entry.get("state_bytes_verified_once_before_timing") is not True
        ):
            raise Phase13PilotError("Phase 13 prefix catalog authority differs")
        catalog_index[key] = entry
    for record in runs:
        if record.get("status") != "completed":
            continue
        result_path = record.get("result_path")
        if not isinstance(result_path, str):
            raise Phase13PilotError("completed Pilot result is absent")
        result = _strict_json(root / result_path)
        restore = result.get("prefix_state_restore")
        entry = catalog_index.get(
            (str(record["method_config_id"]), int(record["context_label"]))
        )
        if (
            not isinstance(restore, dict)
            or entry is None
            or restore.get("state_file_sha256") != entry.get("state_file_sha256")
            or restore.get("source_batch") != entry.get("source_batch")
            or restore.get("target_batch") != record.get("batch_size")
            or restore.get("fresh_target_allocation") is not True
            or restore.get("runtime_prefix_sharing") is not False
            or restore.get("restore_outside_timing") is not True
        ):
            raise Phase13PilotError("Pilot prefix restoration receipt differs")
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
    actions.add_argument("--build-prefix-state", action="store_true")
    actions.add_argument("--validate-prefix-equivalence", action="store_true")
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
    parser.add_argument("--prefix-root", type=Path)
    parser.add_argument("--prefix-state-root", type=Path)
    parser.add_argument("--prefix-state-sha256")
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--snapshot-id")
    parser.add_argument("--source-batch", type=int)
    parser.add_argument("--build-root", type=Path)
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
        if (
            args.stage is None
            or args.campaign_id is None
            or args.git_sha is None
        ):
            raise Phase13PilotError("stage, campaign ID, and Git SHA are required")
        print(
            json.dumps(
                run_campaign(
                    stage=args.stage,
                    campaign_id=args.campaign_id,
                    git_sha=args.git_sha,
                    prefix_root=(
                        args.prefix_root
                        if args.prefix_root is not None
                        else Path("/opt/kvbench-prefix-states")
                    ),
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
            args.prefix_state_root,
            args.prefix_state_sha256,
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
            prefix_state_root=Path(args.prefix_state_root),
            prefix_state_sha256=str(args.prefix_state_sha256),
        )
        print(WORKER_PREFIX + json.dumps(payload, sort_keys=True, separators=(",", ":")))
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
            raise Phase13PilotError("prefix builder identity arguments are required")
        payload = _build_prefix_state_worker(
            snapshot_id=str(args.snapshot_id),
            configuration=str(args.configuration),
            source_batch=int(args.source_batch),
            context_label=int(args.context_label),
            git_sha=str(args.git_sha),
            build_root=Path(args.build_root),
            output=Path(args.output),
        )
        print(
            PREFIX_BUILDER_PREFIX
            + json.dumps(payload, sort_keys=True, separators=(",", ":"))
        )
        return 0
    if args.validate_prefix_equivalence:
        if args.output is None or args.scratch_root is None or args.git_sha is None:
            raise Phase13PilotError(
                "equivalence output, scratch root, and Git SHA are required"
            )
        print(
            json.dumps(
                run_prefix_equivalence(
                    output=args.output,
                    scratch_root=args.scratch_root,
                    git_sha=args.git_sha,
                ),
                sort_keys=True,
            )
        )
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
