"""Narrow successor geometry authority for Phase 16G."""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any


PHASE16G_CONFIGURATIONS = (
    "bf16",
    "tq_4bit_nc",
    "tq_k3v4_nc",
    "tq_3bit_nc",
    "k4v4",
    "k2v4",
    "k2v2",
    "kvq4",
    "kvq3",
    "kvq2",
)
PHASE16G_NEW_BATCH_SIZES = (2, 16)
PHASE16G_ADMITTED_BATCH_SIZES = (1, 2, 4, 8, 16)
PHASE16G_CONTEXT_LENGTH = 4096
PHASE16G_CONTAINER_DIGEST = (
    "sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e"
)
PHASE16G_INDEX_SCHEMA = "kvbench-phase16g-geometry-admission-index-1.0.0"
PHASE16G_PREFIX_SCHEMA = "kvbench-phase16g-prefix-state-3.0.0"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class Phase16GGeometryError(RuntimeError):
    """A geometry key, record, or successor authority is invalid."""


def geometry_key(configuration: str, batch_size: int) -> str:
    if configuration not in PHASE16G_CONFIGURATIONS:
        raise Phase16GGeometryError("geometry configuration is unknown")
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise Phase16GGeometryError("geometry batch size is invalid")
    return f"{configuration}/B{batch_size}"


def validate_geometry_index(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate exact B=2/B=16 evidence without rewriting B=1/4/8 records."""

    records = payload.get("new_geometry_records")
    expected_keys = {
        geometry_key(configuration, batch)
        for configuration in PHASE16G_CONFIGURATIONS
        for batch in PHASE16G_NEW_BATCH_SIZES
    }
    if (
        payload.get("schema_version") != PHASE16G_INDEX_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("decision_id") != "0039"
        or payload.get("authorized_container_digest")
        != PHASE16G_CONTAINER_DIGEST
        or payload.get("prefix_format_version") != PHASE16G_PREFIX_SCHEMA
        or payload.get("admitted_full_scan_batches")
        != list(PHASE16G_ADMITTED_BATCH_SIZES)
        or payload.get("existing_geometry_batches_unchanged") != [1, 4, 8]
        or not isinstance(records, Mapping)
        or set(records) != expected_keys
    ):
        raise Phase16GGeometryError("geometry admission index contract differs")
    for key, record in records.items():
        if not isinstance(record, Mapping):
            raise Phase16GGeometryError("geometry admission record differs")
        expected = geometry_key(
            str(record.get("configuration")),
            int(record.get("batch_size", 0)),
        )
        if (
            key != expected
            or record.get("batch_size") not in PHASE16G_NEW_BATCH_SIZES
            or record.get("status") != "PASS"
            or record.get("short_eager") != "PASS"
            or record.get("short_cuda_graph") != "PASS"
            or record.get("prefix_restore") != "PASS"
            or record.get("allocation") != "PASS"
            or record.get("execution_path") != "PASS"
        ):
            raise Phase16GGeometryError("geometry admission result differs")
        for field in (
            "method_config_fingerprint",
            "adapter_config_fingerprint",
            "cache_layout_fingerprint",
        ):
            value = record.get(field)
            if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                raise Phase16GGeometryError(f"geometry {field} differs")
    return dict(payload)


def require_admitted_geometry(
    payload: Mapping[str, Any],
    *,
    configuration: str,
    batch_size: int,
) -> str:
    """Keep admission policy outside the shape-faithful prefix loader."""

    validated = validate_geometry_index(payload)
    key = geometry_key(configuration, batch_size)
    if batch_size in (1, 4, 8):
        authorities = validated.get("existing_geometry_authorities")
        if not isinstance(authorities, Mapping) or configuration not in authorities:
            raise Phase16GGeometryError("existing geometry authority is absent")
        return key
    record = validated["new_geometry_records"].get(key)
    if not isinstance(record, Mapping) or record.get("status") != "PASS":
        raise Phase16GGeometryError("requested geometry is not admitted")
    return key
