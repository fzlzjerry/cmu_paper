"""Pilot-only deterministic prefix-state snapshot and restore helpers.

The format contains only historical cache state.  It deliberately excludes
model weights, decode workspaces, Graph objects, and live CUDA pointers.  A
timing worker always allocates a new adapter-owned cache and copies this state
into those caller-owned buffers before Graph capture and before timing.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any


PREFIX_STATE_SCHEMA = "kvbench-phase13-prefix-state-1.0.0"
PREFIX_STATE_FILE = "state.safetensors"
PREFIX_STATE_MANIFEST = "manifest.json"
PREFIX_STATE_COMPLETE = "COMPLETE"


class Phase13PrefixStateError(RuntimeError):
    """A prefix snapshot is incomplete, incompatible, or altered."""


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_spec(
    tensors: Mapping[str, Any], batch_axes: Mapping[str, int | None]
) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "shape": [int(item) for item in tensor.shape],
            "dtype": str(tensor.dtype),
            "batch_axis": batch_axes[name],
        }
        for name, tensor in sorted(tensors.items())
    ]


def _bf16_views(cache: Any, historical: int) -> tuple[dict[str, Any], dict[str, int | None]]:
    tensors = {
        "keys": cache.keys[:, :, :, :historical, :],
        "values": cache.values[:, :, :, :historical, :],
    }
    return tensors, {name: 1 for name in tensors}


def _turboquant_views(
    cache: Any, historical: int
) -> tuple[dict[str, Any], dict[str, int | None]]:
    packed = cache.packed_cache.reshape(
        len(cache.compressed_layers),
        cache.batch_size,
        cache.block_count,
        cache.block_size,
        cache.num_kv_heads,
        cache.slot_size,
    )
    tensors = {
        "packed_cache": packed,
        "bf16_keys": cache.bf16_cache.keys[:, :, :, :historical, :],
        "bf16_values": cache.bf16_cache.values[:, :, :, :historical, :],
    }
    return tensors, {name: 1 for name in tensors}


def _kivi_views(cache: Any) -> tuple[dict[str, Any], dict[str, int | None]]:
    batch_names = (
        "packed_key_history",
        "packed_value_history",
        "key_scales",
        "key_minimums",
        "value_scales",
        "value_minimums",
        "key_residual",
        "value_residual_ring",
    )
    ledger_names = (
        "key_history_token_indices",
        "key_residual_token_indices",
        "value_history_token_indices",
        "value_residual_token_indices",
    )
    tensors = {
        name: getattr(cache, name) for name in (*batch_names, *ledger_names)
    }
    axes = {name: 1 for name in batch_names}
    axes.update({name: None for name in ledger_names})
    return tensors, axes


def _kvquant_views(cache: Any) -> tuple[dict[str, Any], dict[str, int | None]]:
    axis_one = (
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
    axis_zero = (
        "value_store_lower_bounds",
        "value_store_upper_bounds",
    )
    tensors = {name: getattr(cache, name) for name in (*axis_one, *axis_zero)}
    axes = {name: 1 for name in axis_one}
    axes.update({name: 0 for name in axis_zero})
    return tensors, axes


def prefix_state_views(
    cache: Any, *, family: str, historical: int
) -> tuple[dict[str, Any], dict[str, int | None]]:
    """Return only state written by prefill and needed by fixed-L decode."""

    if not isinstance(historical, int) or isinstance(historical, bool) or historical <= 0:
        raise Phase13PrefixStateError("historical context is invalid")
    if historical >= int(cache.capacity):
        raise Phase13PrefixStateError("prefix snapshot requires one scratch slot")
    if family == "bf16":
        return _bf16_views(cache, historical)
    if family == "turboquant":
        return _turboquant_views(cache, historical)
    if family == "kivi":
        return _kivi_views(cache)
    if family == "kvquant":
        return _kvquant_views(cache)
    raise Phase13PrefixStateError("prefix snapshot method family is unsupported")


def _lifecycle(cache: Any, family: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "active_context": int(cache.active_context),
        "mode": str(cache.mode),
    }
    if family == "kivi":
        payload.update(
            {
                "key_history_counts": list(cache._key_history_counts),
                "key_residual_counts": list(cache._key_residual_counts),
                "value_history_counts": list(cache._value_history_counts),
                "value_residual_counts": list(cache._value_residual_counts),
                "value_residual_heads": list(cache._value_residual_heads),
                "fixed_scratch_tokens": list(cache._fixed_scratch_tokens),
            }
        )
    elif family == "kvquant":
        payload["known_key_active_entries"] = cache._known_key_active_entries
    return payload


def _restore_lifecycle(cache: Any, family: str, lifecycle: Mapping[str, Any]) -> None:
    if family == "kivi":
        fields = {
            "_key_history_counts": "key_history_counts",
            "_key_residual_counts": "key_residual_counts",
            "_value_history_counts": "value_history_counts",
            "_value_residual_counts": "value_residual_counts",
            "_value_residual_heads": "value_residual_heads",
            "_fixed_scratch_tokens": "fixed_scratch_tokens",
        }
        for attribute, key in fields.items():
            values = lifecycle.get(key)
            if (
                not isinstance(values, list)
                or len(values) != int(cache.num_layers)
                or any(not isinstance(item, int) or isinstance(item, bool) for item in values)
            ):
                raise Phase13PrefixStateError("KIVI prefix lifecycle differs")
            getattr(cache, attribute)[:] = values
    elif family == "kvquant":
        value = lifecycle.get("known_key_active_entries")
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 0
        ):
            raise Phase13PrefixStateError("KVQuant prefix lifecycle differs")
        cache._known_key_active_entries = value


def save_prefix_state(
    *,
    cache: Any,
    family: str,
    configuration: str,
    historical: int,
    source_batch: int,
    output: Path,
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Write one safe-format snapshot after a direct untimed prefill."""

    from safetensors.torch import save_file

    if output.exists() or output.is_symlink():
        raise Phase13PrefixStateError("prefix state output already exists")
    if int(cache.batch_size) != source_batch or int(cache.active_context) != historical:
        raise Phase13PrefixStateError("direct prefix cache geometry differs")
    if str(cache.mode) != "ready":
        raise Phase13PrefixStateError("direct prefix cache is not ready")
    output.mkdir(parents=False)
    tensors, axes = prefix_state_views(cache, family=family, historical=historical)
    cpu_tensors = {
        name: tensor.detach().to(device="cpu", copy=True).contiguous()
        for name, tensor in tensors.items()
    }
    state_path = output / PREFIX_STATE_FILE
    save_file(cpu_tensors, str(state_path))
    state_sha256 = _sha256_file(state_path)
    lifecycle = _lifecycle(cache, family)
    manifest = {
        "schema_version": PREFIX_STATE_SCHEMA,
        "configuration": configuration,
        "family": family,
        "historical_context": historical,
        "capacity": int(cache.capacity),
        "source_batch": source_batch,
        "source_layout_fingerprint": cache.layout_fingerprint(),
        "state_file": PREFIX_STATE_FILE,
        "state_file_bytes": state_path.stat().st_size,
        "state_file_sha256": state_sha256,
        "tensors": _tensor_spec(cpu_tensors, axes),
        "lifecycle": lifecycle,
        "authority": dict(authority),
        "runtime_prefix_sharing": False,
        "fresh_target_allocation_required": True,
        "restore_outside_timing": True,
    }
    (output / PREFIX_STATE_MANIFEST).write_bytes(_canonical_bytes(manifest))
    (output / PREFIX_STATE_COMPLETE).write_text(
        state_sha256 + "\n", encoding="ascii"
    )
    return manifest


def validate_prefix_state(
    root: Path,
    *,
    configuration: str | None = None,
    historical: int | None = None,
    verify_state_bytes: bool = True,
) -> dict[str, Any]:
    """Validate the manifest and, once per catalog, its complete state bytes."""

    if not root.is_dir() or root.is_symlink():
        raise Phase13PrefixStateError("prefix state root is invalid")
    expected_names = {PREFIX_STATE_FILE, PREFIX_STATE_MANIFEST, PREFIX_STATE_COMPLETE}
    observed_names = {item.name for item in root.iterdir()}
    if observed_names != expected_names:
        raise Phase13PrefixStateError("prefix state file set differs")
    try:
        manifest = json.loads((root / PREFIX_STATE_MANIFEST).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase13PrefixStateError("prefix state manifest is invalid") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != PREFIX_STATE_SCHEMA:
        raise Phase13PrefixStateError("prefix state schema differs")
    if configuration is not None and manifest.get("configuration") != configuration:
        raise Phase13PrefixStateError("prefix state configuration differs")
    if historical is not None and manifest.get("historical_context") != historical:
        raise Phase13PrefixStateError("prefix state context differs")
    state_path = root / PREFIX_STATE_FILE
    if (
        state_path.stat().st_size != manifest.get("state_file_bytes")
        or (root / PREFIX_STATE_COMPLETE).read_text(encoding="ascii")
        != str(manifest.get("state_file_sha256")) + "\n"
    ):
        raise Phase13PrefixStateError("prefix state size or COMPLETE differs")
    if verify_state_bytes and _sha256_file(state_path) != manifest.get("state_file_sha256"):
        raise Phase13PrefixStateError("prefix state checksum differs")
    tensors = manifest.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        raise Phase13PrefixStateError("prefix state tensor inventory differs")
    names = [item.get("name") for item in tensors if isinstance(item, dict)]
    if len(names) != len(tensors) or len(set(names)) != len(names):
        raise Phase13PrefixStateError("prefix state tensor names differ")
    source_batch = manifest.get("source_batch")
    if source_batch not in {1, 4, 8}:
        raise Phase13PrefixStateError("prefix state source batch differs")
    for item in tensors:
        shape = item.get("shape")
        axis = item.get("batch_axis")
        if (
            not isinstance(shape, list)
            or not shape
            or any(not isinstance(value, int) or value < 0 for value in shape)
            or (axis is not None and (not isinstance(axis, int) or axis < 0 or axis >= len(shape)))
            or (axis is not None and shape[axis] != source_batch)
        ):
            raise Phase13PrefixStateError("prefix state batch geometry differs")
    if not isinstance(manifest.get("authority"), dict) or not isinstance(
        manifest.get("lifecycle"), dict
    ):
        raise Phase13PrefixStateError("prefix state authority or lifecycle differs")
    return manifest


def restored_prefix_witness(manifest: Mapping[str, Any], *, target_batch: int) -> str:
    payload = {
        "schema_version": "kvbench-phase13-restored-prefix-witness-1.0.0",
        "state_file_sha256": manifest.get("state_file_sha256"),
        "configuration": manifest.get("configuration"),
        "historical_context": manifest.get("historical_context"),
        "source_batch": manifest.get("source_batch"),
        "target_batch": target_batch,
        "batch_selection": "leading_independent_rows",
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def restore_prefix_state(
    *,
    cache: Any,
    family: str,
    configuration: str,
    historical: int,
    root: Path,
    expected_state_sha256: str,
) -> dict[str, Any]:
    """Restore into newly allocated caller-owned cache tensors, outside timing."""

    from safetensors import safe_open

    manifest = validate_prefix_state(
        root,
        configuration=configuration,
        historical=historical,
        verify_state_bytes=False,
    )
    if manifest.get("state_file_sha256") != expected_state_sha256:
        raise Phase13PrefixStateError("prefix catalog binding differs")
    source_batch = manifest.get("source_batch")
    target_batch = int(cache.batch_size)
    if (
        not isinstance(source_batch, int)
        or isinstance(source_batch, bool)
        or target_batch not in {1, 4, 8}
        or source_batch not in {1, 4, 8}
        or target_batch > source_batch
        or int(cache.capacity) != manifest.get("capacity")
    ):
        raise Phase13PrefixStateError("prefix restore geometry differs")
    target_tensors, target_axes = prefix_state_views(
        cache, family=family, historical=historical
    )
    specs = {item["name"]: item for item in manifest["tensors"]}
    if set(specs) != set(target_tensors):
        raise Phase13PrefixStateError("prefix restore tensor inventory differs")
    cache.prepare_prefill(historical)
    with safe_open(str(root / PREFIX_STATE_FILE), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(target_tensors):
            raise Phase13PrefixStateError("safe tensor inventory differs")
        for name in sorted(target_tensors):
            target = target_tensors[name]
            source = handle.get_tensor(name)
            spec = specs[name]
            axis = target_axes[name]
            if axis != spec.get("batch_axis") or str(source.dtype) != spec.get("dtype"):
                raise Phase13PrefixStateError("prefix tensor metadata differs")
            if [int(item) for item in source.shape] != spec.get("shape"):
                raise Phase13PrefixStateError("prefix source shape differs")
            if axis is not None:
                source = source.narrow(axis, 0, target_batch)
            if tuple(source.shape) != tuple(target.shape) or source.dtype != target.dtype:
                raise Phase13PrefixStateError("prefix target shape or dtype differs")
            target.copy_(source, non_blocking=False)
    _restore_lifecycle(cache, family, manifest["lifecycle"])
    cache.complete_prefill()
    if int(cache.active_context) != historical or str(cache.mode) != "ready":
        raise Phase13PrefixStateError("restored cache lifecycle differs")
    if getattr(cache.device, "type", None) == "cuda":
        import torch

        torch.cuda.synchronize(device=cache.device)
    return {
        "state_file_sha256": expected_state_sha256,
        "source_batch": source_batch,
        "target_batch": target_batch,
        "witness_sha256": restored_prefix_witness(
            manifest, target_batch=target_batch
        ),
        "fresh_target_allocation": True,
        "runtime_prefix_sharing": False,
        "restore_outside_timing": True,
    }
