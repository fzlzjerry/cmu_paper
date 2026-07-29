#!/usr/bin/env python3
"""Create the narrow Decision 0025 mixed-provenance fixture bundle."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import struct
import tempfile
from typing import Any, Mapping

from reference.kvquant import generate_fixtures as legacy
from scripts.validate_kvquant_graphsafe_patch import validate as validate_source


FIXTURE_ID = "kvqref-2e0a0e9022c50cbc6fb497d88cae973e"
OLD_FIXTURE_ID = "kvqref-a50af6511c314b6394e58a7f81ceefb8"
OLD_ROOT = "32cdf465a361dd6695b66ccbea0a462bddc075fd9778d0aa8cdaa3f94e6f63ab"
PATCH_SHA256 = "23a15db86790299392412c3ce2da7d971f4f073cfaf6839d82d3746c8b56b551"
PATCHED_COMMIT = "0d9df350bd1788284e1ce76a8bf6e886beca5efa"
PATCHED_TREE = "a85cf7bf093982a4bf89c33d4e6794d9a85f846d"
EXTENSION_SHA256 = "46c41aad8f56d58608d4c1273bd3a72fd36c8f69f9ca2c5a046f0c811631bf51"
MEASUREMENT_CONTAINER = (
    "sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e"
)
OLD_FAMILY_SHA256 = {
    "kvq4": "52bfbdf2c29f546b391b6079497aaf5c3d17a3a125dd0d096c748cc0fae2e0a8",
    "kvq2": "7625600b6b0a5341d542a40edf13693354a9e6a11847a82440349eaed2d927ac",
}


class GenerationError(RuntimeError):
    """Raised when the corrected bundle would violate its narrow contract."""


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_digest(root: Path) -> tuple[str, int]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(f"{_sha256_file(path)}  {relative}\n".encode("utf-8"))
    return digest.hexdigest(), len(files)


def _copy_family_exact(source: Path, destination: Path, family: str) -> dict[str, Any]:
    expected, count = _tree_digest(source)
    if expected != OLD_FAMILY_SHA256[family]:
        raise GenerationError(f"old {family} family digest mismatch")
    shutil.copytree(source, destination, copy_function=shutil.copy2)
    observed, observed_count = _tree_digest(destination)
    if observed != expected or observed_count != count:
        raise GenerationError(f"copied {family} bytes differ")
    for original in sorted(path for path in source.rglob("*") if path.is_file()):
        relative = original.relative_to(source)
        copied = destination / relative
        if original.is_symlink() or copied.is_symlink():
            raise GenerationError("symlinks are forbidden in reused fixtures")
        if original.stat().st_ino == copied.stat().st_ino:
            raise GenerationError("reused fixture must be an independent file copy")
    return {
        "source_fixture_id": OLD_FIXTURE_ID,
        "source_root_sha256": OLD_ROOT,
        "family_tree_sha256": observed,
        "file_count": observed_count,
        "copy_mode": "ordinary_files_byte_identical",
    }


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _scalar_codes(
    values: Any,
    lookup: Any,
    lower: Any,
    upper: Any,
) -> Any:
    import torch

    values = values.detach().float().cpu().contiguous()
    lookup = lookup.detach().float().cpu().contiguous()
    lower = lower.detach().float().cpu().contiguous()
    upper = upper.detach().float().cpu().contiguous()
    tokens = values.shape[0]
    codes = torch.empty(
        (tokens, legacy.NUM_KV_HEADS, legacy.HEAD_DIM),
        dtype=torch.int64,
    )
    for token in range(tokens):
        lower_value = float(lower[token].item())
        upper_value = float(upper[token].item())
        for head in range(legacy.NUM_KV_HEADS):
            for channel in range(legacy.HEAD_DIM):
                value = float(values[token, head, channel].item())
                if value < lower_value or value > upper_value:
                    code = 3
                else:
                    code = 0
                    best = abs(_f32(value - float(lookup[token, 0].item())))
                    for candidate in range(1, 8):
                        distance = abs(
                            _f32(value - float(lookup[token, candidate].item()))
                        )
                        if distance < best:
                            best = distance
                            code = candidate
                codes[token, head, channel] = code
    return codes


def _scalar_pack(codes: Any) -> Any:
    import torch

    codes = codes.detach().to(dtype=torch.int64, device="cpu").contiguous()
    tokens = codes.shape[0]
    packed = torch.zeros(
        (legacy.NUM_KV_HEADS, 12, tokens),
        dtype=torch.int64,
    )
    mask32 = (1 << 32) - 1
    for token in range(tokens):
        for head in range(legacy.NUM_KV_HEADS):
            for channel in range(legacy.HEAD_DIM):
                code = int(codes[token, head, channel].item())
                bit_offset = channel * 3
                word = bit_offset // 32
                shift = bit_offset % 32
                packed[head, word, token] |= (code << shift) & mask32
                if shift + 3 > 32:
                    packed[head, word + 1, token] |= code >> (32 - shift)
    signed = packed.clone()
    signed[signed >= (1 << 31)] -= 1 << 32
    return signed.to(torch.int32).contiguous()


def _scalar_controls(result: Mapping[str, Any]) -> tuple[Any, Any]:
    import torch

    values = (
        result["inputs"]["value_after_v_proj"][
            0, :, legacy.SINK_TOKENS : legacy.TOTAL_CONTEXT, :
        ]
        .permute(1, 0, 2)
        .contiguous()
    )
    lookup = result["quant_v"].lookup_table[: legacy.QUANTIZED_CONTEXT]
    selection = result["value_selection"]
    lower = selection.dense_lower_bound[
        legacy.SINK_TOKENS : legacy.TOTAL_CONTEXT
    ]
    upper = selection.dense_upper_bound[
        legacy.SINK_TOKENS : legacy.TOTAL_CONTEXT
    ]
    packed_all = _scalar_pack(_scalar_codes(values, lookup, lower, upper))
    packed_store = packed_all[:, :, : legacy.STORE_QUANTIZED_CONTEXT]
    actual_store = result["store_state"]["v_dense_allocated"][
        :, :, : legacy.STORE_QUANTIZED_CONTEXT
    ].cpu()
    actual_all = result["append_state"]["v_dense_allocated"][
        :, :, : legacy.QUANTIZED_CONTEXT
    ].cpu()
    if not torch.equal(packed_store, actual_store):
        raise GenerationError("kvq3 store differs from scalar control")
    if not torch.equal(packed_all, actual_all):
        raise GenerationError("kvq3 append differs from scalar control")
    return packed_store, packed_all


def _write_corrected_fixture(
    root: Path,
    result: Mapping[str, Any],
    authority_hashes: Mapping[str, str],
    scalar_store: Any,
    scalar_append: Any,
) -> None:
    tensors = legacy._fixture_tensors(result)
    tensors["dense_payload.safetensors"][
        "value_scalar_control_after_store"
    ] = scalar_store
    tensors["dense_payload.safetensors"][
        "value_scalar_control_after_append"
    ] = scalar_append
    byte_breakdown = legacy._byte_breakdown(result)
    manifest = legacy._manifest_for_fixture(
        result,
        tensors,
        byte_breakdown,
        authority_hashes,
    )
    manifest["schema_version"] = "kvbench-phase11pr-kvquant-fixture-1.0.0"
    manifest["packing"]["three_bit_parallel_value_source_behavior"] = (
        "deterministic_per_token_per_channel_nearest"
    )
    manifest["numerical_control"]["kvq3_scalar_code_pack"] = "PASS"
    manifest["source"]["decision"] = "0025"
    root.mkdir(parents=True, exist_ok=False)
    legacy._write_exclusive(
        root / "fixture_manifest.json",
        _canonical_json(manifest),
    )
    for filename, payload in tensors.items():
        legacy._save_safetensors(root / filename, payload)
    legacy._write_exclusive(
        root / "byte_breakdown.json",
        _canonical_json(byte_breakdown),
    )
    ledger = "".join(
        f"{_sha256_file(path)}  {path.name}\n"
        for path in sorted(root.iterdir())
        if path.name != "checksums.sha256"
    ).encode("utf-8")
    legacy._write_exclusive(root / "checksums.sha256", ledger)
    if {path.name for path in root.iterdir()} != set(legacy.FIXTURE_MEMBERS):
        raise GenerationError("corrected fixture member set mismatch")


def _authority_files(
    source_report: Mapping[str, Any],
    extension: Path,
    patch_manifest: Path,
) -> dict[str, bytes]:
    source = {
        "schema_version": "kvbench-phase11pr-source-1.0.0",
        "status": "PASS",
        "decision": "0025",
        "upstream_base_commit": legacy.UPSTREAM_BASE_COMMIT,
        "upstream_base_tree": legacy.UPSTREAM_BASE_TREE,
        "aggregate_patch_sha256": PATCH_SHA256,
        "patched_commit": PATCHED_COMMIT,
        "patched_tree": PATCHED_TREE,
        "patch_manifest_sha256": _sha256_file(patch_manifest),
        "reconstruction": dict(source_report["reconstruction"]),
        "source_mount": "read_only",
        "official_author_gqa_support_claimed": False,
    }
    environment = {
        "schema_version": "kvbench-phase11pr-environment-1.0.0",
        "status": "PASS",
        "fixture_generation_image": legacy.REFERENCE_IMAGE_CONFIG_DIGEST,
        "cuda_validation_container": MEASUREMENT_CONTAINER,
        "network": "disabled",
        "source_mount": "read_only",
        "calibration_mount": "read_only",
        "credentials_in_container": False,
    }
    calibration_path = Path("/repo/reference/kvquant/calibration_manifest.json")
    calibration = calibration_path.read_bytes()
    build = {
        "schema_version": "kvbench-phase11pr-build-1.0.0",
        "status": "PASS",
        "authorized_measurement_container": MEASUREMENT_CONTAINER,
        "patched_tree": PATCHED_TREE,
        "extension_sha256": _sha256_file(extension),
        "sm_120_cubin": True,
        "compute_120_ptx": True,
        "extension_published": False,
    }
    return {
        "source_manifest.json": _canonical_json(source),
        "environment.json": _canonical_json(environment),
        "calibration_manifest.json": calibration,
        "build_manifest.json": _canonical_json(build),
    }


def _root_manifest(
    authority_hashes: Mapping[str, str],
    reuse: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "kvbench-phase11pr-kvquant-reference-bundle-1.0.0",
        "run_id": FIXTURE_ID,
        "status": "completed",
        "phase": "phase11p_r_kvq3_value_pack_correction",
        "run_kind": "reference_fixture_correction",
        "decision": "0025",
        "method_identifier": "kvquant_gqa_graphsafe_kvq3_v2",
        "source": {
            "upstream_base_commit": legacy.UPSTREAM_BASE_COMMIT,
            "upstream_base_tree": legacy.UPSTREAM_BASE_TREE,
            "aggregate_patch_sha256": PATCH_SHA256,
            "patched_commit": PATCHED_COMMIT,
            "patched_tree": PATCHED_TREE,
            "extension_sha256": EXTENSION_SHA256,
        },
        "calibration": {
            "calibration_id": legacy.CALIBRATION_ID,
            "root_sha256": legacy.CALIBRATION_ROOT,
            "fisher_regenerated": False,
            "quantizers_regenerated": False,
        },
        "authority_manifest_sha256": dict(authority_hashes),
        "fixture_matrix": {
            "families": ["kvq4", "kvq3", "kvq2"],
            "cases": [case for case, _ in legacy.CASES],
            "total": 9,
            "legacy_ambiguous_aliases": False,
        },
        "family_provenance": {
            "kvq4": dict(reuse["kvq4"]),
            "kvq3": {
                "fixture_id": FIXTURE_ID,
                "mode": "regenerated_deterministic_kvq3_only",
                "source_decision": "0025",
                "scalar_control": "PASS",
            },
            "kvq2": dict(reuse["kvq2"]),
        },
        "sparse_contract": {
            "key_counts": [0, 6, 12],
            "value_non_sink_count": 12,
            "value_sink_count": 0,
            "value_selection": "six_lowest_plus_six_highest",
        },
        "gates": {
            "g2_kvq": "NOT_EVALUATED",
            "global_g2_g5": "NOT_EVALUATED",
            "full_scan": "CLOSED",
            "quality_execution": "LOCKED",
            "performance_data_frozen_present": False,
        },
        "performance_measurement": False,
        "profiler_execution": False,
        "quality_evaluation": False,
    }


def generate(arguments: argparse.Namespace) -> dict[str, Any]:
    source = Path(arguments.source_root).resolve(strict=True)
    calibration = Path(arguments.calibration_root).resolve(strict=True)
    extension = Path(arguments.extension).resolve(strict=True)
    old = Path(arguments.old_fixtures).resolve(strict=True)
    destination = Path(arguments.destination).resolve()
    patch_manifest = Path(arguments.patch_manifest).resolve(strict=True)
    if destination.exists() or destination.is_symlink():
        raise GenerationError("refusing to overwrite corrected fixture bundle")
    if _sha256_file(extension) != EXTENSION_SHA256:
        raise GenerationError("corrected extension digest mismatch")
    source_report = validate_source(source)
    calibration_report = legacy._validate_calibration(calibration)
    if calibration_report.get("status") != "PASS":
        raise GenerationError("calibration validation failed")

    legacy.FIXTURE_ID = FIXTURE_ID
    legacy.PATCH_SHA256 = PATCH_SHA256
    legacy.PATCHED_COMMIT = PATCHED_COMMIT
    legacy.PATCHED_TREE = PATCHED_TREE
    legacy.DECISION = "0025"

    authority_files = _authority_files(source_report, extension, patch_manifest)
    authority_hashes = {
        "source_manifest": hashlib.sha256(
            authority_files["source_manifest.json"]
        ).hexdigest(),
        "environment": hashlib.sha256(
            authority_files["environment.json"]
        ).hexdigest(),
        "calibration_manifest": hashlib.sha256(
            authority_files["calibration_manifest.json"]
        ).hexdigest(),
        "build_manifest": hashlib.sha256(
            authority_files["build_manifest.json"]
        ).hexdigest(),
    }
    runtime = legacy._runtime()
    import quant_cuda

    if Path(quant_cuda.__file__).resolve() != extension:
        raise GenerationError("loaded extension path mismatch")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".phase11pr-fixtures-", dir=destination.parent)
    )
    stage = temporary / "fixtures"
    stage.mkdir()
    try:
        reuse = {
            family: _copy_family_exact(
                old / family,
                stage / family,
                family,
            )
            for family in ("kvq4", "kvq2")
        }
        authority_root = stage / "authority"
        authority_root.mkdir()
        for name, payload in authority_files.items():
            legacy._write_exclusive(authority_root / name, payload)

        rope = legacy._make_rope(runtime)
        quantizer = legacy._load_layer_zero_quantizer(calibration, "kvq3")
        for case_name, key_count in legacy.CASES:
            with runtime["torch"].inference_mode():
                result = legacy._execute_fixture(
                    runtime,
                    family="kvq3",
                    bit_width=3,
                    case_name=case_name,
                    expected_key_count=key_count,
                    quantizer=quantizer,
                    rope=rope,
                )
                repeat = legacy._execute_fixture(
                    runtime,
                    family="kvq3",
                    bit_width=3,
                    case_name=case_name,
                    expected_key_count=key_count,
                    quantizer=quantizer,
                    rope=rope,
                )
            scalar_store, scalar_append = _scalar_controls(result)
            if (
                result["value_store_matches_nearest"] is not True
                or result["value_append_matches_nearest"] is not True
            ):
                raise GenerationError("kvq3 nearest-code control did not pass")
            first_store = result["store_state"]["v_dense_allocated"].cpu()
            second_store = repeat["store_state"]["v_dense_allocated"].cpu()
            first_append = result["append_state"]["v_dense_allocated"].cpu()
            second_append = repeat["append_state"]["v_dense_allocated"].cpu()
            if (
                not runtime["torch"].equal(first_store, second_store)
                or not runtime["torch"].equal(first_append, second_append)
            ):
                raise GenerationError("kvq3 output remains history dependent")
            _write_corrected_fixture(
                stage / "kvq3" / case_name,
                result,
                authority_hashes,
                scalar_store,
                scalar_append,
            )
            del result, repeat
            gc.collect()
            runtime["torch"].cuda.empty_cache()

        reuse_proof = {
            "schema_version": "kvbench-phase11pr-reuse-proof-1.0.0",
            "old_fixture_id": OLD_FIXTURE_ID,
            "old_root_sha256": OLD_ROOT,
            "families": reuse,
            "kvq3_regenerated_cases": [
                case for case, _ in legacy.CASES
            ],
        }
        legacy._write_exclusive(
            stage / "reuse_proof.json",
            _canonical_json(reuse_proof),
        )
        old_trace = legacy._reference_trace
        legacy._reference_trace = lambda: {
            "schema_version": "kvbench-phase11pr-reference-trace-1.0.0",
            "run_id": FIXTURE_ID,
            "run_kind": "reference_trace",
            "timing_fields_present": False,
            "operations": {
                "kvq4": "byte_reused",
                "kvq2": "byte_reused",
                "kvq3": "deterministic_per_token_per_channel_pack",
                "backend_fallback": False,
                "complete_prefix_temporary": False,
            },
        }
        try:
            root = legacy._finalize_bundle(
                stage,
                _root_manifest(authority_hashes, reuse),
            )
        finally:
            legacy._reference_trace = old_trace
        legacy._install_no_replace(stage, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
    return {
        "status": "PASS",
        "fixture_id": FIXTURE_ID,
        "local_root_sha256": root,
        "fixture_count": 9,
        "destination": str(destination),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--calibration-root", required=True)
    parser.add_argument("--extension", required=True)
    parser.add_argument("--patch-manifest", required=True)
    parser.add_argument("--old-fixtures", required=True)
    parser.add_argument("--destination", required=True)
    arguments = parser.parse_args()
    try:
        result = generate(arguments)
    except (GenerationError, legacy.ReferenceGenerationError, OSError, RuntimeError, ValueError) as error:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "error_type": type(error).__name__,
                    "reason": str(error),
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
