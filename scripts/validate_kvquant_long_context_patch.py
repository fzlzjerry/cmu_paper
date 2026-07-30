#!/usr/bin/env python3
"""Validate the Decision 0027 deterministic long-context KVQuant patch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import scripts.validate_kvquant_gqa_patch as base


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT
    / "third_party/patches/kvquant/"
    "deterministic-long-context-manifest.json"
)
PATCH = (
    "third_party/patches/kvquant/"
    "0003-deterministic-long-context-value-decode.patch"
)
EVIDENCE = ROOT / "docs/evidence/phase11d/cuda-validation.json"
PARENT_COMMIT = "0d9df350bd1788284e1ce76a8bf6e886beca5efa"
PARENT_TREE = "a85cf7bf093982a4bf89c33d4e6794d9a85f846d"
BASE_COMMIT = "57a238357f0ffe50084670fcd5781c9848f80ea2"
BASE_TREE = "094e0f736f77ee327e5350cbd1eefb1c936aa77b"
SOURCE_REPOSITORY = "https://github.com/SqueezeAILab/KVQuant.git"

CPP_PATH = "deployment/kvquant/quant_cuda.cpp"
CUDA_PATH = "deployment/kvquant/quant_cuda_kernel.cu"
PARENT_CHANGED_PATHS = [CPP_PATH, CUDA_PATH]

API = (
    "vecquant4matmul_nuq_perchannel_transposed_mha_batched_fused_"
    "opt2_deterministic_out"
)
TILE_KERNEL = (
    "VecQuant4MatMulKernelNUQPerChannelTransposedMHABatchedFused"
    "OptDeterministicTiles"
)
REDUCE_KERNEL = (
    "VecQuant4MatMulKernelNUQPerChannelTransposedMHABatchedFused"
    "OptDeterministicReduce"
)
LEGACY_VALUE_KERNELS = {
    4: (
        "VecQuant4MatMulKernelNUQPerChannelTransposedMHABatched"
        "FusedOpt("
    ),
    3: (
        "VecQuant3MatMulKernelNUQPerChannelTransposedMHABatched"
        "FusedOpt("
    ),
    2: (
        "VecQuant2MatMulKernelNUQPerChannelTransposedMHABatched"
        "FusedOpt("
    ),
}

SCHEMA_VERSION = (
    "kvbench-kvquant-long-context-source-patch-manifest-1.0.0"
)

PRESERVED_FILE_AUTHORITY = {
    "decision_0021_sha256": (
        ROOT / "docs/decisions/0021-kvquant-patch-main-repository-custody.md",
        "e09cb0f7c59c07eb04ec28319d6705c436c9c25d466bbe63e2f1859cf75d4daf",
    ),
    "decision_0022_sha256": (
        ROOT / "docs/decisions/0022-phase9-blocked-report-custody.md",
        "e500d82839935109dc3d789abc3bc167449838293ab7254a539176654bd3676f",
    ),
    "decision_0023_sha256": (
        ROOT
        / "docs/decisions/"
        "0023-phase10-kvquant-source-faithful-sparse-fixture-semantics.md",
        "e212b01fb286013a054567b7707a375447849986281d934c7ab17a73156bada3",
    ),
    "decision_0024_sha256": (
        ROOT
        / "docs/decisions/0024-kvquant-graph-safe-caller-owned-cuda-apis.md",
        "1117bea675bbc74af873674a3c0757d93e20bb4297ec0ee3ce99418f0fc46111",
    ),
    "decision_0025_sha256": (
        ROOT / "docs/decisions/0025-kvquant-deterministic-kvq3-value-pack.md",
        "06655f71fa5aef2077adeb40f2c1362efc27be9b42961dc8586c34d366eb0e5e",
    ),
    "decision_0026_sha256": (
        ROOT / "docs/decisions/0026-kvquant-pre-rope-adapter-boundary.md",
        "eee3fb412111b658ecedaecddae2844161d08fc04bb2c88e158bd808f6bfe6f2",
    ),
    "phase11_blocked_report_sha256": (
        ROOT / "docs/phase_reports/phase11-kvquant-measurement-adapter.md",
        "2f291fbcdd639c8854d9543fbacc458993da699d3fb6b4fee2c620981f4c5b88",
    ),
    "decision_0025_manifest_sha256": (
        ROOT / "third_party/patches/kvquant/graphsafe-kvq3-manifest.json",
        "d04a580a4a4cd0fcaa6f0880eadb46f9479ff2f7c6a250eb9568f9f520e2da6e",
    ),
    "decision_0025_patch_sha256": (
        ROOT
        / "third_party/patches/kvquant/"
        "0002-graphsafe-kvq3-deterministic.patch",
        "23a15db86790299392412c3ce2da7d971f4f073cfaf6839d82d3746c8b56b551",
    ),
    "adapter_sha256": (
        ROOT / "src/kvbench/adapters/kvquant.py",
        "897a94541924c4222f0aeb02b9ad190504bc850809c9687cdd93c93da0245fe3",
    ),
}
PRESERVED_IDENTITY_AUTHORITY = {
    "calibration_root_sha256": (
        "8148306d08205af376994b022f189a0d6837915cd279ca8af6b104e1f4b46ccf"
    ),
    "fixture_root_sha256": (
        "c28682d58706b58812dc1db69ba5eb4982339ba13f39bf67f751794cdaabfdec"
    ),
    "authorized_container_digest": (
        "sha256:"
        "059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e"
    ),
}


def _require_sha256(value: object, description: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise base.ValidationError(f"{description} must be a lowercase SHA-256")
    return value


def _require_git_oid(value: object, description: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[0-9a-f]{40}", value) is None
    ):
        raise base.ValidationError(
            f"{description} must be a lowercase 40-character Git object ID"
        )
    return value


def _decision_0027(
    manifest: dict[str, Any],
    *,
    evidence_finalized: bool,
) -> tuple[Path, str]:
    matches = sorted((ROOT / "docs/decisions").glob("0027-*.md"))
    if len(matches) != 1:
        raise base.ValidationError(
            "exactly one Decision 0027 record is required"
        )
    text = matches[0].read_text(encoding="utf-8")
    required = (
        "# Decision 0027",
        "Decision 0025",
        "Measurement-only",
        "caller-owned float32 workspace",
        "current PyTorch CUDA",
    )
    missing = [marker for marker in required if marker not in text]
    if missing:
        raise base.ValidationError(
            f"Decision 0027 contract marker missing: {missing[0]}"
        )
    bound_identities = (
        manifest["source"]["parent_commit"],
        manifest["source"]["parent_tree"],
        manifest["source"]["patched_commit"],
        manifest["source"]["patched_tree"],
        manifest["patch"]["sha256"],
        manifest["parent_delta"]["sha256"],
    )
    missing_identities = [
        value for value in bound_identities if value not in text
    ]
    if missing_identities:
        raise base.ValidationError(
            "Decision 0027 source identity marker missing: "
            f"{missing_identities[0]}"
        )
    status_match = re.search(
        r"^- Status: (Proposed|Accepted)$",
        text,
        flags=re.MULTILINE,
    )
    if status_match is None:
        raise base.ValidationError("Decision 0027 status is invalid")
    status = status_match.group(1)
    if evidence_finalized and status != "Accepted":
        raise base.ValidationError(
            "finalized Decision 0027 evidence requires Accepted status"
        )
    return matches[0], status


def _validate_preserved_authority(manifest: dict[str, Any]) -> None:
    observed = manifest.get("preserved_authority")
    if not isinstance(observed, dict):
        raise base.ValidationError("preserved_authority must be an object")
    expected_values = {
        name: expected
        for name, (_, expected) in PRESERVED_FILE_AUTHORITY.items()
    }
    expected_values.update(PRESERVED_IDENTITY_AUTHORITY)
    if observed != expected_values:
        raise base.ValidationError("preserved authority digest inventory mismatch")
    for name, (path, expected) in PRESERVED_FILE_AUTHORITY.items():
        if base._sha256(path.read_bytes()) != expected:
            raise base.ValidationError(f"preserved authority changed: {name}")


def _extract_function(source: str, marker: str) -> str:
    search_from = 0
    start = -1
    brace = -1
    while True:
        candidate = source.find(marker, search_from)
        if candidate < 0:
            break
        candidate_brace = source.find("{", candidate)
        candidate_semicolon = source.find(";", candidate)
        if candidate_brace >= 0 and (
            candidate_semicolon < 0 or candidate_brace < candidate_semicolon
        ):
            start = candidate
            brace = candidate_brace
            break
        search_from = candidate + len(marker)
    if start < 0:
        raise base.ValidationError(
            f"source function definition missing: {marker}"
        )
    depth = 0
    for index in range(brace, len(source)):
        character = source[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise base.ValidationError(f"unterminated function body after: {marker}")


def _validate_tail_initialization(cuda_source: str) -> None:
    for width, marker in LEGACY_VALUE_KERNELS.items():
        body = _extract_function(cuda_source, marker)
        levels = {4: 16, 3: 8, 2: 4}[width]
        required = (
            f"for (int val = 0; val < {levels}; val += 1)",
            "deq2[val][off] = 0.0f;",
            "blockvec[threadIdx.x] = 0.0f;",
        )
        if any(value not in body for value in required):
            raise base.ValidationError(
                f"kvq{width} tail shared-memory initialization is incomplete"
            )


def _validate_new_source_contract(source_root: Path) -> None:
    cpp_source = (source_root / CPP_PATH).read_text(encoding="utf-8")
    cuda_source = (source_root / CUDA_PATH).read_text(encoding="utf-8")

    cpp_wrapper = _extract_function(cpp_source, f"void {API}(")
    if (
        f"{API}_cuda(" not in cpp_wrapper
        or "const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));"
        not in cpp_wrapper
        or f'm.def("{API}", &{API}' not in cpp_source
    ):
        raise base.ValidationError(
            "deterministic caller-owned pybind API contract mismatch"
        )

    tile_body = _extract_function(cuda_source, TILE_KERNEL)
    reduce_body = _extract_function(cuda_source, REDUCE_KERNEL)
    launch_body = _extract_function(cuda_source, f"void {API}_cuda(")
    new_path = "\n".join((cpp_wrapper, tile_body, reduce_body, launch_body))

    if (
        "for (int token = token_start; token < token_stop; ++token)"
        not in tile_body
        or "for (int slot = 0; slot < num_outliers; ++slot)"
        not in tile_body
        or "for (int tile = 0; tile < num_tiles; ++tile)"
        not in reduce_body
    ):
        raise base.ValidationError("deterministic fixed-order loops changed")
    if (
        "workspace[workspace_offset] = dense_partial + sparse_partial;"
        not in tile_body
        or "mul[(batch * num_query_heads + query_head) * headdim + channel] = total;"
        not in reduce_body
    ):
        raise base.ValidationError("caller-owned output write contract changed")
    if launch_body.count("at::cuda::getCurrentCUDAStream()") != 2:
        raise base.ValidationError(
            "deterministic launches must use the current CUDA stream"
        )
    if launch_body.count("C10_CUDA_KERNEL_LAUNCH_CHECK();") != 2:
        raise base.ValidationError(
            "deterministic launches must each have an error check"
        )

    forbidden = (
        "atomicAdd",
        "atomicCAS",
        ".item(",
        ".cpu(",
        "torch::empty",
        "torch::zeros",
        "torch::full",
        "at::empty",
        "at::zeros",
        "cudaMalloc",
        "cudaFree",
        "cudaDeviceSynchronize",
        "cudaStreamSynchronize",
        "getDefaultCUDAStream",
        "cudaStreamDefault",
        "std::vector",
    )
    for marker in forbidden:
        if marker in new_path:
            raise base.ValidationError(
                f"forbidden deterministic-path source marker: {marker}"
            )
    if re.search(r"\bnew\s+", new_path):
        raise base.ValidationError(
            "dynamic C++ allocation is forbidden in the deterministic path"
        )
    if re.search(r"\batomic[A-Za-z_]*\b", new_path):
        raise base.ValidationError(
            "atomic operations are forbidden in the deterministic path"
        )
    if "blockvec" in tile_body or "deq2" in tile_body:
        raise base.ValidationError(
            "deterministic tile kernel must not use legacy shared-memory state"
        )

    _validate_tail_initialization(cuda_source)


def _parent_delta_bytes(source_root: Path, head: str) -> bytes:
    result = base._run_git(
        source_root,
        "diff",
        "--binary",
        "--full-index",
        PARENT_COMMIT,
        head,
        "--",
        *PARENT_CHANGED_PATHS,
        binary=True,
    )
    if not isinstance(result, bytes):
        raise AssertionError("binary Git command returned text")
    return result


def _git_object_bytes(source_root: Path, revision: str, path: str) -> bytes:
    result = base._run_git(
        source_root,
        "show",
        f"{revision}:{path}",
        binary=True,
    )
    if not isinstance(result, bytes):
        raise AssertionError("binary Git command returned text")
    return result


def _validate_parent_delta(
    source_root: Path,
    manifest: dict[str, Any],
    head: str,
) -> list[str]:
    parent_line = base._run_git(
        source_root,
        "rev-list",
        "--parents",
        "-n",
        "1",
        head,
    )
    if not isinstance(parent_line, str):
        raise AssertionError("text Git command returned bytes")
    if parent_line.strip().split() != [head, PARENT_COMMIT]:
        raise base.ValidationError(
            "patched source must be exactly one single-parent commit above "
            "the Decision 0025 source"
        )
    count = base._run_git(
        source_root,
        "rev-list",
        "--count",
        f"{PARENT_COMMIT}..{head}",
    )
    if not isinstance(count, str) or count.strip() != "1":
        raise base.ValidationError(
            "Decision 0027 source is not exactly one commit above its parent"
        )
    parent_tree = base._run_git(
        source_root,
        "rev-parse",
        f"{PARENT_COMMIT}^{{tree}}",
    )
    if not isinstance(parent_tree, str) or parent_tree.strip() != PARENT_TREE:
        raise base.ValidationError("Decision 0025 parent tree mismatch")

    changed = base._run_git(
        source_root,
        "diff",
        "--name-only",
        PARENT_COMMIT,
        head,
    )
    if not isinstance(changed, str):
        raise AssertionError("text Git command returned bytes")
    changed_paths = changed.splitlines()
    if changed_paths != PARENT_CHANGED_PATHS:
        raise base.ValidationError(
            "parent-relative changed-file scope must be exactly quant_cuda.cpp "
            "and quant_cuda_kernel.cu"
        )

    delta = manifest.get("parent_delta")
    if not isinstance(delta, dict):
        raise base.ValidationError("parent_delta must be an object")
    delta_bytes = _parent_delta_bytes(source_root, head)
    if (
        delta.get("sha256") != base._sha256(delta_bytes)
        or delta.get("size_bytes") != len(delta_bytes)
        or delta.get("changed_file_count") != len(PARENT_CHANGED_PATHS)
    ):
        raise base.ValidationError("parent-relative patch identity mismatch")

    records = delta.get("changed_files")
    if not isinstance(records, list) or len(records) != 2:
        raise base.ValidationError(
            "parent_delta.changed_files must contain exactly two records"
        )
    if [record.get("path") for record in records] != PARENT_CHANGED_PATHS:
        raise base.ValidationError(
            "parent-relative changed-file record order mismatch"
        )

    for record in records:
        path = record["path"]
        before = _git_object_bytes(source_root, PARENT_COMMIT, path)
        after = _git_object_bytes(source_root, head, path)
        before_blob = base._run_git(
            source_root,
            "rev-parse",
            f"{PARENT_COMMIT}:{path}",
        )
        after_blob = base._run_git(
            source_root,
            "rev-parse",
            f"{head}:{path}",
        )
        if not isinstance(before_blob, str) or not isinstance(after_blob, str):
            raise AssertionError("text Git command returned bytes")
        expected = {
            "path": path,
            "before_git_blob": before_blob.strip(),
            "after_git_blob": after_blob.strip(),
            "before_sha256": base._sha256(before),
            "after_sha256": base._sha256(after),
        }
        if record != expected:
            raise base.ValidationError(
                f"parent-relative file identity mismatch: {path}"
            )
    return changed_paths


def _validate_manifest_contract(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise base.ValidationError(
            "unsupported deterministic long-context manifest schema"
        )
    authority = manifest.get("authority")
    if not isinstance(authority, dict):
        raise base.ValidationError("authority must be an object")
    if (
        authority.get("method_identifier")
        != "kvquant_gqa_longctx_deterministic_v3"
        or authority.get("decisions") != ["0021", "0024", "0025", "0027"]
        or authority.get("official_gqa_support_claimed") is not False
    ):
        raise base.ValidationError("Decision 0027 authority mismatch")

    source = manifest.get("source")
    if not isinstance(source, dict):
        raise base.ValidationError("source must be an object")
    expected_source = {
        "repository": SOURCE_REPOSITORY,
        "base_commit": BASE_COMMIT,
        "base_tree": BASE_TREE,
        "parent_commit": PARENT_COMMIT,
        "parent_tree": PARENT_TREE,
    }
    for key, expected in expected_source.items():
        if source.get(key) != expected:
            raise base.ValidationError(f"source {key} mismatch")
    _require_git_oid(source.get("patched_commit"), "source.patched_commit")
    _require_git_oid(source.get("patched_tree"), "source.patched_tree")
    if source.get("durable_execution_authority") != (
        "base_commit_plus_patch_sha256_plus_patched_tree"
    ):
        raise base.ValidationError("durable execution authority mismatch")

    patch = manifest.get("patch")
    if not isinstance(patch, dict):
        raise base.ValidationError("patch must be an object")
    if (
        patch.get("path") != PATCH
        or patch.get("format") != "git_diff_binary_full_index"
        or patch.get("changed_file_count") != 18
    ):
        raise base.ValidationError("aggregate patch contract mismatch")
    _require_sha256(patch.get("sha256"), "patch.sha256")
    if not isinstance(patch.get("size_bytes"), int) or patch["size_bytes"] <= 0:
        raise base.ValidationError("patch.size_bytes must be positive")

    contract = manifest.get("deterministic_value_decode")
    if not isinstance(contract, dict):
        raise base.ValidationError(
            "deterministic_value_decode must be an object"
        )
    expected_contract = {
        "api": API,
        "bit_width": 4,
        "quantized_context": 4092,
        "tile_width": 128,
        "workspace_shape": [1, 32, 32, 128],
        "workspace_dtype": "float32",
        "workspace_bytes": 524288,
        "reduction_order": (
            "token_ascending_slot_ascending_then_tile_ascending"
        ),
        "caller_owned_output": True,
        "caller_owned_workspace": True,
        "current_pytorch_cuda_stream": True,
        "launch_error_checks": True,
        "floating_output_atomics": False,
        "dynamic_allocation": False,
        "host_synchronization": False,
        "tail_initialization_bit_widths": [4, 3, 2],
    }
    if contract != expected_contract:
        raise base.ValidationError(
            "deterministic Value-decode contract mismatch"
        )

    extension = manifest.get("extension")
    if extension != {
        "sha256": (
            "a79644923ba131e56abe95029e669346dbbb11fd210d2b9f8b2086819ffeaad1"
        ),
        "identity_scope": "post_link_stripped_extension",
        "post_link_command": ["/usr/bin/strip", "--strip-unneeded"],
        "reproducibility_reason": "remove_nvcc_tmpxft_process_id_symbols",
    }:
        raise base.ValidationError(
            "deterministic post-link extension identity mismatch"
        )

    _validate_preserved_authority(manifest)


def validate(source_root: Path) -> dict[str, object]:
    if not MANIFEST.is_file():
        raise base.ValidationError(
            "Decision 0027 manifest is absent: "
            "third_party/patches/kvquant/"
            "deterministic-long-context-manifest.json"
        )
    manifest = base._read_json(
        MANIFEST,
        "Decision 0027 deterministic long-context patch manifest",
    )
    _validate_manifest_contract(manifest)
    validation_record = manifest.get("validation")
    if not isinstance(validation_record, dict):
        raise base.ValidationError("validation must be an object")
    evidence_sha256 = validation_record.get("evidence_sha256")
    if evidence_sha256 is not None:
        _require_sha256(
            evidence_sha256,
            "validation.evidence_sha256",
        )
        if (
            not EVIDENCE.is_file()
            or base._sha256(EVIDENCE.read_bytes()) != evidence_sha256
        ):
            raise base.ValidationError(
                "Phase 11D validation evidence identity mismatch"
            )
        evidence = base._read_json(
            EVIDENCE,
            "Phase 11D CUDA validation evidence",
        )
        if (
            evidence.get("status") != "PASS"
            or evidence.get("source", {}).get("patched_commit")
            != manifest["source"]["patched_commit"]
            or evidence.get("source", {}).get("patched_tree")
            != manifest["source"]["patched_tree"]
            or evidence.get("source", {}).get("aggregate_patch_sha256")
            != manifest["patch"]["sha256"]
            or evidence.get("build", {}).get("extension_sha256")
            != manifest["extension"]["sha256"]
            or evidence.get("fixture_preservation", {}).get(
                "fixture_root_sha256"
            )
            != PRESERVED_IDENTITY_AUTHORITY["fixture_root_sha256"]
        ):
            raise base.ValidationError(
                "Phase 11D validation evidence authority mismatch"
            )
    decision, decision_status = _decision_0027(
        manifest,
        evidence_finalized=evidence_sha256 is not None,
    )

    base.MANIFEST_PATH = MANIFEST
    base.PATCH_RELATIVE_PATH = PATCH
    patch_path, patch_bytes = base._resolve_patch(manifest)
    aggregate_paths = base._validate_patch_structure(manifest, patch_bytes)
    reconstruction = base._validate_reconstruction(
        source_root,
        manifest,
        patch_path,
    )

    source_root = source_root.resolve(strict=True)
    head = base._run_git(source_root, "rev-parse", "HEAD")
    tree = base._run_git(source_root, "rev-parse", "HEAD^{tree}")
    if not isinstance(head, str) or not isinstance(tree, str):
        raise AssertionError("text Git command returned bytes")
    head = head.strip()
    tree = tree.strip()
    if (
        head != manifest["source"]["patched_commit"]
        or tree != manifest["source"]["patched_tree"]
    ):
        raise base.ValidationError(
            "local deterministic long-context source identity mismatch"
        )
    status = base._run_git(
        source_root,
        "status",
        "--porcelain",
        "--untracked-files=all",
    )
    if not isinstance(status, str) or status:
        raise base.ValidationError(
            "deterministic long-context source checkout is not clean"
        )

    parent_paths = _validate_parent_delta(source_root, manifest, head)
    _validate_new_source_contract(source_root)

    return {
        "status": "PASS",
        "decision": decision.relative_to(ROOT).as_posix(),
        "decision_status": decision_status,
        "patched_commit": head,
        "patched_tree": tree,
        "aggregate_patch_sha256": base._sha256(patch_bytes),
        "aggregate_changed_paths": aggregate_paths,
        "parent_commit": PARENT_COMMIT,
        "parent_tree": PARENT_TREE,
        "parent_relative_changed_paths": parent_paths,
        "source_contract": "PASS",
        "reconstruction": reconstruction,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        report = validate(arguments.source_root)
    except (OSError, ValueError, KeyError, base.ValidationError) as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, sort_keys=True))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
