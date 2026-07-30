#!/usr/bin/env python3
"""Validate the Decision 0029 deterministic q3/q2 KVQuant source patch."""

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
    "deterministic-long-context-q3-q2-manifest.json"
)
PATCH = (
    "third_party/patches/kvquant/"
    "0004-deterministic-long-context-q3-q2-value-decode.patch"
)
EVIDENCE = ROOT / "docs/evidence/phase11dq23/cuda-validation.json"
BASE_COMMIT = "57a238357f0ffe50084670fcd5781c9848f80ea2"
BASE_TREE = "094e0f736f77ee327e5350cbd1eefb1c936aa77b"
PARENT_COMMIT = "4b8533b29b04f8c4bf55f688a41fefe20487637b"
PARENT_TREE = "46f2149a0369d5c97d9a6bc77d57b5f3a5a5fb3b"
SOURCE_REPOSITORY = "https://github.com/SqueezeAILab/KVQuant.git"
CPP_PATH = "deployment/kvquant/quant_cuda.cpp"
CUDA_PATH = "deployment/kvquant/quant_cuda_kernel.cu"
PARENT_CHANGED_PATHS = [CPP_PATH, CUDA_PATH]
SCHEMA_VERSION = (
    "kvbench-kvquant-q23-long-context-source-patch-manifest-1.0.0"
)
APIS = {
    "kvq3": (
        "vecquant3matmul_nuq_perchannel_transposed_"
        "mha_batched_fused_opt2_deterministic_out"
    ),
    "kvq2": (
        "vecquant2matmul_nuq_perchannel_transposed_"
        "mha_batched_fused_opt2_deterministic_out"
    ),
}

PRESERVED_FILES = {
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
    "decision_0027_sha256": (
        ROOT
        / "docs/decisions/"
        "0027-kvquant-deterministic-long-context-value-decode.md",
        "6c12e93223524cbfe15976459dd13d8ed774b1c2b1f01ed2bdfa3273df3b1693",
    ),
    "decision_0028_sha256": (
        ROOT
        / "docs/decisions/"
        "0028-phase12e-kivi-historical-source-validation.md",
        "23cbcadb0dc357e3ed719f049789d85b752a9ad7bf8df68c798dcee9711d7194",
    ),
    "phase11d_report_sha256": (
        ROOT
        / "docs/phase_reports/"
        "phase11d-kvquant-deterministic-long-context-cuda.md",
        "0e9b29d80466efaea955333337b01fdc18969ae543b14b622c39eee022da82e8",
    ),
    "phase11d_evidence_sha256": (
        ROOT / "docs/evidence/phase11d/cuda-validation.json",
        "18d70be72e3172163ca6cbde46dba6c46e96fd7b9ea22884faac313df1c1eda9",
    ),
    "phase11r_report_sha256": (
        ROOT / "docs/phase_reports/phase11r-kvquant-measurement-adapter.md",
        "bf24d235cb3f1fa80fbe53ae40690c3e4b0451a3b6038eb50b7b60cde19c8b64",
    ),
    "phase11_method_admission_sha256": (
        ROOT / "docs/evidence/phase11/kvquant-method-admission.json",
        "59ef5bfc581a68cdc4d21c4c0a840f046e698633f7475f79906063c6e333ae6a",
    ),
    "phase11_blocked_report_sha256": (
        ROOT / "docs/phase_reports/phase11-kvquant-measurement-adapter.md",
        "2f291fbcdd639c8854d9543fbacc458993da699d3fb6b4fee2c620981f4c5b88",
    ),
    "decision_0025_patch_sha256": (
        ROOT
        / "third_party/patches/kvquant/"
        "0002-graphsafe-kvq3-deterministic.patch",
        "23a15db86790299392412c3ce2da7d971f4f073cfaf6839d82d3746c8b56b551",
    ),
    "decision_0025_manifest_sha256": (
        ROOT
        / "third_party/patches/kvquant/"
        "graphsafe-kvq3-manifest.json",
        "d04a580a4a4cd0fcaa6f0880eadb46f9479ff2f7c6a250eb9568f9f520e2da6e",
    ),
    "decision_0027_patch_sha256": (
        ROOT
        / "third_party/patches/kvquant/"
        "0003-deterministic-long-context-value-decode.patch",
        "bae63bced549479709b10d7f6a8ee35a8f21ec18cc040a7424591cee47c1b0a6",
    ),
    "decision_0027_manifest_sha256": (
        ROOT
        / "third_party/patches/kvquant/"
        "deterministic-long-context-manifest.json",
        "8eea6ac465da795f44724c905e680df67965f55b35ff522d71e492ca6dccb667",
    ),
}
PRESERVED_IDENTITIES = {
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
    "parent_adapter_sha256": (
        "7fef900247f7575437d03a3ee528567da5518fc46305b268710d46ef7b9c7101"
    ),
    "parent_cache_sha256": (
        "c5a1184bbcbe49535bdba0173c16d2e3ce6c6ac8a78d652d713b5b73206b4b8f"
    ),
    "parent_session_sha256": (
        "6884f57336c57ecd3e3325b0b2f582efaa3a7c600350a1a107990d4d60ffe25c"
    ),
}


def _git_bytes(source_root: Path, revision: str, path: str) -> bytes:
    result = base._run_git(
        source_root,
        "show",
        f"{revision}:{path}",
        binary=True,
    )
    if not isinstance(result, bytes):
        raise AssertionError("binary Git command returned text")
    return result


def _git_text(source_root: Path, *arguments: str) -> str:
    result = base._run_git(source_root, *arguments)
    if not isinstance(result, str):
        raise AssertionError("text Git command returned bytes")
    return result.strip()


def _require_sha256(value: object, description: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise base.ValidationError(f"{description} must be a lowercase SHA-256")
    return value


def _decision(
    manifest: dict[str, Any],
    *,
    evidence_finalized: bool,
) -> tuple[Path, str]:
    matches = sorted((ROOT / "docs/decisions").glob("0029-*.md"))
    if len(matches) != 1:
        raise base.ValidationError("exactly one Decision 0029 is required")
    text = matches[0].read_text(encoding="utf-8")
    for marker in (
        "# Decision 0029",
        "caller-owned",
        "current PyTorch CUDA stream",
        "Quantization mathematics",
        "Phase 12 execution is explicitly deferred",
        manifest["source"]["parent_commit"],
        manifest["source"]["parent_tree"],
        manifest["source"]["patched_commit"],
        manifest["source"]["patched_tree"],
        manifest["patch"]["sha256"],
        manifest["parent_delta"]["sha256"],
    ):
        if marker not in text:
            raise base.ValidationError(
                f"Decision 0029 contract marker missing: {marker}"
            )
    status_match = re.search(
        r"^- Status: (Proposed|Accepted)$",
        text,
        flags=re.MULTILINE,
    )
    if status_match is None:
        raise base.ValidationError("Decision 0029 status is invalid")
    status = status_match.group(1)
    if evidence_finalized and status != "Accepted":
        raise base.ValidationError(
            "finalized Decision 0029 evidence requires Accepted status"
        )
    return matches[0], status


def _validate_preservation(manifest: dict[str, Any]) -> None:
    observed = manifest.get("preserved_authority")
    if not isinstance(observed, dict):
        raise base.ValidationError("preserved_authority must be an object")
    expected = dict(PRESERVED_IDENTITIES)
    for name, (path, digest) in PRESERVED_FILES.items():
        if not path.is_file() or base._sha256(path.read_bytes()) != digest:
            raise base.ValidationError(f"preserved authority changed: {name}")
        expected[name] = digest
    if observed != expected:
        raise base.ValidationError("preserved authority manifest differs")


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise base.ValidationError("unsupported Decision 0029 manifest schema")
    authority = manifest.get("authority")
    if not isinstance(authority, dict) or authority.get(
        "method_identifier"
    ) != "kvquant_gqa_longctx_deterministic_q23_v4":
        raise base.ValidationError("Decision 0029 authority differs")
    if authority.get("decisions") != ["0021", "0024", "0025", "0027", "0029"]:
        raise base.ValidationError("Decision 0029 decision chain differs")
    if authority.get("official_gqa_support_claimed") is not False:
        raise base.ValidationError("official GQA support must not be claimed")

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
    for name, expected in expected_source.items():
        if source.get(name) != expected:
            raise base.ValidationError(f"source {name} differs")

    patch = manifest.get("patch")
    if (
        not isinstance(patch, dict)
        or patch.get("path") != PATCH
        or patch.get("format") != "git_diff_binary_full_index"
        or patch.get("changed_file_count") != 18
    ):
        raise base.ValidationError("aggregate patch contract differs")
    _require_sha256(patch.get("sha256"), "patch.sha256")

    contract = manifest.get("deterministic_value_decode")
    if not isinstance(contract, dict) or contract.get("apis") != APIS:
        raise base.ValidationError("q3/q2 deterministic API contract differs")
    expected_contract = {
        "bit_widths": [3, 2],
        "quantized_context": 4092,
        "tile_width": 128,
        "workspace_shape": [1, 32, 31, 128],
        "workspace_dtype": "float32",
        "workspace_bytes": 507904,
        "workspace_additional_persistent_bytes": 0,
        "workspace_alias": "decode_logits",
        "last_tile_computed_by_reduce_kernel": True,
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
        "strided_device_scores": True,
        "q4_path_changed": False,
        "legacy_reference_apis_changed": False,
    }
    for name, expected in expected_contract.items():
        if contract.get(name) != expected:
            raise base.ValidationError(
                f"deterministic Value contract differs: {name}"
            )
    if set(contract) != {"apis", *expected_contract}:
        raise base.ValidationError("deterministic Value contract has extra fields")

    extension = manifest.get("extension")
    if not isinstance(extension, dict):
        raise base.ValidationError("extension must be an object")
    _require_sha256(extension.get("sha256"), "extension.sha256")
    if extension.get("identity_scope") != "post_link_stripped_extension":
        raise base.ValidationError("extension identity scope differs")
    _validate_preservation(manifest)


def _validate_parent_delta(
    source_root: Path,
    manifest: dict[str, Any],
    head: str,
) -> list[str]:
    parent_line = _git_text(
        source_root,
        "rev-list",
        "--parents",
        "-n",
        "1",
        head,
    )
    if parent_line.split() != [head, PARENT_COMMIT]:
        raise base.ValidationError(
            "Decision 0029 source must be one single-parent commit above "
            "Decision 0027"
        )
    if _git_text(source_root, "rev-parse", f"{PARENT_COMMIT}^{{tree}}") != (
        PARENT_TREE
    ):
        raise base.ValidationError("Decision 0027 parent tree differs")
    changed = _git_text(
        source_root,
        "diff",
        "--name-only",
        PARENT_COMMIT,
        head,
    ).splitlines()
    if changed != PARENT_CHANGED_PATHS:
        raise base.ValidationError("Decision 0029 changed-file scope differs")
    numstat = _git_text(
        source_root,
        "diff",
        "--numstat",
        PARENT_COMMIT,
        head,
    ).splitlines()
    if numstat != [
        f"38\t0\t{CPP_PATH}",
        f"344\t0\t{CUDA_PATH}",
    ]:
        raise base.ValidationError(
            "Decision 0029 must be an addition-only two-file source delta"
        )

    delta_bytes = base._run_git(
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
    if not isinstance(delta_bytes, bytes):
        raise AssertionError("binary Git command returned text")
    delta = manifest.get("parent_delta")
    if (
        not isinstance(delta, dict)
        or delta.get("sha256") != base._sha256(delta_bytes)
        or delta.get("size_bytes") != len(delta_bytes)
        or delta.get("changed_file_count") != 2
    ):
        raise base.ValidationError("parent-relative patch identity differs")
    records = delta.get("changed_files")
    if not isinstance(records, list) or [
        item.get("path") for item in records
    ] != PARENT_CHANGED_PATHS:
        raise base.ValidationError("parent-relative file records differ")
    for record in records:
        path = record["path"]
        expected = {
            "path": path,
            "before_git_blob": _git_text(
                source_root,
                "rev-parse",
                f"{PARENT_COMMIT}:{path}",
            ),
            "after_git_blob": _git_text(
                source_root,
                "rev-parse",
                f"{head}:{path}",
            ),
            "before_sha256": base._sha256(
                _git_bytes(source_root, PARENT_COMMIT, path)
            ),
            "after_sha256": base._sha256(
                _git_bytes(source_root, head, path)
            ),
        }
        if record != expected:
            raise base.ValidationError(
                f"parent-relative file identity differs: {path}"
            )
    return changed


def _validate_source_contract(source_root: Path, head: str) -> None:
    cpp = _git_bytes(source_root, head, CPP_PATH).decode("utf-8")
    cuda = _git_bytes(source_root, head, CUDA_PATH).decode("utf-8")
    for api in APIS.values():
        if api not in cpp or f"{api}_cuda" not in cuda:
            raise base.ValidationError(f"q3/q2 API binding absent: {api}")
    start = cuda.find(
        "template <int BITS, int LEVELS, int PACKED_ROWS>\n"
        "__device__ __forceinline__ float VecQuantQ23DeterministicTilePartial"
    )
    stop = cuda.find("// OPTIMIZED FUSED K KERNEL", start)
    if start < 0 or stop < 0:
        raise base.ValidationError("q3/q2 deterministic kernel region absent")
    region = cuda[start:stop]
    required = (
        "VecQuantQ23DeterministicTilePartial",
        "VecQuantQ23MatMulKernelNUQPerChannelTransposedMHABatchedFusedOptDeterministicTiles",
        "VecQuantQ23MatMulKernelNUQPerChannelTransposedMHABatchedFusedOptDeterministicReduce",
        "at::cuda::getCurrentCUDAStream()",
        "C10_CUDA_KERNEL_LAUNCH_CHECK()",
        "code_shift + BITS > 32",
        "for (int token = token_start; token < token_stop; ++token)",
        "for (int slot = 0; slot < num_outliers; ++slot)",
        "for (int tile = 0; tile < stored_tiles; ++tile)",
    )
    for marker in required:
        if marker not in region:
            raise base.ValidationError(
                f"q3/q2 deterministic source marker absent: {marker}"
            )
    if re.search(r"\batomic[A-Za-z_]*\b", region):
        raise base.ValidationError(
            "floating atomics are forbidden in the q3/q2 deterministic path"
        )


def validate(source_root: Path) -> dict[str, object]:
    manifest = base._read_json(MANIFEST, "Decision 0029 patch manifest")
    _validate_manifest(manifest)
    validation = manifest.get("validation")
    if not isinstance(validation, dict):
        raise base.ValidationError("validation must be an object")
    evidence_sha256 = validation.get("evidence_sha256")
    if evidence_sha256 is not None:
        _require_sha256(evidence_sha256, "validation.evidence_sha256")
        if (
            not EVIDENCE.is_file()
            or base._sha256(EVIDENCE.read_bytes()) != evidence_sha256
        ):
            raise base.ValidationError(
                "Phase 11D-Q23 evidence identity differs"
            )
    decision, decision_status = _decision(
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

    resolved = source_root.resolve(strict=True)
    head = _git_text(resolved, "rev-parse", "HEAD")
    tree = _git_text(resolved, "rev-parse", "HEAD^{tree}")
    if (
        head != manifest["source"]["patched_commit"]
        or tree != manifest["source"]["patched_tree"]
    ):
        raise base.ValidationError("local Decision 0029 source identity differs")
    if _git_text(
        resolved,
        "status",
        "--porcelain",
        "--untracked-files=all",
    ):
        raise base.ValidationError("Decision 0029 source checkout is not clean")
    parent_paths = _validate_parent_delta(resolved, manifest, head)
    _validate_source_contract(resolved, head)
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
