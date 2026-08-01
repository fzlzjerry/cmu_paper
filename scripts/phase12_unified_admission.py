"""Narrow Phase 12 G1-G5 aggregation and common G5 coordination.

The module deliberately owns no method implementation.  It validates the four
frozen method-admission authorities, constructs the one preregistered fixed-L
integration point from the existing factory and cache classes, delegates timing
to the unchanged common runner, and manages one append-only 30-process
campaign.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import dataclasses
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import secrets
import stat
import statistics
import subprocess
import sys
import time
from typing import Any

from preflight.run_preflight import (
    configured_secret_value_names,
    json_bytes,
    rename_noreplace,
    write_exclusive,
)
from kvbench.runtime.artifacts import sha256_file
from kvbench.runtime.method_harness import execution_path_audit_facade
from kvbench.runtime.process_supervision import run_supervised_command
from kvbench.runtime.turboquant_session import (
    EndpointSessionError,
    TurboQuantEndpointSession,
)
from kvbench.schema import canonical_json_bytes, GraphMode, RunnerKind, sha256_hex
from kvbench.schema.base import QualityExecutionState
from kvbench.schema.method_admission import (
    MethodAdmissionReport,
    MethodAdmissionReportV2,
)
from kvbench.schema.phase3 import GateDisposition
from kvbench.schema.phase8 import Phase8MethodAdmissionReport
from kvbench.schema.phase11 import Phase11RQ23MethodAdmissionReport
from kvbench.schema.phase13b import Phase13BMethodAdmissionReport
from kvbench.schema.phase12 import (
    PHASE12_AUTHORIZED_CONTAINER_DIGEST,
    PHASE12_BATCH_SIZE,
    PHASE12_CONFIG_FINGERPRINTS,
    PHASE12_CONTEXT_LENGTH,
    PHASE12_CV_THRESHOLD,
    PHASE12_GRAPH_MODE,
    PHASE12_HELD_OUT_CONFIGURATIONS,
    PHASE12_MAIN_CONFIGURATIONS,
    PHASE12_MEASURED_BATCHES,
    PHASE12_MEASURED_STEPS,
    PHASE12_MODEL_ID,
    PHASE12_MODEL_REVISION,
    PHASE12_RANDOMIZATION_SEEDS,
    PHASE12_RANDOMIZED_ORDERS,
    PHASE12_REPLICATES,
    PHASE12_RUNNER_KIND,
    PHASE12_TOKENIZER_ID,
    PHASE12_TOKENIZER_REVISION,
    PHASE12_WARMUP_STEPS,
    Phase12ByteAccounting,
    Phase12ConfigurationAdmission,
    Phase12EvidenceReference,
    Phase12ExcludedConfiguration,
    Phase12G5Disposition,
    Phase12G5Run,
    Phase12G5Statistics,
    Phase12GlobalGates,
    Phase12PriorGateEvidence,
    Phase12PublicationState,
    Phase12RandomizedOrder,
    Phase12UnifiedAdmissionReport,
    derive_phase12_randomized_order,
)
from scripts.r2_artifact import ArtifactValidationError, validate_local_artifact


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PHASE12_ARTIFACT_ROOT = REPOSITORY_ROOT / "artifacts" / "phase12"
PHASE12_PLAN_PATH = Path("docs/plans/phase12-unified-admission.md")
PHASE12_PLAN_SHA256 = (
    "57c4f56fdea64463b7b95484a41a94b7c6644d4c2d502d8c871e915020adf500"
)
PHASE12_R2_CREDENTIAL_NAMES = frozenset(
    {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "CLOUDFLARE_API_TOKEN",
        "KVBENCH_R2_PREFIX",
        "R2_ACCOUNT_ID",
        "R2_BUCKET",
        "R2_ENDPOINT",
    }
)
PHASE12_GPU_UUID = "GPU-75bd273e-6b20-0d22-1b0b-5fbb6fb0025b"
PHASE12_CAMPAIGN_SCHEMA = "kvbench-phase12-campaign-bundle-1.0.0"
PHASE12_LOCAL_REPORT_SCHEMA = (
    Phase12UnifiedAdmissionReport.SCHEMA_VERSION
)
PHASE12_RUN_SCHEMA = "kvbench-phase12-g5-process-run-1.0.0"
PHASE12_PER_CONFIG_SCHEMA = "kvbench-phase12-configuration-admission-1.0.0"
PHASE12_WORKER_RESULT_PREFIX = "PHASE12_WORKER_RESULT="
PHASE12_CHILD_TIMEOUT_SECONDS = 7_200.0
PHASE12_TEST_TIMEOUT_SECONDS = 7_200.0
PHASE12_SETUP_WARMUPS = {
    "bf16": 16,
    "turboquant": 3,
    "kivi": 3,
    "kvquant": 3,
}
PHASE12_INPUT_SCHEMA = "kvbench-phase12-common-input-1.0.0"
PHASE12_INPUT_RECIPE = {
    "schema_version": PHASE12_INPUT_SCHEMA,
    "prefix": "(arange(4096)+12000)%120000+1000",
    "decode": "(12000+4096+257)%120000+1000",
}
PHASE12_INPUT_RECIPE_SHA256 = sha256_hex(
    canonical_json_bytes(PHASE12_INPUT_RECIPE)
)
_CAMPAIGN_ID_RE = re.compile(
    r"phase12-[0-9]{8}t[0-9]{12}z-[0-9a-f]{8}-[0-9a-f]{6}\Z"
)
_RUN_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
_CUDA_GRAPH_POINTER_RE = re.compile(r"0x[0-9A-Fa-f]+")
_CUDA_GRAPH_ID_RE = re.compile(r"graph_[0-9]+")
_CUDA_GRAPH_CLUSTER_RE = re.compile(r"cluster_[0-9]+")
_FORBIDDEN_CHILD_ENVIRONMENT = frozenset(
    {
        "R2_ACCOUNT_ID",
        "R2_BUCKET",
        "R2_ENDPOINT",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "CLOUDFLARE_API_TOKEN",
        "KVBENCH_R2_PREFIX",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
    }
)
_CONTROL_FILES = frozenset(
    {"manifest.json", "artifact_inventory.json", "checksums.sha256", "COMPLETE"}
)
_INVENTORY_EXCLUSIONS = frozenset(
    {"artifact_inventory.json", "checksums.sha256", "COMPLETE"}
)

MAIN_CONFIG_IDS = PHASE12_MAIN_CONFIGURATIONS
HELD_OUT_CONFIG_IDS = PHASE12_HELD_OUT_CONFIGURATIONS
EXPECTED_CONFIG_FINGERPRINTS = dict(PHASE12_CONFIG_FINGERPRINTS)
EXPECTED_REPORT_SHA256S = {
    "bf16": "1362fd1817b8bb5706baaa09ed6e5115789fbc4d35d394f184d0b132a0e58d22",
    "turboquant": (
        "388e8107b649a9093491699357c8b1ad1d8e12c8c75378bce658f8a09bf9ab2a"
    ),
    "kivi": "3a4b63b9da0eab12db9a916ebdc1cffd788ea6f93678d87964a8332ae7cec83a",
    "kvquant": (
        "9cfed618cee9514a1071392d0a2dca327dcf6acd33d81ac72cc477c7880c09e2"
    ),
}
PRIOR_ADMISSION_REPORT_BINDINGS = {
    "bf16": Path("docs/evidence/phase4/method-admission.json"),
    "turboquant": Path(
        "docs/evidence/phase6/turboquant-method-admission.json"
    ),
    "kivi": Path("docs/evidence/phase8/kivi-method-admission.json"),
    "kvquant": Path(
        "docs/evidence/phase11rq23/kvquant-method-admission.json"
    ),
}
PHASE12_KVQUANT_AUTHORITY = {
    "execution_source_identifier": "kvquant_gqa_longctx_deterministic_q23_v4",
    "source_commit": "34b0bdfa83082e1f30387d9ac5cca369006e089c",
    "source_tree": "1f85af65fe03061583ffe8bd91e47d7ecffdd312",
    "aggregate_patch_sha256": (
        "7b9d3cc6773e8ef37697601c885f2c5ec581dffd57cf59424d03e68f147bd55a"
    ),
    "extension_sha256": (
        "b3c33badb8e55b19d6b2ce535182e964ce51e5102d8413b29701dd3d817ad73d"
    ),
    "q23_evidence_root": (
        "8b65112ea2d49b58ee07c1533b429fac1a8af7466e09adad073d9a22ae2ec790"
    ),
    "fixture_root": (
        "c28682d58706b58812dc1db69ba5eb4982339ba13f39bf67f751794cdaabfdec"
    ),
    "decision": "0029",
}
PHASE12_DECISION0026_PATH = Path(
    "docs/decisions/0026-kvquant-pre-rope-adapter-boundary.md"
)
PHASE12_DECISION0026_SHA256 = (
    "eee3fb412111b658ecedaecddae2844161d08fc04bb2c88e158bd808f6bfe6f2"
)
PHASE12_DECISION0026_COMMIT = (
    "781b416748e2bddca8ea5c23cd0f51a63a066276"
)
PHASE12_HISTORICAL_ENDPOINT_BLOB = (
    "853241dd77a6fb7f70cc47894a91c525d7c5f5fe"
)
PHASE12_HISTORICAL_ENDPOINT_SHA256 = (
    "8aa48ec285fb9c7853bc19ae10bd8afc07a04d1d6f522f53e67e705a424a27b9"
)
PHASE12_CURRENT_ENDPOINT_BLOB = (
    "e6967f695540a6a822ffb288f0d9a1f07e905641"
)
PHASE12_CURRENT_ENDPOINT_SHA256 = (
    "9095e9a2a9c01e1ea6afb2f1cefcee46a964a82caae7b819a125757b59244a9b"
)
PHASE12_BF16_PARITY_ARTIFACTS = {
    "eager": {
        "path": Path(
            "artifacts/phase6a/bf16_parity/"
            "phase6a-bf16-parity-eager-"
            "20260724t195409003183z-a6025ae0-d47cea"
        ),
        "root_sha256": (
            "15317bd757ff573483dcac54e4d782134575c40a0b15e60b6d7e01694ca7e9b8"
        ),
        "manifest_sha256": (
            "135bb2aea1a55b555115efb3081b34feab098a1581167d513723d9e2e7c4f6e7"
        ),
        "result_sha256": (
            "a79c5b099f9ad1e7f94cdef95de1c3740047e470bc1f99b6bc3c0e2284e1ab90"
        ),
    },
    "cuda_graph": {
        "path": Path(
            "artifacts/phase6a/bf16_parity/"
            "phase6a-bf16-parity-cuda_graph-"
            "20260724t195438755178z-a6025ae0-71ec78"
        ),
        "root_sha256": (
            "8bd79132354f4b751fde63f07997c5089dd4a31d6eb0ee5c12dee70f4d07ca8b"
        ),
        "manifest_sha256": (
            "0dec3709ad98f48b7535b0425c892aff69183ba8368945b5520b6bd1a072ae43"
        ),
        "result_sha256": (
            "6f92a14002364f6f9ffc2654dfd970c3a2f9db8514cf0932fa660bf3944a543f"
        ),
    },
}
PHASE12_BF16_PARITY_EXECUTION_COMMIT = (
    "a6025ae023e152db0a0813ea8923ffe0fcef3d44"
)
PHASE12_BF16_ADAPTER_SHA256 = (
    "828fc686c1f111ac0168d9c567d5fbf035ba00e770b046c0ed3c1085bf2f8109"
)
PHASE12_BF16_ADAPTER_BLOB = (
    "a8f2a45e0a937bf43fbe39a458a583f28711a783"
)
PHASE12_TURBOQUANT_EXECUTION_COMMIT = (
    "0df5bb4d445d48e6cba17e30723733f8de35cb14"
)
PHASE12_TURBOQUANT_ADAPTER_SHA256 = (
    "53b4d0586750139c62f3a4b6c73a25e3424902af2f49ca297eaced0809adeed7"
)
PHASE12_TURBOQUANT_ADAPTER_BLOB = (
    "085734a9fcd58408fdf48782f6103738c66a8336"
)
PHASE12_TURBOQUANT_SESSION_SHA256 = (
    "959195ae5bd1f9ea2d302e2824bfd8f33459604a5c9c4d37771aa61fd0a86f1b"
)
PHASE12_TURBOQUANT_SESSION_BLOB = (
    "c24b55ce1d91d52890ec0efb7de5507fe9f5c032"
)
PHASE13B_DECISION0030_PATH = Path(
    "docs/decisions/0030-compressed-static-cache-batch-geometry.md"
)
PHASE13B_DECISION0030_SHA256 = (
    "84c2eb943b35afba312eaf599f8ec8f1d4a82169daa2d5c5fc5d127f0a965e62"
)
PHASE13B_DECISION0030_COMMIT = (
    "2af47459e109207ac21167ddd88e8ca79d815490"
)
PHASE13B_SOURCE_AUTHORITY_COMMIT = (
    "b862af64346a0dba2650b2c213ebd1d3b5b99ef2"
)
PHASE13B_TURBOQUANT_REPORT_PATH = Path(
    "docs/evidence/phase13b/turboquant-method-admission.json"
)
PHASE13B_TURBOQUANT_REPORT_SHA256 = (
    "49799ef89646ec008a530c5180fdcef6cd4af9ca0d5772fe2b01d6e775e3b1c0"
)
PHASE13B_TURBOQUANT_SOURCE_AUTHORITY = {
    "src/kvbench/adapters/turboquant.py": {
        "historical_blob": "085734a9fcd58408fdf48782f6103738c66a8336",
        "historical_sha256": PHASE12_TURBOQUANT_ADAPTER_SHA256,
        "authority_blob": "495aea48aca540d37ca5cd1bc1fb0889d542d235",
        "authority_sha256": (
            "b9911379ee0cc79b68691a55e0e5cecd2bfcfa8bbfe35e0120bdbe394f9b7432"
        ),
    },
    "src/kvbench/runtime/turboquant_cache.py": {
        "historical_blob": "5376ffc9b84f925d8b8b7e51db85f0f39fec3754",
        "historical_sha256": (
            "2a94de9ea7baf233c90413abb6e40e5083414f3312da4c19312c9556a98e2274"
        ),
        "authority_blob": "41126285ca3ab22611c242556012f4b7824112f8",
        "authority_sha256": (
            "92551d9daf9c0af2b830655c414c137a52ef5e6dcaf26506efef31200e037b15"
        ),
    },
}
GATE_EVIDENCE_REQUIREMENTS = {
    "bf16": {
        "G1": ("correctness", "execution_path"),
        "G2": ("byte_accounting",),
        "G3": ("execution_path",),
        "G4": ("graph",),
    },
    "turboquant": {
        "G1": (
            "fixture_conformance",
            "store_append_correctness",
            "decode_tolerance",
            "finite_output",
            "no_backend_fallback",
        ),
        "G2": (
            "byte_accounting",
            "static_cache_skip_policy",
            "no_cache_growth",
            "no_unknown_allocation",
        ),
        "G3": (
            "no_full_prefix_dequantization",
            "no_gqa_replication",
            "no_cache_growth",
            "no_unknown_allocation",
            "no_backend_fallback",
        ),
        "G4": (
            "graph_capture_replay",
            "graph_zero_replay_allocation",
            "no_backend_fallback",
        ),
    },
    "kivi": {
        "G1": (
            "fixture_conformance",
            "token_integrity",
            "no_backend_fallback",
        ),
        "G2": (
            "byte_accounting",
            "residual_rollover",
            "static_cache",
            "no_unknown_allocation",
        ),
        "G3": (
            "no_measured_torch_cat",
            "direct_compressed_decode",
            "native_gqa",
            "no_unknown_allocation",
            "no_backend_fallback",
        ),
        "G4": (
            "graph_capture_replay",
            "graph_zero_replay_allocation",
            "no_backend_fallback",
        ),
    },
    "kvquant": {
        "G1": (
            "fixture_conformance",
            "sparse_contract",
            "sink_storage",
            "store_append_correctness",
            "execution_path",
        ),
        "G2": (
            "byte_accounting",
            "no_dynamic_or_unknown_allocation",
        ),
        "G3": (
            "direct_compressed_decode",
            "native_gqa",
            "execution_path",
            "no_dynamic_or_unknown_allocation",
            "no_host_synchronization",
        ),
        "G4": (
            "graph_capture_replay",
            "graph_zero_replay_allocation",
        ),
    },
}

_REPORT_PARSERS = {
    "bf16": MethodAdmissionReport,
    "turboquant": MethodAdmissionReportV2,
    "kivi": Phase8MethodAdmissionReport,
    "kvquant": Phase11RQ23MethodAdmissionReport,
}
_CONFIG_METHOD = {
    "bf16": "bf16",
    "tq_4bit_nc": "turboquant",
    "tq_k3v4_nc": "turboquant",
    "tq_3bit_nc": "turboquant",
    "k4v4": "kivi",
    "k2v4": "kivi",
    "k2v2": "kivi",
    "kvq4": "kvquant",
    "kvq3": "kvquant",
    "kvq2": "kvquant",
}
_TURBOQUANT_REPORT_TO_PHASE12 = {
    "turboquant_4bit_nc": "tq_4bit_nc",
    "turboquant_k3v4_nc": "tq_k3v4_nc",
    "turboquant_3bit_nc": "tq_3bit_nc",
}
_SHA256_CHARACTERS = frozenset("0123456789abcdef")


class Phase12UnifiedAdmissionError(RuntimeError):
    """Phase 12 evidence or execution differs from the frozen contract."""


def _expected_container_runtime_attestation() -> dict[str, Any]:
    return {
        "schema_version": (
            "kvbench-phase12-container-runtime-attestation-1.0.0"
        ),
        "docker_marker_present": True,
        "repository_mount_read_only": True,
        "authorized_container_digest": (
            PHASE12_AUTHORIZED_CONTAINER_DIGEST
        ),
        "execution_environment": "measurement_container",
    }


def _require_authorized_container_runtime() -> dict[str, Any]:
    try:
        repository_flags = os.statvfs(REPOSITORY_ROOT).f_flag
    except OSError as error:
        raise Phase12UnifiedAdmissionError(
            "authorized container repository mount is unavailable"
        ) from error
    observed = {
        "schema_version": (
            "kvbench-phase12-container-runtime-attestation-1.0.0"
        ),
        "docker_marker_present": Path("/.dockerenv").is_file(),
        "repository_mount_read_only": bool(
            repository_flags & os.ST_RDONLY
        ),
        "authorized_container_digest": os.environ.get(
            "KVBENCH_AUTHORIZED_IMAGE_DIGEST"
        ),
        "execution_environment": os.environ.get(
            "KVBENCH_EXECUTION_ENVIRONMENT"
        ),
    }
    if observed != _expected_container_runtime_attestation():
        raise Phase12UnifiedAdmissionError(
            "authorized container runtime attestation differs"
        )
    return observed


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_json,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase12UnifiedAdmissionError(
            f"cannot read strict JSON: {path}"
        ) from error
    if not isinstance(value, dict):
        raise Phase12UnifiedAdmissionError(f"JSON root is not an object: {path}")
    return value


def _reject_duplicate_pairs(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _require_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise Phase12UnifiedAdmissionError(f"{label} is not a SHA-256")
    return value


def _resolve_evidence_path(repo_root: Path, relative: str) -> Path:
    candidate = repo_root / relative
    try:
        resolved_root = repo_root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (OSError, ValueError) as error:
        raise Phase12UnifiedAdmissionError(
            f"referenced evidence is missing or escapes the repository: {relative}"
        ) from error
    if not resolved.is_file() or resolved.is_symlink():
        raise Phase12UnifiedAdmissionError(
            f"referenced evidence is missing or unsafe: {relative}"
        )
    return resolved


def _git_bytes(root: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ("/usr/bin/git", *arguments),
        cwd=root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0 or result.stderr:
        raise Phase12UnifiedAdmissionError(
            f"Git authority lookup failed: {' '.join(arguments)}"
        )
    return result.stdout


def _git_blob_authority(
    root: Path,
    *,
    commit: str,
    path: str,
) -> dict[str, str]:
    blob = _git_bytes(root, "rev-parse", f"{commit}:{path}").decode().strip()
    raw = _git_bytes(root, "cat-file", "blob", f"{commit}:{path}")
    if re.fullmatch(r"[0-9a-f]{40}", blob) is None:
        raise Phase12UnifiedAdmissionError("Git blob identity is invalid")
    return {
        "commit": commit,
        "path": path,
        "blob": blob,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _require_ancestor(root: Path, ancestor: str, descendant: str) -> None:
    result = subprocess.run(
        (
            "/usr/bin/git",
            "merge-base",
            "--is-ancestor",
            ancestor,
            descendant,
        ),
        cwd=root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0 or result.stdout or result.stderr:
        raise Phase12UnifiedAdmissionError(
            "historical execution/source transition is not ancestral"
        )


def _path_transition_commits(
    root: Path,
    *,
    start: str,
    end: str,
    path: str,
) -> tuple[str, ...]:
    output = _git_bytes(
        root,
        "log",
        "--format=%H",
        f"{start}..{end}",
        "--",
        path,
    ).decode()
    return tuple(line for line in output.splitlines() if line)


def _validate_phase13b_turboquant_successor_transition(
    root: Path,
    *,
    execution_commit: str,
) -> dict[str, Any]:
    """Recognize only Decision 0030's checksum-bound static-batch successor."""

    head = _git_bytes(root, "rev-parse", "HEAD").decode().strip()
    decision = _resolve_evidence_path(
        root, PHASE13B_DECISION0030_PATH.as_posix()
    )
    report_path = _resolve_evidence_path(
        root, PHASE13B_TURBOQUANT_REPORT_PATH.as_posix()
    )
    if sha256_file(decision) != PHASE13B_DECISION0030_SHA256:
        raise Phase12UnifiedAdmissionError("Decision 0030 checksum differs")
    if sha256_file(report_path) != PHASE13B_TURBOQUANT_REPORT_SHA256:
        raise Phase12UnifiedAdmissionError(
            "Phase 13B TurboQuant successor report checksum differs"
        )
    try:
        report = Phase13BMethodAdmissionReport.from_dict(
            _strict_json(report_path)
        )
    except (TypeError, ValueError) as error:
        raise Phase12UnifiedAdmissionError(
            "Phase 13B TurboQuant successor report schema differs"
        ) from error
    expected_source_hashes = {
        path: authority["authority_sha256"]
        for path, authority in PHASE13B_TURBOQUANT_SOURCE_AUTHORITY.items()
    }
    if (
        report.method_family != "turboquant"
        or report.decision_id != "0030"
        or report.creation_git_sha != PHASE13B_SOURCE_AUTHORITY_COMMIT
        or report.historical_report_path
        != PRIOR_ADMISSION_REPORT_BINDINGS["turboquant"].as_posix()
        or report.historical_report_sha256
        != EXPECTED_REPORT_SHA256S["turboquant"]
        or report.source_hashes != expected_source_hashes
        or not report.b1_numerical_preserved
        or report.cuda_source_changed
    ):
        raise Phase12UnifiedAdmissionError(
            "Phase 13B TurboQuant successor authority differs"
        )
    _require_ancestor(root, execution_commit, PHASE13B_DECISION0030_COMMIT)
    _require_ancestor(
        root,
        PHASE13B_DECISION0030_COMMIT,
        PHASE13B_SOURCE_AUTHORITY_COMMIT,
    )
    _require_ancestor(root, PHASE13B_SOURCE_AUTHORITY_COMMIT, head)
    sources: dict[str, dict[str, Any]] = {}
    for path, expected in PHASE13B_TURBOQUANT_SOURCE_AUTHORITY.items():
        historical = _git_blob_authority(
            root,
            commit=execution_commit,
            path=path,
        )
        authority = _git_blob_authority(
            root,
            commit=PHASE13B_SOURCE_AUTHORITY_COMMIT,
            path=path,
        )
        current = _git_blob_authority(root, commit=head, path=path)
        if (
            historical["blob"] != expected["historical_blob"]
            or historical["sha256"] != expected["historical_sha256"]
            or authority["blob"] != expected["authority_blob"]
            or authority["sha256"] != expected["authority_sha256"]
            or current["blob"] != authority["blob"]
            or current["sha256"] != authority["sha256"]
            or _path_transition_commits(
                root,
                start=execution_commit,
                end=PHASE13B_SOURCE_AUTHORITY_COMMIT,
                path=path,
            )
            != (PHASE13B_DECISION0030_COMMIT,)
            or _path_transition_commits(
                root,
                start=PHASE13B_SOURCE_AUTHORITY_COMMIT,
                end=head,
                path=path,
            )
        ):
            raise Phase12UnifiedAdmissionError(
                f"Decision 0030 source transition is unrecognized: {path}"
            )
        sources[path] = {
            "historical": historical,
            "authority": authority,
            "current": current,
        }
    return {
        "decision": "0030",
        "decision_path": PHASE13B_DECISION0030_PATH.as_posix(),
        "decision_sha256": PHASE13B_DECISION0030_SHA256,
        "decision_commit": PHASE13B_DECISION0030_COMMIT,
        "source_authority_commit": PHASE13B_SOURCE_AUTHORITY_COMMIT,
        "successor_report_path": PHASE13B_TURBOQUANT_REPORT_PATH.as_posix(),
        "successor_report_sha256": PHASE13B_TURBOQUANT_REPORT_SHA256,
        "sources": sources,
    }


def _validate_decision0026_transition(
    root: Path,
    *,
    execution_commit: str,
    unchanged_paths: Mapping[str, str],
) -> dict[str, Any]:
    """Replay the sole authorized endpoint transition from historical evidence."""

    head = _git_bytes(root, "rev-parse", "HEAD").decode().strip()
    decision = _resolve_evidence_path(
        root, PHASE12_DECISION0026_PATH.as_posix()
    )
    if sha256_file(decision) != PHASE12_DECISION0026_SHA256:
        raise Phase12UnifiedAdmissionError(
            "Decision 0026 checksum differs"
        )
    _require_ancestor(root, execution_commit, PHASE12_DECISION0026_COMMIT)
    _require_ancestor(root, PHASE12_DECISION0026_COMMIT, head)
    endpoint_path = "src/kvbench/runtime/bf16_endpoint.py"
    historical_endpoint = _git_blob_authority(
        root,
        commit=execution_commit,
        path=endpoint_path,
    )
    current_endpoint = _git_blob_authority(
        root,
        commit=head,
        path=endpoint_path,
    )
    if (
        historical_endpoint["blob"] != PHASE12_HISTORICAL_ENDPOINT_BLOB
        or historical_endpoint["sha256"]
        != PHASE12_HISTORICAL_ENDPOINT_SHA256
        or current_endpoint["blob"] != PHASE12_CURRENT_ENDPOINT_BLOB
        or current_endpoint["sha256"] != PHASE12_CURRENT_ENDPOINT_SHA256
        or _path_transition_commits(
            root,
            start=execution_commit,
            end=head,
            path=endpoint_path,
        )
        != (PHASE12_DECISION0026_COMMIT,)
    ):
        raise Phase12UnifiedAdmissionError(
            "Decision 0026 endpoint transition is unrecognized"
        )
    stable: dict[str, dict[str, str]] = {}
    phase13b_successor: dict[str, Any] | None = None
    for path, expected_sha256 in unchanged_paths.items():
        historical = _git_blob_authority(
            root,
            commit=execution_commit,
            path=path,
        )
        current = _git_blob_authority(root, commit=head, path=path)
        historical_differs = (
            historical["sha256"] != expected_sha256
        )
        current_differs = (
            current["sha256"] != expected_sha256
            or historical["blob"] != current["blob"]
            or bool(
                _path_transition_commits(
                    root,
                    start=execution_commit,
                    end=head,
                    path=path,
                )
            )
        )
        if historical_differs:
            raise Phase12UnifiedAdmissionError(
                f"historical method source changed after execution: {path}"
            )
        if current_differs:
            if path != "src/kvbench/adapters/turboquant.py":
                raise Phase12UnifiedAdmissionError(
                    f"historical method source changed after execution: {path}"
                )
            if phase13b_successor is None:
                phase13b_successor = (
                    _validate_phase13b_turboquant_successor_transition(
                        root,
                        execution_commit=execution_commit,
                    )
                )
            if path not in phase13b_successor["sources"]:
                raise Phase12UnifiedAdmissionError(
                    f"historical method source changed after execution: {path}"
                )
        stable[path] = historical
    return {
        "schema_version": (
            "kvbench-phase12-decision0026-transition-authority-1.0.0"
        ),
        "execution_commit": execution_commit,
        "decision": "0026",
        "decision_path": PHASE12_DECISION0026_PATH.as_posix(),
        "decision_sha256": PHASE12_DECISION0026_SHA256,
        "decision_commit": PHASE12_DECISION0026_COMMIT,
        "historical_endpoint": historical_endpoint,
        "current_endpoint": current_endpoint,
        "unchanged_method_sources": stable,
        "semantic_authority": (
            "existing adapters remain behaviorally and allocation equivalent"
        ),
        "recognized_transition_only": True,
    }


def _expected_bf16_container_bridge(root: Path) -> dict[str, Any]:
    return {
        "schema_version": "kvbench-phase12-bf16-container-bridge-1.0.0",
        "artifacts": {
            mode: {
                "path": expected["path"].as_posix(),
                "root_sha256": expected["root_sha256"],
                "manifest_sha256": expected["manifest_sha256"],
                "result_sha256": expected["result_sha256"],
            }
            for mode, expected in PHASE12_BF16_PARITY_ARTIFACTS.items()
        },
        "transition": _validate_decision0026_transition(
            root,
            execution_commit=PHASE12_BF16_PARITY_EXECUTION_COMMIT,
            unchanged_paths={
                "src/kvbench/adapters/bf16.py": (
                    PHASE12_BF16_ADAPTER_SHA256
                )
            },
        ),
    }


def _validate_bf16_container_bridge(root: Path) -> dict[str, Any]:
    observed_artifacts: dict[str, Any] = {}
    for mode, expected in PHASE12_BF16_PARITY_ARTIFACTS.items():
        path = root / expected["path"]
        try:
            validation = validate_local_artifact(path, environ={})
        except (ArtifactValidationError, OSError, ValueError) as error:
            raise Phase12UnifiedAdmissionError(
                f"BF16 {mode} container-parity artifact failed"
            ) from error
        manifest_path = path / "manifest.json"
        manifest = _strict_json(manifest_path)
        result = _strict_json(path / "result.json")
        graph = result.get("graph_validation")
        if (
            validation.root_sha256 != expected["root_sha256"]
            or sha256_file(manifest_path) != expected["manifest_sha256"]
            or manifest.get("execution_git_sha")
            != PHASE12_BF16_PARITY_EXECUTION_COMMIT
            or manifest.get("graph_mode") != mode
            or manifest.get("run", {}).get("status") != "PASS"
            or manifest.get("container", {}).get("image_config_digest")
            != PHASE12_AUTHORIZED_CONTAINER_DIGEST
            or manifest.get("method_identity", {}).get(
                "adapter_implementation_sha256"
            )
            != PHASE12_BF16_ADAPTER_SHA256
            or result.get("passed") is not True
            or result.get("backend_fallback") is not False
            or not isinstance(result.get("numerical"), Mapping)
            or result["numerical"].get("passed") is not True
            or result.get("allocated_cache_bytes") != 17_072_128
            or result.get("logical_bf16_bytes") != 16_908_288
            or result.get("byte_breakdown_sums_to_allocated") is not True
            or (
                mode == "eager"
                and (
                    result.get("output_finite") is not True
                    or result.get("eager_allocation_criterion_passed")
                    is not True
                )
            )
            or (
                mode == "cuda_graph"
                and (
                    not isinstance(graph, Mapping)
                    or graph.get("passed") is not True
                    or graph.get("replay_outputs_exact") is not True
                    or graph.get("cache_pointers_stable") is not True
                    or graph.get("historical_cache_unchanged") is not True
                    or graph.get("replay_allocation", {}).get("passed")
                    is not True
                )
            )
        ):
            raise Phase12UnifiedAdmissionError(
                f"BF16 {mode} container-parity semantics differ"
            )
        observed_artifacts[mode] = {
            "path": expected["path"].as_posix(),
            "root_sha256": validation.root_sha256,
            "manifest_sha256": expected["manifest_sha256"],
            "result_sha256": sha256_file(path / "result.json"),
        }
    expected_bridge = _expected_bf16_container_bridge(root)
    if observed_artifacts != expected_bridge["artifacts"]:
        raise Phase12UnifiedAdmissionError(
            "BF16 container-parity evidence identity differs"
        )
    return expected_bridge


def _validate_turboquant_transition_bridge(
    root: Path,
    *,
    report: Any,
) -> dict[str, Any]:
    if report.creation_git_sha != PHASE12_TURBOQUANT_EXECUTION_COMMIT:
        raise Phase12UnifiedAdmissionError(
            "TurboQuant execution commit differs"
        )
    return {
        "schema_version": (
            "kvbench-phase12-turboquant-transition-bridge-1.0.0"
        ),
        "transition": _validate_decision0026_transition(
            root,
            execution_commit=PHASE12_TURBOQUANT_EXECUTION_COMMIT,
            unchanged_paths={
                "src/kvbench/adapters/turboquant.py": (
                    PHASE12_TURBOQUANT_ADAPTER_SHA256
                ),
                "src/kvbench/runtime/turboquant_session.py": (
                    PHASE12_TURBOQUANT_SESSION_SHA256
                ),
            },
        ),
    }


def _expected_historical_source_bridges(root: Path) -> dict[str, Any]:
    return {
        "bf16": _expected_bf16_container_bridge(root),
        "turboquant": {
            "schema_version": (
                "kvbench-phase12-turboquant-transition-bridge-1.0.0"
            ),
            "transition": _validate_decision0026_transition(
                root,
                execution_commit=PHASE12_TURBOQUANT_EXECUTION_COMMIT,
                unchanged_paths={
                    "src/kvbench/adapters/turboquant.py": (
                        PHASE12_TURBOQUANT_ADAPTER_SHA256
                    ),
                    "src/kvbench/runtime/turboquant_session.py": (
                        PHASE12_TURBOQUANT_SESSION_SHA256
                    ),
                },
            ),
        },
    }


def _method_checks(method: str, report: Any) -> dict[str, str]:
    if method == "bf16":
        return {
            name: getattr(report, name).status.value
            for name in (
                "correctness",
                "byte_accounting",
                "execution_path",
                "graph",
            )
        }
    return {
        check.check_id: check.status.value
        for check in report.checks
    }


def _method_fingerprints(method: str, report: Any) -> dict[str, str]:
    if method == "bf16":
        return {"bf16": report.method_config_fingerprint}
    if method == "turboquant":
        return {
            phase12_id: report.method_config_fingerprints[report_id]
            for report_id, phase12_id in (
                ("turboquant_4bit_nc", "tq_4bit_nc"),
                ("turboquant_k3v4_nc", "tq_k3v4_nc"),
                ("turboquant_3bit_nc", "tq_3bit_nc"),
            )
        }
    if method == "kivi":
        return {
            key: report.method_fingerprints[key]
            for key in ("k4v4", "k2v4", "k2v2")
        }
    return {
        key: report.method_fingerprints[key]
        for key in ("kvq4", "kvq3", "kvq2")
    }


def _validate_current_kvquant_authority(
    *,
    root: Path,
    references: Mapping[str, Mapping[str, str]],
) -> None:
    """Bind the current-source report to Decision 0029 without reinterpretation."""

    required = {"authority", "q23_binding"}
    if not required.issubset(references):
        raise Phase12UnifiedAdmissionError(
            "KVQuant Decision 0029 evidence references are incomplete"
        )
    authority = _strict_json(
        _resolve_evidence_path(root, references["authority"]["path"])
    )
    q23_binding = _strict_json(
        _resolve_evidence_path(root, references["q23_binding"]["path"])
    )
    expected_authority = {
        "execution_source_identifier": PHASE12_KVQUANT_AUTHORITY[
            "execution_source_identifier"
        ],
        "corrected_commit": PHASE12_KVQUANT_AUTHORITY["source_commit"],
        "corrected_tree": PHASE12_KVQUANT_AUTHORITY["source_tree"],
        "aggregate_patch_sha256": PHASE12_KVQUANT_AUTHORITY[
            "aggregate_patch_sha256"
        ],
        "extension_sha256": PHASE12_KVQUANT_AUTHORITY["extension_sha256"],
        "fixture_root": PHASE12_KVQUANT_AUTHORITY["fixture_root"],
        "authorized_container_digest": PHASE12_AUTHORIZED_CONTAINER_DIGEST,
    }
    if any(authority.get(key) != value for key, value in expected_authority.items()):
        raise Phase12UnifiedAdmissionError(
            "KVQuant Decision 0029 authority differs"
        )
    if PHASE12_KVQUANT_AUTHORITY["decision"] not in authority.get(
        "decisions", ()
    ):
        raise Phase12UnifiedAdmissionError(
            "KVQuant Decision 0029 is absent from current authority"
        )
    expected_q23 = {
        "source_profile": "decision0029",
        "source_commit": PHASE12_KVQUANT_AUTHORITY["source_commit"],
        "source_tree": PHASE12_KVQUANT_AUTHORITY["source_tree"],
        "aggregate_patch_sha256": PHASE12_KVQUANT_AUTHORITY[
            "aggregate_patch_sha256"
        ],
        "extension_sha256": PHASE12_KVQUANT_AUTHORITY["extension_sha256"],
        "evidence_root_sha256": PHASE12_KVQUANT_AUTHORITY[
            "q23_evidence_root"
        ],
        "checksums_valid": True,
        "complete_last": True,
        "fixtures_preserved": True,
        "q2_deterministic": True,
        "q3_deterministic": True,
        "graph_passed": True,
        "zero_replay_allocation": True,
        "sanitizer_memcheck_passed": True,
        "sanitizer_initcheck_passed": True,
    }
    if any(q23_binding.get(key) != value for key, value in expected_q23.items()):
        raise Phase12UnifiedAdmissionError(
            "KVQuant Decision 0029 Q23 binding differs"
        )


def load_and_validate_prior_admission_evidence(
    repo_root: Path,
    report_paths: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    """Strictly validate all four reports and every checksum-bound reference."""

    try:
        root = Path(repo_root).resolve(strict=True)
    except OSError as error:
        raise Phase12UnifiedAdmissionError(
            "repository root is unavailable"
        ) from error
    supplied = {} if report_paths is None else dict(report_paths)
    if not set(supplied).issubset(PRIOR_ADMISSION_REPORT_BINDINGS):
        raise Phase12UnifiedAdmissionError(
            "unknown prior-admission report override"
        )

    methods: dict[str, Any] = {}
    observed_fingerprints: dict[str, str] = {}
    observed_report_sha256s: dict[str, str] = {}
    for method, relative in PRIOR_ADMISSION_REPORT_BINDINGS.items():
        report_path = Path(supplied.get(method, root / relative))
        try:
            report_sha256 = sha256_file(report_path)
        except (OSError, ValueError) as error:
            raise Phase12UnifiedAdmissionError(
                f"{method} MethodAdmissionReport is missing"
            ) from error
        if report_sha256 != EXPECTED_REPORT_SHA256S[method]:
            display_name = {
                "bf16": "BF16",
                "turboquant": "TurboQuant",
                "kivi": "KIVI",
                "kvquant": "KVQuant",
            }[method]
            raise Phase12UnifiedAdmissionError(
                f"{display_name} MethodAdmissionReport SHA-256 differs"
            )
        payload = _strict_json(report_path)
        try:
            report = _REPORT_PARSERS[method].from_dict(payload)
        except (TypeError, ValueError) as error:
            raise Phase12UnifiedAdmissionError(
                f"{method} MethodAdmissionReport schema validation failed"
            ) from error
        if report.status.value != "PASS":
            raise Phase12UnifiedAdmissionError(
                f"{method} MethodAdmissionReport is not PASS"
            )
        references: dict[str, dict[str, str]] = {}
        for reference in report.evidence_references:
            if reference.evidence_id in references:
                raise Phase12UnifiedAdmissionError(
                    f"{method} evidence ID is duplicated"
                )
            evidence_path = _resolve_evidence_path(root, reference.path)
            try:
                observed_sha256 = sha256_file(evidence_path)
            except (OSError, ValueError) as error:
                raise Phase12UnifiedAdmissionError(
                    f"{method} referenced evidence is missing: {reference.path}"
                ) from error
            if observed_sha256 != reference.sha256:
                raise Phase12UnifiedAdmissionError(
                    f"{method} referenced evidence SHA-256 differs: "
                    f"{reference.path}"
                )
            references[reference.evidence_id] = {
                "path": reference.path,
                "sha256": reference.sha256,
            }
        if method == "kvquant":
            _validate_current_kvquant_authority(
                root=root,
                references=references,
            )
        authority_bridge: dict[str, Any] | None = None
        if method == "bf16":
            authority_bridge = _validate_bf16_container_bridge(root)
        elif method == "turboquant":
            authority_bridge = _validate_turboquant_transition_bridge(
                root,
                report=report,
            )
        checks = _method_checks(method, report)
        fingerprints = _method_fingerprints(method, report)
        for config_id, fingerprint in fingerprints.items():
            _require_sha256(fingerprint, f"{config_id} fingerprint")
            if (
                config_id not in EXPECTED_CONFIG_FINGERPRINTS
                or fingerprint != EXPECTED_CONFIG_FINGERPRINTS[config_id]
                or config_id in observed_fingerprints
            ):
                raise Phase12UnifiedAdmissionError(
                    f"{method} configuration fingerprint differs"
                )
            observed_fingerprints[config_id] = fingerprint
        methods[method] = {
            "report_path": str(report_path),
            "report_sha256": report_sha256,
            "checks": checks,
            "references": references,
            "all_references_valid": True,
            "config_fingerprints": fingerprints,
            "authority_bridge": authority_bridge,
        }
        observed_report_sha256s[method] = report_sha256

    if (
        tuple(observed_fingerprints) != MAIN_CONFIG_IDS
        or observed_fingerprints != EXPECTED_CONFIG_FINGERPRINTS
    ):
        raise Phase12UnifiedAdmissionError(
            "prior reports do not cover the exact Phase 12 configuration set"
        )
    return {
        "main_configurations": MAIN_CONFIG_IDS,
        "held_out_configurations": HELD_OUT_CONFIG_IDS,
        "config_fingerprints": observed_fingerprints,
        "report_sha256s": observed_report_sha256s,
        "methods": methods,
    }


def aggregate_g1_g4(prior: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize recognized method checks; every configuration must pass."""

    if (
        tuple(prior.get("main_configurations", ())) != MAIN_CONFIG_IDS
        or tuple(prior.get("held_out_configurations", ()))
        != HELD_OUT_CONFIG_IDS
        or prior.get("report_sha256s") != EXPECTED_REPORT_SHA256S
    ):
        raise Phase12UnifiedAdmissionError(
            "prior admission configuration or report authority differs"
        )
    if prior.get("config_fingerprints") != EXPECTED_CONFIG_FINGERPRINTS:
        raise Phase12UnifiedAdmissionError(
            "prior admission configuration fingerprint differs"
        )
    methods = prior.get("methods")
    if not isinstance(methods, Mapping) or set(methods) != set(
        EXPECTED_REPORT_SHA256S
    ):
        raise Phase12UnifiedAdmissionError("prior admission method set differs")

    configurations: dict[str, dict[str, Any]] = {}
    for config_id in MAIN_CONFIG_IDS:
        method = _CONFIG_METHOD[config_id]
        method_record = methods[method]
        if (
            not isinstance(method_record, Mapping)
            or method_record.get("all_references_valid") is not True
            or method_record.get("report_sha256")
            != EXPECTED_REPORT_SHA256S[method]
            or (
                method in {"bf16", "turboquant"}
                and not isinstance(
                    method_record.get("authority_bridge"), Mapping
                )
            )
        ):
            raise Phase12UnifiedAdmissionError(
                f"{method} evidence references are not valid"
            )
        checks = method_record.get("checks")
        if not isinstance(checks, Mapping):
            raise Phase12UnifiedAdmissionError(
                f"{method} admission checks are absent"
            )
        gates: dict[str, str] = {}
        evidence: dict[str, tuple[str, ...]] = {}
        for gate, required_checks in GATE_EVIDENCE_REQUIREMENTS[method].items():
            for check_id in required_checks:
                if checks.get(check_id) != "PASS":
                    raise Phase12UnifiedAdmissionError(
                        f"{method} {check_id} does not pass"
                    )
            gates[gate] = "PASS"
            evidence[gate] = required_checks
        configurations[config_id] = {
            "method": method,
            "fingerprint": EXPECTED_CONFIG_FINGERPRINTS[config_id],
            "gates": gates,
            "evidence": evidence,
        }

    global_gates = {
        gate: (
            "PASS"
            if all(
                record["gates"][gate] == "PASS"
                for record in configurations.values()
            )
            else "FAIL"
        )
        for gate in ("G1", "G2", "G3", "G4")
    }
    return {
        "configurations": configurations,
        "global_gates": global_gates,
        "held_out_configurations": HELD_OUT_CONFIG_IDS,
        "majority_voting": False,
    }


def _expected_entry_g1_g4() -> dict[str, Any]:
    configurations = {
        config_id: {
            "method": _CONFIG_METHOD[config_id],
            "fingerprint": EXPECTED_CONFIG_FINGERPRINTS[config_id],
            "gates": {
                gate: "PASS" for gate in ("G1", "G2", "G3", "G4")
            },
            "evidence": {
                gate: list(checks)
                for gate, checks in GATE_EVIDENCE_REQUIREMENTS[
                    _CONFIG_METHOD[config_id]
                ].items()
            },
        }
        for config_id in MAIN_CONFIG_IDS
    }
    return {
        "configurations": configurations,
        "global_gates": {
            gate: "PASS" for gate in ("G1", "G2", "G3", "G4")
        },
        "held_out_configurations": list(HELD_OUT_CONFIG_IDS),
        "majority_voting": False,
    }


def _validate_current_entry_g1_g4(
    payload: Mapping[str, Any],
    *,
    repo_root: Path = REPOSITORY_ROOT,
) -> None:
    expected = _expected_entry_g1_g4()
    current = json.loads(
        json.dumps(
            aggregate_g1_g4(
                load_and_validate_prior_admission_evidence(repo_root)
            ),
            allow_nan=False,
        )
    )
    if current != expected or payload != expected:
        raise Phase12UnifiedAdmissionError(
            "Phase 12 G1-G4 entry evidence does not replay exactly"
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Phase12OperationKey:
    """One exact structural key for the common Phase 12 G5 point."""

    configuration: str
    runner_kind: RunnerKind
    graph_mode: GraphMode
    historical_context: int
    attended_context: int
    batch_size: int
    capacity: int
    decode_step: int
    operation_fingerprint_sha256: str

    def __post_init__(self) -> None:
        if (
            self.configuration not in MAIN_CONFIG_IDS
            or self.runner_kind is not RunnerKind.FIXED_L
            or self.graph_mode is not GraphMode.CUDA_GRAPH
            or self.historical_context != PHASE12_CONTEXT_LENGTH
            or self.attended_context != PHASE12_CONTEXT_LENGTH + 1
            or self.batch_size != PHASE12_BATCH_SIZE
            or self.capacity != PHASE12_CONTEXT_LENGTH + 1
            or self.decode_step != 0
        ):
            raise ValueError("Phase 12 operation differs from the frozen point")
        _require_sha256(
            self.operation_fingerprint_sha256,
            "Phase 12 operation fingerprint",
        )
        if self.operation_fingerprint_sha256 != self._derive_fingerprint():
            raise ValueError("Phase 12 operation fingerprint differs")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": "kvbench-phase12-operation-key-1.0.0",
            "configuration": self.configuration,
            "runner_kind": self.runner_kind.value,
            "graph_mode": self.graph_mode.value,
            "historical_context": self.historical_context,
            "attended_context": self.attended_context,
            "batch_size": self.batch_size,
            "capacity": self.capacity,
            "decode_step": self.decode_step,
            "input_recipe_sha256": PHASE12_INPUT_RECIPE_SHA256,
        }

    def _derive_fingerprint(self) -> str:
        return sha256_hex(canonical_json_bytes(self._payload()))

    @classmethod
    def create(cls, configuration: str) -> "Phase12OperationKey":
        payload = {
            "schema_version": "kvbench-phase12-operation-key-1.0.0",
            "configuration": configuration,
            "runner_kind": PHASE12_RUNNER_KIND,
            "graph_mode": PHASE12_GRAPH_MODE,
            "historical_context": PHASE12_CONTEXT_LENGTH,
            "attended_context": PHASE12_CONTEXT_LENGTH + 1,
            "batch_size": PHASE12_BATCH_SIZE,
            "capacity": PHASE12_CONTEXT_LENGTH + 1,
            "decode_step": 0,
            "input_recipe_sha256": PHASE12_INPUT_RECIPE_SHA256,
        }
        return cls(
            configuration=configuration,
            runner_kind=RunnerKind.FIXED_L,
            graph_mode=GraphMode.CUDA_GRAPH,
            historical_context=PHASE12_CONTEXT_LENGTH,
            attended_context=PHASE12_CONTEXT_LENGTH + 1,
            batch_size=PHASE12_BATCH_SIZE,
            capacity=PHASE12_CONTEXT_LENGTH + 1,
            decode_step=0,
            operation_fingerprint_sha256=sha256_hex(
                canonical_json_bytes(payload)
            ),
        )


def _endpoint_scratch_pointers(endpoint: Any) -> dict[str, int]:
    pointers = {
        "endpoint_query_rope_scratch_data_ptr": int(
            endpoint.query_rope_scratch.data_ptr()
        ),
        "endpoint_key_rope_scratch_data_ptr": int(
            endpoint.key_rope_scratch.data_ptr()
        ),
    }
    if len(set(pointers.values())) != 2:
        raise Phase12UnifiedAdmissionError("endpoint RoPE scratch aliases")
    return pointers


def _phase12_session_pointers(session: Any) -> dict[str, int]:
    pointers = dict(session.current_cache_pointers())
    scratch = _endpoint_scratch_pointers(session.endpoint)
    for label, pointer in scratch.items():
        if label in pointers and pointers[label] != pointer:
            raise Phase12UnifiedAdmissionError(
                "session endpoint pointer identity differs"
            )
        pointers[label] = pointer
    return pointers


class _Phase12BF16EndpointSession(TurboQuantEndpointSession):
    """Phase-12-local structural bridge over the unchanged BF16 cache."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._pointers = self.current_cache_pointers()

    def current_cache_pointers(self) -> dict[str, int]:
        cache_pointers = self.cache.pointers()
        scratch = _endpoint_scratch_pointers(self.endpoint)
        if set(cache_pointers).intersection(scratch):
            raise EndpointSessionError("BF16 pointer labels overlap")
        return {**cache_pointers, **scratch}

    def current_historical_prefix_sha256(self) -> str:
        from kvbench.runtime.phase3_endpoint_audit import _cache_pair_sha256

        return _cache_pair_sha256(
            self.cache,
            start=0,
            length=self.operation_keys[0].historical_context,
        )

    def gqa_cache_geometry(self) -> dict[str, Any]:
        from kvbench.runtime.gqa_audit import audit_cache_geometry

        return audit_cache_geometry(self.cache, num_query_heads=32)

    def admit(
        self,
        *,
        observed_outputs: Sequence[tuple[str, bool]],
        execution_path_passed: bool,
        allocation_passed: bool,
        graph_passed: bool,
    ) -> None:
        from kvbench.runtime.model_loader import (
            validate_loaded_frozen_model_receipt,
        )

        observed = tuple(observed_outputs)
        verdicts = (
            execution_path_passed,
            allocation_passed,
            graph_passed,
        )
        if (
            self._state != "warmed"
            or any(type(value) is not bool for value in verdicts)
            or len(observed) != 1
        ):
            raise EndpointSessionError("BF16 session admission is invalid")
        digest, finite = observed[0]
        _require_sha256(digest, "BF16 audit output")
        if (
            type(finite) is not bool
            or (digest, finite) != self._warmed_outputs[0]
            or not all(verdicts)
            or not finite
            or self.eager_graph_comparison is None
            or not self.eager_graph_comparison.passed
        ):
            self._state = "failed"
            raise EndpointSessionError("BF16 session audits did not pass")
        validate_loaded_frozen_model_receipt(self.loaded)
        if (
            self.loaded.receipt.receipt_sha256 != self._receipt_sha256
            or self.current_cache_pointers() != self._pointers
            or self.current_historical_prefix_sha256() != self._prefix_sha256
        ):
            raise EndpointSessionError("BF16 session identity changed")
        self._audit_outputs = {0: observed[0]}
        self._state = "ready"


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _capture_process_snapshot(
    *,
    supervised_pid: int | None = None,
    supervised_start_ticks: int | None = None,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "preflight" / "process_query.py"),
    ]
    if supervised_pid is not None or supervised_start_ticks is not None:
        if (
            type(supervised_pid) is not int
            or supervised_pid <= 0
            or type(supervised_start_ticks) is not int
            or supervised_start_ticks < 0
        ):
            raise Phase12UnifiedAdmissionError(
                "supervised GPU-process identity is invalid"
            )
        command.extend(
            (
                "--supervised-root-pid",
                str(supervised_pid),
                "--supervised-root-start-ticks",
                str(supervised_start_ticks),
            )
        )
    environment = {
        "PATH": "/usr/local/cuda-13.0/bin:/usr/local/bin:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": f"{REPOSITORY_ROOT / 'src'}:{REPOSITORY_ROOT}",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TZ": "UTC",
    }
    result = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise Phase12UnifiedAdmissionError(
            "GPU-process snapshot is not JSON"
        ) from error
    if (
        result.returncode != 0
        or not isinstance(payload, dict)
        or payload.get("query_exit_code") != 0
        or payload.get("errors") != []
        or payload.get("foreign_compute_processes") != []
        or payload.get("unknown_processes") != []
    ):
        raise Phase12UnifiedAdmissionError("GPU-process snapshot failed closed")
    return payload


def _require_idle_snapshot(snapshot: Mapping[str, Any]) -> None:
    if snapshot.get("allowed_compute_processes") != []:
        raise Phase12UnifiedAdmissionError(
            "GPU exclusivity check found an active CUDA process"
        )


def _process_start_ticks(pid: int) -> int:
    try:
        raw = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
        closing = raw.rfind(")")
        return int(raw[closing + 2 :].split()[19])
    except (OSError, UnicodeError, ValueError, IndexError) as error:
        raise Phase12UnifiedAdmissionError(
            "worker process identity is unavailable"
        ) from error


def _require_owned_snapshot(
    snapshot: Mapping[str, Any],
    *,
    pid: int,
    start_ticks: int,
) -> None:
    allowed = snapshot.get("allowed_compute_processes")
    if (
        not isinstance(allowed, list)
        or len(allowed) != 1
        or allowed[0].get("gpu_uuid") != PHASE12_GPU_UUID
        or allowed[0].get("pid") != pid
        or allowed[0].get("process_start_time_ticks") != start_ticks
        or allowed[0].get("relationship") != "supervised_child"
    ):
        raise Phase12UnifiedAdmissionError(
            "worker does not exclusively own the authorized GPU"
        )


def _method_family(configuration: str) -> str:
    try:
        return _CONFIG_METHOD[configuration]
    except KeyError as error:
        raise Phase12UnifiedAdmissionError(
            "unknown Phase 12 configuration"
        ) from error


def _validate_live_method_identity(method: Any, configuration: str) -> None:
    """Fail closed if a factory result is mislabeled as another main config."""

    family = _method_family(configuration)
    expected_configuration = {
        "bf16": None,
        "tq_4bit_nc": "turboquant_4bit_nc",
        "tq_k3v4_nc": "turboquant_k3v4_nc",
        "tq_3bit_nc": "turboquant_3bit_nc",
        "k4v4": "k4v4",
        "k2v4": "k2v4",
        "k2v2": "k2v2",
        "kvq4": "kvq4",
        "kvq3": "kvq3",
        "kvq2": "kvq2",
    }[configuration]
    expected_bits = {
        "k4v4": (4, 4),
        "k2v4": (2, 4),
        "k2v2": (2, 2),
        "kvq4": 4,
        "kvq3": 3,
        "kvq2": 2,
    }.get(configuration)
    observed_configuration = getattr(method, "config_name", None)
    observed_bits: object | None = None
    if family == "kivi":
        observed_bits = (
            getattr(method, "k_bits", None),
            getattr(method, "v_bits", None),
        )
    elif family == "kvquant":
        observed_bits = getattr(method, "bits", None)
    if (
        getattr(method, "name", None) != family
        or observed_configuration != expected_configuration
        or (
            expected_bits is not None
            and observed_bits != expected_bits
        )
    ):
        raise Phase12UnifiedAdmissionError(
            "live method factory identity differs from the requested config"
        )


def _build_phase12_session(
    *,
    loaded: Any,
    operation_key: Phase12OperationKey,
    prefix_input_ids: Any,
    decode_input_ids: Any,
) -> Any:
    """Construct one existing adapter/cache/session at only the G5 point."""

    from kvbench.adapters import (
        build_method_adapter,
        declared_bf16_runtime_context,
    )
    from kvbench.runtime.allocation import capture_cuda_memory_snapshot
    from kvbench.runtime.bf16_endpoint import BF16DecodeEndpoint
    from kvbench.runtime.cuda_graph import capture_fixed_graph
    from kvbench.runtime.kivi_session import (
        KIVIEndpointSession,
        PHASE8_DECODE_ATOL,
        PHASE8_DECODE_RTOL,
        _historical_prefix_sha256 as kivi_prefix_sha256,
        kivi_runtime_context,
        load_frozen_kivi_method_config,
    )
    from kvbench.runtime.kvquant_session import (
        KVQuantEndpointSession,
        PHASE11_DECODE_ATOL,
        PHASE11_DECODE_RTOL,
        _historical_prefix_sha256 as kvquant_prefix_sha256,
        kvquant_runtime_context,
        load_frozen_kvquant_method_config,
    )
    from kvbench.runtime.model_loader import (
        validate_loaded_frozen_model_receipt,
    )
    from kvbench.runtime.numerical import (
        compare_tensors_untimed,
        tensor_sha256_untimed,
    )
    from kvbench.runtime.phase3_endpoint_audit import _cache_pair_sha256
    from kvbench.runtime.turboquant_session import turboquant_runtime_context

    import torch

    validate_loaded_frozen_model_receipt(loaded)
    family = _method_family(operation_key.configuration)
    if family == "bf16":
        method = build_method_adapter(
            "bf16",
            declared_bf16_runtime_context(loaded.model),
        )
        workspace_bytes = 32 * 1 * (32 + 8) * 64 * 2
    elif family == "turboquant":
        native_id = {
            "tq_4bit_nc": "turboquant_4bit_nc",
            "tq_k3v4_nc": "turboquant_k3v4_nc",
            "tq_3bit_nc": "turboquant_3bit_nc",
        }[operation_key.configuration]
        method = build_method_adapter(native_id, turboquant_runtime_context())
        workspace_bytes = 0
    elif family == "kivi":
        method = build_method_adapter(
            load_frozen_kivi_method_config(),
            kivi_runtime_context(),
            variant_id=operation_key.configuration,
        )
        method.prepare_runtime()
        workspace_bytes = 0
    else:
        method = build_method_adapter(
            load_frozen_kvquant_method_config(),
            kvquant_runtime_context(),
            variant_id=operation_key.configuration,
        )
        method.prepare_runtime()
        workspace_bytes = 0

    _validate_live_method_identity(method, operation_key.configuration)
    model_memory = capture_cuda_memory_snapshot(
        "model_baseline",
        device=prefix_input_ids.device,
    )
    cache = method.allocate(
        batch_size=PHASE12_BATCH_SIZE,
        capacity=operation_key.capacity,
        device=prefix_input_ids.device,
        workspace_bytes=workspace_bytes,
    )
    if family == "kvquant":
        method.initialize_cache_untimed(cache)
    else:
        cache.initialize_deterministic()
    if family == "kvquant":
        endpoint = BF16DecodeEndpoint(loaded.model, cache, method)
        cache_memory = capture_cuda_memory_snapshot(
            "post_cache_allocation",
            device=prefix_input_ids.device,
        )
    else:
        cache_memory = capture_cuda_memory_snapshot(
            "post_cache_allocation",
            device=prefix_input_ids.device,
        )
        endpoint = BF16DecodeEndpoint(loaded.model, cache, method)
    adapter_fingerprint = method.config_fingerprint(
        cache.layout_fingerprint()
    )
    position = torch.tensor(
        [operation_key.historical_context],
        dtype=torch.long,
        device=prefix_input_ids.device,
    )
    if family == "kvquant":
        cache.bind_fixed_position_tensor_untimed(
            position,
            logical_position=operation_key.historical_context,
        )
    rope = endpoint.prepare_position_embeddings(position.unsqueeze(0))
    token = decode_input_ids[:, :1]

    endpoint.prefill(prefix_input_ids)
    if family == "bf16":
        initial_prefix = _cache_pair_sha256(
            cache,
            start=0,
            length=operation_key.historical_context,
        )
    elif family == "turboquant":
        initial_prefix = cache.history_sha256(
            operation_key.historical_context
        )
    elif family == "kivi":
        initial_prefix = kivi_prefix_sha256(cache, operation_key)
    else:
        initial_prefix = kvquant_prefix_sha256(
            cache,
            operation_key.historical_context,
        )
    cache.prepare_fixed(operation_key.historical_context)

    def fixed_step() -> Any:
        return endpoint.decode(token, position, rope)

    eager_output: Any | None = None
    for _ in range(PHASE12_SETUP_WARMUPS[family]):
        eager_output = fixed_step()
    if eager_output is None:
        raise Phase12UnifiedAdmissionError("setup warmup produced no output")

    def current_prefix() -> str:
        if family == "bf16":
            return _cache_pair_sha256(
                cache,
                start=0,
                length=operation_key.historical_context,
            )
        if family == "turboquant":
            return cache.history_sha256(operation_key.historical_context)
        if family == "kivi":
            return kivi_prefix_sha256(cache, operation_key)
        return kvquant_prefix_sha256(
            cache,
            operation_key.historical_context,
        )

    if current_prefix() != initial_prefix:
        raise Phase12UnifiedAdmissionError("setup warmup changed history")
    eager_reference = eager_output.detach().to(device="cpu", copy=True).clone()
    graph = capture_fixed_graph(
        fixed_step,
        warmup_steps=0,
        device=cache.device,
    )
    if family == "kvquant":
        from kvbench.runtime.kvquant_session import _composite_cache_pointers

        pointers_before = _composite_cache_pointers(cache, endpoint)
    else:
        pointers_before = {
            **cache.pointers(),
            **_endpoint_scratch_pointers(endpoint),
        }
    first_replay = graph.replay().detach().to(device="cpu", copy=True).clone()
    second_replay = graph.replay().detach().to(device="cpu", copy=True).clone()
    torch.cuda.synchronize(device=cache.device)
    atol, rtol = {
        "bf16": (0.02, 0.02),
        "turboquant": (0.02, 0.02),
        "kivi": (PHASE8_DECODE_ATOL, PHASE8_DECODE_RTOL),
        "kvquant": (PHASE11_DECODE_ATOL, PHASE11_DECODE_RTOL),
    }[family]
    comparison = compare_tensors_untimed(
        first_replay,
        eager_reference,
        atol=atol,
        rtol=rtol,
    )
    replay_exact = bool(torch.equal(first_replay, second_replay))
    if family == "kvquant":
        pointers_after = _composite_cache_pointers(cache, endpoint)
    else:
        pointers_after = {
            **cache.pointers(),
            **_endpoint_scratch_pointers(endpoint),
        }
    history_stable = current_prefix() == initial_prefix
    pointers_stable = pointers_before == pointers_after
    if (
        not comparison.passed
        or not replay_exact
        or not history_stable
        or not pointers_stable
        or not bool(torch.isfinite(second_replay).all())
    ):
        raise Phase12UnifiedAdmissionError(
            "setup graph correctness or stability failed"
        )
    graph_evidence = {
        **graph.to_dict(),
        "consecutive_replay_outputs_exact": replay_exact,
        "first_replay_checksum": tensor_sha256_untimed(first_replay),
        "second_replay_checksum": tensor_sha256_untimed(second_replay),
        "cache_pointers_stable": pointers_stable,
        "historical_prefix_unchanged": history_stable,
        "replay_allocation_audited_separately": True,
        "setup_warmups": PHASE12_SETUP_WARMUPS[family],
        "frozen_atol": float(atol),
        "frozen_rtol": float(rtol),
    }
    session_class = {
        "bf16": _Phase12BF16EndpointSession,
        "turboquant": TurboQuantEndpointSession,
        "kivi": KIVIEndpointSession,
        "kvquant": KVQuantEndpointSession,
    }[family]
    return session_class(
        loaded=loaded,
        operation_keys=(operation_key,),
        endpoint=endpoint,
        cache=cache,
        method=method,
        adapter_config_fingerprint=adapter_fingerprint,
        model_memory=model_memory,
        cache_memory=cache_memory,
        fixed_operation=fixed_step,
        graph=graph,
        graph_evidence=graph_evidence,
        eager_graph_comparison=comparison,
        growing_operations=(),
        reset_growing=None,
        warmed_outputs=(
            (
                tensor_sha256_untimed(second_replay),
                bool(torch.isfinite(second_replay).all()),
            ),
        ),
        prefix_sha256=initial_prefix,
    )


def _normalize_runner_result(value: Mapping[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(value, allow_nan=False))
    timing = result.get("timing")
    if (
        result.get("measurement_scope")
        != "measurement_container_admission"
        or result.get("performance_claim_eligible") is not False
        or not isinstance(timing, dict)
        or timing.get("paper_claim_eligible") is not False
        or timing.get("measurement_scope") != "native_host_admission"
        or timing.get("sample_count") != PHASE12_MEASURED_BATCHES
    ):
        raise Phase12UnifiedAdmissionError("common runner governance differs")
    samples = timing.get("samples")
    if (
        not isinstance(samples, list)
        or len(samples) != PHASE12_MEASURED_BATCHES
        or any(
            not isinstance(item, dict)
            or item.get("completed_operations") != PHASE12_MEASURED_STEPS
            or item.get("failed_operations") != 0
            for item in samples
        )
    ):
        raise Phase12UnifiedAdmissionError("common runner timing set differs")
    timing["measurement_scope"] = "measurement_container_admission"
    result.update(
        {
            "quality_status": "unvalidated",
            "claim_eligibility": "performance_only",
            "performance_claim_eligible": False,
            "measurement_scope": "measurement_container_admission",
            "speedup_calculated": False,
            "r_hbm": None,
        }
    )
    return result


def _telemetry_range(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    key: str,
) -> tuple[float, float]:
    values = (before.get(key), after.get(key))
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        for value in values
    ):
        raise Phase12UnifiedAdmissionError(
            f"telemetry {key} is unavailable"
        )
    observed = tuple(float(value) for value in values)
    return min(observed), max(observed)


def _structural_fingerprint(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(payload))).hexdigest()


def _validate_runtime_adapter_fingerprint(
    *,
    method: Any,
    cache: Any,
    observed: object,
) -> str:
    replayed = method.config_fingerprint(cache.layout_fingerprint())
    _require_sha256(replayed, "replayed runtime adapter fingerprint")
    if observed != replayed:
        raise Phase12UnifiedAdmissionError(
            "runtime adapter/layout fingerprint does not replay"
        )
    return replayed


def _expected_prior_g3_binding(family: str) -> dict[str, Any]:
    if (
        family not in PRIOR_ADMISSION_REPORT_BINDINGS
        or family not in GATE_EVIDENCE_REQUIREMENTS
    ):
        raise Phase12UnifiedAdmissionError(
            "prior G3 method family is unsupported"
        )
    return {
        "schema_version": "kvbench-phase12-prior-g3-binding-1.0.0",
        "report_path": PRIOR_ADMISSION_REPORT_BINDINGS[family].as_posix(),
        "report_sha256": EXPECTED_REPORT_SHA256S[family],
        "check_ids": list(GATE_EVIDENCE_REQUIREMENTS[family]["G3"]),
    }


def _validate_prior_g3_binding(
    binding: Mapping[str, Any],
    *,
    family: str,
) -> None:
    expected = _expected_prior_g3_binding(family)
    if (
        dict(binding) != expected
        or sha256_file(REPOSITORY_ROOT / expected["report_path"])
        != expected["report_sha256"]
    ):
        raise Phase12UnifiedAdmissionError(
            "prior immutable G3 execution-path binding differs"
        )


def _validate_execution_path_record(
    record: Mapping[str, Any],
    *,
    family: str,
) -> None:
    required = {
        "passed",
        "backend_identity_verified",
        "device_kernel_family_verified",
        "allocation_categories_verified",
        "temporary_tensor_shapes_verified",
        "gqa_replication_detected",
        "full_prefix_temporary_detected",
        "host_synchronization_detected",
        "backend_fallback_detected",
        "full_prefix_dequantization",
        "evidence_source",
    }
    if set(record) != required:
        raise Phase12UnifiedAdmissionError(
            "runtime execution-path evidence fields differ"
        )
    try:
        replay = execution_path_audit_facade(
            backend_identity_verified=record["backend_identity_verified"],
            device_kernel_family_verified=record[
                "device_kernel_family_verified"
            ],
            allocation_categories_verified=record[
                "allocation_categories_verified"
            ],
            temporary_tensor_shapes_verified=record[
                "temporary_tensor_shapes_verified"
            ],
            gqa_replication_detected=record[
                "gqa_replication_detected"
            ],
            full_prefix_temporary_detected=record[
                "full_prefix_temporary_detected"
            ],
            host_synchronization_detected=record[
                "host_synchronization_detected"
            ],
            backend_fallback_detected=record[
                "backend_fallback_detected"
            ],
            full_prefix_dequantization=record[
                "full_prefix_dequantization"
            ],
        ).to_dict()
    except (TypeError, ValueError) as error:
        raise Phase12UnifiedAdmissionError(
            "runtime execution-path evidence does not replay"
        ) from error
    if (
        replay != record
        or record["passed"] is not True
        or record["evidence_source"] != "existing_phase3_audits"
        or record["full_prefix_dequantization"]
        != ("not_applicable" if family == "bf16" else "verified_false")
    ):
        raise Phase12UnifiedAdmissionError(
            "runtime execution-path verdict differs"
        )


def _gqa_geometry_passes(
    geometry: Mapping[str, Any],
    *,
    family: str,
) -> bool:
    common = bool(
        geometry.get("num_query_heads") == 32
        and geometry.get("num_kv_heads") == 8
        and geometry.get("gqa_group_size", 4) == 4
    )
    family_specific = bool(
        (
            family == "bf16"
            and geometry.get("uses_kv_head_geometry") is True
            and geometry.get("query_head_storage_detected") is False
        )
        or (
            family == "turboquant"
            and geometry.get("native_kv_head_storage") is True
            and geometry.get("gqa_materialized") is False
        )
        or (
            family == "kivi"
            and geometry.get("native_kv_head_storage") is True
            and geometry.get("gqa_materialized") is False
            and geometry.get("expanded_kv_heads") == 0
            and geometry.get("kv_head_mapping") == "query_head // 4"
        )
        or (
            family == "kvquant"
            and geometry.get("native_kv_head_storage") is True
            and geometry.get("query_head_sized_kv_cache") is False
        )
    )
    return common and family_specific


def _normalize_cuda_graph_debug_dot(raw: bytes) -> tuple[bytes, int, int, int]:
    """Remove process-local handles while preserving captured graph structure."""

    try:
        text = raw.decode("utf-8")
    except UnicodeError as error:
        raise Phase12UnifiedAdmissionError(
            "CUDA Graph debug witness is not UTF-8"
        ) from error
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _CUDA_GRAPH_POINTER_RE.sub("<PTR>", normalized)
    normalized = _CUDA_GRAPH_ID_RE.sub("graph", normalized)
    normalized = _CUDA_GRAPH_CLUSTER_RE.sub("cluster", normalized)
    if not normalized.endswith("\n"):
        normalized += "\n"
    node_count = normalized.count('shape="record"')
    kernel_node_count = normalized.count('label="{KERNEL')
    edge_count = normalized.count(" -> ")
    if (
        not normalized.startswith("digraph dot {\n")
        or node_count <= 0
        or kernel_node_count <= 0
        or kernel_node_count > node_count
        or _CUDA_GRAPH_POINTER_RE.search(normalized) is not None
    ):
        raise Phase12UnifiedAdmissionError(
            "CUDA Graph debug witness has an unrecognized structure"
        )
    return (
        normalized.encode("utf-8"),
        node_count,
        kernel_node_count,
        edge_count,
    )


@contextmanager
def _observable_cuda_graph_factory(torch: Any) -> Any:
    """Let the existing graph harness retain its graph for untimed inspection."""

    original = torch.cuda.CUDAGraph
    created: list[Any] = []

    def construct(*args: Any, **kwargs: Any) -> Any:
        if args or kwargs:
            raise Phase12UnifiedAdmissionError(
                "existing CUDA Graph constructor contract changed"
            )
        graph = original(keep_graph=True)
        graph.enable_debug_mode()
        created.append(graph)
        return graph

    torch.cuda.CUDAGraph = construct
    try:
        yield created
    finally:
        torch.cuda.CUDAGraph = original


def _write_cuda_graph_path_witness(
    *,
    graph: Any,
    run_root: Path,
    phase: str,
) -> dict[str, Any]:
    """Persist raw and pointer-normalized topology from the measured graph."""

    resolved = run_root.resolve(strict=True)
    if (
        resolved.name == ""
        or _RUN_ID_RE.fullmatch(resolved.name) is None
        or resolved.is_symlink()
    ):
        raise Phase12UnifiedAdmissionError(
            "CUDA Graph witness output root is unsafe"
        )
    if phase not in {"before", "after"}:
        raise Phase12UnifiedAdmissionError(
            "CUDA Graph witness phase is invalid"
        )
    raw_path = resolved / f"kernel-path.{phase}.raw.dot"
    normalized_path = resolved / f"kernel-path.{phase}.normalized.dot"
    if (
        raw_path.exists()
        or raw_path.is_symlink()
        or normalized_path.exists()
        or normalized_path.is_symlink()
    ):
        raise Phase12UnifiedAdmissionError(
            "CUDA Graph witness output already exists"
        )
    graph.debug_dump(str(raw_path))
    try:
        raw = raw_path.read_bytes()
    except OSError as error:
        raise Phase12UnifiedAdmissionError(
            "CUDA Graph raw path witness was not written"
        ) from error
    normalized, nodes, kernels, edges = _normalize_cuda_graph_debug_dot(raw)
    write_exclusive(normalized_path, normalized)
    return {
        "schema_version": "kvbench-phase12-cuda-graph-path-witness-1.0.0",
        "phase": phase,
        "observation_kind": "cuda_graph_debug_dot",
        "raw_path": raw_path.name,
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "normalized_path": normalized_path.name,
        "normalized_sha256": hashlib.sha256(normalized).hexdigest(),
        "node_count": nodes,
        "kernel_node_count": kernels,
        "edge_count": edges,
        "process_local_handles_normalized": True,
        "timing_collected": False,
        "profiler_used": False,
    }


def _run_g5_worker(
    *,
    run_id: str,
    configuration: str,
    replicate_index: int,
    seed: int,
    order_index: int,
    git_sha: str,
    run_artifact_root: Path,
) -> dict[str, Any]:
    """Run one independent process point; called only in the container child."""

    container_runtime_attestation = _require_authorized_container_runtime()
    pre_snapshot = _capture_process_snapshot()
    _require_idle_snapshot(pre_snapshot)
    pid = os.getpid()
    start_ticks = _process_start_ticks(pid)

    import torch

    from kvbench.runtime.allocation import audit_cuda_allocations
    from kvbench.runtime.backend import forced_flash_execution
    from kvbench.runtime.fixed_l_runner import run_fixed_l
    from kvbench.runtime.model_loader import load_frozen_model
    from kvbench.runtime.numerical import tensor_sha256_untimed
    from kvbench.runtime.timing import warmup_operations

    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
    ):
        raise Phase12UnifiedAdmissionError(
            "worker execution authority differs"
        )
    observed_head = subprocess.run(
        ("/usr/bin/git", "rev-parse", "HEAD"),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    observed_status = subprocess.run(
        (
            "/usr/bin/git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if observed_head != git_sha or observed_status:
        raise Phase12UnifiedAdmissionError(
            "worker Git source authority or cleanliness differs"
        )
    device = torch.device("cuda:0")
    loaded = load_frozen_model(device=device)
    prefix = (
        torch.arange(
            PHASE12_CONTEXT_LENGTH,
            dtype=torch.long,
            device=device,
        ).reshape(1, PHASE12_CONTEXT_LENGTH)
        + 12_000
    ) % 120_000 + 1_000
    decode = torch.tensor(
        [[(12_000 + PHASE12_CONTEXT_LENGTH + 257) % 120_000 + 1_000]],
        dtype=torch.long,
        device=device,
    )
    operation_key = Phase12OperationKey.create(configuration)
    with torch.inference_mode(), forced_flash_execution():
        with _observable_cuda_graph_factory(torch) as observed_graphs:
            session = _build_phase12_session(
                loaded=loaded,
                operation_key=operation_key,
                prefix_input_ids=prefix,
                decode_input_ids=decode,
            )
        if session.graph is None:
            raise Phase12UnifiedAdmissionError("captured graph is absent")
        if (
            len(observed_graphs) != 1
            or session.graph.graph is not observed_graphs[0]
        ):
            raise Phase12UnifiedAdmissionError(
                "measured CUDA Graph is not the observed graph"
            )
        graph_exec_before = int(
            session.graph.graph.raw_cuda_graph_exec()
        )
        graph_path_before = _write_cuda_graph_path_witness(
            graph=session.graph.graph,
            run_root=run_artifact_root,
            phase="before",
        )
        pointers_before = _phase12_session_pointers(session)
        history_before = session.current_historical_prefix_sha256()
        warm_output = warmup_operations(
            session.graph.replay,
            count=PHASE12_WARMUP_STEPS,
            device=session.cache_device,
        )
        warm_copy = warm_output.detach().to(device="cpu", copy=True).clone()
        warm_checksum = tensor_sha256_untimed(warm_copy)
        warm_finite = bool(torch.isfinite(warm_copy).all())
        replay_allocation = audit_cuda_allocations(
            session.graph.replay,
            device=session.cache_device,
        )
        audit_output = (
            session.graph.replay()
            .detach()
            .to(device="cpu", copy=True)
            .clone()
        )
        torch.cuda.synchronize(device=session.cache_device)
        audit_checksum = tensor_sha256_untimed(audit_output)
        audit_finite = bool(torch.isfinite(audit_output).all())
        allocation_record = replay_allocation.to_dict()
        graph_passed = bool(
            session.graph_evidence is not None
            and session.graph_evidence.get("captured") is True
            and session.graph_evidence.get("fallback") is False
            and session.graph_evidence.get(
                "consecutive_replay_outputs_exact"
            )
            is True
            and session.eager_graph_comparison is not None
            and session.eager_graph_comparison.passed
        )
        allocation_passed = bool(
            replay_allocation.audit_available
            and replay_allocation.passed
            and replay_allocation.allocation_event_count == 0
            and replay_allocation.allocation_event_bytes == 0
            and replay_allocation.allocated_after
            == replay_allocation.allocated_before
            and replay_allocation.reserved_after
            == replay_allocation.reserved_before
        )
        family = _method_family(configuration)
        geometry_before = session.gqa_cache_geometry()
        runtime_context = session.method.runtime_context
        prior_g3_binding = _expected_prior_g3_binding(family)
        _validate_prior_g3_binding(
            prior_g3_binding,
            family=family,
        )
        static_g3_bound = True
        replayed_adapter_fingerprint = (
            _validate_runtime_adapter_fingerprint(
                method=session.method,
                cache=session.cache,
                observed=session.adapter_config_fingerprint,
            )
        )
        live_adapter_identity_verified = bool(
            replayed_adapter_fingerprint
        )
        backend_identity_verified = bool(
            runtime_context.model_id == PHASE12_MODEL_ID
            and runtime_context.model_revision == PHASE12_MODEL_REVISION
            and runtime_context.num_layers == 32
            and runtime_context.num_query_heads == 32
            and runtime_context.num_kv_heads == 8
            and runtime_context.head_dim == 128
            and isinstance(runtime_context.backend_id, str)
            and bool(runtime_context.backend_id)
            and re.fullmatch(
                r"[0-9a-f]{64}",
                runtime_context.backend_fingerprint,
            )
            is not None
        )
        runtime_path_audit = execution_path_audit_facade(
            backend_identity_verified=backend_identity_verified,
            device_kernel_family_verified=(
                static_g3_bound
                and live_adapter_identity_verified
                and EXPECTED_REPORT_SHA256S[family]
                == sha256_file(
                    REPOSITORY_ROOT
                    / PRIOR_ADMISSION_REPORT_BINDINGS[family]
                )
            ),
            allocation_categories_verified=allocation_passed,
            temporary_tensor_shapes_verified=(
                pointers_before == _phase12_session_pointers(session)
                and _gqa_geometry_passes(
                    geometry_before,
                    family=family,
                )
            ),
            gqa_replication_detected=not _gqa_geometry_passes(
                geometry_before,
                family=family,
            ),
            full_prefix_temporary_detected=not static_g3_bound,
            host_synchronization_detected=not static_g3_bound,
            backend_fallback_detected=not graph_passed,
            full_prefix_dequantization=(
                "not_applicable"
                if family == "bf16"
                else "verified_false"
            ),
        )
        runtime_path_record = runtime_path_audit.to_dict()
        _validate_execution_path_record(
            runtime_path_record,
            family=family,
        )
        if (
            not warm_finite
            or not audit_finite
            or warm_checksum != audit_checksum
            or pointers_before != _phase12_session_pointers(session)
            or history_before != session.current_historical_prefix_sha256()
        ):
            raise Phase12UnifiedAdmissionError(
                "post-capture warmup or audit drifted"
            )
        session.graph_evidence["replay_allocation"] = allocation_record
        session.graph_evidence["phase12_warmup_replays"] = (
            PHASE12_WARMUP_STEPS
        )
        session.admit(
            observed_outputs=((audit_checksum, audit_finite),),
            execution_path_passed=runtime_path_audit.passed,
            allocation_passed=allocation_passed,
            graph_passed=graph_passed,
        )
        raw_runner = run_fixed_l(
            session,
            measured_steps=PHASE12_MEASURED_STEPS,
            measured_batches=PHASE12_MEASURED_BATCHES,
        ).to_dict()
    runner = _normalize_runner_result(raw_runner)
    graph_exec_after = int(session.graph.graph.raw_cuda_graph_exec())
    graph_path_after = _write_cuda_graph_path_witness(
        graph=session.graph.graph,
        run_root=run_artifact_root,
        phase="after",
    )
    if (
        graph_exec_before <= 0
        or graph_exec_after != graph_exec_before
        or graph_path_before["normalized_sha256"]
        != graph_path_after["normalized_sha256"]
        or graph_path_before["node_count"]
        != graph_path_after["node_count"]
        or graph_path_before["kernel_node_count"]
        != graph_path_after["kernel_node_count"]
        or graph_path_before["edge_count"]
        != graph_path_after["edge_count"]
    ):
        raise Phase12UnifiedAdmissionError(
            "measured CUDA Graph topology or executable changed"
        )
    graph_path_witness = {
        "schema_version": (
            "kvbench-phase12-cuda-graph-path-observation-1.0.0"
        ),
        "observation_kind": "cuda_graph_debug_dot",
        "before": graph_path_before,
        "after": graph_path_after,
        "normalized_sha256": graph_path_before["normalized_sha256"],
        "node_count": graph_path_before["node_count"],
        "kernel_node_count": graph_path_before["kernel_node_count"],
        "edge_count": graph_path_before["edge_count"],
        "graph_exec_pointer_stable": True,
        "topology_stable_within_process": True,
        "timing_collected": False,
        "profiler_used": False,
    }
    memory = runner.get("memory_evidence")
    if (
        runner.get("output_finite") is not True
        or runner.get("cache_pointers_stable") is not True
        or runner.get("historical_cache_unchanged") is not True
        or runner.get("output_checksum") != audit_checksum
        or pointers_before != _phase12_session_pointers(session)
        or history_before != session.current_historical_prefix_sha256()
        or not isinstance(memory, dict)
        or memory.get("timing_allocated_delta_bytes") != 0
        or memory.get("timing_reserved_delta_bytes") != 0
        or runner.get("r_hbm") is not None
    ):
        raise Phase12UnifiedAdmissionError(
            "common G5 run stability or allocation failed"
        )
    geometry = runner.get("gqa_cache_geometry")
    if (
        not isinstance(geometry, Mapping)
        or not _gqa_geometry_passes(geometry, family=family)
        or dict(geometry) != dict(geometry_before)
        or runner.get("adapter_config_fingerprint")
        != session.adapter_config_fingerprint
        or not isinstance(runner.get("graph"), Mapping)
        or runner["graph"].get("captured") is not True
        or runner["graph"].get("fallback") is not False
        or runner["graph"].get("consecutive_replay_outputs_exact") is not True
    ):
        raise Phase12UnifiedAdmissionError("native GQA geometry differs")
    timing_samples = runner["timing"]["samples"]
    process_median = float(
        statistics.median(
            float(item["cuda_ms_per_operation"])
            for item in timing_samples
        )
    )
    telemetry_before = runner["telemetry_before"]
    telemetry_after = runner["telemetry_after"]
    temperature = _telemetry_range(
        telemetry_before,
        telemetry_after,
        "temperature_celsius",
    )
    sm_clock = _telemetry_range(
        telemetry_before,
        telemetry_after,
        "sm_clock_mhz",
    )
    memory_clock = _telemetry_range(
        telemetry_before,
        telemetry_after,
        "memory_clock_mhz",
    )
    power = _telemetry_range(
        telemetry_before,
        telemetry_after,
        "power_watts",
    )
    if any(not value.is_integer() for value in (*sm_clock, *memory_clock)):
        raise Phase12UnifiedAdmissionError("telemetry clocks are not integral")

    owned_snapshot = _capture_process_snapshot(
        supervised_pid=pid,
        supervised_start_ticks=start_ticks,
    )
    _require_owned_snapshot(
        owned_snapshot,
        pid=pid,
        start_ticks=start_ticks,
    )
    allocation_fingerprint = _structural_fingerprint(
        {
            "schema_version": "kvbench-phase12-allocation-fingerprint-1.0.0",
            "configuration": configuration,
            "cache_accounting": runner["cache_accounting"],
            "cache_byte_breakdown": runner["cache_byte_breakdown"],
            "cache_layout_fingerprint": runner["cache_layout_fingerprint"],
            "replay_allocation": allocation_record,
            "normal_timing_allocated_delta_bytes": memory[
                "timing_allocated_delta_bytes"
            ],
            "normal_timing_reserved_delta_bytes": memory[
                "timing_reserved_delta_bytes"
            ],
        }
    )
    method = family
    kernel_path_fingerprint = _structural_fingerprint(
        {
            "schema_version": "kvbench-phase12-kernel-path-fingerprint-1.0.0",
            "configuration": configuration,
            "method": method,
            "prior_method_admission_sha256": EXPECTED_REPORT_SHA256S[method],
            "prior_g3_binding": prior_g3_binding,
            "container_runtime_attestation": (
                container_runtime_attestation
            ),
            "operation_fingerprint_sha256": (
                operation_key.operation_fingerprint_sha256
            ),
            "runtime_adapter_fingerprint": (
                runner["adapter_config_fingerprint"]
            ),
            "cache_layout_fingerprint": (
                runner["cache_layout_fingerprint"]
            ),
            "backend_id": runtime_context.backend_id,
            "backend_fingerprint": runtime_context.backend_fingerprint,
            "observed_execution_path_audit": runtime_path_record,
            "graph_mode": PHASE12_GRAPH_MODE,
            "graph_captured": runner["graph"]["captured"],
            "graph_fallback": runner["graph"]["fallback"],
            "graph_replay_exact": runner["graph"][
                "consecutive_replay_outputs_exact"
            ],
            "native_kv_head_storage": True,
            "observed_cuda_graph_normalized_sha256": (
                graph_path_witness["normalized_sha256"]
            ),
            "observed_cuda_graph_node_count": (
                graph_path_witness["node_count"]
            ),
            "observed_cuda_graph_kernel_node_count": (
                graph_path_witness["kernel_node_count"]
            ),
            "observed_cuda_graph_edge_count": (
                graph_path_witness["edge_count"]
            ),
            "witness_kind": (
                "existing_admission_plus_observed_cuda_graph_topology"
            ),
        }
    )
    return {
        "schema_version": PHASE12_RUN_SCHEMA,
        "run_id": run_id,
        "method_config_id": configuration,
        "method_config_fingerprint": EXPECTED_CONFIG_FINGERPRINTS[
            configuration
        ],
        "method_family": method,
        "replicate_index": replicate_index,
        "seed": seed,
        "order_index": order_index,
        "execution_git_sha": git_sha,
        "authorized_container_digest": PHASE12_AUTHORIZED_CONTAINER_DIGEST,
        "gpu_uuid": PHASE12_GPU_UUID,
        "model_id": PHASE12_MODEL_ID,
        "model_revision": PHASE12_MODEL_REVISION,
        "tokenizer_id": PHASE12_TOKENIZER_ID,
        "tokenizer_revision": PHASE12_TOKENIZER_REVISION,
        "operation_fingerprint_sha256": (
            operation_key.operation_fingerprint_sha256
        ),
        "input_recipe": PHASE12_INPUT_RECIPE,
        "input_recipe_sha256": PHASE12_INPUT_RECIPE_SHA256,
        "runtime_adapter_fingerprint": runner[
            "adapter_config_fingerprint"
        ],
        "cache_layout_fingerprint": runner["cache_layout_fingerprint"],
        "process_median_ms": process_median,
        "output_checksum": runner["output_checksum"],
        "kernel_path_fingerprint": kernel_path_fingerprint,
        "kernel_path_observation": graph_path_witness,
        "kernel_path_witness_kind": (
            "existing_admission_plus_observed_cuda_graph_topology"
        ),
        "prior_g3_binding": prior_g3_binding,
        "container_runtime_attestation": (
            container_runtime_attestation
        ),
        "execution_path_audit": runtime_path_record,
        "runtime_backend_id": runtime_context.backend_id,
        "runtime_backend_fingerprint": (
            runtime_context.backend_fingerprint
        ),
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
        "no_backend_fallback": (
            runtime_path_audit.passed
            and not runtime_path_audit.backend_fallback_detected
        ),
        "allocation_stable": True,
        "kernel_path_stable": runtime_path_audit.passed,
        "gpu_exclusive": True,
        "speedup_calculated": False,
        "quality_status": "unvalidated",
        "claim_eligibility": "performance_only",
        "performance_claim_eligible": False,
        "measurement_scope": "measurement_container_admission",
        "r_hbm": None,
        "setup_warmups": PHASE12_SETUP_WARMUPS[_method_family(configuration)],
        "warmup_replays": PHASE12_WARMUP_STEPS,
        "measured_steps": PHASE12_MEASURED_STEPS,
        "measured_batches": PHASE12_MEASURED_BATCHES,
        "graph_replay_allocation": allocation_record,
        "runner": runner,
        "gpu_process_before_cuda": pre_snapshot,
        "gpu_process_owned_after_measurement": owned_snapshot,
    }


def new_campaign_id(git_sha: str, *, now: datetime | None = None) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", git_sha):
        raise Phase12UnifiedAdmissionError("execution Git SHA is invalid")
    selected = datetime.now(timezone.utc) if now is None else now
    if selected.tzinfo is None:
        raise Phase12UnifiedAdmissionError("campaign timestamp is not UTC-aware")
    timestamp = selected.astimezone(timezone.utc).strftime("%Y%m%dt%H%M%S%f")[
        :21
    ]
    value = f"phase12-{timestamp}z-{git_sha[:8]}-{secrets.token_hex(3)}"
    if _CAMPAIGN_ID_RE.fullmatch(value) is None:
        raise Phase12UnifiedAdmissionError("generated campaign ID is invalid")
    return value


def _validate_campaign_id(value: str) -> str:
    if type(value) is not str or _CAMPAIGN_ID_RE.fullmatch(value) is None:
        raise Phase12UnifiedAdmissionError("Phase 12 campaign ID is invalid")
    return value


def _require_safe_phase12_root(root: Path) -> Path:
    lexical = Path(root).absolute()
    expected_parent = PHASE12_ARTIFACT_ROOT.absolute()
    if lexical.parent != expected_parent or lexical.name.startswith("."):
        raise Phase12UnifiedAdmissionError(
            "Phase 12 artifact is outside the exact campaign root"
        )
    try:
        metadata = lexical.lstat()
    except OSError as error:
        raise Phase12UnifiedAdmissionError(
            "Phase 12 artifact root is absent"
        ) from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise Phase12UnifiedAdmissionError(
            "Phase 12 artifact root is unsafe"
        )
    return lexical


def _entry_authority_contract(
    *,
    campaign_id: str,
    execution_git_sha: str,
    historical_source_bridges: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "kvbench-phase12-entry-authority-1.0.0",
        "campaign_id": campaign_id,
        "execution_git_sha": execution_git_sha,
        "authorized_container_digest": PHASE12_AUTHORIZED_CONTAINER_DIGEST,
        "plan_path": PHASE12_PLAN_PATH.as_posix(),
        "plan_sha256": PHASE12_PLAN_SHA256,
        "method_admission_report_sha256s": EXPECTED_REPORT_SHA256S,
        "configuration_fingerprints": EXPECTED_CONFIG_FINGERPRINTS,
        "kvquant_execution_authority": PHASE12_KVQUANT_AUTHORITY,
        "historical_source_bridges": dict(historical_source_bridges),
        "main_configurations": list(MAIN_CONFIG_IDS),
        "held_out_configurations": list(HELD_OUT_CONFIG_IDS),
        "randomization_seeds": list(PHASE12_RANDOMIZATION_SEEDS),
        "randomized_orders": [
            list(derive_phase12_randomized_order(seed))
            for seed in PHASE12_RANDOMIZATION_SEEDS
        ],
        "quality_execution": "LOCKED",
        "full_scan": "CLOSED",
        "performance_data_frozen": False,
        "pilot": "NOT_STARTED",
        "speedup_calculated": False,
    }


def _expected_entry_authority(
    *,
    campaign_id: str,
    execution_git_sha: str,
    repo_root: Path,
) -> dict[str, Any]:
    """Build entry authority after replaying live Git and artifact custody."""

    return _entry_authority_contract(
        campaign_id=campaign_id,
        execution_git_sha=execution_git_sha,
        historical_source_bridges=_expected_historical_source_bridges(
            repo_root
        ),
    )


def _serialized_transition_authority(
    *,
    execution_commit: str,
    current_commit: str,
    unchanged_sources: Mapping[str, tuple[str, str]],
) -> dict[str, Any]:
    endpoint_path = "src/kvbench/runtime/bf16_endpoint.py"
    return {
        "schema_version": (
            "kvbench-phase12-decision0026-transition-authority-1.0.0"
        ),
        "execution_commit": execution_commit,
        "decision": "0026",
        "decision_path": PHASE12_DECISION0026_PATH.as_posix(),
        "decision_sha256": PHASE12_DECISION0026_SHA256,
        "decision_commit": PHASE12_DECISION0026_COMMIT,
        "historical_endpoint": {
            "commit": execution_commit,
            "path": endpoint_path,
            "blob": PHASE12_HISTORICAL_ENDPOINT_BLOB,
            "sha256": PHASE12_HISTORICAL_ENDPOINT_SHA256,
        },
        "current_endpoint": {
            "commit": current_commit,
            "path": endpoint_path,
            "blob": PHASE12_CURRENT_ENDPOINT_BLOB,
            "sha256": PHASE12_CURRENT_ENDPOINT_SHA256,
        },
        "unchanged_method_sources": {
            path: {
                "commit": execution_commit,
                "path": path,
                "blob": blob,
                "sha256": sha256,
            }
            for path, (blob, sha256) in unchanged_sources.items()
        },
        "semantic_authority": (
            "existing adapters remain behaviorally and allocation equivalent"
        ),
        "recognized_transition_only": True,
    }


def _serialized_historical_source_bridges(
    execution_git_sha: str,
) -> dict[str, Any]:
    """Return the checksum-bound bridges already serialized by reservation.

    Reservation and execution validate these values against live Git and
    artifacts. Payload replay must remain self-contained so a clean retrieval
    can be validated without the historical ignored artifacts or a Git
    checkout.
    """

    return {
        "bf16": {
            "schema_version": (
                "kvbench-phase12-bf16-container-bridge-1.0.0"
            ),
            "artifacts": {
                mode: {
                    "path": expected["path"].as_posix(),
                    "root_sha256": expected["root_sha256"],
                    "manifest_sha256": expected["manifest_sha256"],
                    "result_sha256": expected["result_sha256"],
                }
                for mode, expected in PHASE12_BF16_PARITY_ARTIFACTS.items()
            },
            "transition": _serialized_transition_authority(
                execution_commit=PHASE12_BF16_PARITY_EXECUTION_COMMIT,
                current_commit=execution_git_sha,
                unchanged_sources={
                    "src/kvbench/adapters/bf16.py": (
                        PHASE12_BF16_ADAPTER_BLOB,
                        PHASE12_BF16_ADAPTER_SHA256,
                    )
                },
            ),
        },
        "turboquant": {
            "schema_version": (
                "kvbench-phase12-turboquant-transition-bridge-1.0.0"
            ),
            "transition": _serialized_transition_authority(
                execution_commit=PHASE12_TURBOQUANT_EXECUTION_COMMIT,
                current_commit=execution_git_sha,
                unchanged_sources={
                    "src/kvbench/adapters/turboquant.py": (
                        PHASE12_TURBOQUANT_ADAPTER_BLOB,
                        PHASE12_TURBOQUANT_ADAPTER_SHA256,
                    ),
                    "src/kvbench/runtime/turboquant_session.py": (
                        PHASE12_TURBOQUANT_SESSION_BLOB,
                        PHASE12_TURBOQUANT_SESSION_SHA256,
                    ),
                },
            ),
        },
    }


def _expected_serialized_entry_authority(
    *,
    campaign_id: str,
    execution_git_sha: str,
) -> dict[str, Any]:
    """Replay the exact serialized contract without live repository access."""

    return _entry_authority_contract(
        campaign_id=campaign_id,
        execution_git_sha=execution_git_sha,
        historical_source_bridges=_serialized_historical_source_bridges(
            execution_git_sha
        ),
    )


def reserve_campaign(
    *,
    campaign_id: str,
    git_sha: str,
    repo_root: Path = REPOSITORY_ROOT,
) -> Path:
    """Reserve one append-only ID and create one unique staging directory."""

    identifier = _validate_campaign_id(campaign_id)
    if tuple(
        derive_phase12_randomized_order(seed)
        for seed in PHASE12_RANDOMIZATION_SEEDS
    ) != PHASE12_RANDOMIZED_ORDERS:
        raise Phase12UnifiedAdmissionError(
            "Phase 12 randomized orders do not derive from their seeds"
        )
    if not re.fullmatch(r"[0-9a-f]{40}", git_sha):
        raise Phase12UnifiedAdmissionError("execution Git SHA is invalid")
    plan = repo_root / PHASE12_PLAN_PATH
    if (
        not plan.is_file()
        or plan.is_symlink()
        or PHASE12_PLAN_SHA256 == "TO_BE_RECORDED_AFTER_PLAN_COMMIT"
        or sha256_file(plan) != PHASE12_PLAN_SHA256
    ):
        raise Phase12UnifiedAdmissionError(
            "committed Phase 12 plan identity differs"
        )
    root = repo_root / "artifacts" / "phase12"
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise Phase12UnifiedAdmissionError("Phase 12 artifact root is unsafe")
    final = root / identifier
    reservations = root / ".kvbench-reservations"
    staging = root / ".kvbench-staging"
    reservations.mkdir(mode=0o700, exist_ok=True)
    staging.mkdir(mode=0o700, exist_ok=True)
    if any(path.is_symlink() for path in (reservations, staging)):
        raise Phase12UnifiedAdmissionError(
            "Phase 12 control directory is unsafe"
        )
    if (
        final.exists()
        or final.is_symlink()
        or any(staging.glob(f"{identifier}.*.staging"))
    ):
        raise Phase12UnifiedAdmissionError(
            "Phase 12 campaign ID was already used"
        )
    reservation = reservations / identifier
    try:
        reservation.mkdir(mode=0o500)
    except FileExistsError as error:
        raise Phase12UnifiedAdmissionError(
            "Phase 12 campaign ID is already reserved"
        ) from error
    stage = staging / f"{identifier}.{secrets.token_hex(12)}.staging"
    stage.mkdir(mode=0o700)
    prior = load_and_validate_prior_admission_evidence(repo_root)
    expected_bridges = _expected_historical_source_bridges(repo_root)
    if {
        method: prior["methods"][method]["authority_bridge"]
        for method in ("bf16", "turboquant")
    } != expected_bridges:
        raise Phase12UnifiedAdmissionError(
            "historical source authority bridges differ"
        )
    aggregation = aggregate_g1_g4(prior)
    normalized_aggregation = json.loads(
        json.dumps(aggregation, allow_nan=False)
    )
    if normalized_aggregation != _expected_entry_g1_g4():
        raise Phase12UnifiedAdmissionError(
            "Phase 12 entry aggregation differs from the frozen contract"
        )
    write_exclusive(
        stage / "unified" / "entry-g1-g4.json",
        json_bytes(aggregation),
    )
    write_exclusive(
        stage / "unified" / "entry-authority.json",
        json_bytes(
            _expected_entry_authority(
                campaign_id=identifier,
                execution_git_sha=git_sha,
                repo_root=repo_root,
            )
        ),
    )
    write_exclusive(
        stage / "campaign-reservation.json",
        json_bytes(
            {
                "schema_version": (
                    "kvbench-phase12-campaign-reservation-1.0.0"
                ),
                "campaign_id": identifier,
                "execution_git_sha": git_sha,
                "created_at_utc": _utc_now(),
                "staging_directory": stage.name,
                "reservation_directory": reservation.name,
                "append_only": True,
                "reuse_permitted": False,
            }
        ),
    )
    return stage


def _child_environment() -> dict[str, str]:
    allowed_names = (
        "PATH",
        "LD_LIBRARY_PATH",
        "PYTHONPATH",
        "PYTHONIOENCODING",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONNOUSERSITE",
        "LANG",
        "LC_ALL",
        "TZ",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES",
        "HF_HOME",
        "HF_HUB_OFFLINE",
        "HF_HUB_DISABLE_TELEMETRY",
        "TRANSFORMERS_OFFLINE",
        "TOKENIZERS_PARALLELISM",
        "CUBLAS_WORKSPACE_CONFIG",
        "TORCH_CUDA_ARCH_LIST",
        "TRITON_CACHE_DIR",
        "KVBENCH_KIVI_SOURCE_ROOT",
        "KVBENCH_KVQUANT_SOURCE_ROOT",
        "KVBENCH_KVQUANT_CALIBRATION_ROOT",
        "KVBENCH_KVQUANT_EXTENSION",
        "KVBENCH_KVQUANT_EXTENSION_SHA256",
        "KVBENCH_PHASE11DQ23_EVIDENCE_ROOT",
        "KVBENCH_AUTHORIZED_IMAGE_DIGEST",
        "KVBENCH_EXECUTION_ENVIRONMENT",
    )
    environment = {
        name: os.environ[name]
        for name in allowed_names
        if name in os.environ
    }
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(
                (
                    "/opt/kvbench/.phase3/site-packages",
                    str(REPOSITORY_ROOT / "src"),
                    str(REPOSITORY_ROOT),
                )
            ),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "TZ": "UTC",
            "CUDA_HOME": "/usr/local/cuda-13.0",
            "CC": "/usr/bin/gcc",
            "CXX": "/usr/bin/c++",
            "CUDAARCHS": "120",
            "CMAKE_CUDA_ARCHITECTURES": "120",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
    )
    if (
        any(name in environment for name in _FORBIDDEN_CHILD_ENVIRONMENT)
        or any(name in os.environ for name in _FORBIDDEN_CHILD_ENVIRONMENT)
        or environment.get("KVBENCH_AUTHORIZED_IMAGE_DIGEST")
        != PHASE12_AUTHORIZED_CONTAINER_DIGEST
        or environment.get("KVBENCH_EXECUTION_ENVIRONMENT")
        != "measurement_container"
    ):
        raise Phase12UnifiedAdmissionError(
            "container child environment violates execution or secret policy"
        )
    return environment


def _supervision_passed(result: Any) -> bool:
    record = result.to_dict()
    return bool(
        result.returncode == 0
        and result.timed_out is False
        and record["direct_child"]["verified"] is True
        and record["direct_child"]["process_handle_retained"] is True
        and record["final_reap"]["completed"] is True
        and record["final_reap"]["count"] == 1
    )


def _write_supervised_command_evidence(
    *,
    root: Path,
    prefix: str,
    result: Any,
    pre_snapshot: Mapping[str, Any],
    post_snapshot: Mapping[str, Any],
) -> None:
    write_exclusive(root / f"{prefix}.stdout.txt", result.stdout)
    write_exclusive(root / f"{prefix}.stderr.txt", result.stderr)
    write_exclusive(
        root / f"{prefix}.supervision.json",
        json_bytes(result.to_dict()),
    )
    write_exclusive(
        root / f"{prefix}.gpu-before.json",
        json_bytes(dict(pre_snapshot)),
    )
    write_exclusive(
        root / f"{prefix}.gpu-after.json",
        json_bytes(dict(post_snapshot)),
    )


def _run_container_test(stage: Path, target: str) -> None:
    if target not in {"test-cuda", "test-graph"}:
        raise Phase12UnifiedAdmissionError("unknown Phase 12 CUDA test")
    pre_snapshot = _capture_process_snapshot()
    _require_idle_snapshot(pre_snapshot)
    command = (
        "/usr/bin/make",
        f"PHASE2_PYTHON={sys.executable}",
        f"PHASE3_PYTHON={sys.executable}",
        target,
    )
    result = run_supervised_command(
        command,
        working_directory=str(REPOSITORY_ROOT),
        environment=_child_environment(),
        timeout_seconds=PHASE12_TEST_TIMEOUT_SECONDS,
    )
    post_snapshot = _capture_process_snapshot()
    _require_idle_snapshot(post_snapshot)
    evidence_root = stage / "validation" / target
    _write_supervised_command_evidence(
        root=evidence_root,
        prefix="command",
        result=result,
        pre_snapshot=pre_snapshot,
        post_snapshot=post_snapshot,
    )
    passed = _supervision_passed(result)
    write_exclusive(
        evidence_root / "verdict.json",
        json_bytes(
            {
                "schema_version": (
                    "kvbench-phase12-container-test-verdict-1.0.0"
                ),
                "target": target,
                "authorized_container_digest": (
                    PHASE12_AUTHORIZED_CONTAINER_DIGEST
                ),
                "passed": passed,
                "cuda_executed_on_native_host": False,
            }
        ),
    )
    if not passed:
        raise Phase12UnifiedAdmissionError(
            f"authorized-container {target} failed"
        )


def _parse_worker_result(stdout: bytes) -> dict[str, Any]:
    lines = stdout.decode("utf-8", errors="strict").splitlines()
    matches = [
        line[len(PHASE12_WORKER_RESULT_PREFIX) :]
        for line in lines
        if line.startswith(PHASE12_WORKER_RESULT_PREFIX)
    ]
    if len(matches) != 1:
        raise Phase12UnifiedAdmissionError(
            "worker did not emit exactly one result record"
        )
    try:
        payload = json.loads(
            matches[0],
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise Phase12UnifiedAdmissionError(
            "worker result record is invalid"
        ) from error
    if not isinstance(payload, dict):
        raise Phase12UnifiedAdmissionError("worker result is not an object")
    return payload


def _validate_worker_result(
    payload: Mapping[str, Any],
    *,
    run_id: str,
    configuration: str,
    replicate_index: int,
    seed: int,
    order_index: int,
    git_sha: str,
) -> None:
    expected = {
        "schema_version": PHASE12_RUN_SCHEMA,
        "run_id": run_id,
        "method_config_id": configuration,
        "method_config_fingerprint": EXPECTED_CONFIG_FINGERPRINTS[
            configuration
        ],
        "method_family": _method_family(configuration),
        "replicate_index": replicate_index,
        "seed": seed,
        "order_index": order_index,
        "execution_git_sha": git_sha,
        "authorized_container_digest": PHASE12_AUTHORIZED_CONTAINER_DIGEST,
        "gpu_uuid": PHASE12_GPU_UUID,
        "model_id": PHASE12_MODEL_ID,
        "model_revision": PHASE12_MODEL_REVISION,
        "tokenizer_id": PHASE12_TOKENIZER_ID,
        "tokenizer_revision": PHASE12_TOKENIZER_REVISION,
        "input_recipe_sha256": PHASE12_INPUT_RECIPE_SHA256,
        "kernel_path_witness_kind": (
            "existing_admission_plus_observed_cuda_graph_topology"
        ),
        "finite_output": True,
        "no_backend_fallback": True,
        "allocation_stable": True,
        "kernel_path_stable": True,
        "gpu_exclusive": True,
        "speedup_calculated": False,
        "quality_status": "unvalidated",
        "claim_eligibility": "performance_only",
        "performance_claim_eligible": False,
        "measurement_scope": "measurement_container_admission",
        "r_hbm": None,
        "container_runtime_attestation": (
            _expected_container_runtime_attestation()
        ),
        "warmup_replays": PHASE12_WARMUP_STEPS,
        "measured_steps": PHASE12_MEASURED_STEPS,
        "measured_batches": PHASE12_MEASURED_BATCHES,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise Phase12UnifiedAdmissionError("worker authority or verdict differs")
    for key in (
        "operation_fingerprint_sha256",
        "runtime_adapter_fingerprint",
        "cache_layout_fingerprint",
        "output_checksum",
        "kernel_path_fingerprint",
        "allocation_fingerprint",
    ):
        _require_sha256(payload.get(key), f"worker {key}")
    path_observation = payload.get("kernel_path_observation")
    if (
        not isinstance(path_observation, Mapping)
        or set(path_observation)
        != {
            "schema_version",
            "observation_kind",
            "before",
            "after",
            "normalized_sha256",
            "node_count",
            "kernel_node_count",
            "edge_count",
            "graph_exec_pointer_stable",
            "topology_stable_within_process",
            "timing_collected",
            "profiler_used",
        }
        or path_observation.get("schema_version")
        != "kvbench-phase12-cuda-graph-path-observation-1.0.0"
        or path_observation.get("observation_kind")
        != "cuda_graph_debug_dot"
        or type(path_observation.get("node_count")) is not int
        or type(path_observation.get("kernel_node_count")) is not int
        or type(path_observation.get("edge_count")) is not int
        or path_observation["node_count"] <= 0
        or path_observation["kernel_node_count"] <= 0
        or path_observation["kernel_node_count"]
        > path_observation["node_count"]
        or path_observation["edge_count"] < 0
        or path_observation.get("graph_exec_pointer_stable") is not True
        or path_observation.get("topology_stable_within_process")
        is not True
        or path_observation.get("timing_collected") is not False
        or path_observation.get("profiler_used") is not False
    ):
        raise Phase12UnifiedAdmissionError(
            "worker CUDA Graph path observation differs"
        )
    _require_sha256(
        path_observation.get("normalized_sha256"),
        "worker normalized CUDA Graph path",
    )
    expected_phase_keys = {
        "schema_version",
        "phase",
        "observation_kind",
        "raw_path",
        "raw_sha256",
        "normalized_path",
        "normalized_sha256",
        "node_count",
        "kernel_node_count",
        "edge_count",
        "process_local_handles_normalized",
        "timing_collected",
        "profiler_used",
    }
    for phase in ("before", "after"):
        record = path_observation.get(phase)
        if (
            not isinstance(record, Mapping)
            or set(record) != expected_phase_keys
            or record.get("schema_version")
            != "kvbench-phase12-cuda-graph-path-witness-1.0.0"
            or record.get("phase") != phase
            or record.get("observation_kind")
            != "cuda_graph_debug_dot"
            or record.get("raw_path")
            != f"kernel-path.{phase}.raw.dot"
            or record.get("normalized_path")
            != f"kernel-path.{phase}.normalized.dot"
            or record.get("normalized_sha256")
            != path_observation["normalized_sha256"]
            or record.get("node_count") != path_observation["node_count"]
            or record.get("kernel_node_count")
            != path_observation["kernel_node_count"]
            or record.get("edge_count") != path_observation["edge_count"]
            or record.get("process_local_handles_normalized") is not True
            or record.get("timing_collected") is not False
            or record.get("profiler_used") is not False
        ):
            raise Phase12UnifiedAdmissionError(
                "worker CUDA Graph phase witness differs"
            )
        _require_sha256(
            record.get("raw_sha256"),
            f"worker {phase} raw CUDA Graph path",
        )
    _require_sha256(
        payload.get("runtime_backend_fingerprint"),
        "worker runtime backend",
    )
    path_audit = payload.get("execution_path_audit")
    prior_g3_binding = payload.get("prior_g3_binding")
    if isinstance(path_audit, Mapping):
        _validate_execution_path_record(
            path_audit,
            family=_method_family(configuration),
        )
    if isinstance(prior_g3_binding, Mapping):
        _validate_prior_g3_binding(
            prior_g3_binding,
            family=_method_family(configuration),
        )
    if (
        payload.get("input_recipe") != PHASE12_INPUT_RECIPE
        or payload.get("setup_warmups")
        != PHASE12_SETUP_WARMUPS[_method_family(configuration)]
        or not isinstance(payload.get("runner"), Mapping)
        or not isinstance(payload.get("runtime_backend_id"), str)
        or not payload.get("runtime_backend_id")
        or not isinstance(path_audit, Mapping)
        or not isinstance(prior_g3_binding, Mapping)
        or not isinstance(payload.get("graph_replay_allocation"), Mapping)
        or payload["graph_replay_allocation"].get("passed") is not True
        or payload["graph_replay_allocation"].get(
            "allocation_event_count"
        )
        != 0
        or payload["graph_replay_allocation"].get("allocated_delta") != 0
        or payload["graph_replay_allocation"].get("reserved_delta") != 0
    ):
        raise Phase12UnifiedAdmissionError(
            "worker graph or allocation evidence differs"
        )
    runner = payload["runner"]
    graph = runner.get("graph")
    geometry = runner.get("gqa_cache_geometry")
    family = _method_family(configuration)
    if (
        not isinstance(graph, Mapping)
        or not isinstance(geometry, Mapping)
        or not _gqa_geometry_passes(geometry, family=family)
        or runner.get("adapter_config_fingerprint")
        != payload.get("runtime_adapter_fingerprint")
        or runner.get("cache_layout_fingerprint")
        != payload.get("cache_layout_fingerprint")
    ):
        raise Phase12UnifiedAdmissionError(
            "worker runtime path witness differs"
        )
    expected_kernel_path_fingerprint = _structural_fingerprint(
        {
            "schema_version": (
                "kvbench-phase12-kernel-path-fingerprint-1.0.0"
            ),
            "configuration": configuration,
            "method": family,
            "prior_method_admission_sha256": (
                EXPECTED_REPORT_SHA256S[family]
            ),
            "prior_g3_binding": dict(prior_g3_binding),
            "container_runtime_attestation": (
                _expected_container_runtime_attestation()
            ),
            "operation_fingerprint_sha256": payload[
                "operation_fingerprint_sha256"
            ],
            "runtime_adapter_fingerprint": payload[
                "runtime_adapter_fingerprint"
            ],
            "cache_layout_fingerprint": payload[
                "cache_layout_fingerprint"
            ],
            "backend_id": payload["runtime_backend_id"],
            "backend_fingerprint": payload[
                "runtime_backend_fingerprint"
            ],
            "observed_execution_path_audit": dict(path_audit),
            "graph_mode": PHASE12_GRAPH_MODE,
            "graph_captured": graph.get("captured"),
            "graph_fallback": graph.get("fallback"),
            "graph_replay_exact": graph.get(
                "consecutive_replay_outputs_exact"
            ),
            "native_kv_head_storage": True,
            "observed_cuda_graph_normalized_sha256": (
                path_observation["normalized_sha256"]
            ),
            "observed_cuda_graph_node_count": (
                path_observation["node_count"]
            ),
            "observed_cuda_graph_kernel_node_count": (
                path_observation["kernel_node_count"]
            ),
            "observed_cuda_graph_edge_count": (
                path_observation["edge_count"]
            ),
            "witness_kind": (
                "existing_admission_plus_observed_cuda_graph_topology"
            ),
        }
    )
    if (
        payload.get("kernel_path_fingerprint")
        != expected_kernel_path_fingerprint
    ):
        raise Phase12UnifiedAdmissionError(
            "worker kernel-path fingerprint does not replay"
        )
    numeric = (
        "process_median_ms",
        "temperature_min_c",
        "temperature_max_c",
        "power_min_w",
        "power_max_w",
    )
    if any(
        not isinstance(payload.get(key), (int, float))
        or isinstance(payload.get(key), bool)
        or not math.isfinite(float(payload[key]))
        for key in numeric
    ):
        raise Phase12UnifiedAdmissionError("worker numeric evidence is invalid")


def _run_one_process(
    *,
    stage: Path,
    campaign_id: str,
    configuration: str,
    replicate_index: int,
    seed: int,
    order_index: int,
    git_sha: str,
) -> tuple[dict[str, Any], str, str]:
    run_id = (
        f"{campaign_id}-r{replicate_index}-{order_index:02d}-"
        f"{configuration}"
    )
    if _RUN_ID_RE.fullmatch(run_id) is None:
        raise Phase12UnifiedAdmissionError("Phase 12 run ID is invalid")
    run_root = stage / "runs" / run_id
    try:
        run_root.mkdir(parents=True)
    except FileExistsError as error:
        raise Phase12UnifiedAdmissionError(
            "Phase 12 run ID already exists"
        ) from error
    write_exclusive(
        run_root / "started.json",
        json_bytes(
            {
                "schema_version": "kvbench-phase12-run-start-1.0.0",
                "run_id": run_id,
                "campaign_id": campaign_id,
                "method_config_id": configuration,
                "replicate_index": replicate_index,
                "seed": seed,
                "order_index": order_index,
                "started_at_utc": _utc_now(),
            }
        ),
    )
    pre_snapshot = _capture_process_snapshot()
    _require_idle_snapshot(pre_snapshot)
    command = (
        sys.executable,
        str(REPOSITORY_ROOT / "scripts" / "phase12_unified_admission.py"),
        "--run-worker",
        "--run-id",
        run_id,
        "--configuration",
        configuration,
        "--replicate-index",
        str(replicate_index),
        "--seed",
        str(seed),
        "--order-index",
        str(order_index),
        "--git-sha",
        git_sha,
        "--run-artifact-root",
        str(run_root),
    )
    result = run_supervised_command(
        command,
        working_directory=str(REPOSITORY_ROOT),
        environment=_child_environment(),
        timeout_seconds=PHASE12_CHILD_TIMEOUT_SECONDS,
    )
    post_snapshot = _capture_process_snapshot()
    _require_idle_snapshot(post_snapshot)
    _write_supervised_command_evidence(
        root=run_root,
        prefix="worker",
        result=result,
        pre_snapshot=pre_snapshot,
        post_snapshot=post_snapshot,
    )
    if not _supervision_passed(result):
        write_exclusive(
            run_root / "failure.json",
            json_bytes(
                {
                    "schema_version": "kvbench-phase12-run-failure-1.0.0",
                    "run_id": run_id,
                    "failure_reason": "supervised_worker_failed",
                    "selective_retry_permitted": False,
                    "campaign_preserved": True,
                }
            ),
        )
        raise Phase12UnifiedAdmissionError(
            f"Phase 12 worker failed: {run_id}"
        )
    payload = _parse_worker_result(result.stdout)
    _validate_worker_result(
        payload,
        run_id=run_id,
        configuration=configuration,
        replicate_index=replicate_index,
        seed=seed,
        order_index=order_index,
        git_sha=git_sha,
    )
    path_observation = payload["kernel_path_observation"]
    path_files: dict[str, Path] = {}
    for phase in ("before", "after"):
        record = path_observation[phase]
        raw_path = run_root / str(record["raw_path"])
        normalized_path = run_root / str(record["normalized_path"])
        try:
            raw = raw_path.read_bytes()
            normalized, nodes, kernels, edges = (
                _normalize_cuda_graph_debug_dot(raw)
            )
        except OSError as error:
            raise Phase12UnifiedAdmissionError(
                "worker CUDA Graph path evidence is absent"
            ) from error
        if (
            hashlib.sha256(raw).hexdigest() != record["raw_sha256"]
            or normalized_path.read_bytes() != normalized
            or sha256_file(normalized_path)
            != record["normalized_sha256"]
            or (nodes, kernels, edges)
            != (
                record["node_count"],
                record["kernel_node_count"],
                record["edge_count"],
            )
        ):
            raise Phase12UnifiedAdmissionError(
                "worker CUDA Graph path evidence does not replay"
            )
        path_files[f"{phase}_raw"] = raw_path
        path_files[f"{phase}_normalized"] = normalized_path
    write_exclusive(run_root / "result.json", json_bytes(dict(payload)))
    manifest = {
        "schema_version": "kvbench-phase12-g5-run-manifest-1.0.0",
        "run_id": run_id,
        "campaign_id": campaign_id,
        "status": "completed",
        "method_config_id": configuration,
        "method_config_fingerprint": EXPECTED_CONFIG_FINGERPRINTS[
            configuration
        ],
        "replicate_index": replicate_index,
        "seed": seed,
        "order_index": order_index,
        "execution_git_sha": git_sha,
        "authorized_container_digest": PHASE12_AUTHORIZED_CONTAINER_DIGEST,
        "runner_kind": PHASE12_RUNNER_KIND,
        "graph_mode": PHASE12_GRAPH_MODE,
        "batch_size": PHASE12_BATCH_SIZE,
        "context_length": PHASE12_CONTEXT_LENGTH,
        "warmup_steps": PHASE12_WARMUP_STEPS,
        "measured_steps": PHASE12_MEASURED_STEPS,
        "measured_batches": PHASE12_MEASURED_BATCHES,
        "result_path": f"runs/{run_id}/result.json",
        "result_sha256": sha256_file(run_root / "result.json"),
        "stdout_sha256": sha256_file(run_root / "worker.stdout.txt"),
        "stderr_sha256": sha256_file(run_root / "worker.stderr.txt"),
        "supervision_sha256": sha256_file(
            run_root / "worker.supervision.json"
        ),
        "gpu_before_sha256": sha256_file(
            run_root / "worker.gpu-before.json"
        ),
        "gpu_after_sha256": sha256_file(
            run_root / "worker.gpu-after.json"
        ),
        "kernel_path_before_raw_path": (
            f"runs/{run_id}/kernel-path.before.raw.dot"
        ),
        "kernel_path_before_raw_sha256": sha256_file(
            path_files["before_raw"]
        ),
        "kernel_path_before_normalized_path": (
            f"runs/{run_id}/kernel-path.before.normalized.dot"
        ),
        "kernel_path_before_normalized_sha256": sha256_file(
            path_files["before_normalized"]
        ),
        "kernel_path_after_raw_path": (
            f"runs/{run_id}/kernel-path.after.raw.dot"
        ),
        "kernel_path_after_raw_sha256": sha256_file(
            path_files["after_raw"]
        ),
        "kernel_path_after_normalized_path": (
            f"runs/{run_id}/kernel-path.after.normalized.dot"
        ),
        "kernel_path_after_normalized_sha256": sha256_file(
            path_files["after_normalized"]
        ),
        "quality_status": "unvalidated",
        "claim_eligibility": "performance_only",
        "performance_claim_eligible": False,
        "measurement_scope": "measurement_container_admission",
        "speedup_calculated": False,
        "r_hbm": None,
        "selective_rerun": False,
    }
    write_exclusive(run_root / "manifest.json", json_bytes(manifest))
    return (
        dict(payload),
        f"runs/{run_id}/manifest.json",
        sha256_file(run_root / "manifest.json"),
    )


def _required_breakdown(
    breakdown: Mapping[str, Any],
    expected_keys: set[str],
) -> dict[str, int]:
    if set(breakdown) != expected_keys or any(
        type(value) is not int or value < 0
        for value in breakdown.values()
    ):
        raise Phase12UnifiedAdmissionError(
            "method byte breakdown differs from its admitted schema"
        )
    return {key: int(value) for key, value in breakdown.items()}


def _phase12_byte_accounting(
    configuration: str,
    worker: Mapping[str, Any],
) -> Phase12ByteAccounting:
    runner = worker.get("runner")
    if not isinstance(runner, Mapping):
        raise Phase12UnifiedAdmissionError("worker runner evidence is absent")
    accounting = runner.get("cache_accounting")
    raw_breakdown = runner.get("cache_byte_breakdown")
    if not isinstance(accounting, Mapping) or not isinstance(
        raw_breakdown, Mapping
    ):
        raise Phase12UnifiedAdmissionError(
            "worker accounting evidence is absent"
        )
    family = _method_family(configuration)
    common_accounting_fields = {
        "active_context",
        "allocated_bytes",
        "capacity",
        "measured_tensor_bytes",
        "model_baseline_allocated_bytes",
        "padding_bytes",
        "predicted_tensor_bytes",
        "workspace_bytes",
    }
    extra_accounting_fields = {
        "bf16": set(),
        "turboquant": {
            "rounded_capacity",
            "temporary_peak_bytes",
        },
        "kivi": {
            "active_storage_bytes",
            "logical_bf16_active_bytes",
            "logical_bf16_allocated_bytes",
            "temporary_peak_bytes",
        },
        "kvquant": {
            "active_storage_bytes",
            "endpoint_rope_scratch_bytes",
            "key_active_entries",
            "logical_bf16_active_bytes",
            "logical_bf16_allocated_bytes",
            "relative_error",
            "staging_bytes",
            "temporary_peak_bytes",
        },
    }
    expected_accounting_fields = (
        common_accounting_fields | extra_accounting_fields[family]
    )
    if set(accounting) != expected_accounting_fields:
        raise Phase12UnifiedAdmissionError(
            "runner accounting fields differ from the frozen method contract"
        )
    integer_accounting = {
        key: value
        for key, value in accounting.items()
        if key != "relative_error"
    }
    if any(
        type(value) is not int or value < 0
        for value in integer_accounting.values()
    ):
        raise Phase12UnifiedAdmissionError(
            "runner accounting integers are invalid"
        )
    if (
        accounting["active_context"] != PHASE12_CONTEXT_LENGTH
        or accounting["capacity"] != PHASE12_CONTEXT_LENGTH + 1
        or accounting["model_baseline_allocated_bytes"] <= 0
        or accounting["predicted_tensor_bytes"] <= 0
        or accounting["measured_tensor_bytes"] <= 0
        or accounting["allocated_bytes"] <= 0
    ):
        raise Phase12UnifiedAdmissionError(
            "runner accounting geometry or totals differ"
        )
    if family == "bf16":
        breakdown = _required_breakdown(
            raw_breakdown,
            {
                "data_bytes",
                "workspace_bytes",
                "padding_bytes",
                "scale_bytes",
                "zero_point_bytes",
                "metadata_bytes",
            },
        )
        data = breakdown["data_bytes"]
        metadata = (
            breakdown["metadata_bytes"]
            + breakdown["scale_bytes"]
            + breakdown["zero_point_bytes"]
        )
        sparse = 0
        sink_residual = 0
        padding = breakdown["padding_bytes"]
        workspace = breakdown["workspace_bytes"]
    elif family == "turboquant":
        breakdown = _required_breakdown(
            raw_breakdown,
            {
                "block_rounding_overhead_bytes",
                "compressed_key_payload_bytes",
                "compressed_value_payload_bytes",
                "key_norm_metadata_bytes",
                "mapping_metadata_bytes",
                "persistent_workspace_bytes",
                "skipped_layer_bf16_key_bytes",
                "skipped_layer_bf16_value_bytes",
                "slot_padding_alignment_bytes",
                "value_scale_metadata_bytes",
                "value_zero_point_metadata_bytes",
            },
        )
        data = sum(
            breakdown[key]
            for key in (
                "compressed_key_payload_bytes",
                "compressed_value_payload_bytes",
                "skipped_layer_bf16_key_bytes",
                "skipped_layer_bf16_value_bytes",
            )
        )
        metadata = sum(
            breakdown[key]
            for key in (
                "key_norm_metadata_bytes",
                "mapping_metadata_bytes",
                "value_scale_metadata_bytes",
                "value_zero_point_metadata_bytes",
            )
        )
        sparse = 0
        sink_residual = 0
        padding = (
            breakdown["block_rounding_overhead_bytes"]
            + breakdown["slot_padding_alignment_bytes"]
        )
        workspace = breakdown["persistent_workspace_bytes"]
    elif family == "kivi":
        breakdown = _required_breakdown(
            raw_breakdown,
            {
                "block_group_rounding_bytes",
                "fp16_staging",
                "key_scales",
                "key_zero_points",
                "other_metadata",
                "padding_alignment",
                "persistent_workspace",
                "quantization_staging",
                "quantized_k_payload",
                "quantized_v_payload",
                "residual_k",
                "residual_v",
                "value_rollover_shift_scratch",
                "value_scales",
                "value_zero_points",
            },
        )
        data = breakdown["quantized_k_payload"] + breakdown[
            "quantized_v_payload"
        ]
        metadata = sum(
            breakdown[key]
            for key in (
                "key_scales",
                "key_zero_points",
                "value_scales",
                "value_zero_points",
                "other_metadata",
            )
        )
        sparse = 0
        sink_residual = breakdown["residual_k"] + breakdown["residual_v"]
        padding = (
            breakdown["block_group_rounding_bytes"]
            + breakdown["padding_alignment"]
        )
        workspace = sum(
            breakdown[key]
            for key in (
                "fp16_staging",
                "persistent_workspace",
                "quantization_staging",
                "value_rollover_shift_scratch",
            )
        )
    else:
        breakdown = _required_breakdown(
            raw_breakdown,
            {
                "active_count_mask",
                "dense_k_payload",
                "dense_v_payload",
                "key_metadata",
                "key_sparse_indices",
                "key_sparse_values",
                "padding_alignment",
                "persistent_workspace",
                "sink_k",
                "sink_v",
                "staging",
                "value_metadata",
                "value_sparse_indices",
                "value_sparse_values",
            },
        )
        data = breakdown["dense_k_payload"] + breakdown["dense_v_payload"]
        metadata = (
            breakdown["key_metadata"]
            + breakdown["value_metadata"]
            + breakdown["active_count_mask"]
        )
        sparse = sum(
            breakdown[key]
            for key in (
                "key_sparse_indices",
                "key_sparse_values",
                "value_sparse_indices",
                "value_sparse_values",
            )
        )
        sink_residual = breakdown["sink_k"] + breakdown["sink_v"]
        padding = breakdown["padding_alignment"]
        workspace = (
            breakdown["persistent_workspace"] + breakdown["staging"]
        )
    categories = (
        data,
        metadata,
        sparse,
        sink_residual,
        padding,
        workspace,
    )
    allocated = int(accounting["allocated_bytes"])
    predicted = int(accounting["predicted_tensor_bytes"])
    measured = int(accounting["measured_tensor_bytes"])
    owned_breakdown = sum(categories)
    logical = accounting.get("logical_bf16_allocated_bytes")
    if logical is None:
        logical = (
            2
            * 32
            * PHASE12_BATCH_SIZE
            * 8
            * (PHASE12_CONTEXT_LENGTH + 1)
            * 128
            * 2
        )
    measured_replays = (
        measured + workspace == allocated
        and predicted + padding == measured
        and accounting["workspace_bytes"] == workspace
        if family == "bf16"
        else measured == allocated
    )
    accounting_padding = (
        breakdown["padding_alignment"]
        if family == "kivi"
        else padding
    )
    if (
        type(logical) is not int
        or logical <= 0
        or owned_breakdown != allocated
        or not measured_replays
        or accounting["padding_bytes"] != accounting_padding
    ):
        raise Phase12UnifiedAdmissionError(
            "Phase 12 owned-byte replay differs"
        )
    relative_error = abs(predicted - allocated) / allocated
    if (
        relative_error >= 0.01
        or (
            family == "kvquant"
            and (
                not isinstance(accounting["relative_error"], (int, float))
                or isinstance(accounting["relative_error"], bool)
                or not math.isfinite(float(accounting["relative_error"]))
                or not math.isclose(
                    float(accounting["relative_error"]),
                    relative_error,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            )
        )
    ):
        raise Phase12UnifiedAdmissionError(
            "Phase 12 independent byte prediction differs by at least 1%"
        )
    rho = allocated / logical
    r_alloc = logical / allocated
    return Phase12ByteAccounting(
        data_payload_bytes=data,
        metadata_bytes=metadata,
        sparse_bytes=sparse,
        sink_residual_bytes=sink_residual,
        padding_bytes=padding,
        workspace_bytes=workspace,
        predicted_allocated_bytes=predicted,
        allocated_bytes=allocated,
        logical_bf16_bytes=logical,
        rho_alloc=rho,
        r_alloc=r_alloc,
        predicted_relative_error=relative_error,
        r_hbm=None,
    )


def _configuration_admission(
    configuration: str,
    *,
    prior: Mapping[str, Any],
    worker: Mapping[str, Any],
    entry_authority_sha256: str,
) -> Phase12ConfigurationAdmission:
    method = _method_family(configuration)
    report_path = PRIOR_ADMISSION_REPORT_BINDINGS[method].as_posix()
    report_sha256 = EXPECTED_REPORT_SHA256S[method]
    _require_sha256(
        entry_authority_sha256,
        "Phase 12 entry authority",
    )
    gate_records: list[Phase12PriorGateEvidence] = []
    for gate in ("G1", "G2", "G3", "G4"):
        evidence = [
            Phase12EvidenceReference(
                evidence_id=(
                    f"{configuration}_{gate.lower()}_method_admission"
                ),
                path=report_path,
                sha256=report_sha256,
            )
        ]
        if method in {"bf16", "turboquant"}:
            evidence.append(
                Phase12EvidenceReference(
                    evidence_id=(
                        f"{configuration}_{gate.lower()}_"
                        "historical_source_bridge"
                    ),
                    path="unified/entry-authority.json",
                    sha256=entry_authority_sha256,
                )
            )
        gate_records.append(
            Phase12PriorGateEvidence(
                gate=gate,  # type: ignore[arg-type]
                status=GateDisposition.PASS,
                criteria_satisfied=True,
                evidence=tuple(evidence),
            )
        )
    gates = tuple(gate_records)
    normalized = prior["configurations"][configuration]
    if (
        normalized["method"] != method
        or normalized["fingerprint"]
        != EXPECTED_CONFIG_FINGERPRINTS[configuration]
        or normalized["gates"]
        != {"G1": "PASS", "G2": "PASS", "G3": "PASS", "G4": "PASS"}
    ):
        raise Phase12UnifiedAdmissionError(
            "normalized prior gate evidence differs"
        )
    return Phase12ConfigurationAdmission(
        method_config_id=configuration,
        method_config_fingerprint=EXPECTED_CONFIG_FINGERPRINTS[
            configuration
        ],
        prior_gates=gates,
        byte_accounting=_phase12_byte_accounting(
            configuration,
            worker,
        ),
        no_fallback=True,
        speedup_calculated=False,
    )


def _compact_g5_run(
    *,
    payload: Mapping[str, Any],
    manifest_path: str,
    manifest_sha256: str,
) -> Phase12G5Run:
    return Phase12G5Run(
        run_id=str(payload["run_id"]),
        method_config_id=str(payload["method_config_id"]),
        replicate_index=int(payload["replicate_index"]),
        seed=int(payload["seed"]),
        order_index=int(payload["order_index"]),
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        process_median_ms=float(payload["process_median_ms"]),
        output_checksum=str(payload["output_checksum"]),
        kernel_path_fingerprint=str(payload["kernel_path_fingerprint"]),
        allocation_fingerprint=str(payload["allocation_fingerprint"]),
        temperature_min_c=float(payload["temperature_min_c"]),
        temperature_max_c=float(payload["temperature_max_c"]),
        sm_clock_min_mhz=int(payload["sm_clock_min_mhz"]),
        sm_clock_max_mhz=int(payload["sm_clock_max_mhz"]),
        memory_clock_min_mhz=int(payload["memory_clock_min_mhz"]),
        memory_clock_max_mhz=int(payload["memory_clock_max_mhz"]),
        power_min_w=float(payload["power_min_w"]),
        power_max_w=float(payload["power_max_w"]),
        finite_output=True,
        no_backend_fallback=bool(payload["no_backend_fallback"]),
        allocation_stable=bool(payload["allocation_stable"]),
        kernel_path_stable=bool(payload["kernel_path_stable"]),
        gpu_exclusive=bool(payload["gpu_exclusive"]),
        speedup_calculated=bool(payload["speedup_calculated"]),
    )


def _g5_statistics(
    configuration: str,
    runs: tuple[Phase12G5Run, ...],
) -> Phase12G5Statistics:
    matching = tuple(
        item for item in runs if item.method_config_id == configuration
    )
    if len(matching) != PHASE12_REPLICATES:
        raise Phase12UnifiedAdmissionError(
            "G5 configuration lacks exactly three processes"
        )
    medians = tuple(item.process_median_ms for item in matching)
    mean = statistics.mean(medians)
    deviation = statistics.stdev(medians)
    cv = deviation / mean
    output_agreement = len({item.output_checksum for item in matching}) == 1
    path_agreement = (
        len({item.kernel_path_fingerprint for item in matching}) == 1
    )
    allocation_agreement = (
        len({item.allocation_fingerprint for item in matching}) == 1
    )
    disposition = (
        Phase12G5Disposition.PASS
        if cv <= PHASE12_CV_THRESHOLD
        and output_agreement
        and path_agreement
        and allocation_agreement
        else Phase12G5Disposition.UNSTABLE
    )
    return Phase12G5Statistics(
        method_config_id=configuration,
        run_ids=tuple(item.run_id for item in matching),
        process_medians_ms=medians,
        median_ms=float(statistics.median(medians)),
        minimum_ms=min(medians),
        maximum_ms=max(medians),
        mean_ms=mean,
        standard_deviation_ms=deviation,
        coefficient_of_variation=cv,
        temperature_min_c=min(item.temperature_min_c for item in matching),
        temperature_max_c=max(item.temperature_max_c for item in matching),
        sm_clock_min_mhz=min(item.sm_clock_min_mhz for item in matching),
        sm_clock_max_mhz=max(item.sm_clock_max_mhz for item in matching),
        memory_clock_min_mhz=min(
            item.memory_clock_min_mhz for item in matching
        ),
        memory_clock_max_mhz=max(
            item.memory_clock_max_mhz for item in matching
        ),
        power_min_w=min(item.power_min_w for item in matching),
        power_max_w=max(item.power_max_w for item in matching),
        output_checksum_agreement=output_agreement,
        kernel_path_agreement=path_agreement,
        allocation_agreement=allocation_agreement,
        disposition=disposition,
    )


def _local_global_gates(
    statistics_records: tuple[Phase12G5Statistics, ...],
) -> Phase12GlobalGates:
    all_local_pass = all(
        item.disposition is Phase12G5Disposition.PASS
        for item in statistics_records
    )
    return Phase12GlobalGates(
        g0=GateDisposition.PASS,
        g1=GateDisposition.PASS,
        g2=GateDisposition.PASS,
        g3=GateDisposition.PASS,
        g4=GateDisposition.PASS,
        g5=(
            GateDisposition.NOT_EVALUATED
            if all_local_pass
            else GateDisposition.FAIL
        ),
        pilot_state="NOT_READY",
        full_scan_state="CLOSED",
        quality_execution=QualityExecutionState.LOCKED,
        performance_data_frozen=False,
    )


def run_campaign(
    *,
    stage: Path,
    campaign_id: str,
    git_sha: str,
) -> dict[str, Any]:
    """Run the container tests and exact ordered 30-process matrix."""

    identifier = _validate_campaign_id(campaign_id)
    _require_authorized_container_runtime()
    if (
        any(name in os.environ for name in _FORBIDDEN_CHILD_ENVIRONMENT)
    ):
        raise Phase12UnifiedAdmissionError(
            "Phase 12 coordinator is outside the authorized container"
        )
    if not re.fullmatch(r"[0-9a-f]{40}", git_sha):
        raise Phase12UnifiedAdmissionError("execution Git SHA is invalid")
    resolved_stage = stage.resolve(strict=True)
    if (
        resolved_stage.is_symlink()
        or resolved_stage.name.split(".", 1)[0] != identifier
        or not (resolved_stage / "campaign-reservation.json").is_file()
        or any((resolved_stage / name).exists() for name in _CONTROL_FILES)
    ):
        raise Phase12UnifiedAdmissionError(
            "Phase 12 staging identity differs"
        )
    git_head = subprocess.run(
        ("/usr/bin/git", "rev-parse", "HEAD"),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    git_status = subprocess.run(
        (
            "/usr/bin/git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    plan_probe = subprocess.run(
        ("/usr/bin/git", "cat-file", "-e", f"HEAD:{PHASE12_PLAN_PATH}"),
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
    )
    if (
        git_head != git_sha
        or git_status
        or plan_probe.returncode != 0
        or sha256_file(REPOSITORY_ROOT / PHASE12_PLAN_PATH)
        != PHASE12_PLAN_SHA256
    ):
        raise Phase12UnifiedAdmissionError(
            "container source or committed plan identity differs"
        )
    entry = _strict_json(resolved_stage / "unified" / "entry-g1-g4.json")
    if entry != _expected_entry_g1_g4():
        raise Phase12UnifiedAdmissionError(
            "host-validated entry aggregation differs"
        )
    entry_authority = _strict_json(
        resolved_stage / "unified" / "entry-authority.json"
    )
    if entry_authority != _expected_entry_authority(
        campaign_id=identifier,
        execution_git_sha=git_sha,
        repo_root=REPOSITORY_ROOT,
    ):
        raise Phase12UnifiedAdmissionError(
            "host-validated entry authority differs"
        )

    _run_container_test(resolved_stage, "test-cuda")
    _run_container_test(resolved_stage, "test-graph")

    compact_runs: list[Phase12G5Run] = []
    workers: dict[str, list[dict[str, Any]]] = {
        configuration: [] for configuration in MAIN_CONFIG_IDS
    }
    for replicate_index, (seed, order) in enumerate(
        zip(
            PHASE12_RANDOMIZATION_SEEDS,
            PHASE12_RANDOMIZED_ORDERS,
            strict=True,
        )
    ):
        for order_index, configuration in enumerate(order):
            payload, manifest_path, manifest_sha256 = _run_one_process(
                stage=resolved_stage,
                campaign_id=identifier,
                configuration=configuration,
                replicate_index=replicate_index,
                seed=seed,
                order_index=order_index,
                git_sha=git_sha,
            )
            workers[configuration].append(payload)
            compact_runs.append(
                _compact_g5_run(
                    payload=payload,
                    manifest_path=manifest_path,
                    manifest_sha256=manifest_sha256,
                )
            )
    run_tuple = tuple(compact_runs)
    statistics_records = tuple(
        _g5_statistics(configuration, run_tuple)
        for configuration in MAIN_CONFIG_IDS
    )
    configurations = tuple(
        _configuration_admission(
            configuration,
            prior=entry,
            worker=workers[configuration][0],
            entry_authority_sha256=sha256_file(
                resolved_stage / "unified" / "entry-authority.json"
            ),
        )
        for configuration in MAIN_CONFIG_IDS
    )
    randomized_orders = tuple(
        Phase12RandomizedOrder(
            replicate_index=index,
            seed=seed,
            configurations=PHASE12_RANDOMIZED_ORDERS[index],
        )
        for index, seed in enumerate(PHASE12_RANDOMIZATION_SEEDS)
    )
    local_report = Phase12UnifiedAdmissionReport(
        schema_version=Phase12UnifiedAdmissionReport.SCHEMA_VERSION,
        created_at_utc=_utc_now(),
        campaign_id=identifier,
        execution_git_sha=git_sha,
        authorized_container_digest=PHASE12_AUTHORIZED_CONTAINER_DIGEST,
        model_id=PHASE12_MODEL_ID,
        model_revision=PHASE12_MODEL_REVISION,
        tokenizer_id=PHASE12_TOKENIZER_ID,
        tokenizer_revision=PHASE12_TOKENIZER_REVISION,
        runner_kind=PHASE12_RUNNER_KIND,
        graph_mode=PHASE12_GRAPH_MODE,
        batch_size=PHASE12_BATCH_SIZE,
        context_length=PHASE12_CONTEXT_LENGTH,
        warmup_steps=PHASE12_WARMUP_STEPS,
        measured_steps=PHASE12_MEASURED_STEPS,
        measured_batches=PHASE12_MEASURED_BATCHES,
        independent_process_replicates=PHASE12_REPLICATES,
        cv_threshold=float(PHASE12_CV_THRESHOLD),
        configurations=configurations,
        excluded_configurations=tuple(
            Phase12ExcludedConfiguration(
                method_config_id=configuration,
                reason="validation_only_control",
            )
            for configuration in HELD_OUT_CONFIG_IDS
        ),
        randomized_orders=randomized_orders,
        runs=run_tuple,
        g5_statistics=statistics_records,
        publication_state=Phase12PublicationState.PENDING,
        publication_receipt=None,
        published_root_sha256=None,
        r2_uri=None,
        object_count=None,
        complete_last=False,
        clean_retrieval=False,
        gates=_local_global_gates(statistics_records),
        speedup_calculated=False,
    )
    for configuration, admission, summary in zip(
        MAIN_CONFIG_IDS,
        configurations,
        statistics_records,
        strict=True,
    ):
        method = _method_family(configuration)
        report_payload = {
            "schema_version": PHASE12_PER_CONFIG_SCHEMA,
            "campaign_id": identifier,
            "method_config_id": configuration,
            "method_family": method,
            "configuration_admission": admission.to_dict(),
            "prior_gate_check_ids": entry["configurations"][configuration][
                "evidence"
            ],
            "prior_method_admission_path": (
                PRIOR_ADMISSION_REPORT_BINDINGS[method].as_posix()
            ),
            "prior_method_admission_sha256": (
                EXPECTED_REPORT_SHA256S[method]
            ),
            "g5_statistics": summary.to_dict(),
            "gates": {
                "G1": "PASS",
                "G2": "PASS",
                "G3": "PASS",
                "G4": "PASS",
                "G5": summary.disposition.value,
            },
            "speedup_calculated": False,
            "r_hbm": None,
        }
        write_exclusive(
            resolved_stage
            / "admission"
            / configuration
            / "report.json",
            json_bytes(report_payload),
        )
    write_exclusive(
        resolved_stage / "unified" / "local-admission.json",
        json_bytes(local_report.to_dict()),
    )
    local_pass = all(
        item.disposition is Phase12G5Disposition.PASS
        for item in statistics_records
    )
    campaign_result = {
        "schema_version": "kvbench-phase12-local-campaign-result-1.0.0",
        "campaign_id": identifier,
        "execution_git_sha": git_sha,
        "container_digest": PHASE12_AUTHORIZED_CONTAINER_DIGEST,
        "expected_runs": 30,
        "completed_runs": 30,
        "failed_runs": 0,
        "unstable_configurations": [
            item.method_config_id
            for item in statistics_records
            if item.disposition is Phase12G5Disposition.UNSTABLE
        ],
        "local_g1_g4": "PASS",
        "local_g5_reproducibility": "PASS" if local_pass else "unstable",
        "durable_publication": "PENDING_HOST_SIDE",
        "global_g5": "NOT_EVALUATED" if local_pass else "FAIL",
        "pilot": "NOT_READY",
        "full_scan": "CLOSED",
        "quality_execution": "LOCKED",
        "performance_data_frozen": False,
        "selective_reruns": 0,
        "speedup_calculated": False,
        "r_hbm": None,
        "local_admission_path": "unified/local-admission.json",
        "local_admission_sha256": sha256_file(
            resolved_stage / "unified" / "local-admission.json"
        ),
    }
    write_exclusive(
        resolved_stage / "unified" / "campaign-result.json",
        json_bytes(campaign_result),
    )
    return campaign_result


def _payload_paths(root: Path, excluded: set[str]) -> list[Path]:
    files: list[Path] = []
    for path in sorted(
        root.rglob("*"),
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise Phase12UnifiedAdmissionError(
                "Phase 12 campaign contains a symlink"
            )
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise Phase12UnifiedAdmissionError(
                "Phase 12 campaign contains an unsafe file"
            )
        relative = path.relative_to(root).as_posix()
        if relative not in excluded:
            files.append(path)
    return files


def _artifact_role(relative: str) -> str:
    if relative == "manifest.json":
        return "manifest"
    if relative.startswith("runs/"):
        return "g5_process_run"
    if relative.startswith("admission/"):
        return "configuration_admission"
    if relative.startswith("unified/"):
        return "unified_admission"
    if relative.startswith("validation/"):
        return "container_validation"
    if relative == "campaign-reservation.json":
        return "append_only_reservation"
    return "phase12_campaign"


def _expected_payload_paths(
    campaign_id: str,
    report: Phase12UnifiedAdmissionReport,
) -> set[str]:
    paths = {
        "campaign-reservation.json",
        "unified/entry-g1-g4.json",
        "unified/entry-authority.json",
        "unified/local-admission.json",
        "unified/campaign-result.json",
    }
    for target in ("test-cuda", "test-graph"):
        paths.update(
            {
                f"validation/{target}/command.stdout.txt",
                f"validation/{target}/command.stderr.txt",
                f"validation/{target}/command.supervision.json",
                f"validation/{target}/command.gpu-before.json",
                f"validation/{target}/command.gpu-after.json",
                f"validation/{target}/verdict.json",
            }
        )
    for configuration in MAIN_CONFIG_IDS:
        paths.add(f"admission/{configuration}/report.json")
    for run in report.runs:
        if not run.run_id.startswith(f"{campaign_id}-"):
            raise Phase12UnifiedAdmissionError(
                "Phase 12 run is outside its campaign ID"
            )
        paths.update(
            {
                f"runs/{run.run_id}/started.json",
                f"runs/{run.run_id}/worker.stdout.txt",
                f"runs/{run.run_id}/worker.stderr.txt",
                f"runs/{run.run_id}/worker.supervision.json",
                f"runs/{run.run_id}/worker.gpu-before.json",
                f"runs/{run.run_id}/worker.gpu-after.json",
                f"runs/{run.run_id}/kernel-path.before.raw.dot",
                f"runs/{run.run_id}/kernel-path.before.normalized.dot",
                f"runs/{run.run_id}/kernel-path.after.raw.dot",
                f"runs/{run.run_id}/kernel-path.after.normalized.dot",
                f"runs/{run.run_id}/result.json",
                f"runs/{run.run_id}/manifest.json",
            }
        )
    return paths


def _validate_idle_snapshot_payload(payload: Mapping[str, Any]) -> None:
    if (
        payload.get("query_exit_code") != 0
        or payload.get("errors") != []
        or payload.get("allowed_compute_processes") != []
        or payload.get("foreign_compute_processes") != []
        or payload.get("unknown_processes") != []
    ):
        raise Phase12UnifiedAdmissionError(
            "preserved GPU idle snapshot differs"
        )


def _validate_supervision_payload(payload: Mapping[str, Any]) -> None:
    if (
        payload.get("schema_version")
        != "kvbench-generic-supervised-command-result-1.0.0"
        or payload.get("returncode") != 0
        or not isinstance(payload.get("timeout"), Mapping)
        or payload["timeout"].get("timed_out") is not False
        or not isinstance(payload.get("direct_child"), Mapping)
        or payload["direct_child"].get("verified") is not True
        or payload["direct_child"].get("process_handle_retained") is not True
        or not isinstance(payload.get("final_reap"), Mapping)
        or payload["final_reap"].get("completed") is not True
        or payload["final_reap"].get("count") != 1
    ):
        raise Phase12UnifiedAdmissionError(
            "preserved process supervision differs"
        )


def validate_phase12_payload(
    root: Path,
    *,
    expected_campaign_id: str | None = None,
) -> tuple[Phase12UnifiedAdmissionReport, dict[str, Any]]:
    """Validate the entire scientific payload before or after finalization."""

    entry = _strict_json(root / "unified" / "entry-g1-g4.json")
    _validate_current_entry_g1_g4(entry)
    campaign_result = _strict_json(root / "unified" / "campaign-result.json")
    campaign_id = campaign_result.get("campaign_id")
    if (
        not isinstance(campaign_id, str)
        or _CAMPAIGN_ID_RE.fullmatch(campaign_id) is None
        or (
            expected_campaign_id is not None
            and campaign_id != expected_campaign_id
        )
    ):
        raise Phase12UnifiedAdmissionError(
            "Phase 12 campaign result identity differs"
        )
    local_payload = _strict_json(root / "unified" / "local-admission.json")
    try:
        report = Phase12UnifiedAdmissionReport.from_dict(local_payload)
    except (TypeError, ValueError) as error:
        raise Phase12UnifiedAdmissionError(
            "Phase 12 local admission schema failed"
        ) from error
    if (
        report.campaign_id != campaign_id
        or report.publication_state is not Phase12PublicationState.PENDING
        or report.execution_git_sha
        != campaign_result.get("execution_git_sha")
        or report.authorized_container_digest
        != PHASE12_AUTHORIZED_CONTAINER_DIGEST
        or campaign_result.get("expected_runs") != 30
        or campaign_result.get("completed_runs") != 30
        or campaign_result.get("failed_runs") != 0
        or campaign_result.get("selective_reruns") != 0
        or campaign_result.get("speedup_calculated") is not False
        or campaign_result.get("r_hbm") is not None
        or campaign_result.get("full_scan") != "CLOSED"
        or campaign_result.get("quality_execution") != "LOCKED"
        or campaign_result.get("performance_data_frozen") is not False
        or campaign_result.get("pilot") != "NOT_READY"
        or campaign_result.get("durable_publication")
        != "PENDING_HOST_SIDE"
        or campaign_result.get("local_admission_sha256")
        != sha256_file(root / "unified" / "local-admission.json")
    ):
        raise Phase12UnifiedAdmissionError(
            "Phase 12 local campaign governance differs"
        )
    entry_authority = _strict_json(
        root / "unified" / "entry-authority.json"
    )
    if entry_authority != _expected_serialized_entry_authority(
        campaign_id=campaign_id,
        execution_git_sha=report.execution_git_sha,
    ):
        raise Phase12UnifiedAdmissionError(
            "Phase 12 entry authority does not replay"
        )
    local_pass = all(
        item.disposition is Phase12G5Disposition.PASS
        for item in report.g5_statistics
    )
    if (
        campaign_result.get("local_g5_reproducibility")
        != ("PASS" if local_pass else "unstable")
        or campaign_result.get("global_g5")
        != ("NOT_EVALUATED" if local_pass else "FAIL")
    ):
        raise Phase12UnifiedAdmissionError(
            "Phase 12 local G5 decision differs"
        )
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in _payload_paths(root, set(_CONTROL_FILES))
    }
    expected_paths = _expected_payload_paths(campaign_id, report)
    if actual_paths != expected_paths:
        raise Phase12UnifiedAdmissionError(
            "Phase 12 payload topology differs"
        )
    for target in ("test-cuda", "test-graph"):
        base = root / "validation" / target
        verdict = _strict_json(base / "verdict.json")
        if (
            verdict.get("target") != target
            or verdict.get("authorized_container_digest")
            != PHASE12_AUTHORIZED_CONTAINER_DIGEST
            or verdict.get("passed") is not True
            or verdict.get("cuda_executed_on_native_host") is not False
        ):
            raise Phase12UnifiedAdmissionError(
                f"Phase 12 {target} evidence differs"
            )
        _validate_supervision_payload(
            _strict_json(base / "command.supervision.json")
        )
        _validate_idle_snapshot_payload(
            _strict_json(base / "command.gpu-before.json")
        )
        _validate_idle_snapshot_payload(
            _strict_json(base / "command.gpu-after.json")
        )
    for compact in report.runs:
        manifest_path = root / compact.manifest_path
        if sha256_file(manifest_path) != compact.manifest_sha256:
            raise Phase12UnifiedAdmissionError(
                "Phase 12 run-manifest checksum differs"
            )
        manifest = _strict_json(manifest_path)
        run_root = manifest_path.parent
        if (
            manifest.get("schema_version")
            != "kvbench-phase12-g5-run-manifest-1.0.0"
            or manifest.get("run_id") != compact.run_id
            or manifest.get("campaign_id") != campaign_id
            or manifest.get("status") != "completed"
            or manifest.get("method_config_id")
            != compact.method_config_id
            or manifest.get("method_config_fingerprint")
            != EXPECTED_CONFIG_FINGERPRINTS[compact.method_config_id]
            or manifest.get("replicate_index")
            != compact.replicate_index
            or manifest.get("seed") != compact.seed
            or manifest.get("order_index") != compact.order_index
            or manifest.get("execution_git_sha")
            != report.execution_git_sha
            or manifest.get("authorized_container_digest")
            != PHASE12_AUTHORIZED_CONTAINER_DIGEST
            or manifest.get("runner_kind") != PHASE12_RUNNER_KIND
            or manifest.get("graph_mode") != PHASE12_GRAPH_MODE
            or manifest.get("batch_size") != PHASE12_BATCH_SIZE
            or manifest.get("context_length") != PHASE12_CONTEXT_LENGTH
            or manifest.get("warmup_steps") != PHASE12_WARMUP_STEPS
            or manifest.get("measured_steps") != PHASE12_MEASURED_STEPS
            or manifest.get("measured_batches") != PHASE12_MEASURED_BATCHES
            or manifest.get("speedup_calculated") is not False
            or manifest.get("r_hbm") is not None
            or manifest.get("selective_rerun") is not False
        ):
            raise Phase12UnifiedAdmissionError(
                "Phase 12 run manifest differs"
            )
        result_path = root / str(manifest.get("result_path"))
        if (
            result_path != run_root / "result.json"
            or sha256_file(result_path) != manifest.get("result_sha256")
            or sha256_file(run_root / "worker.stdout.txt")
            != manifest.get("stdout_sha256")
            or sha256_file(run_root / "worker.stderr.txt")
            != manifest.get("stderr_sha256")
            or sha256_file(run_root / "worker.supervision.json")
            != manifest.get("supervision_sha256")
            or sha256_file(run_root / "worker.gpu-before.json")
            != manifest.get("gpu_before_sha256")
            or sha256_file(run_root / "worker.gpu-after.json")
            != manifest.get("gpu_after_sha256")
        ):
            raise Phase12UnifiedAdmissionError(
                "Phase 12 run evidence checksum differs"
            )
        worker = _strict_json(result_path)
        observation = worker.get("kernel_path_observation")
        if not isinstance(observation, Mapping):
            raise Phase12UnifiedAdmissionError(
                "Phase 12 observed kernel-path binding differs"
            )
        for phase in ("before", "after"):
            record = observation.get(phase)
            if not isinstance(record, Mapping):
                raise Phase12UnifiedAdmissionError(
                    "Phase 12 observed kernel-path phase is absent"
                )
            raw_key = f"kernel_path_{phase}_raw"
            normalized_key = f"kernel_path_{phase}_normalized"
            kernel_raw_path = root / str(
                manifest.get(f"{raw_key}_path")
            )
            kernel_normalized_path = root / str(
                manifest.get(f"{normalized_key}_path")
            )
            if (
                kernel_raw_path
                != run_root / f"kernel-path.{phase}.raw.dot"
                or kernel_normalized_path
                != run_root / f"kernel-path.{phase}.normalized.dot"
                or sha256_file(kernel_raw_path)
                != manifest.get(f"{raw_key}_sha256")
                or sha256_file(kernel_normalized_path)
                != manifest.get(f"{normalized_key}_sha256")
                or record.get("raw_sha256")
                != manifest.get(f"{raw_key}_sha256")
                or record.get("normalized_sha256")
                != manifest.get(f"{normalized_key}_sha256")
            ):
                raise Phase12UnifiedAdmissionError(
                    "Phase 12 observed kernel-path binding differs"
                )
            normalized, nodes, kernels, edges = (
                _normalize_cuda_graph_debug_dot(
                    kernel_raw_path.read_bytes()
                )
            )
            if (
                kernel_normalized_path.read_bytes() != normalized
                or (nodes, kernels, edges)
                != (
                    record.get("node_count"),
                    record.get("kernel_node_count"),
                    record.get("edge_count"),
                )
            ):
                raise Phase12UnifiedAdmissionError(
                    "Phase 12 observed kernel path does not replay"
                )
        _validate_worker_result(
            worker,
            run_id=compact.run_id,
            configuration=compact.method_config_id,
            replicate_index=compact.replicate_index,
            seed=compact.seed,
            order_index=compact.order_index,
            git_sha=report.execution_git_sha,
        )
        reconstructed = _compact_g5_run(
            payload=worker,
            manifest_path=compact.manifest_path,
            manifest_sha256=compact.manifest_sha256,
        )
        if reconstructed != compact:
            raise Phase12UnifiedAdmissionError(
                "Phase 12 compact run does not replay"
            )
        _validate_supervision_payload(
            _strict_json(run_root / "worker.supervision.json")
        )
        _validate_idle_snapshot_payload(
            _strict_json(run_root / "worker.gpu-before.json")
        )
        _validate_idle_snapshot_payload(
            _strict_json(run_root / "worker.gpu-after.json")
        )
    for admission, summary in zip(
        report.configurations,
        report.g5_statistics,
        strict=True,
    ):
        config_path = (
            root
            / "admission"
            / admission.method_config_id
            / "report.json"
        )
        payload = _strict_json(config_path)
        method = _method_family(admission.method_config_id)
        expected_gate_checks = {
            gate: list(checks)
            for gate, checks in GATE_EVIDENCE_REQUIREMENTS[method].items()
        }
        if (
            payload.get("schema_version") != PHASE12_PER_CONFIG_SCHEMA
            or payload.get("campaign_id") != campaign_id
            or payload.get("method_config_id")
            != admission.method_config_id
            or payload.get("method_family") != method
            or payload.get("configuration_admission")
            != admission.to_dict()
            or payload.get("g5_statistics") != summary.to_dict()
            or payload.get("prior_method_admission_path")
            != PRIOR_ADMISSION_REPORT_BINDINGS[method].as_posix()
            or payload.get("prior_method_admission_sha256")
            != EXPECTED_REPORT_SHA256S[method]
            or payload.get("prior_gate_check_ids")
            != expected_gate_checks
            or payload.get("speedup_calculated") is not False
            or payload.get("r_hbm") is not None
        ):
            raise Phase12UnifiedAdmissionError(
                "Phase 12 per-configuration report differs"
            )
        entry_sha256 = sha256_file(
            root / "unified" / "entry-authority.json"
        )
        for gate in admission.prior_gates:
            expected_paths = [PRIOR_ADMISSION_REPORT_BINDINGS[method].as_posix()]
            expected_shas = [EXPECTED_REPORT_SHA256S[method]]
            if method in {"bf16", "turboquant"}:
                expected_paths.append("unified/entry-authority.json")
                expected_shas.append(entry_sha256)
            if (
                [item.path for item in gate.evidence] != expected_paths
                or [item.sha256 for item in gate.evidence]
                != expected_shas
            ):
                raise Phase12UnifiedAdmissionError(
                    "Phase 12 prior-gate authority binding differs"
                )
    return report, campaign_result


def _seal_phase12_artifact(
    root: Path,
    *,
    run_id: str,
    status: str,
) -> tuple[str, int]:
    inventory_items = []
    for path in _payload_paths(root, set(_INVENTORY_EXCLUSIONS)):
        relative = path.relative_to(root).as_posix()
        inventory_items.append(
            {
                "path": relative,
                "role": _artifact_role(relative),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    write_exclusive(
        root / "artifact_inventory.json",
        json_bytes(
            {
                "schema_version": "kvbench-artifact-inventory-1.0.0",
                "run_id": run_id,
                "files": inventory_items,
                "excluded_control_files": [
                    "artifact_inventory.json",
                    "checksums.sha256",
                    "COMPLETE",
                ],
            }
        ),
    )
    ledger_paths = _payload_paths(
        root,
        {"checksums.sha256", "COMPLETE"},
    )
    ledger = "".join(
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n"
        for path in ledger_paths
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
                "checksum_ledger_sha256": sha256_file(
                    root / "checksums.sha256"
                ),
                "written_last": True,
            }
        ),
    )
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
    root.chmod(0o555)
    artifact = validate_local_artifact(root, environ={})
    return artifact.root_sha256, len(artifact.files)


def _finalize_campaign_stage(
    *,
    stage: Path,
    campaign_id: str,
) -> tuple[Path, str, int]:
    identifier = _validate_campaign_id(campaign_id)
    stage_path = stage.absolute()
    staging_root = PHASE12_ARTIFACT_ROOT / ".kvbench-staging"
    try:
        stage_path.resolve(strict=True).relative_to(
            staging_root.resolve(strict=True)
        )
    except (OSError, ValueError) as error:
        raise Phase12UnifiedAdmissionError(
            "Phase 12 stage is outside the reserved staging root"
        ) from error
    if (
        stage_path.is_symlink()
        or not stage_path.is_dir()
        or not stage_path.name.startswith(f"{identifier}.")
        or not stage_path.name.endswith(".staging")
    ):
        raise Phase12UnifiedAdmissionError("Phase 12 stage identity differs")
    report, result = validate_phase12_payload(
        stage_path,
        expected_campaign_id=identifier,
    )
    final = PHASE12_ARTIFACT_ROOT / identifier
    if final.exists() or final.is_symlink():
        raise Phase12UnifiedAdmissionError(
            "Phase 12 final campaign already exists"
        )
    local_pass = all(
        item.disposition is Phase12G5Disposition.PASS
        for item in report.g5_statistics
    )
    manifest = {
        "schema_version": PHASE12_CAMPAIGN_SCHEMA,
        "run_id": identifier,
        "campaign_id": identifier,
        "status": "completed" if local_pass else "unstable",
        "created_at_utc": _strict_json(
            stage_path / "campaign-reservation.json"
        )["created_at_utc"],
        "finalized_at_utc": _utc_now(),
        "execution_git_sha": report.execution_git_sha,
        "authorized_container_digest": PHASE12_AUTHORIZED_CONTAINER_DIGEST,
        "model_id": PHASE12_MODEL_ID,
        "model_revision": PHASE12_MODEL_REVISION,
        "tokenizer_id": PHASE12_TOKENIZER_ID,
        "tokenizer_revision": PHASE12_TOKENIZER_REVISION,
        "main_configurations": list(MAIN_CONFIG_IDS),
        "held_out_configurations": list(HELD_OUT_CONFIG_IDS),
        "configuration_fingerprints": EXPECTED_CONFIG_FINGERPRINTS,
        "method_admission_report_sha256s": EXPECTED_REPORT_SHA256S,
        "runner_kind": PHASE12_RUNNER_KIND,
        "graph_mode": PHASE12_GRAPH_MODE,
        "batch_size": PHASE12_BATCH_SIZE,
        "context_length": PHASE12_CONTEXT_LENGTH,
        "warmup_steps": PHASE12_WARMUP_STEPS,
        "measured_steps": PHASE12_MEASURED_STEPS,
        "measured_batches": PHASE12_MEASURED_BATCHES,
        "independent_process_replicates": PHASE12_REPLICATES,
        "randomization_seeds": list(PHASE12_RANDOMIZATION_SEEDS),
        "randomized_orders": [
            list(order) for order in PHASE12_RANDOMIZED_ORDERS
        ],
        "run_ids": [item.run_id for item in report.runs],
        "local_admission_path": "unified/local-admission.json",
        "local_admission_sha256": sha256_file(
            stage_path / "unified" / "local-admission.json"
        ),
        "campaign_result_path": "unified/campaign-result.json",
        "campaign_result_sha256": sha256_file(
            stage_path / "unified" / "campaign-result.json"
        ),
        "local_g1_g4": "PASS",
        "local_g5_reproducibility": result[
            "local_g5_reproducibility"
        ],
        "durable_publication": "PENDING_HOST_SIDE",
        "global_g5": result["global_g5"],
        "pilot": "NOT_READY",
        "full_scan": "CLOSED",
        "quality_execution": "LOCKED",
        "performance_data_frozen": False,
        "selective_reruns": 0,
        "speedup_calculated": False,
        "r_hbm": None,
    }
    write_exclusive(stage_path / "manifest.json", json_bytes(manifest))
    root_sha256, object_count = _seal_phase12_artifact(
        stage_path,
        run_id=identifier,
        status=manifest["status"],
    )
    rename_noreplace(stage_path, final)
    final.parent.mkdir(parents=True, exist_ok=True)
    directory_descriptor = os.open(
        final.parent, os.O_RDONLY | os.O_DIRECTORY
    )
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    validated = validate_phase12_campaign(final)
    if (
        validated["root_sha256"] != root_sha256
        or validated["object_count"] != object_count
    ):
        raise Phase12UnifiedAdmissionError(
            "final Phase 12 artifact identity changed"
        )
    return final, root_sha256, object_count


def finalize_failed_phase12_campaign(
    *,
    stage: Path,
    campaign_id: str,
    git_sha: str,
    failure_code: int,
) -> dict[str, Any]:
    """Seal an interrupted campaign once; it can never be resumed or reused."""

    identifier = _validate_campaign_id(campaign_id)
    if (
        re.fullmatch(r"[0-9a-f]{40}", git_sha) is None
        or type(failure_code) is not int
        or not 1 <= failure_code <= 255
    ):
        raise Phase12UnifiedAdmissionError(
            "failed Phase 12 terminal identity is invalid"
        )
    stage_path = stage.absolute()
    staging_root = PHASE12_ARTIFACT_ROOT / ".kvbench-staging"
    try:
        stage_path.resolve(strict=True).relative_to(
            staging_root.resolve(strict=True)
        )
    except (OSError, ValueError) as error:
        raise Phase12UnifiedAdmissionError(
            "failed Phase 12 stage is outside its reserved root"
        ) from error
    reservation = _strict_json(stage_path / "campaign-reservation.json")
    if (
        stage_path.is_symlink()
        or not stage_path.is_dir()
        or not stage_path.name.startswith(f"{identifier}.")
        or not stage_path.name.endswith(".staging")
        or reservation.get("campaign_id") != identifier
        or reservation.get("execution_git_sha") != git_sha
        or any((stage_path / name).exists() for name in _CONTROL_FILES)
    ):
        raise Phase12UnifiedAdmissionError(
            "failed Phase 12 stage cannot be terminally sealed"
        )
    run_ids = sorted(
        path.name
        for path in (stage_path / "runs").iterdir()
        if path.is_dir() and not path.is_symlink()
    ) if (stage_path / "runs").is_dir() else []
    failure = {
        "schema_version": "kvbench-phase12-terminal-failure-1.0.0",
        "campaign_id": identifier,
        "execution_git_sha": git_sha,
        "failed_at_utc": _utc_now(),
        "failure_code": failure_code,
        "preserved_run_ids": run_ids,
        "resume_permitted": False,
        "selective_rerun_permitted": False,
        "pilot": "NOT_READY",
        "full_scan": "CLOSED",
        "quality_execution": "LOCKED",
        "performance_data_frozen": False,
        "speedup_calculated": False,
    }
    write_exclusive(
        stage_path / "failure.json",
        json_bytes(failure),
    )
    manifest = {
        "schema_version": "kvbench-phase12-failed-campaign-bundle-1.0.0",
        "run_id": identifier,
        "campaign_id": identifier,
        "status": "failed",
        "created_at_utc": reservation["created_at_utc"],
        "finalized_at_utc": _utc_now(),
        "execution_git_sha": git_sha,
        "authorized_container_digest": (
            PHASE12_AUTHORIZED_CONTAINER_DIGEST
        ),
        "preserved_run_ids": run_ids,
        "failure_path": "failure.json",
        "failure_sha256": sha256_file(stage_path / "failure.json"),
        "append_only": True,
        "resume_permitted": False,
        "selective_reruns": 0,
        "global_g5": "NOT_EVALUATED",
        "pilot": "NOT_READY",
        "full_scan": "CLOSED",
        "quality_execution": "LOCKED",
        "performance_data_frozen": False,
        "speedup_calculated": False,
        "r_hbm": None,
    }
    write_exclusive(stage_path / "manifest.json", json_bytes(manifest))
    root_sha256, object_count = _seal_phase12_artifact(
        stage_path,
        run_id=identifier,
        status="failed",
    )
    final = PHASE12_ARTIFACT_ROOT / identifier
    if final.exists() or final.is_symlink():
        raise Phase12UnifiedAdmissionError(
            "failed Phase 12 final campaign already exists"
        )
    rename_noreplace(stage_path, final)
    directory_descriptor = os.open(
        final.parent, os.O_RDONLY | os.O_DIRECTORY
    )
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    artifact = validate_local_artifact(final, environ={})
    if (
        artifact.root_sha256 != root_sha256
        or len(artifact.files) != object_count
    ):
        raise Phase12UnifiedAdmissionError(
            "failed Phase 12 terminal artifact changed"
        )
    return {
        "status": "FAILED_PRESERVED",
        "campaign_id": identifier,
        "artifact_path": str(final),
        "root_sha256": root_sha256,
        "object_count": object_count,
        "preserved_run_ids": run_ids,
        "resume_permitted": False,
    }


def validate_phase12_campaign(path: Path) -> dict[str, Any]:
    root = _require_safe_phase12_root(path)
    artifact = validate_local_artifact(root, environ={})
    manifest = _strict_json(root / "manifest.json")
    campaign_id = root.name
    report, result = validate_phase12_payload(
        root,
        expected_campaign_id=campaign_id,
    )
    local_pass = all(
        item.disposition is Phase12G5Disposition.PASS
        for item in report.g5_statistics
    )
    expected_status = "completed" if local_pass else "unstable"
    if (
        manifest.get("schema_version") != PHASE12_CAMPAIGN_SCHEMA
        or manifest.get("run_id") != campaign_id
        or manifest.get("campaign_id") != campaign_id
        or manifest.get("status") != expected_status
        or manifest.get("execution_git_sha") != report.execution_git_sha
        or manifest.get("authorized_container_digest")
        != PHASE12_AUTHORIZED_CONTAINER_DIGEST
        or tuple(manifest.get("main_configurations", ())) != MAIN_CONFIG_IDS
        or tuple(manifest.get("held_out_configurations", ()))
        != HELD_OUT_CONFIG_IDS
        or manifest.get("configuration_fingerprints")
        != EXPECTED_CONFIG_FINGERPRINTS
        or manifest.get("method_admission_report_sha256s")
        != EXPECTED_REPORT_SHA256S
        or tuple(manifest.get("run_ids", ()))
        != tuple(item.run_id for item in report.runs)
        or manifest.get("local_admission_sha256")
        != sha256_file(root / "unified" / "local-admission.json")
        or manifest.get("campaign_result_sha256")
        != sha256_file(root / "unified" / "campaign-result.json")
        or manifest.get("local_g5_reproducibility")
        != result["local_g5_reproducibility"]
        or manifest.get("durable_publication") != "PENDING_HOST_SIDE"
        or manifest.get("global_g5") != result["global_g5"]
        or manifest.get("pilot") != "NOT_READY"
        or manifest.get("full_scan") != "CLOSED"
        or manifest.get("quality_execution") != "LOCKED"
        or manifest.get("performance_data_frozen") is not False
        or manifest.get("selective_reruns") != 0
        or manifest.get("speedup_calculated") is not False
        or manifest.get("r_hbm") is not None
    ):
        raise Phase12UnifiedAdmissionError(
            "Phase 12 finalized manifest differs"
        )
    return {
        "status": "PASS" if local_pass else "UNSTABLE",
        "campaign_id": campaign_id,
        "artifact_path": str(root),
        "root_sha256": artifact.root_sha256,
        "object_count": len(artifact.files),
        "local_g5_reproducibility": result[
            "local_g5_reproducibility"
        ],
        "global_g5": "NOT_EVALUATED_PUBLICATION_PENDING",
        "pilot": "NOT_READY",
        "full_scan": "CLOSED",
        "quality_execution": "LOCKED",
        "speedup_calculated": False,
    }


def _stable_bucket_lock(payload: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "provider": "cloudflare_r2",
        "endpoint_class": "cloudflare_r2_s3",
        "bucket": "kvbench-artifacts",
        "bucket_exists": True,
        "bucket_public": False,
        "managed_r2_dev_enabled": False,
        "public_r2_dev": False,
        "custom_domain_count": 0,
        "enabled_custom_domain_count": 0,
        "public_custom_domain": False,
        "public_state_result": "PASS",
        "verification_result": "PASS",
        "enabled": True,
        "lock_rule_id": "kvbench-evidence-indefinite",
        "lock_rule_name": None,
        "covered_prefix": "kvbench/sha256/",
        "lock_prefix": "kvbench/sha256/",
        "lock_scope": "exact",
        "retention_type": "Indefinite",
        "retention_condition": "Indefinite",
        "endpoint": "https://<R2_ACCOUNT_ID>.r2.cloudflarestorage.com",
    }
    expected_keys = {*expected, "verified_at_utc"}
    if set(payload) != expected_keys or any(
        payload.get(key) != value for key, value in expected.items()
    ):
        raise Phase12UnifiedAdmissionError("R2 Bucket Lock authority differs")
    verified_at = payload.get("verified_at_utc")
    if (
        not isinstance(verified_at, str)
        or not verified_at.endswith("Z")
    ):
        raise Phase12UnifiedAdmissionError(
            "R2 Bucket Lock endpoint or timestamp differs"
        )
    return {
        key: payload.get(key)
        for key in sorted(expected)
    }


def _validate_r2_tool_result(
    *,
    path: Path,
    operation: str,
    root_sha256: str,
    object_count: int,
) -> dict[str, Any]:
    payload = _strict_json(path)
    required_statuses = payload.get("required_variables")
    expected_top_level = {
        "status",
        "required_variables",
        "r2",
        "bucket_lock",
        operation,
    }
    if (
        operation not in {"publish", "verify"}
        or set(payload) != expected_top_level
        or payload.get("status") != "PASS"
        or not isinstance(required_statuses, Mapping)
        or set(required_statuses) != PHASE12_R2_CREDENTIAL_NAMES
        or set(required_statuses.values()) != {"PRESENT"}
        or payload.get("r2")
        != {
            "bucket": "kvbench-artifacts",
            "endpoint": (
                "https://<R2_ACCOUNT_ID>.r2.cloudflarestorage.com"
            ),
            "endpoint_class": "cloudflare_r2_s3",
            "prefix": "kvbench/sha256",
            "provider": "cloudflare_r2",
            "region": "auto",
        }
        or not isinstance(payload.get("bucket_lock"), Mapping)
        or not isinstance(payload.get(operation), Mapping)
    ):
        raise Phase12UnifiedAdmissionError(
            f"R2 {operation} tool evidence differs"
        )
    _stable_bucket_lock(payload["bucket_lock"])
    result = payload[operation]
    expected_result_keys = (
        {
            "provider",
            "root_sha256",
            "uri",
            "object_count",
            "complete_last",
            "uploaded_count",
            "verified_existing_count",
            "published_at_utc",
            "publication_order_sha256",
        }
        if operation == "publish"
        else {
            "provider",
            "root_sha256",
            "uri",
            "object_count",
            "verification_result",
            "checksum_ledger_valid",
            "complete_marker_valid",
            "inventory_valid",
            "unexpected_objects",
            "retrieved_at_utc",
        }
    )
    if set(result) != expected_result_keys:
        raise Phase12UnifiedAdmissionError(
            f"R2 {operation} result schema differs"
        )
    uri = (
        "r2://kvbench-artifacts/kvbench/sha256/"
        f"{root_sha256}/"
    )
    common = {
        "provider": "cloudflare_r2",
        "root_sha256": root_sha256,
        "uri": uri,
        "object_count": object_count,
    }
    if any(result.get(key) != value for key, value in common.items()):
        raise Phase12UnifiedAdmissionError(
            f"R2 {operation} root identity differs"
        )
    if operation == "publish":
        uploaded = result.get("uploaded_count")
        verified = result.get("verified_existing_count")
        if (
            result.get("complete_last") is not True
            or type(uploaded) is not int
            or type(verified) is not int
            or uploaded < 0
            or verified < 0
            or uploaded + verified != object_count
            or not isinstance(result.get("published_at_utc"), str)
        ):
            raise Phase12UnifiedAdmissionError(
                "R2 publication order or object accounting differs"
            )
        _require_sha256(
            result.get("publication_order_sha256"),
            "R2 publication order",
        )
    elif (
        result.get("verification_result") != "PASS"
        or result.get("checksum_ledger_valid") is not True
        or result.get("complete_marker_valid") is not True
        or result.get("inventory_valid") is not True
        or result.get("unexpected_objects") is not False
        or not isinstance(result.get("retrieved_at_utc"), str)
    ):
        raise Phase12UnifiedAdmissionError(
            "R2 clean retrieval evidence differs"
        )
    return payload


def _secret_free_r2_evidence_bytes(path: Path) -> bytes:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise Phase12UnifiedAdmissionError(
            "R2 tool evidence is unavailable"
        ) from error
    if configured_secret_value_names(raw, os.environ):
        raise Phase12UnifiedAdmissionError(
            "R2 tool evidence contains a configured secret value"
        )
    return raw


def _render_phase12_markdown(
    report: Phase12UnifiedAdmissionReport,
    *,
    root_sha256: str,
    object_count: int,
) -> str:
    lines = [
        "# Phase 12 — Unified Admission Gates",
        "",
        "Status: PASS",
        "",
        f"- Campaign: `{report.campaign_id}`",
        f"- Execution Git SHA: `{report.execution_git_sha}`",
        f"- Container: `{report.authorized_container_digest}`",
        (
            "- Common point: `fixed_l`, `cuda_graph`, "
            "B=1, L=4096, warmup=64, measured_steps=128"
        ),
        "- Independent processes: 3 per configuration (30 total)",
        "- Selective reruns: 0",
        "",
        "## Configuration results",
        "",
        "| Configuration | CV | G1 | G2 | G3 | G4 | G5 |",
        "| --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for configuration, summary in zip(
        report.configurations,
        report.g5_statistics,
        strict=True,
    ):
        lines.append(
            "| "
            f"`{configuration.method_config_id}` | "
            f"{summary.coefficient_of_variation:.8f} | "
            "PASS | PASS | PASS | PASS | "
            f"{summary.disposition.value} |"
        )
    lines.extend(
        [
            "",
            "Held-out controls `turboquant_k8v4` and `k4v2` did not "
            "participate in the main gate.",
            "",
            "## Global decision and custody",
            "",
            "- G0: PASS",
            "- G1: PASS",
            "- G2: PASS",
            "- G3: PASS",
            "- G4: PASS",
            "- G5: PASS",
            "- Pilot: READY",
            "- Full Scan: CLOSED",
            "- Quality execution: LOCKED",
            "- PERFORMANCE_DATA_FROZEN: absent",
            f"- Published campaign root: `{root_sha256}`",
            f"- Published campaign R2 URI: `{report.r2_uri}`",
            f"- Published campaign objects: {object_count}",
            "- COMPLETE uploaded last: yes",
            "- Clean retrieval: PASS",
            "",
            "Adapters, CUDA, cache layouts, calibration, fixtures, and "
            "method-specific evidence were not changed. This report "
            "establishes only unified numerical, memory, execution-path, "
            "graph, and three-process reproducibility admission; it makes "
            "no speedup, HBM, knee, capacity, performance, or quality claim.",
            "",
            "Phase 13 is deferred to a separate task.",
            "",
        ]
    )
    return "\n".join(lines)




def _derive_final_phase12_report(
    closure_candidate: Phase12UnifiedAdmissionReport,
    *,
    receipt_sha256: str,
    campaign_root_sha256: str,
    campaign_r2_uri: str,
    campaign_object_count: int,
) -> Phase12UnifiedAdmissionReport:
    if (
        closure_candidate.publication_state
        is not Phase12PublicationState.PENDING
        or closure_candidate.publication_receipt is not None
        or closure_candidate.published_root_sha256 is not None
        or closure_candidate.r2_uri is not None
        or closure_candidate.object_count is not None
        or closure_candidate.complete_last
        or closure_candidate.clean_retrieval
        or closure_candidate.gates.g5
        is not GateDisposition.NOT_EVALUATED
        or closure_candidate.gates.pilot_state != "NOT_READY"
    ):
        raise Phase12UnifiedAdmissionError(
            "Phase 12 campaign closure candidate cannot derive final PASS"
        )
    _require_sha256(receipt_sha256, "Phase 12 publication receipt")
    _require_sha256(campaign_root_sha256, "Phase 12 campaign root")
    if (
        campaign_r2_uri
        != (
            "r2://kvbench-artifacts/kvbench/sha256/"
            f"{campaign_root_sha256}/"
        )
        or type(campaign_object_count) is not int
        or campaign_object_count <= 0
    ):
        raise Phase12UnifiedAdmissionError(
            "Phase 12 campaign publication identity differs"
        )
    return dataclasses.replace(
        closure_candidate,
        publication_state=Phase12PublicationState.PASS,
        publication_receipt=Phase12EvidenceReference(
            evidence_id="phase12_r2_publication",
            path="docs/evidence/phase12/r2-publication.json",
            sha256=receipt_sha256,
        ),
        published_root_sha256=campaign_root_sha256,
        r2_uri=campaign_r2_uri,
        object_count=campaign_object_count,
        complete_last=True,
        clean_retrieval=True,
        gates=dataclasses.replace(
            closure_candidate.gates,
            g5=GateDisposition.PASS,
            pilot_state="READY",
        ),
    )




def close_phase12_campaign_publication(
    *,
    artifact: Path,
    publish_result_path: Path,
    verify_result_path: Path,
    receipt_output: Path,
    report_output: Path,
    markdown_output: Path,
) -> dict[str, Any]:
    """Close the single published campaign after exactly one clean retrieval."""

    validation = validate_phase12_campaign(artifact)
    if (
        validation["status"] != "PASS"
        or validation["local_g5_reproducibility"] != "PASS"
    ):
        raise Phase12UnifiedAdmissionError(
            "only a locally stable campaign may be published"
        )
    root = Path(validation["artifact_path"])
    root_sha256 = str(validation["root_sha256"])
    object_count = int(validation["object_count"])
    publish_raw = _secret_free_r2_evidence_bytes(publish_result_path)
    verify_raw = _secret_free_r2_evidence_bytes(verify_result_path)
    publish = _validate_r2_tool_result(
        path=publish_result_path,
        operation="publish",
        root_sha256=root_sha256,
        object_count=object_count,
    )
    verify = _validate_r2_tool_result(
        path=verify_result_path,
        operation="verify",
        root_sha256=root_sha256,
        object_count=object_count,
    )
    if (
        _stable_bucket_lock(publish["bucket_lock"])
        != _stable_bucket_lock(verify["bucket_lock"])
        or publish_result_path.read_bytes() != publish_raw
        or verify_result_path.read_bytes() != verify_raw
    ):
        raise Phase12UnifiedAdmissionError(
            "campaign publication authority changed during validation"
        )
    closure_candidate = Phase12UnifiedAdmissionReport.from_dict(
        _strict_json(root / "unified" / "local-admission.json")
    )
    expected_receipt = (
        REPOSITORY_ROOT / "docs/evidence/phase12/r2-publication.json"
    )
    expected_report = (
        REPOSITORY_ROOT / "docs/evidence/phase12/unified-admission.json"
    )
    expected_markdown = (
        REPOSITORY_ROOT
        / "docs/phase_reports/phase12-unified-admission.md"
    )
    supplied_paths = (
        receipt_output.absolute(),
        report_output.absolute(),
        markdown_output.absolute(),
    )
    if supplied_paths != (
        expected_receipt,
        expected_report,
        expected_markdown,
    ):
        raise Phase12UnifiedAdmissionError(
            "Phase 12 final evidence output path differs"
        )
    receipt = {
        "schema_version": "kvbench-phase12-r2-publication-1.0.0",
        "recorded_at_utc": _utc_now(),
        "campaign_id": validation["campaign_id"],
        "source_git_sha": closure_candidate.execution_git_sha,
        "local_validation": {
            **validation,
            "artifact_path": root.relative_to(REPOSITORY_ROOT).as_posix(),
        },
        "publication": {
            **publish["publish"],
            "conditional_writes": True,
            "content_addressed": True,
        },
        "clean_retrieval": {
            **verify["verify"],
            "destination_initially_empty": True,
            "result": "PASS",
        },
        "publication_attempt_count": 1,
        "clean_retrieval_count": 1,
        "bucket_lock": verify["bucket_lock"],
        "credential_statuses": verify["required_variables"],
        "credential_values_recorded": False,
        "env_file_read": False,
        "env_file_hashed": False,
        "credentials_passed_to_measurement_container": False,
        "complete_last": True,
        "self_reference_control": {
            "included_in_bundle": False,
            "receipt_path": "docs/evidence/phase12/r2-publication.json",
        },
        "global_g5": "PASS",
        "pilot": "READY",
        "full_scan": "CLOSED",
        "quality_execution": "LOCKED",
        "performance_data_frozen": False,
        "selective_reruns": 0,
        "speedup_calculated": False,
        "r_hbm": None,
    }
    write_exclusive(expected_receipt, json_bytes(receipt))
    final_report = _derive_final_phase12_report(
        closure_candidate,
        receipt_sha256=sha256_file(expected_receipt),
        campaign_root_sha256=root_sha256,
        campaign_r2_uri=verify["verify"]["uri"],
        campaign_object_count=object_count,
    )
    write_exclusive(expected_report, json_bytes(final_report.to_dict()))
    write_exclusive(
        expected_markdown,
        _render_phase12_markdown(
            final_report,
            root_sha256=root_sha256,
            object_count=object_count,
        ).encode("utf-8"),
    )
    return validate_phase12_campaign_final_evidence(
        artifact=root,
        receipt=expected_receipt,
        report=expected_report,
        markdown=expected_markdown,
    )


def validate_phase12_campaign_final_evidence(
    *,
    artifact: Path,
    receipt: Path,
    report: Path,
    markdown: Path,
) -> dict[str, Any]:
    """Replay the single-bundle custody and final gate derivation."""

    validation = validate_phase12_campaign(artifact)
    if validation["status"] != "PASS":
        raise Phase12UnifiedAdmissionError(
            "final Phase 12 campaign is not locally stable"
        )
    root = Path(validation["artifact_path"])
    root_sha256 = str(validation["root_sha256"])
    object_count = int(validation["object_count"])
    closure_candidate = Phase12UnifiedAdmissionReport.from_dict(
        _strict_json(root / "unified" / "local-admission.json")
    )
    receipt_payload = _strict_json(receipt)
    final_report = Phase12UnifiedAdmissionReport.from_dict(
        _strict_json(report)
    )
    publication = receipt_payload.get("publication")
    retrieval = receipt_payload.get("clean_retrieval")
    credentials = receipt_payload.get("credential_statuses")
    bucket_lock = receipt_payload.get("bucket_lock")
    expected_uri = (
        "r2://kvbench-artifacts/kvbench/sha256/"
        f"{root_sha256}/"
    )
    expected_receipt_keys = {
        "schema_version",
        "recorded_at_utc",
        "campaign_id",
        "source_git_sha",
        "local_validation",
        "publication",
        "clean_retrieval",
        "publication_attempt_count",
        "clean_retrieval_count",
        "bucket_lock",
        "credential_statuses",
        "credential_values_recorded",
        "env_file_read",
        "env_file_hashed",
        "credentials_passed_to_measurement_container",
        "complete_last",
        "self_reference_control",
        "global_g5",
        "pilot",
        "full_scan",
        "quality_execution",
        "performance_data_frozen",
        "selective_reruns",
        "speedup_calculated",
        "r_hbm",
    }
    expected_publication_keys = {
        "provider",
        "root_sha256",
        "uri",
        "object_count",
        "complete_last",
        "uploaded_count",
        "verified_existing_count",
        "published_at_utc",
        "publication_order_sha256",
        "conditional_writes",
        "content_addressed",
    }
    expected_retrieval_keys = {
        "provider",
        "root_sha256",
        "uri",
        "object_count",
        "verification_result",
        "checksum_ledger_valid",
        "complete_marker_valid",
        "inventory_valid",
        "unexpected_objects",
        "retrieved_at_utc",
        "destination_initially_empty",
        "result",
    }
    if not isinstance(bucket_lock, Mapping):
        raise Phase12UnifiedAdmissionError(
            "final Phase 12 Bucket Lock evidence differs"
        )
    _stable_bucket_lock(bucket_lock)
    if (
        set(receipt_payload) != expected_receipt_keys
        or not isinstance(publication, Mapping)
        or set(publication) != expected_publication_keys
        or not isinstance(retrieval, Mapping)
        or set(retrieval) != expected_retrieval_keys
        or not isinstance(credentials, Mapping)
        or set(credentials) != PHASE12_R2_CREDENTIAL_NAMES
        or set(credentials.values()) != {"PRESENT"}
    ):
        raise Phase12UnifiedAdmissionError(
            "final Phase 12 publication receipt schema differs"
        )
    uploaded = publication.get("uploaded_count")
    existing = publication.get("verified_existing_count")
    expected_local_validation = {
        **validation,
        "artifact_path": root.relative_to(REPOSITORY_ROOT).as_posix(),
    }
    if (
        receipt_payload.get("schema_version")
        != "kvbench-phase12-r2-publication-1.0.0"
        or not isinstance(receipt_payload.get("recorded_at_utc"), str)
        or not receipt_payload["recorded_at_utc"].endswith("Z")
        or receipt_payload.get("campaign_id") != validation["campaign_id"]
        or receipt_payload.get("source_git_sha")
        != closure_candidate.execution_git_sha
        or receipt_payload.get("local_validation")
        != expected_local_validation
        or receipt_payload.get("publication_attempt_count") != 1
        or receipt_payload.get("clean_retrieval_count") != 1
        or receipt_payload.get("credential_values_recorded") is not False
        or receipt_payload.get("env_file_read") is not False
        or receipt_payload.get("env_file_hashed") is not False
        or receipt_payload.get(
            "credentials_passed_to_measurement_container"
        )
        is not False
        or receipt_payload.get("complete_last") is not True
        or receipt_payload.get("self_reference_control")
        != {
            "included_in_bundle": False,
            "receipt_path": "docs/evidence/phase12/r2-publication.json",
        }
        or receipt_payload.get("global_g5") != "PASS"
        or receipt_payload.get("pilot") != "READY"
        or receipt_payload.get("full_scan") != "CLOSED"
        or receipt_payload.get("quality_execution") != "LOCKED"
        or receipt_payload.get("performance_data_frozen") is not False
        or receipt_payload.get("selective_reruns") != 0
        or receipt_payload.get("speedup_calculated") is not False
        or receipt_payload.get("r_hbm") is not None
        or publication.get("provider") != "cloudflare_r2"
        or publication.get("root_sha256") != root_sha256
        or publication.get("uri") != expected_uri
        or publication.get("object_count") != object_count
        or publication.get("complete_last") is not True
        or type(uploaded) is not int
        or type(existing) is not int
        or uploaded < 0
        or existing < 0
        or uploaded + existing != object_count
        or publication.get("conditional_writes") is not True
        or publication.get("content_addressed") is not True
        or not isinstance(publication.get("published_at_utc"), str)
        or retrieval.get("provider") != "cloudflare_r2"
        or retrieval.get("root_sha256") != root_sha256
        or retrieval.get("uri") != expected_uri
        or retrieval.get("object_count") != object_count
        or retrieval.get("verification_result") != "PASS"
        or retrieval.get("checksum_ledger_valid") is not True
        or retrieval.get("complete_marker_valid") is not True
        or retrieval.get("inventory_valid") is not True
        or retrieval.get("unexpected_objects") is not False
        or not isinstance(retrieval.get("retrieved_at_utc"), str)
        or retrieval.get("destination_initially_empty") is not True
        or retrieval.get("result") != "PASS"
    ):
        raise Phase12UnifiedAdmissionError(
            "final Phase 12 publication receipt differs"
        )
    _require_sha256(
        publication.get("publication_order_sha256"),
        "final Phase 12 publication order",
    )
    expected_final = _derive_final_phase12_report(
        closure_candidate,
        receipt_sha256=sha256_file(receipt),
        campaign_root_sha256=root_sha256,
        campaign_r2_uri=expected_uri,
        campaign_object_count=object_count,
    )
    if (
        final_report.to_dict() != expected_final.to_dict()
        or report.read_bytes() != json_bytes(expected_final.to_dict())
        or markdown.read_text(encoding="utf-8")
        != _render_phase12_markdown(
            expected_final,
            root_sha256=root_sha256,
            object_count=object_count,
        )
    ):
        raise Phase12UnifiedAdmissionError(
            "final Phase 12 unified report differs"
        )
    return {
        "status": "PASS",
        "campaign_id": final_report.campaign_id,
        "root_sha256": root_sha256,
        "object_count": object_count,
        "r2_uri": expected_uri,
        "complete_last": True,
        "clean_retrieval": True,
        "clean_retrieval_count": 1,
        "g0": "PASS",
        "g1": "PASS",
        "g2": "PASS",
        "g3": "PASS",
        "g4": "PASS",
        "g5": "PASS",
        "pilot": "READY",
        "full_scan": "CLOSED",
        "quality_execution": "LOCKED",
        "speedup_calculated": False,
    }


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_mutually_exclusive_group(required=True)
    commands.add_argument(
        "--validate-prior-only",
        action="store_true",
        help="validate and print the normalized G1-G4 aggregation",
    )
    commands.add_argument(
        "--new-campaign-id",
        action="store_true",
        help="generate one new append-only Phase 12 campaign ID",
    )
    commands.add_argument(
        "--reserve-campaign",
        action="store_true",
        help="reserve one campaign ID and print its staging path",
    )
    commands.add_argument(
        "--run-campaign",
        action="store_true",
        help="run the exact container-only validation and 30-run matrix",
    )
    commands.add_argument(
        "--run-worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    commands.add_argument(
        "--finalize-staged-campaign",
        action="store_true",
        help="finalize one completed append-only staging directory",
    )
    commands.add_argument(
        "--finalize-failed-campaign",
        action="store_true",
        help="terminally seal one interrupted campaign without resuming it",
    )
    commands.add_argument(
        "--validate-campaign",
        action="store_true",
        help="validate one finalized Phase 12 campaign without regeneration",
    )
    commands.add_argument(
        "--close-publication",
        action="store_true",
        help="bind the single campaign publication and close global gates",
    )
    commands.add_argument(
        "--validate-final-evidence",
        action="store_true",
        help="validate final publication and unified gate evidence",
    )
    parser.add_argument("--campaign-id")
    parser.add_argument("--git-sha")
    parser.add_argument("--stage", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--configuration", choices=MAIN_CONFIG_IDS)
    parser.add_argument("--replicate-index", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--order-index", type=int)
    parser.add_argument("--run-artifact-root", type=Path)
    parser.add_argument("--failure-code", type=int)
    parser.add_argument("--publish-result", type=Path)
    parser.add_argument("--verify-result", type=Path)
    parser.add_argument("--receipt-output", type=Path)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    if arguments.validate_prior_only:
        prior = load_and_validate_prior_admission_evidence(REPOSITORY_ROOT)
        print(
            json.dumps(
                aggregate_g1_g4(prior),
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return 0
    if arguments.new_campaign_id:
        if arguments.git_sha is None:
            raise Phase12UnifiedAdmissionError("--git-sha is required")
        print(new_campaign_id(arguments.git_sha))
        return 0
    if arguments.reserve_campaign:
        if arguments.campaign_id is None or arguments.git_sha is None:
            raise Phase12UnifiedAdmissionError(
                "--campaign-id and --git-sha are required"
            )
        stage = reserve_campaign(
            campaign_id=arguments.campaign_id,
            git_sha=arguments.git_sha,
        )
        print(stage)
        return 0
    if arguments.run_campaign:
        if (
            arguments.stage is None
            or arguments.campaign_id is None
            or arguments.git_sha is None
        ):
            raise Phase12UnifiedAdmissionError(
                "--stage, --campaign-id, and --git-sha are required"
            )
        result = run_campaign(
            stage=arguments.stage,
            campaign_id=arguments.campaign_id,
            git_sha=arguments.git_sha,
        )
        print(
            json.dumps(
                result,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return 0
    if arguments.run_worker:
        required = (
            arguments.run_id,
            arguments.configuration,
            arguments.replicate_index,
            arguments.seed,
            arguments.order_index,
            arguments.git_sha,
            arguments.run_artifact_root,
        )
        if any(value is None for value in required):
            raise Phase12UnifiedAdmissionError(
                "worker identity arguments are required"
            )
        assert arguments.run_id is not None
        assert arguments.configuration is not None
        assert arguments.replicate_index is not None
        assert arguments.seed is not None
        assert arguments.order_index is not None
        assert arguments.git_sha is not None
        assert arguments.run_artifact_root is not None
        if (
            _RUN_ID_RE.fullmatch(arguments.run_id) is None
            or not 0 <= arguments.replicate_index < PHASE12_REPLICATES
            or arguments.seed
            != PHASE12_RANDOMIZATION_SEEDS[arguments.replicate_index]
            or not 0 <= arguments.order_index < len(MAIN_CONFIG_IDS)
            or PHASE12_RANDOMIZED_ORDERS[arguments.replicate_index][
                arguments.order_index
            ]
            != arguments.configuration
        ):
            raise Phase12UnifiedAdmissionError(
                "worker identity differs from the frozen order"
            )
        payload = _run_g5_worker(
            run_id=arguments.run_id,
            configuration=arguments.configuration,
            replicate_index=arguments.replicate_index,
            seed=arguments.seed,
            order_index=arguments.order_index,
            git_sha=arguments.git_sha,
            run_artifact_root=arguments.run_artifact_root,
        )
        _validate_worker_result(
            payload,
            run_id=arguments.run_id,
            configuration=arguments.configuration,
            replicate_index=arguments.replicate_index,
            seed=arguments.seed,
            order_index=arguments.order_index,
            git_sha=arguments.git_sha,
        )
        print(
            PHASE12_WORKER_RESULT_PREFIX
            + json.dumps(
                payload,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return 0
    if arguments.finalize_staged_campaign:
        if arguments.stage is None or arguments.campaign_id is None:
            raise Phase12UnifiedAdmissionError(
                "--stage and --campaign-id are required"
            )
        final, root_sha256, object_count = _finalize_campaign_stage(
            stage=arguments.stage,
            campaign_id=arguments.campaign_id,
        )
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "artifact": str(final),
                    "root_sha256": root_sha256,
                    "object_count": object_count,
                },
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return 0
    if arguments.finalize_failed_campaign:
        if (
            arguments.stage is None
            or arguments.campaign_id is None
            or arguments.git_sha is None
            or arguments.failure_code is None
        ):
            raise Phase12UnifiedAdmissionError(
                "failed-campaign terminal identity is required"
            )
        print(
            json.dumps(
                finalize_failed_phase12_campaign(
                    stage=arguments.stage,
                    campaign_id=arguments.campaign_id,
                    git_sha=arguments.git_sha,
                    failure_code=arguments.failure_code,
                ),
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return 0
    if arguments.validate_campaign:
        if arguments.artifact is None:
            raise Phase12UnifiedAdmissionError("--artifact is required")
        validation = validate_phase12_campaign(arguments.artifact)
        print(
            json.dumps(
                validation,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return 0 if validation["status"] == "PASS" else 2
    if arguments.close_publication:
        required = (
            arguments.artifact,
            arguments.publish_result,
            arguments.verify_result,
            arguments.receipt_output,
            arguments.report_output,
            arguments.markdown_output,
        )
        if any(value is None for value in required):
            raise Phase12UnifiedAdmissionError(
                "publication closure paths are required"
            )
        assert arguments.artifact is not None
        assert arguments.publish_result is not None
        assert arguments.verify_result is not None
        assert arguments.receipt_output is not None
        assert arguments.report_output is not None
        assert arguments.markdown_output is not None
        print(
            json.dumps(
                close_phase12_campaign_publication(
                    artifact=arguments.artifact,
                    publish_result_path=arguments.publish_result,
                    verify_result_path=arguments.verify_result,
                    receipt_output=arguments.receipt_output,
                    report_output=arguments.report_output,
                    markdown_output=arguments.markdown_output,
                ),
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return 0
    if arguments.validate_final_evidence:
        required = (
            arguments.artifact,
            arguments.receipt_output,
            arguments.report_output,
            arguments.markdown_output,
        )
        if any(value is None for value in required):
            raise Phase12UnifiedAdmissionError(
                "final evidence paths are required"
            )
        assert arguments.artifact is not None
        assert arguments.receipt_output is not None
        assert arguments.report_output is not None
        assert arguments.markdown_output is not None
        print(
            json.dumps(
                validate_phase12_campaign_final_evidence(
                    artifact=arguments.artifact,
                    receipt=arguments.receipt_output,
                    report=arguments.report_output,
                    markdown=arguments.markdown_output,
                ),
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return 0
    raise Phase12UnifiedAdmissionError("unreachable Phase 12 mode")


if __name__ == "__main__":
    raise SystemExit(main())
