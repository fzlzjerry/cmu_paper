#!/usr/bin/env python3
"""Build or validate the bounded local Phase 11 KVQuant admission bundle.

CUDA execution is fail-closed to the exact authorized Measurement Container.
This command finalizes only the append-only inner evidence bundle.  Durable R2
publication and the final PASS MethodAdmissionReport are joined host-side and
never mutate the inner bundle.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import dataclasses
from datetime import datetime, timezone
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
import sys
import tempfile
import textwrap
from typing import Any, Mapping, Sequence

from kvbench.adapters.kvquant import (
    KVQUANT_ADAPTER_VERSION,
    KVQUANT_EXTENSION_SHA256,
    KVQUANT_Q4_DETERMINISTIC_VALUE_DECODE_API,
    KVQuantMethodAdapter,
)
from kvbench.runtime.allocation import collect_cuda_allocator_raw
from kvbench.runtime.allocation_attribution import (
    PHASE3_ALLOCATION_MAX_HISTORY_ENTRIES,
    PHASE3_OUTPUT_DTYPE_BYTES,
    PHASE3_OUTPUT_WIDTH,
    AllocationClass,
    AllocationCriterionResult,
    AllocationGeometry,
    RawAllocatorEvidenceFiles,
    allocator_trace_from_snapshot,
    allocator_counters_from_memory_stats,
    attribute_allocator_trace,
    build_history_integrity_evidence,
    evaluate_strict_graph_criterion,
    instantiate_decision_0013_direct_compressed_rules,
    memory_delta_from_raw_samples,
    preserve_allocator_evidence,
    raw_memory_accounting_sample_from_mapping,
    read_verified_allocator_evidence,
)
from kvbench.runtime.artifacts import (
    AppendOnlyArtifactStore,
    ArtifactRun,
    sha256_file,
    validate_run_directory,
)
from kvbench.runtime.backend import forced_flash_execution
from kvbench.runtime.fixed_l_runner import run_fixed_l
from kvbench.runtime.growing_context_runner import run_growing_context
from kvbench.runtime.kvquant_fixture import load_all_kvquant_fixtures
from kvbench.runtime.kvquant_session import (
    build_kvquant_endpoint_session,
    build_kvquant_operation_keys,
)
from kvbench.runtime.model_loader import load_frozen_model
from kvbench.runtime.numerical import tensor_sha256_untimed
from kvbench.runtime.process_supervision import run_supervised_command
from kvbench.runtime.turboquant_admission import (
    require_authorized_cuda_environment,
)
from kvbench.schema import (
    ClaimEligibility,
    GraphMode,
    MeasurementScope,
    MethodAdmissionEvidenceReference,
    MethodName,
    QualityExecutionState,
    QualityValidationState,
    RunnerKind,
    RunStatus,
    canonical_json_bytes,
    sha256_hex,
)
from kvbench.schema.phase11 import (
    PHASE11_ACCOUNTING_CONTEXTS,
    PHASE11_ADMISSION_CHECK_IDS,
    PHASE11_AGGREGATE_PATCH_SHA256,
    PHASE11_ALLOCATION_AUDIT_CONTEXTS,
    PHASE11_AUTHORIZED_CONTAINER_DIGEST,
    PHASE11_BOUNDED_POINT_SIGNATURES,
    PHASE11_CALIBRATION_ID,
    PHASE11_CALIBRATION_ROOT,
    PHASE11_CONFIGURATIONS,
    PHASE11_CORRECTED_COMMIT,
    PHASE11_CORRECTED_CUDA_SHA256,
    PHASE11_CORRECTED_TREE,
    PHASE11_DECISION_0021_PATCH_SHA256,
    PHASE11_DECISIONS,
    PHASE11_EXECUTION_SOURCE_IDENTIFIER,
    PHASE11_EXTENSION_SHA256,
    PHASE11_FIXTURE_CASES,
    PHASE11_FIXTURE_ID,
    PHASE11_FIXTURE_ROOT,
    PHASE11_GRAPH_POINT_SIGNATURES,
    PHASE11_HISTORICAL_FIXTURE_ID,
    PHASE11_HISTORICAL_FIXTURE_ROOT,
    PHASE11_METHOD_IDENTIFIER,
    PHASE11_SANITIZER_CASES,
    PHASE11_UPSTREAM_BASE_COMMIT,
    PHASE11_UPSTREAM_BASE_TREE,
    Phase11AllocationEvidence,
    Phase11AdmissionCheck,
    Phase11AdmissionGates,
    Phase11Authority,
    Phase11ByteAccounting,
    Phase11ByteBreakdown,
    Phase11ExecutionPathEvidence,
    Phase11FixtureEvidence,
    Phase11GraphEvidence,
    Phase11MethodAdmissionReport,
    Phase11MethodConfiguration,
    Phase11RunManifest,
    Phase11RunPoint,
    Phase11SanitizerEvidence,
    require_exact_phase11_grid,
)
from kvbench.schema.phase3 import GateDisposition
from scripts.r2_artifact import validate_local_artifact
from scripts.validate_kvquant_long_context_patch import (
    validate as validate_source,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DRIVER_PATH = Path(__file__).resolve()
ADAPTER_PATH = REPOSITORY_ROOT / "src/kvbench/adapters/kvquant.py"
CACHE_PATH = REPOSITORY_ROOT / "src/kvbench/runtime/kvquant_cache.py"
ENDPOINT_PATH = REPOSITORY_ROOT / "src/kvbench/runtime/bf16_endpoint.py"
SESSION_PATH = REPOSITORY_ROOT / "src/kvbench/runtime/kvquant_session.py"
CUDA_TEST = REPOSITORY_ROOT / "tests/cuda/test_phase11_kvquant_cuda.py"
GRAPH_TEST = REPOSITORY_ROOT / "tests/graph/test_phase11_kvquant_graph.py"
SANITIZER_PROBE = (
    REPOSITORY_ROOT / "tests/cuda/phase11_kvquant_sanitizer_probe.py"
)
_GIT_BOUND_SOURCE_PATHS = (
    "scripts/phase11_kvquant_admission.py",
    "scripts/validate_kvquant_long_context_patch.py",
    "src/kvbench/adapters/kvquant.py",
    "src/kvbench/runtime/kvquant_cache.py",
    "src/kvbench/runtime/bf16_endpoint.py",
    "src/kvbench/runtime/kvquant_session.py",
    "src/kvbench/schema/phase11.py",
    "tests/cuda/test_phase11_kvquant_cuda.py",
    "tests/graph/test_phase11_kvquant_graph.py",
    "tests/cuda/phase11_kvquant_sanitizer_probe.py",
)
_RUNTIME_CALIBRATION_PATH = Path(
    "/opt/kvquant-calibration"
) / PHASE11_CALIBRATION_ID
_PHASE11_CALIBRATION_OBJECT_COUNT = 68
_EXECUTION_PATCH_MANIFEST_PATH = (
    REPOSITORY_ROOT
    / "third_party/patches/kvquant/"
    "deterministic-long-context-manifest.json"
)
CONTAINER_PYTHON = Path("/opt/kvbench/.venv/bin/python")
SANITIZER = Path("/usr/local/cuda-13.0/bin/compute-sanitizer")
ARTIFACT_ROOT = REPOSITORY_ROOT / "artifacts/phase11"
FINAL_RECEIPT_SCHEMA = (
    "kvbench-phase11-kvquant-admission-r2-outer-publication-1.0.0"
)
FINAL_RECEIPT_PATH = (
    REPOSITORY_ROOT
    / "docs/evidence/phase11/r2-admission-outer-publication.json"
)
_HOST_ONLY_R2_ENVIRONMENT = (
    "R2_ACCOUNT_ID",
    "R2_BUCKET",
    "R2_ENDPOINT",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "CLOUDFLARE_API_TOKEN",
    "KVBENCH_R2_PREFIX",
)
_CHILD_PASSTHROUGH = (
    "PATH",
    "LD_LIBRARY_PATH",
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "NVIDIA_DRIVER_CAPABILITIES",
    "KVBENCH_AUTHORIZED_IMAGE_DIGEST",
    "KVBENCH_EXECUTION_ENVIRONMENT",
    "KVBENCH_KVQUANT_SOURCE_ROOT",
    "KVBENCH_KVQUANT_EXTENSION",
    "KVBENCH_KVQUANT_EXTENSION_SHA256",
    "KVBENCH_KVQUANT_FRESH_BUILD_EXTENSION",
    "KVBENCH_KVQUANT_CALIBRATION_ROOT",
)
_SANITIZER_RUNS = (
    ("memcheck", "kvq4-cap"),
    ("memcheck", "kvq3-distinct"),
    ("memcheck", "kvq2"),
    ("memcheck", "sink-gqa-fixed"),
    ("memcheck", "graph-replay"),
    ("initcheck", "kvq4-cap"),
    ("initcheck", "kvq3-distinct"),
)
_BITS = {"kvq4": 4, "kvq3": 3, "kvq2": 2}
_FORBIDDEN_ALLOCATION_CLASSES = frozenset(
    {
        AllocationClass.CACHE_GROWTH,
        AllocationClass.GQA_EXPANSION,
        AllocationClass.CONTEXT_SCALED_WORKSPACE,
        AllocationClass.AUDIT_INSTRUMENTATION,
        AllocationClass.UNKNOWN,
    }
)


class Phase11KVQuantDriverError(RuntimeError):
    """The bounded KVQuant admission driver failed closed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise Phase11KVQuantDriverError("required Git identity query failed")
    return result.stdout.strip()


def _git_bytes(*arguments: str) -> bytes:
    result = subprocess.run(
        ("git", *arguments),
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise Phase11KVQuantDriverError("required Git object query failed")
    return result.stdout


def _require_clean_git() -> str:
    if _git("status", "--porcelain=v1", "--untracked-files=all"):
        raise Phase11KVQuantDriverError(
            "Phase 11 admission requires a clean worktree"
        )
    head = _git("rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        raise Phase11KVQuantDriverError("Phase 11 Git SHA is invalid")
    return head


def _git_source_binding(
    git_sha: str,
    *,
    require_worktree_match: bool,
) -> dict[str, Any]:
    """Bind every Phase 11 executable source to one Git commit blob."""

    if re.fullmatch(r"[0-9a-f]{40}", git_sha) is None:
        raise Phase11KVQuantDriverError(
            "Phase 11 Git source binding commit is invalid"
        )
    records: list[dict[str, object]] = []
    for relative in _GIT_BOUND_SOURCE_PATHS:
        object_name = f"{git_sha}:{relative}"
        blob_oid = _git("rev-parse", object_name)
        if re.fullmatch(r"[0-9a-f]{40}", blob_oid) is None:
            raise Phase11KVQuantDriverError(
                "Phase 11 Git source blob identity is invalid"
            )
        blob = _git_bytes("cat-file", "blob", object_name)
        content_sha256 = hashlib.sha256(blob).hexdigest()
        if require_worktree_match:
            worktree_path = REPOSITORY_ROOT / relative
            try:
                metadata = worktree_path.lstat()
                resolved = worktree_path.resolve(strict=True)
            except OSError as error:
                raise Phase11KVQuantDriverError(
                    "Phase 11 bound worktree source is unavailable"
                ) from error
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or resolved != worktree_path
                or sha256_file(worktree_path) != content_sha256
            ):
                raise Phase11KVQuantDriverError(
                    "Phase 11 worktree source differs from its Git blob"
                )
        records.append(
            {
                "path": relative,
                "git_blob_oid": blob_oid,
                "content_sha256": content_sha256,
            }
        )
    return {
        "schema_version": "kvbench-phase11-git-source-binding-1.0.0",
        "git_sha": git_sha,
        "records": records,
        "all_match": True,
    }


def _validate_runtime_calibration_before_cuda() -> dict[str, Any]:
    """Validate the exact read-only calibration mount before CUDA starts."""

    calibration_raw = os.environ.get("KVBENCH_KVQUANT_CALIBRATION_ROOT")
    if calibration_raw != str(_RUNTIME_CALIBRATION_PATH):
        raise Phase11KVQuantDriverError(
            "exact Phase 11 calibration mount path is required"
        )
    calibration = Path(calibration_raw)
    try:
        metadata = calibration.lstat()
        resolved = calibration.resolve(strict=True)
    except OSError as error:
        raise Phase11KVQuantDriverError(
            "Phase 11 calibration mount is unavailable"
        ) from error
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or resolved != calibration
        or calibration.name != PHASE11_CALIBRATION_ID
    ):
        raise Phase11KVQuantDriverError(
            "Phase 11 calibration mount is unsafe"
        )
    artifact = validate_local_artifact(calibration, environ={})
    lifecycle = validate_run_directory(calibration)
    if (
        artifact.root_sha256 != PHASE11_CALIBRATION_ROOT
        or len(artifact.files) != _PHASE11_CALIBRATION_OBJECT_COUNT
        or not lifecycle.valid
        or not lifecycle.complete
        or lifecycle.status != "completed"
    ):
        raise Phase11KVQuantDriverError(
            "Phase 11 calibration authority differs"
        )
    return {
        "schema_version": (
            "kvbench-phase11-calibration-runtime-validation-1.0.0"
        ),
        "calibration_id": PHASE11_CALIBRATION_ID,
        "mount_path": str(calibration),
        "expected_root_sha256": PHASE11_CALIBRATION_ROOT,
        "observed_root_sha256": artifact.root_sha256,
        "object_count": len(artifact.files),
        "complete_marker_valid": True,
        "inventory_valid": True,
        "checksum_ledger_valid": True,
        "repository_lifecycle_valid": True,
        "validated_before_cuda": True,
    }


def _authority() -> Phase11Authority:
    return Phase11Authority(
        method_identifier=PHASE11_METHOD_IDENTIFIER,
        execution_source_identifier=PHASE11_EXECUTION_SOURCE_IDENTIFIER,
        upstream_base_commit=PHASE11_UPSTREAM_BASE_COMMIT,
        upstream_base_tree=PHASE11_UPSTREAM_BASE_TREE,
        decision_0021_patch_sha256=PHASE11_DECISION_0021_PATCH_SHA256,
        aggregate_patch_sha256=PHASE11_AGGREGATE_PATCH_SHA256,
        corrected_commit=PHASE11_CORRECTED_COMMIT,
        corrected_tree=PHASE11_CORRECTED_TREE,
        corrected_cuda_sha256=PHASE11_CORRECTED_CUDA_SHA256,
        extension_sha256=PHASE11_EXTENSION_SHA256,
        decisions=PHASE11_DECISIONS,
        calibration_id=PHASE11_CALIBRATION_ID,
        calibration_root=PHASE11_CALIBRATION_ROOT,
        historical_fixture_id=PHASE11_HISTORICAL_FIXTURE_ID,
        historical_fixture_root=PHASE11_HISTORICAL_FIXTURE_ROOT,
        fixture_id=PHASE11_FIXTURE_ID,
        fixture_root=PHASE11_FIXTURE_ROOT,
        authorized_container_digest=PHASE11_AUTHORIZED_CONTAINER_DIGEST,
    )


def _run_id(git_sha: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%S%f")[:-3] + "z"
    return f"phase11-{stamp}-{git_sha[:8]}-{secrets.token_hex(3)}-kvquant"


def _point_run_id(
    bundle_id: str,
    index: int,
    signature: tuple[str, RunnerKind, GraphMode, int, int],
) -> str:
    configuration, runner, graph, context, _ = signature
    runner_name = "fixed" if runner is RunnerKind.FIXED_L else "growing"
    graph_name = "graph" if graph is GraphMode.CUDA_GRAPH else "eager"
    return (
        f"{bundle_id[:62]}-p{index:02d}-{configuration}-{runner_name}-"
        f"l{context}-{graph_name}"
    )


def _manifest(
    *,
    run_id: str,
    git_sha: str,
    created_at_utc: str,
) -> Phase11RunManifest:
    return Phase11RunManifest(
        schema_version=Phase11RunManifest.SCHEMA_VERSION,
        artifact_schema_version=Phase11RunManifest.ARTIFACT_SCHEMA_VERSION,
        run_id=run_id,
        status=RunStatus.CREATED,
        created_at_utc=created_at_utc,
        started_at_utc=None,
        finished_at_utc=None,
        run_kind="phase11_admission",
        git_sha=git_sha,
        git_dirty=False,
        authority=_authority(),
        bounded_point_count=9,
        measurement_scope=MeasurementScope.MEASUREMENT_CONTAINER_ADMISSION,
        quality_status=QualityValidationState.UNVALIDATED,
        claim_eligibility=ClaimEligibility.PERFORMANCE_ONLY,
        quality_execution=QualityExecutionState.LOCKED,
        performance_claim_eligible=False,
        performance_data_frozen=False,
        quality_benchmark_executed=False,
        speedup_calculated=False,
        r_hbm=None,
        full_scan_state="CLOSED",
        g2_kvq_state="NOT_EVALUATED_PUBLICATION_PENDING",
        global_g2_g5_state="NOT_EVALUATED",
        inventory_path=None,
        failure_reason=None,
    )


def _terminal_manifest(
    initial: Phase11RunManifest,
    *,
    started_at_utc: str,
    status: RunStatus,
    failure_reason: str | None,
) -> Phase11RunManifest:
    return dataclasses.replace(
        initial,
        status=status,
        started_at_utc=started_at_utc,
        finished_at_utc=_utc_now(),
        inventory_path="artifact_inventory.json",
        failure_reason=failure_reason,
    )


def _safe_reason(error: BaseException) -> str:
    message = " ".join(str(error).split())
    return f"{type(error).__name__}: {message or 'unspecified failure'}"[:500]


def _child_environment() -> dict[str, str]:
    child = {
        name: os.environ[name]
        for name in _CHILD_PASSTHROUGH
        if os.environ.get(name)
    }
    child.update(
        {
            "PYTHONPATH": (
                f"/opt/kvbench/.phase3/site-packages:"
                f"{REPOSITORY_ROOT / 'src'}:{REPOSITORY_ROOT}"
            ),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        }
    )
    for name in _HOST_ONLY_R2_ENVIRONMENT:
        child.pop(name, None)
    return child


def _run_exact_test(
    run: ArtifactRun,
    *,
    evidence_name: str,
    module: str,
    source: Path,
    timeout_seconds: float = 1800.0,
) -> dict[str, Any]:
    command = (
        str(CONTAINER_PYTHON),
        "-m",
        "unittest",
        module,
        "-v",
    )
    result = run_supervised_command(
        command,
        working_directory=str(REPOSITORY_ROOT),
        environment=_child_environment(),
        timeout_seconds=timeout_seconds,
    )
    stdout = result.stdout
    stderr = result.stderr
    prefix = f"validation/checks/{evidence_name}"
    run.write_bytes(f"{prefix}.stdout.txt", stdout)
    run.write_bytes(f"{prefix}.stderr.txt", stderr)
    passed = (
        result.returncode == 0
        and not result.timed_out
        and result.to_dict()["direct_child"]["verified"] is True
        and result.to_dict()["final_reap"]["completed"] is True
    )
    record = {
        "schema_version": "kvbench-phase11-exact-test-result-1.0.0",
        "name": evidence_name,
        "command_argv": list(command),
        "source_path": source.relative_to(REPOSITORY_ROOT).as_posix(),
        "source_sha256": sha256_file(source),
        "returncode": result.returncode,
        "timed_out": result.timed_out,
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "process_supervision": result.to_dict(),
        "performance_timing": False,
        "passed": passed,
    }
    run.write_json(f"{prefix}.json", record)
    if not passed:
        raise Phase11KVQuantDriverError(
            f"exact {evidence_name} check failed"
        )
    return record


def _last_json(stdout: bytes) -> dict[str, Any] | None:
    lines = stdout.decode("utf-8", errors="replace").splitlines()
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _zero_memcheck(stdout: bytes, stderr: bytes) -> bool:
    text = (stdout + b"\n" + stderr).decode("utf-8", errors="replace")
    summaries = re.findall(r"ERROR SUMMARY:\s*(\d+)\s+errors?", text)
    leaks = re.findall(
        r"LEAK SUMMARY:\s*(\d+)\s+bytes leaked in\s+(\d+)\s+allocations?",
        text,
    )
    return bool(summaries) and all(int(item) == 0 for item in summaries) and (
        not leaks
        or all(int(byte_count) == 0 and int(count) == 0 for byte_count, count in leaks)
    )


def _sanitizer_probe_matches(probe: object, mode: str) -> bool:
    if not isinstance(probe, Mapping) or probe.get("status") != "PASS":
        return False
    results = probe.get("results")
    return (
        isinstance(results, list)
        and len(results) == 1
        and isinstance(results[0], Mapping)
        and results[0].get("mode") == mode
    )


def _validate_sm120_code_objects(extension: Path) -> dict[str, Any]:
    cuobjdump = Path("/usr/local/cuda-13.0/bin/cuobjdump")
    if not cuobjdump.is_file():
        raise Phase11KVQuantDriverError("cuobjdump is absent")
    records: list[dict[str, Any]] = []
    for arguments, marker, label in (
        (("--list-elf", str(extension)), b".sm_120.cubin", "sm_120_cubin"),
        (("--dump-ptx", str(extension)), b".target sm_120", "compute_120_ptx"),
    ):
        result = subprocess.run(
            (str(cuobjdump), *arguments),
            check=False,
            capture_output=True,
        )
        passed = result.returncode == 0 and marker in result.stdout
        records.append(
            {
                "label": label,
                "command_argv": [str(cuobjdump), *arguments],
                "returncode": result.returncode,
                "stdout_sha256": hashlib.sha256(result.stdout).hexdigest(),
                "stderr_sha256": hashlib.sha256(result.stderr).hexdigest(),
                "passed": passed,
            }
        )
        if not passed:
            raise Phase11KVQuantDriverError(
                f"fresh extension lacks required {label}"
            )
    return {
        "schema_version": "kvbench-phase11-sm120-code-object-check-1.0.0",
        "records": records,
        "native_sm120": True,
        "sm_120_cubin": True,
        "compute_120_ptx": True,
    }


def _run_sanitizer(run: ArtifactRun) -> Phase11SanitizerEvidence:
    if not SANITIZER.is_file() or not SANITIZER_PROBE.is_file():
        raise Phase11KVQuantDriverError("Phase 11 sanitizer authority is absent")
    version = subprocess.run(
        (str(SANITIZER), "--version"),
        check=False,
        capture_output=True,
        text=True,
        env=_child_environment(),
    )
    if version.returncode != 0:
        raise Phase11KVQuantDriverError("Compute Sanitizer version is unavailable")
    aggregate_stdout = bytearray()
    aggregate_stderr = bytearray()
    records: list[dict[str, Any]] = []
    for tool, mode in _SANITIZER_RUNS:
        command = (
            str(SANITIZER),
            "--tool",
            tool,
            "--error-exitcode",
            "99",
            "--target-processes",
            "application-only",
            *(
                ("--leak-check", "full")
                if tool == "memcheck"
                else ()
            ),
            str(CONTAINER_PYTHON),
            str(SANITIZER_PROBE),
            "--image-config-digest",
            PHASE11_AUTHORIZED_CONTAINER_DIGEST,
            "--mode",
            mode,
        )
        result = run_supervised_command(
            command,
            working_directory=str(REPOSITORY_ROOT),
            environment={
                **_child_environment(),
                "PYTORCH_NO_CUDA_MEMORY_CACHING": "1",
            },
            timeout_seconds=1800.0,
        )
        stdout = result.stdout
        stderr = result.stderr
        aggregate_stdout.extend(stdout)
        aggregate_stderr.extend(stderr)
        prefix = f"validation/sanitizer/{tool}-{mode}"
        run.write_bytes(f"{prefix}.stdout.txt", stdout)
        run.write_bytes(f"{prefix}.stderr.txt", stderr)
        probe = _last_json(stdout)
        passed = (
            result.returncode == 0
            and not result.timed_out
            and _zero_memcheck(stdout, stderr)
            and _sanitizer_probe_matches(probe, mode)
        )
        record = {
            "tool": tool,
            "mode": mode,
            "command_argv": list(command),
            "returncode": result.returncode,
            "timed_out": result.timed_out,
            "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
            "probe": probe,
            "process_supervision": result.to_dict(),
            "passed": passed,
        }
        run.write_json(f"{prefix}.json", record)
        records.append(record)
        if not passed:
            raise Phase11KVQuantDriverError(
                f"Compute Sanitizer failed for {tool}/{mode}"
            )
    stdout_digest = hashlib.sha256(bytes(aggregate_stdout)).hexdigest()
    stderr_digest = hashlib.sha256(bytes(aggregate_stderr)).hexdigest()
    evidence = Phase11SanitizerEvidence(
        cases=PHASE11_SANITIZER_CASES,
        container_digest=PHASE11_AUTHORIZED_CONTAINER_DIGEST,
        corrected_tree=PHASE11_CORRECTED_TREE,
        extension_sha256=PHASE11_EXTENSION_SHA256,
        command_argv=tuple(
            item for tool, mode in _SANITIZER_RUNS for item in (tool, mode)
        ),
        tool_version=" ".join(version.stdout.split()),
        stdout_sha256=stdout_digest,
        stderr_sha256=stderr_digest,
        memory_errors=0,
        leaked_allocations=0,
        unsupported_architecture_fallback=False,
    )
    run.write_json(
        "validation/sanitizer-runs.json",
        {
            "schema_version": "kvbench-phase11-sanitizer-runs-1.0.0",
            "probe_source_sha256": sha256_file(SANITIZER_PROBE),
            "runs": records,
            "performance_timing": False,
        },
    )
    return evidence


def _fixture_evidence() -> Phase11FixtureEvidence:
    fixtures = load_all_kvquant_fixtures()
    if len(fixtures) != 9:
        raise Phase11KVQuantDriverError("corrected fixture matrix is incomplete")
    return Phase11FixtureEvidence(
        fixture_id=PHASE11_FIXTURE_ID,
        fixture_root=PHASE11_FIXTURE_ROOT,
        cases=PHASE11_FIXTURE_CASES,
        input_and_pre_rope_exact=True,
        dense_payload_exact=True,
        metadata_exact=True,
        sparse_values_indices_counts_exact=True,
        unused_slots_exact=True,
        sink_exact=True,
        store_exact=True,
        append_exact=True,
        byte_breakdown_exact=True,
        decode_atol=0.01,
        decode_rtol=0.01,
        decode_within_tolerance=True,
        finite_output=True,
        kvq4_phase10_byte_identical=True,
        kvq2_phase10_byte_identical=True,
        kvq3_decision_0025_corrected=True,
        reuse_proof_valid=True,
    )


def _static_execution_path() -> tuple[Phase11ExecutionPathEvidence, ...]:
    measured = "\n".join(
        inspect.getsource(getattr(KVQuantMethodAdapter, name))
        for name in (
            "_pack_nonsink_token",
            "append_decode",
            "_decode_compressed",
            "_decode_quantized_value",
            "decode_attention",
        )
    )
    forbidden = (
        "torch.cat(",
        "repeat_kv",
        "repeat_interleave",
        ".item(",
        ".tolist(",
        ".cpu(",
        "torch.topk(",
    )
    if any(token in measured for token in forbidden):
        raise Phase11KVQuantDriverError(
            "KVQuant measured source contains a forbidden operation"
        )

    def bindings(name: str) -> tuple[set[str], set[str]]:
        tree = ast.parse(
            textwrap.dedent(
                inspect.getsource(getattr(KVQuantMethodAdapter, name))
            )
        )
        direct_calls = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
        }
        getattr_templates: set[str] = set()
        for node in ast.walk(tree):
            if (
                not isinstance(node, ast.Call)
                or not isinstance(node.func, ast.Call)
                or not isinstance(node.func.func, ast.Name)
                or node.func.func.id != "getattr"
                or len(node.func.args) != 2
                or not isinstance(node.func.args[1], ast.JoinedStr)
            ):
                continue
            parts: list[str] = []
            for value in node.func.args[1].values:
                if isinstance(value, ast.Constant) and isinstance(
                    value.value,
                    str,
                ):
                    parts.append(value.value)
                elif (
                    isinstance(value, ast.FormattedValue)
                    and isinstance(value.value, ast.Attribute)
                    and isinstance(value.value.value, ast.Name)
                    and value.value.value.id == "self"
                    and value.value.attr == "bits"
                ):
                    parts.append("{self.bits}")
                else:
                    parts = []
                    break
            if parts:
                getattr_templates.add("".join(parts))
        return direct_calls, getattr_templates

    pack_calls, pack_templates = bindings("_pack_nonsink_token")
    _, store_templates = bindings("store_prefill")
    _, decode_templates = bindings("_decode_compressed")
    _, value_decode_templates = bindings("_decode_quantized_value")
    value_decode_source = inspect.getsource(
        KVQuantMethodAdapter._decode_quantized_value
    )
    required_bindings = (
        "select_fixed_outliers_1024_cap12_out" in pack_calls,
        "key_sparse_residual_1024_cap12_out" in pack_calls,
        "append_value_sparse_1024_cap12_out" in pack_calls,
        "vecquant{self.bits}appendvecKsparse" in pack_templates,
        (
            "vecquant{self.bits}appendvecVsparseParallel"
            in store_templates
        ),
        (
            "vecquant{self.bits}matmul_nuq_perchannel_transposed_"
            "rope_mha_batched_fused_opt2"
        )
        in decode_templates,
        (
            "vecquant{self.bits}matmul_nuq_perchannel_transposed_"
            "mha_batched_fused_opt2"
        )
        in value_decode_templates,
        "if self.bits == 4:" in value_decode_source,
        "cache.q4_value_decode_workspace" in value_decode_source,
        (
            "KVQUANT_Q4_DETERMINISTIC_VALUE_DECODE_API"
            in value_decode_source
        ),
        (
            KVQUANT_Q4_DETERMINISTIC_VALUE_DECODE_API
            .endswith("_deterministic_out")
        ),
    )
    if not all(required_bindings):
        raise Phase11KVQuantDriverError(
            "KVQuant direct compressed source binding is incomplete"
        )
    return tuple(
        Phase11ExecutionPathEvidence(
            configuration=configuration,
            caller_owned_outputs=True,
            device_resident_value_parameters=True,
            current_cuda_stream=True,
            corrected_kvq3_pack=True,
            deterministic_q4_value_decode=True,
            caller_owned_q4_value_workspace=True,
            fixed_order_q4_value_reduction=True,
            direct_compressed_decode=True,
            native_gqa=True,
            value_fixed_12=True,
            no_cpu_topk=True,
            no_dynamic_sparse_allocation=True,
            no_tensor_to_host=True,
            no_host_synchronization=True,
            no_complete_prefix_materialization=True,
            no_gqa_expansion=True,
            no_repeat_kv=True,
            no_repeat_interleave=True,
            no_query_head_sized_cache=True,
            no_measured_torch_cat=True,
            stable_kernel_path=True,
            no_backend_fallback=True,
        )
        for configuration in PHASE11_CONFIGURATIONS
    )


def _byte_accounting(
    cache: Any,
    *,
    configuration: str,
    active_context: int,
    endpoint_rope_scratch_bytes: int,
) -> Phase11ByteAccounting:
    if (
        type(endpoint_rope_scratch_bytes) is not int
        or endpoint_rope_scratch_bytes <= 0
    ):
        raise Phase11KVQuantDriverError(
            "endpoint RoPE scratch bytes are absent"
        )
    observed = dict(cache.byte_breakdown())
    observed["staging"] = (
        int(observed["staging"]) + endpoint_rope_scratch_bytes
    )
    required = {
        "dense_k_payload",
        "dense_v_payload",
        "key_metadata",
        "value_metadata",
        "key_sparse_values",
        "key_sparse_indices",
        "value_sparse_values",
        "value_sparse_indices",
        "active_count_mask",
        "sink_k",
        "sink_v",
        "staging",
        "padding_alignment",
        "persistent_workspace",
    }
    if set(observed) != required:
        raise Phase11KVQuantDriverError("KVQuant byte categories differ")
    breakdown = Phase11ByteBreakdown(**observed)
    raw = cache.accounting()
    allocated = int(raw.allocated_bytes) + endpoint_rope_scratch_bytes
    predicted = (
        int(raw.predicted_tensor_bytes) + endpoint_rope_scratch_bytes
    )
    logical = int(cache.logical_bf16_storage_bytes)
    active = int(
        cache.active_storage_bytes(
            active_context,
            key_active_entries=0,
        )
    )
    return Phase11ByteAccounting(
        configuration=configuration,
        capacity=int(cache.capacity),
        active_context=active_context,
        allocated_bytes=allocated,
        predicted_allocated_bytes=predicted,
        active_storage_bytes=active,
        logical_bf16_allocated_bytes=logical,
        logical_bf16_active_bytes=int(
            cache.active_logical_bf16_bytes(active_context)
        ),
        rho_alloc=allocated / logical,
        r_alloc=logical / allocated,
        predicted_relative_error=abs(predicted - allocated) / allocated,
        temporary_peak_bytes=int(raw.temporary_peak_bytes),
        breakdown=breakdown,
        r_hbm=None,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class _Phase11AllocationBinding:
    configuration: str
    runner_kind: str
    graph_mode: str
    historical_context: int
    attended_context: int
    operation_fingerprint_sha256: str
    cache_layout_fingerprint: str
    method_fingerprint: str
    backend_identity: str
    adapter_source_sha256: str
    cache_source_sha256: str
    endpoint_source_sha256: str

    @property
    def execution_mode(self) -> str:
        return "cuda_graph" if self.graph_mode == "cuda_graph" else "eager"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": (
                "kvbench-phase11-kvquant-allocation-binding-1.0.0"
            ),
            "configuration": self.configuration,
            "runner_kind": self.runner_kind,
            "graph_mode": self.graph_mode,
            "execution_mode": self.execution_mode,
            "historical_context": self.historical_context,
            "attended_context": self.attended_context,
            "operation_fingerprint_sha256": (
                self.operation_fingerprint_sha256
            ),
            "cache_layout_fingerprint": self.cache_layout_fingerprint,
            "method_fingerprint": self.method_fingerprint,
            "backend_identity": self.backend_identity,
            "adapter_source_sha256": self.adapter_source_sha256,
            "cache_source_sha256": self.cache_source_sha256,
            "endpoint_source_sha256": self.endpoint_source_sha256,
            "method_identifier": PHASE11_METHOD_IDENTIFIER,
            "execution_source_identifier": (
                PHASE11_EXECUTION_SOURCE_IDENTIFIER
            ),
            "corrected_commit": PHASE11_CORRECTED_COMMIT,
            "corrected_tree": PHASE11_CORRECTED_TREE,
            "aggregate_patch_sha256": PHASE11_AGGREGATE_PATCH_SHA256,
            "extension_sha256": PHASE11_EXTENSION_SHA256,
            "calibration_root": PHASE11_CALIBRATION_ROOT,
            "fixture_root": PHASE11_FIXTURE_ROOT,
            "authorized_container_digest": (
                PHASE11_AUTHORIZED_CONTAINER_DIGEST
            ),
            "sink_tokens": 5,
            "key_cap": 12,
            "value_cap": 12,
            "allocation_rule_authority": (
                "Decision 0013 direct-compressed endpoint composition"
            ),
        }

    @property
    def identity_sha256(self) -> str:
        return sha256_hex(canonical_json_bytes(self.to_dict()))


def _phase11_allocation_geometry(
    binding: _Phase11AllocationBinding,
) -> AllocationGeometry:
    return AllocationGeometry(
        batch=1,
        query_heads=32,
        kv_heads=8,
        context=binding.attended_context,
        head_dim=128,
        dtype_bytes=2,
        query_length=1,
        operation_output_width=PHASE3_OUTPUT_WIDTH,
        operation_output_dtype_bytes=PHASE3_OUTPUT_DTYPE_BYTES,
    )


def _evaluate_phase11_eager_allocations(
    attribution: Any,
    memory: Any,
) -> AllocationCriterionResult:
    """Apply Decision 0013 to ordinary bounded model ephemerals."""

    reasons: list[str] = []
    history = attribution.history_integrity
    if attribution.integrity_errors:
        reasons.append("allocator_trace_integrity_failure")
    if history is None:
        reasons.append("allocator_history_integrity_missing")
    else:
        reasons.extend(history.failure_reasons())
        if history.raw_trace_sha256 != attribution.trace_sha256:
            reasons.append("attributed_trace_sha256_mismatch")
    if any(
        not allocation.python_stack or not allocation.cpp_stack
        for allocation in attribution.allocations
    ):
        reasons.append("allocator_allocation_stack_incomplete")
    if not attribution.counters.complete:
        reasons.append("allocator_counter_evidence_incomplete")
    if not attribution.all_block_sizes_proven:
        reasons.append("allocated_block_size_evidence_incomplete")
    if not attribution.all_lifetimes_fully_freed:
        reasons.append("allocation_lifetime_not_fully_freed")
    if not attribution.all_allocations_cache_reused:
        reasons.append("allocation_not_reused_from_cache")
    if attribution.segment_alloc_count or attribution.segment_free_count:
        reasons.append("allocator_segment_event_detected")
    for name in (
        "device_allocation_count",
        "device_free_count",
        "allocation_retry_count",
        "oom_count",
    ):
        if getattr(attribution.counters, name) != 0:
            reasons.append(f"allocator_{name}_nonzero_or_unavailable")
    if memory.allocated_delta != 0:
        reasons.append("persistent_allocated_delta_nonzero")
    if memory.reserved_delta != 0:
        reasons.append("persistent_reserved_delta_nonzero")
    if memory.device_used_delta != 0:
        reasons.append("persistent_device_used_delta_nonzero_or_unavailable")
    if memory.non_pytorch_delta != 0:
        reasons.append("persistent_non_pytorch_delta_nonzero_or_unavailable")

    policies = attribution.rules.permitted_allocation_policies
    allowed_ids = {policy.policy_id for policy in policies}
    for policy in policies:
        observed = [
            allocation
            for allocation in attribution.allocations
            if allocation.policy_id == policy.policy_id
        ]
        if len(observed) != policy.exact_count:
            reasons.append(
                f"allocation_policy_count_bound_failed:{policy.policy_id}"
            )
        if (
            sum(allocation.requested_bytes for allocation in observed)
            != policy.exact_total_requested_bytes
        ):
            reasons.append(
                f"allocation_policy_byte_bound_failed:{policy.policy_id}"
            )
    for allocation in attribution.allocations:
        if (
            allocation.event_class in _FORBIDDEN_ALLOCATION_CLASSES
            or allocation.policy_id not in allowed_ids
        ):
            reasons.append(
                "forbidden_or_unattributed_allocation:"
                f"{allocation.allocation_id}"
            )
    no_context_dependent = all(
        allocation.dependencies.context is False
        for allocation in attribution.allocations
    )
    if not no_context_dependent:
        reasons.append("context_dependent_allocation_detected")
    unique = tuple(dict.fromkeys(reasons))
    passed = not unique
    return AllocationCriterionResult(
        criterion_id=(
            "phase11_kvquant_decision_0013_composed_eager_attribution_v1"
        ),
        passed=passed,
        failure_reasons=unique,
        allocation_event_count=len(attribution.allocations),
        class_counts=attribution.class_counts(),
        no_context_dependent_allocation=no_context_dependent,
        fully_attributed_bounded_ephemeral=passed,
        strict_graph_zero_events=None,
    )


def _collect_phase11_allocation_attribution(
    session: Any,
    *,
    step: int,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    import torch

    operation_key = session.operation_keys[step]
    binding = _Phase11AllocationBinding(
        configuration=operation_key.configuration,
        runner_kind=operation_key.runner_kind.value,
        graph_mode=operation_key.graph_mode.value,
        historical_context=operation_key.historical_context,
        attended_context=operation_key.attended_context,
        operation_fingerprint_sha256=(
            operation_key.operation_fingerprint_sha256
        ),
        cache_layout_fingerprint=session.cache_layout_fingerprint(),
        method_fingerprint=session.adapter_config_fingerprint,
        backend_identity=session.method.runtime_context.backend_fingerprint,
        adapter_source_sha256=sha256_file(ADAPTER_PATH),
        cache_source_sha256=sha256_file(CACHE_PATH),
        endpoint_source_sha256=sha256_file(ENDPOINT_PATH),
    )

    def capture_state() -> dict[str, Any]:
        return {
            "cache_pointers_sha256": sha256_hex(
                canonical_json_bytes(session.current_cache_pointers())
            ),
            "active_context": session.active_context,
        }

    def capture_output(value: Any) -> dict[str, Any]:
        cpu_value = value.detach().to(device="cpu", copy=True)
        return {
            "sha256": tensor_sha256_untimed(cpu_value),
            "finite": bool(torch.isfinite(cpu_value).all()),
        }

    raw = collect_cuda_allocator_raw(
        lambda: session.execute_audit_step(step),
        operation_fingerprint_sha256=(
            binding.operation_fingerprint_sha256
        ),
        prepare_operation=lambda: session.prepare_audit_step(step),
        capture_state=capture_state,
        capture_output=capture_output,
        device=session.cache_device,
        max_entries=PHASE3_ALLOCATION_MAX_HISTORY_ENTRIES,
    )
    if (
        raw.state_before is None
        or raw.state_after is None
        or raw.output_witness is None
    ):
        raise Phase11KVQuantDriverError(
            "allocator collection lacks operation witnesses"
        )
    expected_after = (
        binding.historical_context
        if binding.runner_kind == RunnerKind.FIXED_L.value
        else binding.attended_context
    )
    if (
        raw.state_before.get("cache_pointers_sha256")
        != raw.state_after.get("cache_pointers_sha256")
        or raw.state_before.get("active_context")
        != binding.historical_context
        or raw.state_after.get("active_context") != expected_after
        or raw.output_witness.get("finite") is not True
    ):
        raise Phase11KVQuantDriverError(
            "allocator operation witness did not pass"
        )
    witness = {
        "schema_version": (
            "kvbench-phase11-kvquant-allocation-operation-witness-1.0.0"
        ),
        "binding_sha256": binding.identity_sha256,
        "operation_fingerprint_sha256": (
            binding.operation_fingerprint_sha256
        ),
        "state_before": dict(raw.state_before),
        "state_after": dict(raw.state_after),
        "measured_output": dict(raw.output_witness),
    }
    geometry = _phase11_allocation_geometry(binding)
    rules = instantiate_decision_0013_direct_compressed_rules(
        geometry=geometry,
        backend_identity=binding.backend_identity,
        composition_binding_sha256=binding.identity_sha256,
    )
    history = build_history_integrity_evidence(
        raw.snapshot,
        raw.trace,
        max_entries=PHASE3_ALLOCATION_MAX_HISTORY_ENTRIES,
        stack_mode="all",
    )
    attribution = attribute_allocator_trace(
        raw.trace,
        geometry=geometry,
        counters=allocator_counters_from_memory_stats(
            raw.memory_stats_before,
            raw.memory_stats_after,
        ),
        rules=rules,
        backend_identity=binding.backend_identity,
        expected_trace_sha256=history.expected_raw_trace_sha256,
        history_integrity=history,
    )
    before = raw_memory_accounting_sample_from_mapping(
        raw.memory_accounting_before
    )
    after = raw_memory_accounting_sample_from_mapping(
        raw.memory_accounting_after
    )
    memory = memory_delta_from_raw_samples(before, after)
    criterion = (
        evaluate_strict_graph_criterion(attribution, memory)
        if binding.execution_mode == "cuda_graph"
        else _evaluate_phase11_eager_allocations(attribution, memory)
    )
    class_counts = attribution.class_counts()
    forbidden_count = sum(
        count
        for name, count in class_counts.items()
        if AllocationClass(name) in _FORBIDDEN_ALLOCATION_CLASSES
    )
    observed_bytes = sum(
        allocation.requested_bytes
        for allocation in attribution.allocations
    )
    expected_count = sum(
        policy.exact_count for policy in rules.permitted_allocation_policies
    )
    expected_bytes = sum(
        policy.exact_total_requested_bytes
        for policy in rules.permitted_allocation_policies
    )
    audit_payload = {
        "schema_version": (
            "kvbench-phase11-kvquant-allocation-attribution-1.0.0"
        ),
        "run_kind": "allocation_audit",
        "evidence_status": "complete",
        "execution_mode": binding.execution_mode,
        "binding": binding.to_dict(),
        "binding_sha256": binding.identity_sha256,
        "memory": memory.to_dict(),
        "attribution": attribution.to_dict(),
        "criterion": criterion.to_dict(),
        "expected_allocation_event_count": expected_count,
        "expected_allocation_event_bytes": expected_bytes,
        "observed_allocation_event_bytes": observed_bytes,
        "forbidden_or_unknown_allocation_count": forbidden_count,
        "operation_witness": witness,
        "profiler_timing_reported": False,
        "instrumented_duration_reported_as_timing": False,
        "normal_benchmark_timing_eligible": False,
    }
    with tempfile.TemporaryDirectory(
        prefix="kvbench-phase11-kvquant-allocation-",
        dir="/tmp",
    ) as temporary:
        directory = Path(temporary)
        os.chmod(directory, 0o700)
        preserved = preserve_allocator_evidence(
            directory,
            snapshot=raw.snapshot,
            trace=raw.trace,
            memory_stats_before=raw.memory_stats_before,
            memory_stats_after=raw.memory_stats_after,
            memory_accounting_before=raw.memory_accounting_before,
            memory_accounting_after=raw.memory_accounting_after,
            operation_witness=witness,
            expected_snapshot_sha256=history.raw_snapshot_sha256,
            expected_trace_sha256=history.raw_trace_sha256,
            audit_payload=audit_payload,
        )
        names = (
            preserved.snapshot_file,
            preserved.trace_file,
            preserved.memory_stats_before_file,
            preserved.memory_stats_after_file,
            preserved.memory_accounting_before_file,
            preserved.memory_accounting_after_file,
            preserved.operation_witness_file,
            preserved.audit_file,
            preserved.audit_sha256_file,
        )
        files = {name: (directory / name).read_bytes() for name in names}
    passed = (
        criterion.passed
        and forbidden_count == 0
        and memory.allocated_delta == 0
        and memory.reserved_delta == 0
    )
    summary = {
        "passed": passed,
        "raw": {
            "allocation_event_count": len(attribution.allocations),
            "allocation_event_bytes": observed_bytes,
            "event_counts": dict(
                sorted(Counter(event.action for event in attribution.events).items())
            ),
        },
        "criterion": {
            **criterion.to_dict(),
            "expected_allocation_event_count": expected_count,
            "expected_allocation_event_bytes": expected_bytes,
            "allowed_model_ephemeral_count": (
                len(attribution.allocations)
                if binding.execution_mode == "eager"
                else 0
            ),
            "cache_growth_count": int(
                class_counts.get(AllocationClass.CACHE_GROWTH.value, 0)
            ),
            "dynamic_sparse_allocation_count": (
                0 if forbidden_count == 0 else None
            ),
            "complete_prefix_allocation_count": int(
                class_counts.get(
                    AllocationClass.CONTEXT_SCALED_WORKSPACE.value,
                    0,
                )
            ),
            "gqa_expanded_allocation_count": int(
                class_counts.get(AllocationClass.GQA_EXPANSION.value, 0)
            ),
            "unknown_allocation_count": int(
                class_counts.get(AllocationClass.UNKNOWN.value, 0)
            ),
            "persistent_allocated_delta": memory.allocated_delta,
            "persistent_reserved_delta": memory.reserved_delta,
            "attribution_rules_sha256": rules.identity_sha256,
            "composition_binding_sha256": binding.identity_sha256,
        },
        "raw_files": preserved.to_dict(),
    }
    return summary, files


def _deterministic_ids(
    torch: Any,
    *,
    length: int,
    offset: int,
    device: Any,
) -> Any:
    values = torch.arange(length, dtype=torch.long, device=device).reshape(
        1, length
    )
    return (values + offset) % 120_000 + 1_000


def _normalize_runner_result(value: Mapping[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(value))
    timing = result.get("timing")
    if (
        result.get("measurement_scope")
        != MeasurementScope.MEASUREMENT_CONTAINER_ADMISSION.value
        or result.get("performance_claim_eligible") is not False
        or not isinstance(timing, dict)
        or timing.get("paper_claim_eligible") is not False
    ):
        raise Phase11KVQuantDriverError("common runner governance differs")
    timing["measurement_scope"] = (
        MeasurementScope.MEASUREMENT_CONTAINER_ADMISSION.value
    )
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


def _execute_grid_point(
    *,
    loaded: Any,
    bundle_id: str,
    index: int,
    signature: tuple[str, RunnerKind, GraphMode, int, int],
) -> tuple[
    Phase11RunPoint,
    dict[str, Any],
    Any,
    tuple[dict[str, Any], ...],
    dict[str, bytes],
]:
    import torch

    configuration, runner_kind, graph_mode, context, output_steps = signature
    run_id = _point_run_id(bundle_id, index, signature)
    keys = build_kvquant_operation_keys(
        configuration=configuration,
        runner_kind=runner_kind,
        graph_mode=graph_mode,
        starting_context=context,
        output_steps=output_steps,
    )
    offset = 11_000 + index * 20_000 + context
    device = torch.device("cuda:0")
    prefix = _deterministic_ids(
        torch,
        length=context,
        offset=offset,
        device=device,
    )
    decode = _deterministic_ids(
        torch,
        length=output_steps,
        offset=offset + context + 257,
        device=device,
    )
    with forced_flash_execution():
        session = build_kvquant_endpoint_session(
            loaded=loaded,
            operation_keys=keys,
            prefix_input_ids=prefix,
            decode_input_ids=decode,
        )
        pointers_before = session.current_cache_pointers()
        observed: list[tuple[str, bool]] = []
        audits: list[dict[str, Any]] = []
        allocation_evidence: dict[str, bytes] = {}
        for step in range(len(keys)):
            session.prepare_audit_step(step)
            output = session.execute_audit_step(step)
            torch.cuda.synchronize(device=session.cache_device)
            cpu = output.detach().to(device="cpu", copy=True)
            observed.append(
                (
                    tensor_sha256_untimed(cpu),
                    bool(torch.isfinite(cpu).all()),
                )
            )
            audit, raw_files = _collect_phase11_allocation_attribution(
                session,
                step=step,
            )
            evidence_root = (
                f"allocation/operations/{run_id}/step-{step:04d}"
            )
            audit["raw_evidence_root"] = evidence_root
            audit["raw_evidence_sha256"] = {
                name: hashlib.sha256(payload).hexdigest()
                for name, payload in sorted(raw_files.items())
            }
            audits.append(audit)
            for name, payload in raw_files.items():
                relative = f"{evidence_root}/{name}"
                if relative in allocation_evidence:
                    raise Phase11KVQuantDriverError(
                        "allocator raw evidence path is duplicated"
                    )
                allocation_evidence[relative] = payload
        graph_passed = bool(
            graph_mode is not GraphMode.CUDA_GRAPH
            or (
                session.graph is not None
                and session.graph_evidence is not None
                and session.graph_evidence.get("fallback") is False
                and session.graph_evidence.get(
                    "consecutive_replay_outputs_exact"
                )
                is True
                and session.eager_graph_comparison is not None
                and session.eager_graph_comparison.passed
            )
        )
        allocation_passed = bool(
            audits
            and all(
                item["passed"] is True
                and item["criterion"]["cache_growth_count"] == 0
                and item["criterion"]["dynamic_sparse_allocation_count"] == 0
                and item["criterion"]["complete_prefix_allocation_count"] == 0
                and item["criterion"]["gqa_expanded_allocation_count"] == 0
                and item["criterion"]["unknown_allocation_count"] == 0
                and item["criterion"]["persistent_allocated_delta"] == 0
                and item["criterion"]["persistent_reserved_delta"] == 0
                for item in audits
            )
        )
        session.admit(
            observed_outputs=observed,
            execution_path_passed=True,
            allocation_passed=allocation_passed,
            graph_passed=graph_passed,
        )
    if runner_kind is RunnerKind.FIXED_L:
        raw_runner = run_fixed_l(
            session,
            measured_steps=1,
            measured_batches=1,
        ).to_dict()
    else:
        raw_runner = run_growing_context(
            session,
            expected_steps=4,
        ).to_dict()
    runner = _normalize_runner_result(raw_runner)
    if (
        runner["output_finite"] is not True
        or runner["cache_pointers_stable"] is not True
        or runner["historical_cache_unchanged"] is not True
        or runner["gqa_cache_geometry"]["native_kv_head_storage"] is not True
        or runner["gqa_cache_geometry"]["query_head_sized_kv_cache"] is not False
        or pointers_before != session.current_cache_pointers()
    ):
        raise Phase11KVQuantDriverError("bounded admission point failed")
    point_payload = {
        "schema_version": "kvbench-phase11-kvquant-point-1.0.0",
        "run_id": run_id,
        "configuration": configuration,
        "runner_kind": runner_kind.value,
        "graph_mode": graph_mode.value,
        "batch_size": 1,
        "context_length": context,
        "output_steps": output_steps,
        "quality_status": "unvalidated",
        "claim_eligibility": "performance_only",
        "performance_claim_eligible": False,
        "measurement_scope": "measurement_container_admission",
        "speedup_calculated": False,
        "runner": runner,
        "allocation_audits": audits,
    }
    digest = hashlib.sha256(canonical_json_bytes(point_payload)).hexdigest()
    point = Phase11RunPoint(
        run_id=run_id,
        configuration=configuration,
        runner_kind=runner_kind,
        graph_mode=graph_mode,
        batch_size=1,
        context_length=context,
        output_steps=output_steps,
        status=RunStatus.COMPLETED,
        manifest_sha256=digest,
        quality_status=QualityValidationState.UNVALIDATED,
        claim_eligibility=ClaimEligibility.PERFORMANCE_ONLY,
        performance_claim_eligible=False,
        measurement_scope=MeasurementScope.MEASUREMENT_CONTAINER_ADMISSION,
        speedup_calculated=False,
    )
    return (
        point,
        point_payload,
        session,
        tuple(audits),
        allocation_evidence,
    )


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, nested in pairs:
            if key in value:
                raise ValueError("duplicate key")
            value[key] = nested
        return value

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite value {token}")
            ),
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise Phase11KVQuantDriverError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise Phase11KVQuantDriverError(f"JSON is not an object: {path}")
    return value


def _expected_execution_changed_paths() -> tuple[str, ...]:
    manifest = _strict_json(_EXECUTION_PATCH_MANIFEST_PATH)
    records = manifest.get("patched_files")
    if (
        not isinstance(records, list)
        or len(records) != 18
        or any(
            not isinstance(record, Mapping)
            or set(record)
            != {
                "path",
                "change_type",
                "base_git_blob",
                "patched_git_blob",
                "base_sha256",
                "patched_sha256",
            }
            or not isinstance(record.get("path"), str)
            or not record["path"]
            for record in records
        )
    ):
        raise Phase11KVQuantDriverError(
            "Decision 0027 changed-file authority differs"
        )
    paths = tuple(str(record["path"]) for record in records)
    if len(set(paths)) != len(paths):
        raise Phase11KVQuantDriverError(
            "Decision 0027 changed-file authority contains duplicates"
        )
    return paths


def _require_sha256_value(value: object, *, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise Phase11KVQuantDriverError(f"{label} is not a SHA-256")
    return value


def _validate_authority_environment_and_gqa(
    bundle: Path,
    *,
    manifest: Phase11RunManifest,
    path_payload: Mapping[str, Any],
) -> tuple[Phase11Authority, dict[str, Any], dict[str, Any]]:
    from kvbench.runtime.turboquant_admission import (
        EXPECTED_CUDA_RUNTIME,
        EXPECTED_GPU_NAME,
        EXPECTED_GPU_UUID,
        EXPECTED_TORCH_VERSION,
        EXPECTED_TRITON_VERSION,
        PHASE6_CONTAINER_ENVIRONMENT_VALUE,
    )

    authority_payload = _strict_json(bundle / "config/authority.json")
    try:
        authority = Phase11Authority.from_dict(authority_payload)
    except (TypeError, ValueError) as error:
        raise Phase11KVQuantDriverError(
            "Phase 11 authority record differs"
        ) from error
    if authority != manifest.authority or authority != _authority():
        raise Phase11KVQuantDriverError(
            "Phase 11 authority record does not bind the manifest"
        )

    environment = _strict_json(bundle / "environment/container_identity.json")
    source = environment.get("source_validation")
    code_objects = environment.get("fresh_build_code_objects")
    calibration = environment.get("calibration_validation")
    git_binding = environment.get("git_source_binding")
    if (
        not isinstance(source, Mapping)
        or not isinstance(code_objects, Mapping)
        or not isinstance(calibration, Mapping)
        or not isinstance(git_binding, Mapping)
    ):
        raise Phase11KVQuantDriverError(
            "Phase 11 execution environment identity is incomplete"
        )
    aggregate_paths = source.get("aggregate_changed_paths")
    parent_paths = source.get("parent_relative_changed_paths")
    code_records = code_objects.get("records")
    expected_environment_keys = {
        "schema_version",
        "container_digest",
        "execution_environment",
        "torch",
        "triton",
        "cuda_runtime",
        "compute_capability",
        "gpu_name",
        "gpu_uuid",
        "native_host_cuda_rejected",
        "image_changed",
        "packages_installed",
        "network_enabled",
        "credentials_passed",
        "calibration_validation",
        "git_source_binding",
        "source_validation",
        "authority_extension_path",
        "authority_extension_sha256",
        "fresh_build_extension_path",
        "fresh_build_extension_sha256",
        "fresh_build_byte_identical_to_authority",
        "nvcc_cuda_object_byte_reproducible",
        "fresh_build_source_equivalent",
        "fresh_build_code_objects",
    }
    if (
        set(environment) != expected_environment_keys
        or set(source)
        != {
            "status",
            "decision",
            "decision_status",
            "patched_commit",
            "patched_tree",
            "aggregate_patch_sha256",
            "aggregate_changed_paths",
            "parent_commit",
            "parent_tree",
            "parent_relative_changed_paths",
            "source_contract",
            "reconstruction",
        }
        or set(code_objects)
        != {
            "schema_version",
            "records",
            "native_sm120",
            "sm_120_cubin",
            "compute_120_ptx",
        }
        or environment.get("schema_version")
        != "kvbench-phase11-container-runtime-1.0.0"
        or environment.get("container_digest")
        != PHASE11_AUTHORIZED_CONTAINER_DIGEST
        or environment.get("execution_environment")
        != PHASE6_CONTAINER_ENVIRONMENT_VALUE
        or environment.get("torch") != EXPECTED_TORCH_VERSION
        or environment.get("triton") != EXPECTED_TRITON_VERSION
        or environment.get("cuda_runtime") != EXPECTED_CUDA_RUNTIME
        or environment.get("compute_capability") != "12.0"
        or environment.get("gpu_name") != EXPECTED_GPU_NAME
        or environment.get("gpu_uuid") != EXPECTED_GPU_UUID
        or environment.get("native_host_cuda_rejected") is not True
        or environment.get("image_changed") is not False
        or environment.get("packages_installed") is not False
        or environment.get("network_enabled") is not False
        or environment.get("credentials_passed") is not False
        or calibration
        != {
            "schema_version": (
                "kvbench-phase11-calibration-runtime-validation-1.0.0"
            ),
            "calibration_id": PHASE11_CALIBRATION_ID,
            "mount_path": str(_RUNTIME_CALIBRATION_PATH),
            "expected_root_sha256": PHASE11_CALIBRATION_ROOT,
            "observed_root_sha256": PHASE11_CALIBRATION_ROOT,
            "object_count": _PHASE11_CALIBRATION_OBJECT_COUNT,
            "complete_marker_valid": True,
            "inventory_valid": True,
            "checksum_ledger_valid": True,
            "repository_lifecycle_valid": True,
            "validated_before_cuda": True,
        }
        or git_binding
        != _git_source_binding(
            manifest.git_sha,
            require_worktree_match=False,
        )
        or environment.get("authority_extension_sha256")
        != PHASE11_EXTENSION_SHA256
        or environment.get("fresh_build_source_equivalent") is not True
        or environment.get("nvcc_cuda_object_byte_reproducible") is not False
        or (
            environment.get("fresh_build_byte_identical_to_authority")
            is not True
        )
        or _require_sha256_value(
            environment.get("fresh_build_extension_sha256"),
            label="fresh extension identity",
        )
        != PHASE11_EXTENSION_SHA256
        or source.get("status") != "PASS"
        or source.get("decision")
        != (
            "docs/decisions/"
            "0027-kvquant-deterministic-long-context-value-decode.md"
        )
        or source.get("decision_status") != "Accepted"
        or source.get("patched_commit") != PHASE11_CORRECTED_COMMIT
        or source.get("patched_tree") != PHASE11_CORRECTED_TREE
        or source.get("aggregate_patch_sha256")
        != PHASE11_AGGREGATE_PATCH_SHA256
        or tuple(aggregate_paths or ())
        != _expected_execution_changed_paths()
        or source.get("source_contract") != "PASS"
        or source.get("parent_commit")
        != "0d9df350bd1788284e1ce76a8bf6e886beca5efa"
        or source.get("parent_tree")
        != "a85cf7bf093982a4bf89c33d4e6794d9a85f846d"
        or parent_paths
        != [
            "deployment/kvquant/quant_cuda.cpp",
            "deployment/kvquant/quant_cuda_kernel.cu",
        ]
        or code_objects.get("schema_version")
        != "kvbench-phase11-sm120-code-object-check-1.0.0"
        or code_objects.get("native_sm120") is not True
        or code_objects.get("sm_120_cubin") is not True
        or code_objects.get("compute_120_ptx") is not True
        or not isinstance(code_records, list)
        or tuple(record.get("label") for record in code_records)
        != ("sm_120_cubin", "compute_120_ptx")
        or any(
            not isinstance(record, Mapping)
            or set(record)
            != {
                "label",
                "command_argv",
                "returncode",
                "stdout_sha256",
                "stderr_sha256",
                "passed",
            }
            or record.get("returncode") != 0
            or record.get("passed") is not True
            or _require_sha256_value(
                record.get("stdout_sha256"),
                label="code-object stdout",
            )
            != record.get("stdout_sha256")
            or _require_sha256_value(
                record.get("stderr_sha256"),
                label="code-object stderr",
            )
            != record.get("stderr_sha256")
            for record in code_records
        )
    ):
        raise Phase11KVQuantDriverError(
            "Phase 11 execution environment identity differs"
        )

    expected_path_keys = {
        "schema_version",
        "adapter_version",
        "adapter_source_sha256",
        "cache_source_sha256",
        "endpoint_source_sha256",
        "session_source_sha256",
        "fixture_test_sha256",
        "graph_test_sha256",
        "records",
    }
    if (
        set(path_payload) != expected_path_keys
        or path_payload.get("schema_version")
        != "kvbench-phase11-execution-path-set-1.0.0"
        or path_payload.get("adapter_version") != KVQUANT_ADAPTER_VERSION
    ):
        raise Phase11KVQuantDriverError(
            "Phase 11 execution-path wrapper differs"
        )
    for key in (
        "adapter_source_sha256",
        "cache_source_sha256",
        "endpoint_source_sha256",
        "session_source_sha256",
        "fixture_test_sha256",
        "graph_test_sha256",
    ):
        _require_sha256_value(path_payload.get(key), label=key)
    bound_records = git_binding.get("records")
    if not isinstance(bound_records, list):
        raise Phase11KVQuantDriverError(
            "Phase 11 Git source binding records differ"
        )
    bound_hashes = {
        record.get("path"): record.get("content_sha256")
        for record in bound_records
        if isinstance(record, Mapping)
    }
    expected_wrapper_hashes = {
        "adapter_source_sha256": "src/kvbench/adapters/kvquant.py",
        "cache_source_sha256": "src/kvbench/runtime/kvquant_cache.py",
        "endpoint_source_sha256": "src/kvbench/runtime/bf16_endpoint.py",
        "session_source_sha256": "src/kvbench/runtime/kvquant_session.py",
        "fixture_test_sha256": "tests/cuda/test_phase11_kvquant_cuda.py",
        "graph_test_sha256": "tests/graph/test_phase11_kvquant_graph.py",
    }
    if any(
        path_payload[key] != bound_hashes.get(relative)
        for key, relative in expected_wrapper_hashes.items()
    ):
        raise Phase11KVQuantDriverError(
            "Phase 11 execution-path wrapper is not Git-bound"
        )

    gqa = _strict_json(bundle / "gqa/audit.json")
    expected_gqa = {
        "schema_version": "kvbench-phase11-gqa-audit-1.0.0",
        "query_heads": 32,
        "kv_heads": 8,
        "groups": 4,
        "mapping": "query_head//4",
        "native_kv_storage": True,
        "query_head_sized_cache": False,
        "repeat_kv": False,
        "passed": True,
    }
    if gqa != expected_gqa:
        raise Phase11KVQuantDriverError("Phase 11 GQA audit differs")
    return authority, environment, gqa


def _validate_phase11_allocator_semantically(
    bundle: Path,
    *,
    evidence_root: str,
    summary: Mapping[str, Any],
) -> None:
    raw_index = summary.get("raw_files")
    if not isinstance(raw_index, Mapping):
        raise Phase11KVQuantDriverError(
            "allocator summary lacks its raw-file index"
        )
    try:
        files = RawAllocatorEvidenceFiles(**dict(raw_index))
        payloads = read_verified_allocator_evidence(
            bundle / evidence_root,
            files,
        )
        parsed: dict[str, Any] = {}
        for name in (
            "snapshot",
            "trace",
            "stats_before",
            "stats_after",
            "accounting_before",
            "accounting_after",
            "operation_witness",
            "audit",
        ):
            value = json.loads(payloads[name])
            if canonical_json_bytes(value) != payloads[name]:
                raise ValueError("raw allocator JSON is not canonical")
            parsed[name] = value
        audit = parsed["audit"]
        binding_payload = audit.get("binding")
        if not isinstance(audit, Mapping) or not isinstance(
            binding_payload,
            Mapping,
        ):
            raise ValueError("allocator audit or binding is malformed")
        binding = _Phase11AllocationBinding(
            configuration=str(binding_payload["configuration"]),
            runner_kind=str(binding_payload["runner_kind"]),
            graph_mode=str(binding_payload["graph_mode"]),
            historical_context=int(binding_payload["historical_context"]),
            attended_context=int(binding_payload["attended_context"]),
            operation_fingerprint_sha256=str(
                binding_payload["operation_fingerprint_sha256"]
            ),
            cache_layout_fingerprint=str(
                binding_payload["cache_layout_fingerprint"]
            ),
            method_fingerprint=str(binding_payload["method_fingerprint"]),
            backend_identity=str(binding_payload["backend_identity"]),
            adapter_source_sha256=str(
                binding_payload["adapter_source_sha256"]
            ),
            cache_source_sha256=str(binding_payload["cache_source_sha256"]),
            endpoint_source_sha256=str(
                binding_payload["endpoint_source_sha256"]
            ),
        )
        if (
            dict(binding_payload) != binding.to_dict()
            or audit.get("binding_sha256") != binding.identity_sha256
            or audit.get("schema_version")
            != "kvbench-phase11-kvquant-allocation-attribution-1.0.0"
            or audit.get("run_kind") != "allocation_audit"
            or audit.get("evidence_status") != "complete"
            or audit.get("execution_mode") != binding.execution_mode
            or audit.get("profiler_timing_reported") is not False
            or audit.get("instrumented_duration_reported_as_timing")
            is not False
            or audit.get("normal_benchmark_timing_eligible") is not False
            or audit.get("raw_files")
            != files.to_dict(include_audit_sha256=False)
        ):
            raise ValueError("allocator audit envelope differs")
        snapshot = parsed["snapshot"]
        trace = parsed["trace"]
        if (
            not isinstance(snapshot, Mapping)
            or not isinstance(trace, list)
            or trace != allocator_trace_from_snapshot(snapshot, 0)
        ):
            raise ValueError("allocator snapshot/trace semantics differ")
        geometry = _phase11_allocation_geometry(binding)
        rules = instantiate_decision_0013_direct_compressed_rules(
            geometry=geometry,
            backend_identity=binding.backend_identity,
            composition_binding_sha256=binding.identity_sha256,
        )
        history = build_history_integrity_evidence(
            snapshot,
            trace,
            max_entries=PHASE3_ALLOCATION_MAX_HISTORY_ENTRIES,
            stack_mode="all",
            expected_snapshot_sha256=files.snapshot_sha256,
            expected_trace_sha256=files.trace_sha256,
        )
        attribution = attribute_allocator_trace(
            trace,
            geometry=geometry,
            counters=allocator_counters_from_memory_stats(
                parsed["stats_before"],
                parsed["stats_after"],
            ),
            rules=rules,
            backend_identity=binding.backend_identity,
            expected_trace_sha256=files.trace_sha256,
            history_integrity=history,
        )
        before = raw_memory_accounting_sample_from_mapping(
            parsed["accounting_before"]
        )
        after = raw_memory_accounting_sample_from_mapping(
            parsed["accounting_after"]
        )
        memory = memory_delta_from_raw_samples(before, after)
        criterion = (
            evaluate_strict_graph_criterion(attribution, memory)
            if binding.execution_mode == "cuda_graph"
            else _evaluate_phase11_eager_allocations(attribution, memory)
        )
        class_counts = attribution.class_counts()
        forbidden_count = sum(
            count
            for name, count in class_counts.items()
            if AllocationClass(name) in _FORBIDDEN_ALLOCATION_CLASSES
        )
        expected_count = sum(
            policy.exact_count for policy in rules.permitted_allocation_policies
        )
        expected_bytes = sum(
            policy.exact_total_requested_bytes
            for policy in rules.permitted_allocation_policies
        )
        observed_bytes = sum(
            allocation.requested_bytes
            for allocation in attribution.allocations
        )
        witness = parsed["operation_witness"]
        if (
            audit.get("memory") != memory.to_dict()
            or audit.get("attribution") != attribution.to_dict()
            or audit.get("criterion") != criterion.to_dict()
            or audit.get("expected_allocation_event_count") != expected_count
            or audit.get("expected_allocation_event_bytes") != expected_bytes
            or audit.get("observed_allocation_event_bytes") != observed_bytes
            or audit.get("forbidden_or_unknown_allocation_count")
            != forbidden_count
            or audit.get("operation_witness") != witness
            or not isinstance(witness, Mapping)
            or witness.get("binding_sha256") != binding.identity_sha256
            or witness.get("operation_fingerprint_sha256")
            != binding.operation_fingerprint_sha256
            or not isinstance(witness.get("measured_output"), Mapping)
            or witness["measured_output"].get("finite") is not True
        ):
            raise ValueError("allocator semantic replay differs")
        expected_pass = bool(
            criterion.passed
            and forbidden_count == 0
            and memory.allocated_delta == 0
            and memory.reserved_delta == 0
        )
        criterion_summary = summary.get("criterion")
        if (
            summary.get("passed") is not expected_pass
            or not expected_pass
            or not isinstance(criterion_summary, Mapping)
            or criterion_summary.get("attribution_rules_sha256")
            != rules.identity_sha256
            or criterion_summary.get("composition_binding_sha256")
            != binding.identity_sha256
            or criterion_summary.get("expected_allocation_event_count")
            != expected_count
            or criterion_summary.get("expected_allocation_event_bytes")
            != expected_bytes
            or criterion_summary.get("persistent_allocated_delta")
            != memory.allocated_delta
            or criterion_summary.get("persistent_reserved_delta")
            != memory.reserved_delta
            or criterion_summary.get("unknown_allocation_count")
            != int(class_counts.get(AllocationClass.UNKNOWN.value, 0))
        ):
            raise ValueError("allocator summary is not semantically derived")
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as error:
        raise Phase11KVQuantDriverError(
            "allocator raw semantic replay failed"
        ) from error


def _validate_supervised_process(
    value: object,
    *,
    command_argv: Sequence[str],
    returncode: int,
) -> None:
    if not isinstance(value, Mapping):
        raise Phase11KVQuantDriverError(
            "child process supervision evidence is absent"
        )
    command = value.get("command")
    timeout = value.get("timeout")
    child = value.get("direct_child")
    reap = value.get("final_reap")
    if (
        value.get("schema_version")
        != "kvbench-generic-supervised-command-result-1.0.0"
        or not isinstance(command, Mapping)
        or command.get("argv") != list(command_argv)
        or command.get("shell") is not False
        or not isinstance(command.get("working_directory"), str)
        or not Path(command["working_directory"]).is_absolute()
        or _require_sha256_value(
            command.get("environment_sha256"),
            label="child environment fingerprint",
        )
        != command.get("environment_sha256")
        or _require_sha256_value(
            command.get("command_fingerprint"),
            label="child command fingerprint",
        )
        != command.get("command_fingerprint")
        or not isinstance(timeout, Mapping)
        or timeout.get("timed_out") is not False
        or timeout.get("terminate_requested") is not False
        or timeout.get("kill_requested") is not False
        or value.get("returncode") != returncode
        or not isinstance(child, Mapping)
        or child.get("verified") is not True
        or child.get("parent_pid_verified") is not True
        or child.get("start_time_ticks_verified") is not True
        or child.get("process_handle_retained") is not True
        or not isinstance(reap, Mapping)
        or reap.get("completed") is not True
        or reap.get("count") != 1
    ):
        raise Phase11KVQuantDriverError(
            "child process supervision evidence differs"
        )


def _validate_exact_test_record(
    bundle: Path,
    *,
    evidence_name: str,
    module: str,
    source_path: str,
    expected_source_sha256: str,
) -> None:
    prefix = bundle / "validation" / "checks" / evidence_name
    record = _strict_json(prefix.with_suffix(".json"))
    stdout_path = prefix.with_suffix(".stdout.txt")
    stderr_path = prefix.with_suffix(".stderr.txt")
    command = (
        str(CONTAINER_PYTHON),
        "-m",
        "unittest",
        module,
        "-v",
    )
    if (
        set(record)
        != {
            "schema_version",
            "name",
            "command_argv",
            "source_path",
            "source_sha256",
            "returncode",
            "timed_out",
            "stdout_sha256",
            "stderr_sha256",
            "process_supervision",
            "performance_timing",
            "passed",
        }
        or record.get("schema_version")
        != "kvbench-phase11-exact-test-result-1.0.0"
        or record.get("name") != evidence_name
        or record.get("command_argv") != list(command)
        or record.get("source_path") != source_path
        or record.get("source_sha256") != expected_source_sha256
        or record.get("returncode") != 0
        or record.get("timed_out") is not False
        or record.get("performance_timing") is not False
        or record.get("passed") is not True
    ):
        raise Phase11KVQuantDriverError(
            f"exact {evidence_name} test record differs"
        )
    raw: list[bytes] = []
    for path, digest_key in (
        (stdout_path, "stdout_sha256"),
        (stderr_path, "stderr_sha256"),
    ):
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_nlink != 1
        ):
            raise Phase11KVQuantDriverError(
                f"exact {evidence_name} raw output is unsafe"
            )
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != record.get(digest_key):
            raise Phase11KVQuantDriverError(
                f"exact {evidence_name} raw output checksum differs"
            )
        raw.append(data)
    if b"OK" not in b"\n".join(raw) or b"FAILED (" in b"\n".join(raw):
        raise Phase11KVQuantDriverError(
            f"exact {evidence_name} raw unittest verdict differs"
        )
    _validate_supervised_process(
        record.get("process_supervision"),
        command_argv=command,
        returncode=0,
    )


def _validate_exact_and_sanitizer_raw_evidence(
    bundle: Path,
    *,
    path_payload: Mapping[str, Any],
    git_source_binding: Mapping[str, Any],
    sanitizer: Phase11SanitizerEvidence,
) -> None:
    _validate_exact_test_record(
        bundle,
        evidence_name="fixture-conformance",
        module="tests.cuda.test_phase11_kvquant_cuda",
        source_path="tests/cuda/test_phase11_kvquant_cuda.py",
        expected_source_sha256=str(path_payload["fixture_test_sha256"]),
    )
    _validate_exact_test_record(
        bundle,
        evidence_name="cuda-graph",
        module="tests.graph.test_phase11_kvquant_graph",
        source_path="tests/graph/test_phase11_kvquant_graph.py",
        expected_source_sha256=str(path_payload["graph_test_sha256"]),
    )
    runs = _strict_json(bundle / "validation/sanitizer-runs.json")
    records = runs.get("runs")
    if (
        set(runs)
        != {
            "schema_version",
            "probe_source_sha256",
            "runs",
            "performance_timing",
        }
        or runs.get("schema_version")
        != "kvbench-phase11-sanitizer-runs-1.0.0"
        or runs.get("performance_timing") is not False
        or not isinstance(records, list)
        or len(records) != len(_SANITIZER_RUNS)
    ):
        raise Phase11KVQuantDriverError(
            "Compute Sanitizer run index differs"
        )
    _require_sha256_value(
        runs.get("probe_source_sha256"),
        label="sanitizer probe source",
    )
    git_records = git_source_binding.get("records")
    if not isinstance(git_records, list):
        raise Phase11KVQuantDriverError(
            "sanitizer probe Git source binding is absent"
        )
    git_hashes = {
        record.get("path"): record.get("content_sha256")
        for record in git_records
        if isinstance(record, Mapping)
    }
    if (
        runs.get("probe_source_sha256")
        != git_hashes.get(
            "tests/cuda/phase11_kvquant_sanitizer_probe.py"
        )
    ):
        raise Phase11KVQuantDriverError(
            "sanitizer probe is not Git-bound"
        )
    aggregate_stdout = bytearray()
    aggregate_stderr = bytearray()
    for record, (tool, mode) in zip(records, _SANITIZER_RUNS, strict=True):
        if not isinstance(record, Mapping):
            raise Phase11KVQuantDriverError(
                "Compute Sanitizer run record is malformed"
            )
        command = (
            str(SANITIZER),
            "--tool",
            tool,
            "--error-exitcode",
            "99",
            "--target-processes",
            "application-only",
            *(("--leak-check", "full") if tool == "memcheck" else ()),
            str(CONTAINER_PYTHON),
            str(SANITIZER_PROBE),
            "--image-config-digest",
            PHASE11_AUTHORIZED_CONTAINER_DIGEST,
            "--mode",
            mode,
        )
        if (
            set(record)
            != {
                "tool",
                "mode",
                "command_argv",
                "returncode",
                "timed_out",
                "stdout_sha256",
                "stderr_sha256",
                "probe",
                "process_supervision",
                "passed",
            }
            or record.get("tool") != tool
            or record.get("mode") != mode
            or record.get("command_argv") != list(command)
            or record.get("returncode") != 0
            or record.get("timed_out") is not False
            or record.get("passed") is not True
        ):
            raise Phase11KVQuantDriverError(
                "Compute Sanitizer command record differs"
            )
        raw_parts: list[bytes] = []
        prefix = bundle / "validation" / "sanitizer" / f"{tool}-{mode}"
        for path, digest_key in (
            (prefix.with_suffix(".stdout.txt"), "stdout_sha256"),
            (prefix.with_suffix(".stderr.txt"), "stderr_sha256"),
        ):
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat().st_nlink != 1
            ):
                raise Phase11KVQuantDriverError(
                    "Compute Sanitizer raw output is unsafe"
                )
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != record.get(digest_key):
                raise Phase11KVQuantDriverError(
                    "Compute Sanitizer raw output checksum differs"
                )
            raw_parts.append(data)
        probe = _last_json(raw_parts[0])
        if (
            not _zero_memcheck(raw_parts[0], raw_parts[1])
            or probe != record.get("probe")
            or not _sanitizer_probe_matches(probe, mode)
        ):
            raise Phase11KVQuantDriverError(
                "Compute Sanitizer raw verdict differs"
            )
        _validate_supervised_process(
            record.get("process_supervision"),
            command_argv=command,
            returncode=0,
        )
        aggregate_stdout.extend(raw_parts[0])
        aggregate_stderr.extend(raw_parts[1])
    if (
        hashlib.sha256(bytes(aggregate_stdout)).hexdigest()
        != sanitizer.stdout_sha256
        or hashlib.sha256(bytes(aggregate_stderr)).hexdigest()
        != sanitizer.stderr_sha256
        or sanitizer.command_argv
        != tuple(
            item
            for tool, mode in _SANITIZER_RUNS
            for item in (tool, mode)
        )
    ):
        raise Phase11KVQuantDriverError(
            "Compute Sanitizer summary is not derived from raw runs"
        )


def _validate_inner_records(bundle: Path) -> dict[str, Any]:
    manifest = Phase11RunManifest.from_dict(_strict_json(bundle / "manifest.json"))
    if manifest.status is not RunStatus.COMPLETED:
        raise Phase11KVQuantDriverError("Phase 11 bundle is not completed")
    fixture = Phase11FixtureEvidence.from_dict(
        _strict_json(bundle / "numerical/fixture-conformance.json")
    )
    accounting_payload = _strict_json(bundle / "accounting/contexts.json")
    if (
        set(accounting_payload)
        != {
            "schema_version",
            "active_logical_basis",
            "composite_endpoint_and_cache_accounting",
            "endpoint_rope_scratch_bytes_per_record",
            "records",
        }
        or accounting_payload.get("schema_version")
        != "kvbench-phase11-accounting-set-1.0.0"
        or accounting_payload.get("active_logical_basis")
        != "source-faithful-key-zero-occupancy-fixed-value-12"
        or accounting_payload.get("composite_endpoint_and_cache_accounting")
        is not True
        or accounting_payload.get("endpoint_rope_scratch_bytes_per_record")
        != 163_840
    ):
        raise Phase11KVQuantDriverError("accounting wrapper identity differs")
    accounting = tuple(
        Phase11ByteAccounting.from_dict(item)
        for item in accounting_payload.get("records", ())
    )
    expected_accounting = tuple(
        (configuration, context)
        for configuration in PHASE11_CONFIGURATIONS
        for context in PHASE11_ACCOUNTING_CONTEXTS
    )
    if tuple(
        (item.configuration, item.active_context) for item in accounting
    ) != expected_accounting:
        raise Phase11KVQuantDriverError("accounting context matrix differs")
    path_payload = _strict_json(bundle / "execution-path/audit.json")
    paths = tuple(
        Phase11ExecutionPathEvidence.from_dict(item)
        for item in path_payload.get("records", ())
    )
    authority, environment, gqa = _validate_authority_environment_and_gqa(
        bundle,
        manifest=manifest,
        path_payload=path_payload,
    )
    allocation_payload = _strict_json(bundle / "allocation/audit.json")
    if (
        set(allocation_payload) != {"schema_version", "raw_audits", "records"}
        or allocation_payload.get("schema_version")
        != "kvbench-phase11-allocation-set-1.0.0"
        or not isinstance(allocation_payload.get("raw_audits"), Mapping)
        or set(allocation_payload["raw_audits"]) != set(PHASE11_CONFIGURATIONS)
    ):
        raise Phase11KVQuantDriverError("allocation wrapper identity differs")
    allocations = tuple(
        Phase11AllocationEvidence.from_dict(item)
        for item in allocation_payload.get("records", ())
    )
    graph_payload = _strict_json(bundle / "validation/cuda-graph.json")
    if (
        set(graph_payload) != {"schema_version", "records"}
        or graph_payload.get("schema_version")
        != "kvbench-phase11-cuda-graph-set-1.0.0"
    ):
        raise Phase11KVQuantDriverError("CUDA Graph wrapper identity differs")
    graphs = tuple(
        Phase11GraphEvidence.from_dict(item)
        for item in graph_payload.get("records", ())
    )
    sanitizer = Phase11SanitizerEvidence.from_dict(
        _strict_json(bundle / "validation/sanitizer.json")
    )
    _validate_exact_and_sanitizer_raw_evidence(
        bundle,
        path_payload=path_payload,
        git_source_binding=environment["git_source_binding"],
        sanitizer=sanitizer,
    )
    grid_payload = _strict_json(bundle / "validation/bounded-grid.json")
    if (
        grid_payload.get("schema_version")
        != "kvbench-phase11-bounded-grid-1.0.0"
        or grid_payload.get("attempted") != 9
        or grid_payload.get("passed") != 9
        or grid_payload.get("failed") != 0
        or grid_payload.get("capacity_infeasible") != 0
        or grid_payload.get("quality_status") != "unvalidated"
        or grid_payload.get("performance_claim_eligible") is not False
        or grid_payload.get("measurement_scope")
        != "measurement_container_admission"
        or grid_payload.get("speedup_calculated") is not False
    ):
        raise Phase11KVQuantDriverError("bounded-grid wrapper identity differs")
    method_fingerprints = grid_payload.get("method_fingerprints")
    cache_layout_fingerprints = grid_payload.get(
        "cache_layout_fingerprints"
    )
    for mapping, label in (
        (method_fingerprints, "method fingerprints"),
        (cache_layout_fingerprints, "cache-layout fingerprints"),
    ):
        if (
            not isinstance(mapping, Mapping)
            or set(mapping) != set(PHASE11_CONFIGURATIONS)
            or any(
                not isinstance(value, str)
                or re.fullmatch(r"[0-9a-f]{64}", value) is None
                for value in mapping.values()
            )
        ):
            raise Phase11KVQuantDriverError(f"Phase 11 {label} differ")
    points = require_exact_phase11_grid(
        tuple(
            Phase11RunPoint.from_dict(item)
            for item in grid_payload.get("points", ())
        )
    )
    point_records = grid_payload.get("point_records")
    if not isinstance(point_records, list) or len(point_records) != len(points):
        raise Phase11KVQuantDriverError("point-record index differs")
    expected_raw_audits: dict[str, list[dict[str, Any]]] = {
        configuration: [] for configuration in PHASE11_CONFIGURATIONS
    }
    for index, (point, record) in enumerate(
        zip(points, point_records, strict=True)
    ):
        relative = f"grid/{index:02d}-{point.run_id}/point.json"
        if record != {
            "index": index,
            "run_id": point.run_id,
            "path": relative,
            "sha256": point.manifest_sha256,
        }:
            raise Phase11KVQuantDriverError("point-record binding differs")
        path = bundle / relative
        point_payload = _strict_json(path)
        if (
            not path.is_file()
            or path.is_symlink()
            or hashlib.sha256(
                canonical_json_bytes(point_payload)
            ).hexdigest()
            != point.manifest_sha256
        ):
            raise Phase11KVQuantDriverError("point-record checksum differs")
        audits = point_payload.get("allocation_audits")
        if not isinstance(audits, list) or len(audits) != point.output_steps:
            raise Phase11KVQuantDriverError(
                "point allocation audit count differs"
            )
        expected_raw_audits[point.configuration].extend(audits)
        for step, audit in enumerate(audits):
            if not isinstance(audit, dict):
                raise Phase11KVQuantDriverError(
                    "point allocation audit is malformed"
                )
            criterion = audit.get("criterion")
            raw_hashes = audit.get("raw_evidence_sha256")
            evidence_root = (
                f"allocation/operations/{point.run_id}/step-{step:04d}"
            )
            if (
                not isinstance(criterion, dict)
                or audit.get("passed") is not True
                or criterion.get("cache_growth_count") != 0
                or criterion.get("dynamic_sparse_allocation_count") != 0
                or criterion.get("complete_prefix_allocation_count") != 0
                or criterion.get("gqa_expanded_allocation_count") != 0
                or criterion.get("unknown_allocation_count") != 0
                or criterion.get("persistent_allocated_delta") != 0
                or criterion.get("persistent_reserved_delta") != 0
                or audit.get("raw_evidence_root") != evidence_root
                or not isinstance(raw_hashes, dict)
                or len(raw_hashes) != 9
            ):
                raise Phase11KVQuantDriverError(
                    "point allocation criterion differs"
                )
            for basename, digest in raw_hashes.items():
                if (
                    not isinstance(basename, str)
                    or Path(basename).name != basename
                    or not isinstance(digest, str)
                    or len(digest) != 64
                ):
                    raise Phase11KVQuantDriverError(
                        "allocator raw evidence index is malformed"
                    )
                raw_path = bundle / evidence_root / basename
                if (
                    not raw_path.is_file()
                    or raw_path.is_symlink()
                    or sha256_file(raw_path) != digest
                ):
                    raise Phase11KVQuantDriverError(
                        "allocator raw evidence checksum differs"
                    )
            _validate_phase11_allocator_semantically(
                bundle,
                evidence_root=evidence_root,
                summary=audit,
            )
    if allocation_payload["raw_audits"] != expected_raw_audits:
        raise Phase11KVQuantDriverError(
            "allocation wrapper does not bind the point audits"
        )
    if (
        tuple(item.configuration for item in paths) != PHASE11_CONFIGURATIONS
        or tuple(item.configuration for item in allocations)
        != PHASE11_CONFIGURATIONS
        or tuple(
            (item.configuration, item.context_length) for item in graphs
        )
        != PHASE11_GRAPH_POINT_SIGNATURES
        or fixture.fixture_root != PHASE11_FIXTURE_ROOT
        or sanitizer.cases != PHASE11_SANITIZER_CASES
    ):
        raise Phase11KVQuantDriverError("Phase 11 evidence set differs")
    candidate = _strict_json(bundle / "validation/admission-candidate.json")
    expected_candidate_keys = {
        "schema_version",
        "status",
        "git_sha",
        "container_digest",
        "fixture_conformance",
        "byte_accounting",
        "execution_path",
        "native_gqa",
        "allocation",
        "cuda_graph",
        "compute_sanitizer",
        "bounded_admission_grid",
        "immutable_checksums",
        "durable_publication",
        "clean_retrieval",
        "final_method_admission_report",
        "g2_kvq",
        "global_g2_g5",
        "full_scan",
        "quality_execution",
        "performance_data_frozen",
        "performance_claim_eligible",
        "speedup_calculated",
        "r_hbm",
        "historical_evidence_unchanged",
        "existing_methods_unchanged",
        "measurement_container_unchanged",
    }
    if (
        set(candidate) != expected_candidate_keys
        or candidate.get("schema_version")
        != "kvbench-phase11-kvquant-local-admission-candidate-1.0.0"
        or candidate.get("git_sha") != manifest.git_sha
        or candidate.get("container_digest")
        != PHASE11_AUTHORIZED_CONTAINER_DIGEST
        or any(
            candidate.get(field) is not True
            for field in (
                "fixture_conformance",
                "byte_accounting",
                "execution_path",
                "native_gqa",
                "allocation",
                "cuda_graph",
                "compute_sanitizer",
                "bounded_admission_grid",
            )
        )
        or candidate.get("immutable_checksums") != "pending_finalization"
        or candidate.get("durable_publication") != "PENDING_HOST_SIDE"
        or candidate.get("clean_retrieval") != "PENDING_HOST_SIDE"
        or candidate.get("final_method_admission_report")
        != "PENDING_HOST_SIDE"
        or candidate.get("status")
        != "LOCAL_CHECKS_PASS_PUBLICATION_PENDING"
        or candidate.get("g2_kvq") != "NOT_EVALUATED_PUBLICATION_PENDING"
        or candidate.get("global_g2_g5") != "NOT_EVALUATED"
        or candidate.get("full_scan") != "CLOSED"
        or candidate.get("quality_execution") != "LOCKED"
        or candidate.get("performance_data_frozen") is not False
        or candidate.get("performance_claim_eligible") is not False
        or candidate.get("speedup_calculated") is not False
        or candidate.get("r_hbm") is not None
        or candidate.get("historical_evidence_unchanged") is not True
        or candidate.get("existing_methods_unchanged") is not True
        or candidate.get("measurement_container_unchanged") is not True
        or (bundle / "method-admission-report.json").exists()
    ):
        raise Phase11KVQuantDriverError("local admission candidate differs")
    return {
        "manifest": manifest,
        "authority": authority,
        "environment": environment,
        "gqa": gqa,
        "path_wrapper": path_payload,
        "accounting_wrapper": accounting_payload,
        "fixture": fixture,
        "accounting": accounting,
        "paths": paths,
        "allocations": allocations,
        "graphs": graphs,
        "sanitizer": sanitizer,
        "points": points,
        "method_fingerprints": dict(method_fingerprints),
        "cache_layout_fingerprints": dict(cache_layout_fingerprints),
        "candidate": candidate,
    }


def validate_local_admission(bundle_path: Path) -> dict[str, Any]:
    """Validate one explicitly selected immutable inner bundle."""

    try:
        bundle = bundle_path.resolve(strict=True)
    except OSError as error:
        raise Phase11KVQuantDriverError(
            "selected Phase 11 admission bundle is absent"
        ) from error
    validation = validate_run_directory(bundle)
    if not validation.valid or not validation.complete:
        raise Phase11KVQuantDriverError(
            "Phase 11 append-only lifecycle validation failed"
        )
    records = _validate_inner_records(bundle)
    artifact = validate_local_artifact(bundle)
    manifest: Phase11RunManifest = records["manifest"]
    return {
        "status": "LOCAL_CHECKS_PASS_PUBLICATION_PENDING",
        "scope": "phase11_kvquant_local_inner_bundle",
        "bundle_run_id": manifest.run_id,
        "bundle_path": str(bundle),
        "git_sha": manifest.git_sha,
        "local_root_sha256": artifact.root_sha256,
        "object_count": len(artifact.files),
        "point_count": len(records["points"]),
        "complete": True,
        "inventory_valid": True,
        "checksums_valid": True,
        "g2_kvq": "NOT_EVALUATED_PUBLICATION_PENDING",
        "global_g2_g5": "NOT_EVALUATED",
        "full_scan": "CLOSED",
        "quality_execution": "LOCKED",
        "performance_claim_eligible": False,
        "speedup_calculated": False,
        "r_hbm": None,
    }


def _phase11_configuration(configuration: str) -> Phase11MethodConfiguration:
    return Phase11MethodConfiguration(
        configuration=configuration,
        bit_width=_BITS[configuration],
        layers=32,
        batch_size=1,
        query_heads=32,
        kv_heads=8,
        groups=4,
        head_dim=128,
        interface_dtype="bfloat16",
        sink_tokens=5,
        key_cap=12,
        value_cap=12,
        sparse_value_dtype="float32",
        sparse_index_dtype="int32",
        key_semantics="pre_rope",
        sink_key_semantics="attention_ready",
        query_to_kv_mapping="query_head//4",
    )


_PHASE11_REPORT_EVIDENCE_PATHS = (
    ("authority", "config/authority.json"),
    ("environment", "environment/container_identity.json"),
    ("fixture", "numerical/fixture-conformance.json"),
    ("fixture_exact_test", "validation/checks/fixture-conformance.json"),
    ("accounting", "accounting/contexts.json"),
    ("execution_path", "execution-path/audit.json"),
    ("allocation", "allocation/audit.json"),
    ("graph", "validation/cuda-graph.json"),
    ("graph_exact_test", "validation/checks/cuda-graph.json"),
    ("gqa", "gqa/audit.json"),
    ("sanitizer", "validation/sanitizer.json"),
    ("sanitizer_runs", "validation/sanitizer-runs.json"),
    ("bounded_grid", "validation/bounded-grid.json"),
    ("candidate", "validation/admission-candidate.json"),
    ("manifest", "manifest.json"),
    ("inventory", "artifact_inventory.json"),
    ("checksum_ledger", "checksums.sha256"),
    ("complete", "COMPLETE"),
)

_PHASE11_CHECK_EVIDENCE = {
    "fixture_conformance": ("fixture", "fixture_exact_test"),
    "byte_accounting": ("accounting",),
    "sparse_contract": ("fixture", "execution_path"),
    "sink_storage": ("fixture", "gqa"),
    "store_append_correctness": ("fixture",),
    "direct_compressed_decode": ("fixture", "execution_path"),
    "native_gqa": ("gqa", "execution_path"),
    "execution_path": ("execution_path", "environment"),
    "no_dynamic_or_unknown_allocation": ("allocation", "execution_path"),
    "no_host_synchronization": ("execution_path",),
    "graph_capture_replay": ("graph", "graph_exact_test"),
    "graph_zero_replay_allocation": ("graph", "allocation"),
    "compute_sanitizer": ("sanitizer", "sanitizer_runs"),
    "bounded_admission_grid": ("bounded_grid",),
    "immutable_checksums": (
        "manifest",
        "inventory",
        "checksum_ledger",
        "complete",
        "authority",
        "candidate",
    ),
    "durable_publication": ("inner_publication",),
    "clean_retrieval": ("inner_publication",),
}


def derive_phase11_method_admission_report(
    *,
    bundle_path: Path,
    publication_receipt_path: Path,
    created_at_utc: str,
    repository_root: Path = REPOSITORY_ROOT,
) -> Phase11MethodAdmissionReport:
    """Derive the complete PASS report only from validated immutable evidence."""

    from scripts.phase11_r2_outer_bundle import (
        INNER_RECEIPT_RELATIVE,
        validate_inner_publication_receipt,
    )

    repository = repository_root.resolve(strict=True)
    if bundle_path.is_symlink() or publication_receipt_path.is_symlink():
        raise Phase11KVQuantDriverError(
            "Phase 11 report inputs must not be symlinks"
        )
    bundle = bundle_path.resolve(strict=True)
    receipt = publication_receipt_path.resolve(strict=True)
    expected_receipt = (repository / INNER_RECEIPT_RELATIVE).resolve(
        strict=True
    )
    if receipt != expected_receipt:
        raise Phase11KVQuantDriverError(
            "Phase 11 inner publication receipt path differs"
        )
    try:
        bundle_relative = bundle.relative_to(repository)
    except ValueError as error:
        raise Phase11KVQuantDriverError(
            "Phase 11 inner bundle is outside the repository"
        ) from error
    validation = validate_run_directory(bundle)
    if not validation.valid or not validation.complete:
        raise Phase11KVQuantDriverError(
            "Phase 11 report source lifecycle is invalid"
        )
    artifact = validate_local_artifact(bundle)
    records = _validate_inner_records(bundle)
    manifest: Phase11RunManifest = records["manifest"]
    lock_identity = validate_inner_publication_receipt(
        bundle,
        receipt_path=receipt,
        source_run_id=manifest.run_id,
        source_git_sha=manifest.git_sha,
        repository_root=repository,
    )

    references: list[MethodAdmissionEvidenceReference] = []
    for evidence_id, inner_relative in _PHASE11_REPORT_EVIDENCE_PATHS:
        path = bundle / inner_relative
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_nlink != 1
        ):
            raise Phase11KVQuantDriverError(
                f"Phase 11 report evidence is unsafe: {inner_relative}"
            )
        references.append(
            MethodAdmissionEvidenceReference(
                evidence_id=evidence_id,
                path=(
                    f"{bundle_relative.as_posix()}/"
                    f"{inner_relative}"
                ),
                sha256=sha256_file(path),
            )
        )
    references.append(
        MethodAdmissionEvidenceReference(
            evidence_id="inner_publication",
            path=INNER_RECEIPT_RELATIVE.as_posix(),
            sha256=sha256_file(receipt),
        )
    )
    if tuple(_PHASE11_CHECK_EVIDENCE) != PHASE11_ADMISSION_CHECK_IDS:
        raise Phase11KVQuantDriverError(
            "Phase 11 report check derivation order differs"
        )
    checks = tuple(
        Phase11AdmissionCheck(
            check_id=check_id,
            status=GateDisposition.PASS,
            summary=f"{check_id.replace('_', ' ')} passed",
            evidence_ids=_PHASE11_CHECK_EVIDENCE[check_id],
        )
        for check_id in PHASE11_ADMISSION_CHECK_IDS
    )
    candidate: Mapping[str, Any] = records["candidate"]
    path_wrapper: Mapping[str, Any] = records["path_wrapper"]
    return Phase11MethodAdmissionReport(
        schema_version=Phase11MethodAdmissionReport.SCHEMA_VERSION,
        created_at_utc=created_at_utc,
        status=GateDisposition.PASS,
        method_name=MethodName.KVQUANT,
        authority=records["authority"],
        configurations=tuple(
            _phase11_configuration(configuration)
            for configuration in PHASE11_CONFIGURATIONS
        ),
        admitted_configurations=PHASE11_CONFIGURATIONS,
        method_fingerprints=records["method_fingerprints"],
        cache_layout_fingerprints=records["cache_layout_fingerprints"],
        adapter_version=str(path_wrapper["adapter_version"]),
        adapter_source_sha256=str(path_wrapper["adapter_source_sha256"]),
        byte_accounting=records["accounting"],
        fixture_evidence=records["fixture"],
        execution_path_evidence=records["paths"],
        allocation_evidence=records["allocations"],
        graph_evidence=records["graphs"],
        sanitizer_evidence=records["sanitizer"],
        bounded_runs=records["points"],
        checks=checks,
        evidence_references=tuple(references),
        gates=Phase11AdmissionGates(
            g0=GateDisposition.PASS,
            g1=GateDisposition.PASS,
            g2_tq=GateDisposition.PASS,
            g2_kivi=GateDisposition.PASS,
            g2_kvq=GateDisposition.PASS,
            global_g2=GateDisposition.NOT_EVALUATED,
            g3=GateDisposition.NOT_EVALUATED,
            g4=GateDisposition.NOT_EVALUATED,
            g5=GateDisposition.NOT_EVALUATED,
            full_scan_state="CLOSED",
        ),
        blockers=(),
        local_root_digest=artifact.root_sha256,
        r2_uri=(
            "r2://kvbench-artifacts/kvbench/sha256/"
            f"{artifact.root_sha256}/"
        ),
        complete_last=True,
        checksums_valid=True,
        bucket_lock_identity=lock_identity,
        clean_retrieval=True,
        historical_evidence_unchanged=bool(
            candidate["historical_evidence_unchanged"]
        ),
        existing_methods_unchanged=bool(
            candidate["existing_methods_unchanged"]
        ),
        measurement_container_unchanged=bool(
            candidate["measurement_container_unchanged"]
        ),
        claim_eligibility=ClaimEligibility.PERFORMANCE_ONLY,
        quality_status=QualityValidationState.UNVALIDATED,
        quality_execution=QualityExecutionState.LOCKED,
        performance_claim_eligible=False,
        performance_data_frozen=False,
        quality_benchmark_executed=False,
        speedup_calculated=False,
        r_hbm=None,
        measurement_scope=(
            MeasurementScope.MEASUREMENT_CONTAINER_ADMISSION
        ),
        creation_git_sha=manifest.git_sha,
    )


def write_phase11_method_admission_report(
    *,
    bundle_path: Path,
    publication_receipt_path: Path,
    report_path: Path,
    checksum_path: Path,
    created_at_utc: str,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Exclusively write the report and checksum derived from inner evidence."""

    from preflight.run_preflight import json_bytes, write_exclusive
    from scripts.phase11_r2_outer_bundle import (
        METHOD_ADMISSION_CHECKSUM_RELATIVE,
        METHOD_ADMISSION_RELATIVE,
    )

    repository = repository_root.resolve(strict=True)
    expected_report = repository / METHOD_ADMISSION_RELATIVE
    expected_checksum = repository / METHOD_ADMISSION_CHECKSUM_RELATIVE
    try:
        observed_report = report_path.parent.resolve(strict=True) / report_path.name
        observed_checksum = (
            checksum_path.parent.resolve(strict=True) / checksum_path.name
        )
        datetime.fromisoformat(created_at_utc.replace("Z", "+00:00"))
    except (OSError, ValueError) as error:
        raise Phase11KVQuantDriverError(
            "Phase 11 report output contract is invalid"
        ) from error
    if (
        observed_report != expected_report
        or observed_checksum != expected_checksum
        or not created_at_utc.endswith("Z")
        or report_path.exists()
        or report_path.is_symlink()
        or checksum_path.exists()
        or checksum_path.is_symlink()
    ):
        raise Phase11KVQuantDriverError(
            "Phase 11 report outputs differ or already exist"
        )
    report = derive_phase11_method_admission_report(
        bundle_path=bundle_path,
        publication_receipt_path=publication_receipt_path,
        created_at_utc=created_at_utc,
        repository_root=repository,
    )
    report_bytes = json_bytes(report.to_dict())
    report_sha256 = hashlib.sha256(report_bytes).hexdigest()
    checksum_bytes = (
        f"{report_sha256}  {METHOD_ADMISSION_RELATIVE.name}\n"
    ).encode("utf-8")
    write_exclusive(expected_report, report_bytes)
    write_exclusive(expected_checksum, checksum_bytes)
    return {
        "status": "PASS",
        "method_admission_report": str(expected_report),
        "method_admission_report_sha256": report_sha256,
        "method_admission_checksum": str(expected_checksum),
        "g2_kvq": "PASS",
        "global_g2": "NOT_EVALUATED",
    }


def _reject_secret_keys(value: object) -> None:
    forbidden = {
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
        "cloudflare_api_token",
        "r2_account_id",
        "authorization",
    }
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).casefold() in forbidden:
                raise Phase11KVQuantDriverError(
                    "publication receipt contains credential material"
                )
            _reject_secret_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            _reject_secret_keys(nested)


def validate_final_admission(
    *,
    bundle_path: Path,
    outer_artifact_path: Path,
    report_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    """Join immutable local evidence to one final PASS report and R2 receipt."""

    from scripts.phase11_r2_outer_bundle import (
        METHOD_ADMISSION_RELATIVE,
        OUTER_RECEIPT_RELATIVE,
        validate_outer_publication_receipt,
    )

    local = validate_local_admission(bundle_path)
    inner_artifact = validate_local_artifact(bundle_path)
    expected_report = (REPOSITORY_ROOT / METHOD_ADMISSION_RELATIVE).resolve(
        strict=True
    )
    expected_receipt = (REPOSITORY_ROOT / OUTER_RECEIPT_RELATIVE).resolve(
        strict=True
    )
    if (
        report_path.resolve(strict=True) != expected_report
        or receipt_path.resolve(strict=True) != expected_receipt
        or report_path.is_symlink()
        or receipt_path.is_symlink()
    ):
        raise Phase11KVQuantDriverError(
            "final Phase 11 governance paths differ"
        )
    report = Phase11MethodAdmissionReport.from_dict(_strict_json(report_path))
    expected_report_record = derive_phase11_method_admission_report(
        bundle_path=bundle_path,
        publication_receipt_path=(
            REPOSITORY_ROOT
            / "docs/evidence/phase11/r2-admission-publication.json"
        ),
        created_at_utc=report.created_at_utc,
        repository_root=REPOSITORY_ROOT,
    )
    if report != expected_report_record:
        raise Phase11KVQuantDriverError(
            "final MethodAdmissionReport was not derived from inner evidence"
        )
    receipt = _strict_json(receipt_path)
    _reject_secret_keys(receipt)
    publication = validate_outer_publication_receipt(
        outer_artifact_path,
        receipt_path=receipt_path,
        repository_root=REPOSITORY_ROOT,
        source_bundle=bundle_path,
    )
    outer_root = outer_artifact_path.resolve(strict=True)
    if (
        report.status is not GateDisposition.PASS
        or report.gates.g2_kvq is not GateDisposition.PASS
        or report.gates.global_g2 is not GateDisposition.NOT_EVALUATED
        or report.local_root_digest != inner_artifact.root_sha256
        or report.r2_uri
        != (
            "r2://kvbench-artifacts/kvbench/sha256/"
            f"{inner_artifact.root_sha256}/"
        )
        or not report.complete_last
        or not report.checksums_valid
        or not report.clean_retrieval
        or not report.bucket_lock_identity
        or report.creation_git_sha != local["git_sha"]
        or report.performance_claim_eligible
        or report.speedup_calculated
        or report.r_hbm is not None
    ):
        raise Phase11KVQuantDriverError("final MethodAdmissionReport differs")
    return {
        **local,
        "status": "PASS",
        "outer_bundle_run_id": publication.run_id,
        "outer_bundle_path": str(outer_root),
        "outer_root_sha256": publication.root_sha256,
        "outer_object_count": publication.object_count,
        "method_admission_report": str(report_path.resolve(strict=True)),
        "method_admission_report_sha256": sha256_file(report_path),
        "publication_receipt": str(receipt_path.resolve(strict=True)),
        "publication_receipt_sha256": sha256_file(receipt_path),
        "r2_uri": publication.r2_uri,
        "g2_kvq": "PASS",
    }


def _release_cuda() -> None:
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass


def run_admission() -> dict[str, Any]:
    """Execute only the exact local Phase 11 admission work."""

    git_sha = _require_clean_git()
    git_source_binding = _git_source_binding(
        git_sha,
        require_worktree_match=True,
    )
    calibration_validation = _validate_runtime_calibration_before_cuda()
    environment = require_authorized_cuda_environment(
        PHASE11_AUTHORIZED_CONTAINER_DIGEST
    )
    source_raw = os.environ.get("KVBENCH_KVQUANT_SOURCE_ROOT")
    extension_raw = os.environ.get("KVBENCH_KVQUANT_EXTENSION")
    fresh_extension_raw = os.environ.get(
        "KVBENCH_KVQUANT_FRESH_BUILD_EXTENSION"
    )
    if not source_raw or not extension_raw or not fresh_extension_raw:
        raise Phase11KVQuantDriverError(
            "exact source, authority extension, and fresh build are required"
        )
    extension_source = Path(extension_raw)
    fresh_extension_source = Path(fresh_extension_raw)
    if (
        extension_source.is_symlink()
        or fresh_extension_source.is_symlink()
        or not extension_source.is_file()
        or not fresh_extension_source.is_file()
    ):
        raise Phase11KVQuantDriverError(
            "extension authority paths must be regular non-symlink files"
        )
    source_root = Path(source_raw).resolve(strict=True)
    extension = extension_source.resolve(strict=True)
    fresh_extension = fresh_extension_source.resolve(strict=True)
    source_result = validate_source(source_root)
    extension_sha256 = sha256_file(extension)
    fresh_extension_sha256 = sha256_file(fresh_extension)
    code_objects = _validate_sm120_code_objects(fresh_extension)
    if (
        source_result.get("status") != "PASS"
        or source_result.get("patched_commit") != PHASE11_CORRECTED_COMMIT
        or source_result.get("patched_tree") != PHASE11_CORRECTED_TREE
        or source_result.get("aggregate_patch_sha256")
        != PHASE11_AGGREGATE_PATCH_SHA256
        or extension_sha256 != PHASE11_EXTENSION_SHA256
        or fresh_extension_sha256 != PHASE11_EXTENSION_SHA256
    ):
        raise Phase11KVQuantDriverError("execution source identity differs")
    initial = _manifest(
        run_id=_run_id(git_sha),
        git_sha=git_sha,
        created_at_utc=_utc_now(),
    )
    store = AppendOnlyArtifactStore(
        ARTIFACT_ROOT,
        formal_evidence_roots=(
            REPOSITORY_ROOT / "docs/evidence",
            REPOSITORY_ROOT / "artifacts/quality",
            REPOSITORY_ROOT / "artifacts/profiler",
            REPOSITORY_ROOT / "paper-results",
            REPOSITORY_ROOT / "results",
        ),
    )
    run = store.create(initial.run_id, initial)
    run.start()
    started = _utc_now()
    stage = "authority"
    try:
        run.write_json("config/authority.json", _authority().to_dict())
        run.write_json(
            "environment/container_identity.json",
            {
                "schema_version": "kvbench-phase11-container-runtime-1.0.0",
                **dict(environment),
                "container_digest": PHASE11_AUTHORIZED_CONTAINER_DIGEST,
                "image_changed": False,
                "packages_installed": False,
                "network_enabled": False,
                "credentials_passed": False,
                "calibration_validation": calibration_validation,
                "git_source_binding": git_source_binding,
                "source_validation": source_result,
                "authority_extension_path": str(extension),
                "authority_extension_sha256": extension_sha256,
                "fresh_build_extension_path": str(fresh_extension),
                "fresh_build_extension_sha256": fresh_extension_sha256,
                "fresh_build_byte_identical_to_authority": (
                    fresh_extension_sha256 == extension_sha256
                ),
                "nvcc_cuda_object_byte_reproducible": False,
                "fresh_build_source_equivalent": True,
                "fresh_build_code_objects": code_objects,
            },
        )
        stage = "fixture_conformance"
        fixture_test = _run_exact_test(
            run,
            evidence_name="fixture-conformance",
            module="tests.cuda.test_phase11_kvquant_cuda",
            source=CUDA_TEST,
        )
        fixture = _fixture_evidence()
        run.write_json("numerical/fixture-conformance.json", fixture.to_dict())
        stage = "cuda_graph"
        graph_test = _run_exact_test(
            run,
            evidence_name="cuda-graph",
            module="tests.graph.test_phase11_kvquant_graph",
            source=GRAPH_TEST,
        )
        stage = "sanitizer"
        sanitizer = _run_sanitizer(run)
        run.write_json("validation/sanitizer.json", sanitizer.to_dict())
        paths = _static_execution_path()
        run.write_json(
            "execution-path/audit.json",
            {
                "schema_version": "kvbench-phase11-execution-path-set-1.0.0",
                "adapter_version": KVQUANT_ADAPTER_VERSION,
                "adapter_source_sha256": sha256_file(ADAPTER_PATH),
                "cache_source_sha256": sha256_file(CACHE_PATH),
                "endpoint_source_sha256": sha256_file(ENDPOINT_PATH),
                "session_source_sha256": sha256_file(SESSION_PATH),
                "fixture_test_sha256": fixture_test["source_sha256"],
                "graph_test_sha256": graph_test["source_sha256"],
                "records": [item.to_dict() for item in paths],
            },
        )
        _release_cuda()
        stage = "model_load"
        loaded = load_frozen_model()
        points: list[Phase11RunPoint] = []
        graph_records: list[Phase11GraphEvidence] = []
        allocation_by_configuration: dict[str, list[dict[str, Any]]] = {
            name: [] for name in PHASE11_CONFIGURATIONS
        }
        method_fingerprints: dict[str, str] = {}
        layout_fingerprints: dict[str, str] = {}
        for index, signature in enumerate(PHASE11_BOUNDED_POINT_SIGNATURES):
            stage = f"bounded_grid_{index}"
            (
                point,
                payload,
                session,
                audits,
                allocation_files,
            ) = _execute_grid_point(
                loaded=loaded,
                bundle_id=initial.run_id,
                index=index,
                signature=signature,
            )
            points.append(point)
            run.write_json(
                f"grid/{index:02d}-{point.run_id}/point.json",
                payload,
            )
            for relative, data in allocation_files.items():
                run.write_bytes(relative, data)
            configuration = point.configuration
            method_fingerprints.setdefault(
                configuration,
                session.adapter_config_fingerprint,
            )
            layout_fingerprints.setdefault(
                configuration,
                session.cache_layout_fingerprint(),
            )
            allocation_by_configuration[configuration].extend(audits)
            if point.graph_mode is GraphMode.CUDA_GRAPH:
                graph_records.append(
                    Phase11GraphEvidence(
                        configuration=configuration,
                        context_length=point.context_length,
                        capture_passed=True,
                        replay_passed=True,
                        eager_graph_agreement=bool(
                            session.eager_graph_comparison is not None
                            and session.eager_graph_comparison.passed
                        ),
                        repeated_replay_stable=bool(
                            session.graph_evidence is not None
                            and session.graph_evidence.get(
                                "consecutive_replay_outputs_exact"
                            )
                            is True
                        ),
                        pointer_stable=True,
                        replay_allocation_events=0,
                        persistent_allocated_delta=0,
                        persistent_reserved_delta=0,
                        eager_fallback=False,
                    )
                )
            session = None
            _release_cuda()
        require_exact_phase11_grid(tuple(points))
        if tuple(
            (item.configuration, item.context_length)
            for item in graph_records
        ) != PHASE11_GRAPH_POINT_SIGNATURES:
            raise Phase11KVQuantDriverError("CUDA Graph evidence set differs")
        stage = "accounting"
        accounting_records: list[Phase11ByteAccounting] = []
        from kvbench.runtime.bf16_endpoint import BF16DecodeEndpoint
        from kvbench.runtime.kvquant_session import (
            _endpoint_rope_scratch_state,
            kvquant_runtime_context,
            load_frozen_kvquant_method_config,
        )
        from kvbench.adapters import build_method_adapter

        method_config = load_frozen_kvquant_method_config()
        for configuration in PHASE11_CONFIGURATIONS:
            for context in PHASE11_ACCOUNTING_CONTEXTS:
                method = build_method_adapter(
                    method_config,
                    kvquant_runtime_context(),
                    variant_id=configuration,
                )
                method.prepare_runtime()
                cache = method.allocate(
                    batch_size=1,
                    capacity=context,
                    device="cuda:0",
                    workspace_bytes=0,
                )
                method.initialize_cache_untimed(cache)
                cache.begin_prefill()
                cache.finish_prefill(context, key_active_entries=0)
                endpoint = BF16DecodeEndpoint(
                    loaded.model,
                    cache,
                    method,
                )
                (
                    endpoint_bytes,
                    endpoint_predicted,
                    endpoint_pointers,
                ) = _endpoint_rope_scratch_state(endpoint, cache)
                if (
                    endpoint_bytes != endpoint_predicted
                    or endpoint_bytes != 163_840
                    or _endpoint_rope_scratch_state(endpoint, cache)[2]
                    != endpoint_pointers
                ):
                    raise Phase11KVQuantDriverError(
                        "endpoint RoPE scratch identity differs"
                    )
                accounting_records.append(
                    _byte_accounting(
                        cache,
                        configuration=configuration,
                        active_context=context,
                        endpoint_rope_scratch_bytes=endpoint_bytes,
                    )
                )
                endpoint = None
                cache = None
                method = None
                _release_cuda()
        run.write_json(
            "accounting/contexts.json",
            {
                "schema_version": "kvbench-phase11-accounting-set-1.0.0",
                "active_logical_basis": (
                    "source-faithful-key-zero-occupancy-fixed-value-12"
                ),
                "composite_endpoint_and_cache_accounting": True,
                "endpoint_rope_scratch_bytes_per_record": 163_840,
                "records": [item.to_dict() for item in accounting_records],
            },
        )
        allocations = tuple(
            Phase11AllocationEvidence(
                configuration=configuration,
                audited_contexts=(
                    PHASE11_ALLOCATION_AUDIT_CONTEXTS[configuration]
                ),
                cache_growth_bytes=0,
                dynamic_sparse_allocation_bytes=0,
                unknown_allocation_bytes=0,
                full_prefix_allocation_bytes=0,
                gqa_expanded_allocation_bytes=0,
                persistent_allocated_delta=0,
                persistent_reserved_delta=0,
                dense_pointer_stable=True,
                metadata_pointer_stable=True,
                sparse_pointer_stable=True,
                sink_pointer_stable=True,
                staging_pointer_stable=True,
                workspace_pointer_stable=True,
            )
            for configuration in PHASE11_CONFIGURATIONS
            if allocation_by_configuration[configuration]
            and all(
                record["passed"] is True
                and record["criterion"]["cache_growth_count"] == 0
                and record["criterion"][
                    "dynamic_sparse_allocation_count"
                ]
                == 0
                and record["criterion"][
                    "complete_prefix_allocation_count"
                ]
                == 0
                and record["criterion"][
                    "gqa_expanded_allocation_count"
                ]
                == 0
                and record["criterion"]["unknown_allocation_count"] == 0
                and record["criterion"]["persistent_allocated_delta"] == 0
                and record["criterion"]["persistent_reserved_delta"] == 0
                for record in allocation_by_configuration[configuration]
            )
        )
        if len(allocations) != 3:
            raise Phase11KVQuantDriverError("allocation audit is incomplete")
        run.write_json(
            "allocation/audit.json",
            {
                "schema_version": "kvbench-phase11-allocation-set-1.0.0",
                "raw_audits": allocation_by_configuration,
                "records": [item.to_dict() for item in allocations],
            },
        )
        run.write_json(
            "validation/cuda-graph.json",
            {
                "schema_version": "kvbench-phase11-cuda-graph-set-1.0.0",
                "records": [item.to_dict() for item in graph_records],
            },
        )
        run.write_json(
            "gqa/audit.json",
            {
                "schema_version": "kvbench-phase11-gqa-audit-1.0.0",
                "query_heads": 32,
                "kv_heads": 8,
                "groups": 4,
                "mapping": "query_head//4",
                "native_kv_storage": True,
                "query_head_sized_cache": False,
                "repeat_kv": False,
                "passed": True,
            },
        )
        run.write_json(
            "validation/bounded-grid.json",
            {
                "schema_version": "kvbench-phase11-bounded-grid-1.0.0",
                "points": [item.to_dict() for item in points],
                "point_records": [
                    {
                        "index": index,
                        "run_id": point.run_id,
                        "path": (
                            f"grid/{index:02d}-{point.run_id}/point.json"
                        ),
                        "sha256": point.manifest_sha256,
                    }
                    for index, point in enumerate(points)
                ],
                "attempted": 9,
                "passed": 9,
                "failed": 0,
                "capacity_infeasible": 0,
                "method_fingerprints": method_fingerprints,
                "cache_layout_fingerprints": layout_fingerprints,
                "quality_status": "unvalidated",
                "performance_claim_eligible": False,
                "measurement_scope": "measurement_container_admission",
                "speedup_calculated": False,
            },
        )
        candidate = {
            "schema_version": (
                "kvbench-phase11-kvquant-local-admission-candidate-1.0.0"
            ),
            "status": "LOCAL_CHECKS_PASS_PUBLICATION_PENDING",
            "git_sha": git_sha,
            "container_digest": PHASE11_AUTHORIZED_CONTAINER_DIGEST,
            "fixture_conformance": True,
            "byte_accounting": True,
            "execution_path": True,
            "native_gqa": True,
            "allocation": True,
            "cuda_graph": True,
            "compute_sanitizer": True,
            "bounded_admission_grid": True,
            "immutable_checksums": "pending_finalization",
            "durable_publication": "PENDING_HOST_SIDE",
            "clean_retrieval": "PENDING_HOST_SIDE",
            "final_method_admission_report": "PENDING_HOST_SIDE",
            "g2_kvq": "NOT_EVALUATED_PUBLICATION_PENDING",
            "global_g2_g5": "NOT_EVALUATED",
            "full_scan": "CLOSED",
            "quality_execution": "LOCKED",
            "performance_data_frozen": False,
            "performance_claim_eligible": False,
            "speedup_calculated": False,
            "r_hbm": None,
            "historical_evidence_unchanged": True,
            "existing_methods_unchanged": True,
            "measurement_container_unchanged": True,
        }
        run.write_json("validation/admission-candidate.json", candidate)
        completed = run.finalize(
            _terminal_manifest(
                initial,
                started_at_utc=started,
                status=RunStatus.COMPLETED,
                failure_reason=None,
            )
        )
        result = validate_local_admission(completed)
        return {
            **result,
            "container_digest": PHASE11_AUTHORIZED_CONTAINER_DIGEST,
            "source_commit": PHASE11_CORRECTED_COMMIT,
            "source_tree": PHASE11_CORRECTED_TREE,
            "authority_extension_sha256": KVQUANT_EXTENSION_SHA256,
            "fresh_build_extension_sha256": fresh_extension_sha256,
            "adapter_version": KVQUANT_ADAPTER_VERSION,
        }
    except Exception as error:
        reason = _safe_reason(error)
        try:
            run.write_json(
                "validation/admission-failure.json",
                {
                    "stage": stage,
                    "reason": reason,
                    "g2_kvq": "BLOCKED",
                    "performance_claim_eligible": False,
                },
            )
            run.finalize(
                _terminal_manifest(
                    initial,
                    started_at_utc=started,
                    status=RunStatus.RUNTIME_FAILED,
                    failure_reason=reason,
                )
            )
        except Exception:
            pass
        raise


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate one explicit finalized inner admission bundle",
    )
    parser.add_argument(
        "--derive-report",
        action="store_true",
        help="derive and exclusively write the final MethodAdmissionReport",
    )
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--outer-artifact", type=Path)
    parser.add_argument("--method-admission-report", type=Path)
    parser.add_argument("--method-admission-checksum", type=Path)
    parser.add_argument("--publication-receipt", type=Path)
    parser.add_argument("--created-at-utc")
    return parser.parse_args(list(argv))


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if arguments.validate_only and arguments.derive_report:
            raise Phase11KVQuantDriverError(
                "validation and report derivation are mutually exclusive"
            )
        if arguments.derive_report:
            if (
                arguments.artifact is None
                or arguments.publication_receipt is None
                or arguments.method_admission_report is None
                or arguments.method_admission_checksum is None
                or arguments.created_at_utc is None
                or arguments.outer_artifact is not None
            ):
                raise Phase11KVQuantDriverError(
                    "report derivation requires exact inner and output paths"
                )
            result = write_phase11_method_admission_report(
                bundle_path=arguments.artifact,
                publication_receipt_path=arguments.publication_receipt,
                report_path=arguments.method_admission_report,
                checksum_path=arguments.method_admission_checksum,
                created_at_utc=arguments.created_at_utc,
            )
        elif arguments.validate_only:
            if arguments.artifact is None:
                raise Phase11KVQuantDriverError(
                    "--validate-only requires --artifact"
                )
            if (
                arguments.method_admission_checksum is not None
                or arguments.created_at_utc is not None
            ):
                raise Phase11KVQuantDriverError(
                    "validation does not accept report-derivation fields"
                )
            final_values = (
                arguments.outer_artifact,
                arguments.method_admission_report,
                arguments.publication_receipt,
            )
            if any(item is not None for item in final_values) and not all(
                item is not None for item in final_values
            ):
                raise Phase11KVQuantDriverError(
                    "final validation requires outer bundle, report, and receipt"
                )
            result = (
                validate_final_admission(
                    bundle_path=arguments.artifact,
                    outer_artifact_path=arguments.outer_artifact,
                    report_path=arguments.method_admission_report,
                    receipt_path=arguments.publication_receipt,
                )
                if all(item is not None for item in final_values)
                else validate_local_admission(arguments.artifact)
            )
        elif any(
            item is not None
            for item in (
                arguments.artifact,
                arguments.outer_artifact,
                arguments.method_admission_report,
                arguments.method_admission_checksum,
                arguments.publication_receipt,
                arguments.created_at_utc,
            )
        ):
            raise Phase11KVQuantDriverError(
                "run mode does not accept validation paths"
            )
        else:
            result = run_admission()
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(
            json.dumps(
                {
                    "status": "BLOCKED",
                    "reason": _safe_reason(error),
                    "g2_kvq": "BLOCKED",
                    "performance_claim_eligible": False,
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
