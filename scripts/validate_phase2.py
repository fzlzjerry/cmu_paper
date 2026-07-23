#!/usr/bin/python3
"""Dependency-free, read-only repository governance validation.

These checks never execute CUDA, timing, profiler, model, or quality code and
never install packages. They protect the certified E00 evidence/environment
while preserving the completed Phase 2 audit boundary and validating the
approved Phase 3 implementation separately.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tomllib
import typing
from collections.abc import Callable, Sequence
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PHASE2_ENTRY_COMMIT = "aba70be8220972c068c6fbeac279d54e34cddbde"
PHASE2_FINAL_COMMIT = "c16139b0f365eaa052b17cff2fd19c1d4c62a4d1"
QUALITY_COMMIT = "6535a6f6a4e5caa53213e917e9fcf8fc9c0f0190"
ENVIRONMENT_COMMIT = "6442ba1f7554ea0ebf0b3bb1a920c94567cab689"
EVIDENCE_COMMIT = "aba70be8220972c068c6fbeac279d54e34cddbde"
PLAN_COMMIT = "f313817"

QUALITY_PROTOCOL_HASHES = {
    "CODEX_QUALITY_EVALUATION_ADDENDUM.md": (
        "62a8978e04732caff101487275d8b22f14358254538a7b377db2153597a1f332"
    ),
    "CODEX_POST_PERFORMANCE_QUALITY_VALIDATION.md": (
        "b6178566f239ca6ae598b477754f2ebb9d34d0f44c4fd25593b7ea58aa844620"
    ),
}
E00_RUNS = {
    "e00-20260722T041628.190813Z-980eff7b6f59-0dd71f2d": {
        "status": "FAIL",
        "manifest_sha256": (
            "0720734d29c90f609e51cf4c5e4f0b1fadce220e23e146e566f860bb962c0035"
        ),
        "ledger_sha256": (
            "8716fc317747e7e9b5c06017cb8e5339df610c5a89d0d7fbee82ad07fbc68b52"
        ),
    },
    "e00-20260722T050632.375718Z-6442ba1f7554-02d5bd32": {
        "status": "PASS",
        "manifest_sha256": (
            "d054df714bb5eea1f114bf10a03a2879f56ec8d17d3b07e24fe6efcaba6b7aca"
        ),
        "ledger_sha256": (
            "5a610162163979aca97beb2b7b0b480befb85d0b4e63b77c26ec46c36864eca8"
        ),
    },
}
PHASE2_CONFIG_PATHS = (
    "configs/hardware/rtx_pro_6000.yaml",
    "configs/models/primary_gqa_model.yaml",
    "configs/methods/bf16.yaml",
    "configs/methods/turboquant.yaml",
    "configs/methods/kivi.yaml",
    "configs/methods/kvquant.yaml",
    "configs/plans/smoke.yaml",
    "configs/plans/pilot.yaml",
    "configs/plans/graph_ab.yaml",
    "configs/plans/profiler_subset.yaml",
    "configs/plans/full_scan.yaml",
)
PHASE3_CONFIG_PATHS = (
    "configs/plans/phase3_bf16_fixed_l.yaml",
    "configs/plans/phase3_bf16_growing.yaml",
)
CONFIG_PATHS = (*PHASE2_CONFIG_PATHS, *PHASE3_CONFIG_PATHS)
PHASE2_ALLOWED_PATHS = frozenset(
    {
        ".gitignore",
        "Makefile",
        "README.md",
        "pyproject.toml",
        "analysis/README.md",
        "artifacts/README.md",
        "calibration/README.md",
        "reference/README.md",
        *PHASE2_CONFIG_PATHS,
        "docs/experiment_contract.md",
        "docs/measurement_protocol.md",
        "docs/method_notes/README.md",
        "docs/decisions/0006-phase2-schema-tooling-and-artifact-boundary.md",
        "docs/phase2_implementation_plan.md",
        "docs/phase_reports/phase2.md",
        "docs/artifact_policy.md",
        "docs/status.md",
        "docs/blockers.md",
        "docs/risk_register.md",
        "docs/tasks.md",
        "src/kvbench/__init__.py",
        "src/kvbench/__main__.py",
        "src/kvbench/errors.py",
        "src/kvbench/config.py",
        "src/kvbench/validation.py",
        "src/kvbench/cli.py",
        "src/kvbench/schema/__init__.py",
        "src/kvbench/schema/base.py",
        "src/kvbench/schema/config.py",
        "src/kvbench/schema/result.py",
        "src/kvbench/runtime/__init__.py",
        "src/kvbench/runtime/artifacts.py",
        "src/kvbench/runtime/command.py",
        "scripts/validate_phase2.py",
        "tests/schema/__init__.py",
        "tests/schema/test_config_schema.py",
        "tests/unit/test_phase2_artifacts.py",
        "tests/unit/test_phase2_cli.py",
        "tests/unit/test_phase2_governance.py",
    }
)
PHASE3_ALLOWED_PATHS = frozenset(
    {
        ".gitignore",
        "Makefile",
        "configs/models/primary_gqa_model.yaml",
        "configs/methods/bf16.yaml",
        "configs/plans/phase3_bf16_fixed_l.yaml",
        "configs/plans/phase3_bf16_growing.yaml",
        "docs/blockers.md",
        "docs/decisions/0007-phase3-primary-model-and-bf16-backend.md",
        "docs/decisions/0008-phase3-gqa-evidence-taxonomy.md",
        "docs/decisions/0009-phase3-eager-allocation-criterion.md",
        "docs/decisions/0010-phase3-audit-to-measurement-session.md",
        "docs/decisions/0011-phase3-run-session-and-control-join.md",
        "docs/decisions/0012-phase3-minimum-remediation-scope.md",
        "docs/decisions/0013-phase3-source-backed-eager-allocation-catalog.md",
        "docs/evidence/phase3/backend-identity.md",
        "docs/evidence/phase3/g1-admission.json",
        "docs/evidence/phase3/g1-remediation-admission.json",
        "docs/evidence/phase3/model-identity.md",
        "docs/experiment_contract.md",
        "docs/measurement_protocol.md",
        "docs/plans/phase3-bf16-baseline.md",
        "docs/phase_reports/phase3.md",
        "docs/phase_reports/phase3-remediation.md",
        "docs/risk_register.md",
        "docs/status.md",
        "docs/tasks.md",
        "preflight/requirements-phase3.txt",
        "scripts/bootstrap_phase3.py",
        "scripts/validate_phase2.py",
        "src/kvbench/cli.py",
        "src/kvbench/config.py",
        "src/kvbench/schema/__init__.py",
        "src/kvbench/schema/base.py",
        "src/kvbench/schema/phase3.py",
        "src/kvbench/schema/result.py",
        "src/kvbench/runtime/__init__.py",
        "src/kvbench/runtime/allocation.py",
        "src/kvbench/runtime/allocation_attribution.py",
        "src/kvbench/runtime/artifacts.py",
        "src/kvbench/runtime/backend.py",
        "src/kvbench/runtime/bf16_endpoint.py",
        "src/kvbench/runtime/command.py",
        "src/kvbench/runtime/cuda_graph.py",
        "src/kvbench/runtime/fixed_l_runner.py",
        "src/kvbench/runtime/gqa_audit.py",
        "src/kvbench/runtime/gqa_device_dispatch.py",
        "src/kvbench/runtime/gqa_taxonomy.py",
        "src/kvbench/runtime/growing_context_runner.py",
        "src/kvbench/runtime/model_loader.py",
        "src/kvbench/runtime/numerical.py",
        "src/kvbench/runtime/phase3_allocator_controls.py",
        "src/kvbench/runtime/phase3_coordinator.py",
        "src/kvbench/runtime/phase3_campaign.py",
        "src/kvbench/runtime/phase3_audit_operation.py",
        "src/kvbench/runtime/phase3_endpoint_audit.py",
        "src/kvbench/runtime/phase3_raw_audit_evidence.py",
        "src/kvbench/runtime/phase3_report.py",
        "src/kvbench/runtime/phase3_report_publication.py",
        "src/kvbench/runtime/phase3_worker.py",
        "src/kvbench/runtime/phase3_worker_channels.py",
        "src/kvbench/runtime/process_supervision.py",
        "src/kvbench/runtime/static_cache.py",
        "src/kvbench/runtime/telemetry.py",
        "src/kvbench/runtime/timing.py",
        "tests/schema/test_config_schema.py",
        "tests/schema/test_phase3_schema.py",
        "tests/unit/test_model_loader_receipt.py",
        "tests/unit/test_phase3_artifacts.py",
        "tests/unit/test_phase3_cli.py",
        "tests/unit/test_phase3_governance.py",
        "tests/unit/test_phase3_campaign.py",
        "tests/unit/test_phase3_allocator_controls.py",
        "tests/unit/test_phase3_endpoint_audit.py",
        "tests/unit/test_phase3_audit_operation.py",
        "tests/unit/test_phase3_report.py",
        "tests/unit/test_phase3_raw_audit_evidence.py",
        "tests/unit/test_phase3_raw_audit_ipc.py",
        "tests/unit/test_phase3_runtime.py",
        "tests/unit/test_allocation_attribution.py",
        "tests/unit/test_gqa_device_dispatch.py",
        "tests/unit/test_gqa_taxonomy.py",
        "tests/unit/test_phase3_gqa_device_dispatch_geometry.py",
        "tests/unit/test_phase3_report_publication.py",
        "tests/unit/test_process_supervision.py",
        "tests/cuda/test_phase3_allocation_attribution.py",
        "tests/cuda/test_phase3_endpoint_audit.py",
        "tests/cuda/test_phase3_gqa_device_dispatch.py",
        "tests/cuda/test_phase3_process_supervision.py",
        "tests/cuda/test_phase3_runtime_cuda.py",
        "tests/cuda/test_phase3_full_model.py",
        "tests/graph/test_phase3_full_model_graph.py",
        "tests/graph/test_phase3_runtime_graph.py",
    }
)
RAW_RESULT_SUFFIXES = {
    ".bin",
    ".cc",
    ".cpp",
    ".cu",
    ".cuh",
    ".gguf",
    ".ncu-rep",
    ".nsys-rep",
    ".parquet",
    ".safetensors",
    ".pt",
    ".pth",
}
BANNED_IMPORTS = {
    "datasets",
    "evaluate",
    "lm_eval",
    "longbench",
    "ruler",
    "triton",
    "vllm",
}
PHASE3_EXTERNAL_IMPORTS = {
    "src/kvbench/runtime/allocation.py": {"torch"},
    "src/kvbench/runtime/backend.py": {"torch"},
    "src/kvbench/runtime/bf16_endpoint.py": {"torch"},
    "src/kvbench/runtime/cuda_graph.py": {"torch"},
    "src/kvbench/runtime/fixed_l_runner.py": {"torch"},
    "src/kvbench/runtime/gqa_audit.py": {"torch"},
    "src/kvbench/runtime/growing_context_runner.py": {"torch"},
    "src/kvbench/runtime/model_loader.py": {"torch", "transformers"},
    "src/kvbench/runtime/numerical.py": {"torch"},
    "src/kvbench/runtime/static_cache.py": {"torch"},
    "src/kvbench/runtime/timing.py": {"torch"},
}
HOT_PATH_FUNCTIONS = {
    "src/kvbench/runtime/backend.py": {
        "flash_attention_forward",
    },
    "src/kvbench/runtime/static_cache.py": {"update"},
    "src/kvbench/runtime/bf16_endpoint.py": {
        "rotate_half_in_place",
        "_attention",
        "_base_forward",
        "decode",
    },
    "src/kvbench/runtime/cuda_graph.py": {"replay"},
    "src/kvbench/runtime/phase3_endpoint_audit.py": {
        "fixed_step",
        "growing_step",
    },
    "src/kvbench/runtime/growing_context_runner.py": {
        "measured_step",
    },
}
HOT_PATH_BANNED_CALLS = {
    "cat",
    "cpu",
    "expand",
    "item",
    "numpy",
    "repeat_interleave",
    "repeat_kv",
    "synchronize",
    "tolist",
}
E00_PROTECTED_PATHS = (
    "preflight/README.md",
    "preflight/__init__.py",
    "preflight/audit_checkpoint.py",
    "preflight/e00_cuda/binding.cpp",
    "preflight/e00_cuda/build.py",
    "preflight/e00_cuda/xor_kernel.cu",
    "preflight/e00_cuda/xor_kernel.h",
    "preflight/e00_manifest.schema.json",
    "preflight/process_query.py",
    "preflight/python_integrity_probe.py",
    "preflight/python_probe.py",
    "preflight/requirements-e00.txt",
    "preflight/run_preflight.py",
    "preflight/system-packages.lock.json",
    "scripts/preflight.sh",
    "tests/allocation/test_e00_allocation.py",
    "tests/cuda/e00_runtime_probe.py",
    "tests/cuda/e00_sanitizer_probe.py",
    "tests/golden/test_e00_numerical.py",
    "tests/graph_capture/test_e00_graph.py",
    "tests/unit/test_preflight_unit.py",
)
METHOD_NAMES = {"bf16", "turboquant", "kivi", "kvquant"}
SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
REQUIREMENT = re.compile(
    r"([A-Za-z0-9_.-]+)==([^\s]+) --hash=sha256:([0-9a-f]{64})"
)


class ValidationFailure(RuntimeError):
    """A check could not be completed safely."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def run(
    argv: Sequence[str],
    *,
    cwd: Path = ROOT,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        cwd=cwd,
        env=environment,
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def git(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment.update(
        {
            "LC_ALL": "C",
            "LANG": "C",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
        }
    )
    return run(("/usr/bin/git", *argv), environment=environment)


def git_output(argv: Sequence[str]) -> str:
    result = git(argv)
    if result.returncode != 0:
        raise ValidationFailure(
            f"git command failed with exit status {result.returncode}: "
            + " ".join(argv)
        )
    return result.stdout


def git_paths(argv: Sequence[str]) -> set[str]:
    return {item for item in git_output(argv).split("\0") if item}


def report(name: str, errors: Sequence[str], *, note: str | None = None) -> int:
    if errors:
        print(f"[phase2:{name}] FAIL ({len(errors)} error(s))", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    suffix = f"; {note}" if note else ""
    print(f"[phase2:{name}] PASS{suffix}")
    return 0


def historical_phase2_paths() -> set[str]:
    return git_paths(
        (
            "diff",
            "--name-only",
            "-z",
            PHASE2_ENTRY_COMMIT,
            PHASE2_FINAL_COMMIT,
            "--",
        )
    )


def current_phase3_paths() -> set[str]:
    changed = git_paths(
        ("diff", "--name-only", "-z", PHASE2_FINAL_COMMIT, "--")
    )
    untracked = git_paths(
        ("ls-files", "--others", "--exclude-standard", "-z", "--")
    )
    return changed | untracked


def changed_paths() -> set[str]:
    return current_phase3_paths()


def repository_python_paths() -> list[Path]:
    paths: set[Path] = set()
    if SRC.is_dir():
        paths.update(SRC.rglob("*.py"))
    paths.add(ROOT / "scripts" / "validate_phase2.py")
    schema_tests = ROOT / "tests" / "schema"
    if schema_tests.is_dir():
        paths.update(schema_tests.rglob("*.py"))
    unit_tests = ROOT / "tests" / "unit"
    if unit_tests.is_dir():
        paths.update(unit_tests.glob("test_phase2_*.py"))
        paths.update(unit_tests.glob("test_phase3_*.py"))
        paths.update(
            unit_tests / name
            for name in (
                "test_allocation_attribution.py",
                "test_gqa_device_dispatch.py",
                "test_gqa_taxonomy.py",
                "test_process_supervision.py",
            )
        )
    cuda_tests = ROOT / "tests" / "cuda"
    if cuda_tests.is_dir():
        paths.update(cuda_tests.glob("test_phase3_*.py"))
    graph_tests = ROOT / "tests" / "graph"
    if graph_tests.is_dir():
        paths.update(graph_tests.glob("test_phase3_*.py"))
    return sorted(path for path in paths if path.is_file())


def check_format() -> int:
    errors: list[str] = []
    text_suffixes = {".md", ".py", ".toml", ".yaml", ".json", ".txt"}
    candidates = {
        ROOT / relative
        for relative in changed_paths()
        if Path(relative).suffix in text_suffixes
        or relative in {".gitignore", "Makefile"}
    }
    candidates.update(repository_python_paths())
    for path in sorted(candidates):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        data = path.read_bytes()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            errors.append(f"non-UTF-8 Phase 2 text file: {relative}")
            continue
        if b"\r" in data:
            errors.append(f"CR/CRLF line ending is not canonical: {relative}")
        if data and not data.endswith(b"\n"):
            errors.append(f"missing terminal newline: {relative}")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if line.rstrip(" \t") != line:
                errors.append(f"trailing whitespace: {relative}:{line_number}")
            if relative.endswith(".py") and "\t" in line:
                errors.append(f"tab indentation in Python: {relative}:{line_number}")
    return report(
        "format",
        errors,
        note="UTF-8/newline/whitespace check (no formatter dependency)",
    )


def check_lint() -> int:
    errors: list[str] = []
    paths = repository_python_paths()
    if not any(path.is_relative_to(SRC) for path in paths):
        errors.append("src/kvbench Python package is missing")
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=relative)
            compile(tree, relative, "exec", dont_inherit=True)
        except (SyntaxError, UnicodeError) as error:
            errors.append(f"AST compilation failed: {relative}: {error}")
            continue
        defines_typed_code = any(
            isinstance(node, (ast.AnnAssign, ast.ClassDef, ast.FunctionDef))
            for node in tree.body
        )
        if defines_typed_code and (
            path.is_relative_to(SRC) or path.name == "validate_phase2.py"
        ):
            has_future_annotations = any(
                isinstance(node, ast.ImportFrom)
                and node.module == "__future__"
                and any(alias.name == "annotations" for alias in node.names)
                for node in tree.body
            )
            if not has_future_annotations:
                errors.append(f"missing future annotations import: {relative}")
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and any(
                alias.name == "*" for alias in node.names
            ):
                errors.append(f"wildcard import: {relative}:{node.lineno}")
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                errors.append(f"bare except: {relative}:{node.lineno}")
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defaults = [*node.args.defaults, *node.args.kw_defaults]
                if any(
                    isinstance(default, (ast.Dict, ast.List, ast.Set))
                    for default in defaults
                ):
                    errors.append(
                        f"mutable function default: {relative}:{node.lineno}"
                    )
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            imported: set[str] = set()
            if isinstance(node, ast.Import):
                imported.update(
                    alias.name.split(".", 1)[0] for alias in node.names
                )
            elif node.level == 0 and node.module:
                imported.add(node.module.split(".", 1)[0])
            banned = imported & BANNED_IMPORTS
            if banned:
                errors.append(
                    f"out-of-scope import {sorted(banned)!r}: "
                    f"{relative}:{node.lineno}"
                )
            external = {
                name
                for name in imported
                if name not in sys.stdlib_module_names and name != "kvbench"
            }
            allowed_external = PHASE3_EXTERNAL_IMPORTS.get(relative, set())
            undeclared = external - allowed_external
            if undeclared and (
                path.is_relative_to(SRC) or path.name == "validate_phase2.py"
            ):
                errors.append(
                    f"undeclared non-stdlib import {sorted(undeclared)!r}: "
                    f"{relative}:{node.lineno}"
                )
    return report(
        "lint",
        errors,
        note="AST compilation and dependency/scope lint",
    )


def _call_leaf(node: ast.Call) -> str | None:
    function = node.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return None


def _contains_forbidden_expand_reshape(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if _call_leaf(node) not in {"reshape", "view"}:
        return False
    function = node.func
    if not isinstance(function, ast.Attribute):
        return False
    return any(
        isinstance(inner, ast.Call) and _call_leaf(inner) == "expand"
        for inner in ast.walk(function.value)
    )


def _function_definitions(tree: ast.AST) -> dict[str, list[ast.AST]]:
    definitions: dict[str, list[ast.AST]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            definitions.setdefault(node.name, []).append(node)
    return definitions


def _hot_node_errors(
    relative: str,
    function_name: str,
    node: ast.AST,
) -> list[str]:
    errors: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            leaf = _call_leaf(child)
            if leaf in HOT_PATH_BANNED_CALLS:
                errors.append(
                    f"forbidden hot-path call {leaf}: "
                    f"{relative}:{getattr(child, 'lineno', 0)}:"
                    f"{function_name}"
                )
            if leaf in {
                "run",
                "Popen",
                "collect_telemetry",
                "print",
                "debug",
                "info",
                "warning",
                "error",
                "critical",
            }:
                errors.append(
                    f"logging, telemetry, or subprocess call in hot path: "
                    f"{relative}:{getattr(child, 'lineno', 0)}:"
                    f"{function_name}"
                )
            if _contains_forbidden_expand_reshape(child):
                errors.append(
                    f"expand-plus-reshape in hot path: "
                    f"{relative}:{getattr(child, 'lineno', 0)}:"
                    f"{function_name}"
                )
        if isinstance(child, ast.Name) and child.id == "DynamicCache":
            errors.append(
                f"DynamicCache in hot path: {relative}:"
                f"{getattr(child, 'lineno', 0)}:{function_name}"
            )
    return errors


def check_hot_path() -> int:
    errors: list[str] = []
    trees: dict[str, ast.AST] = {}
    for relative, expected_names in HOT_PATH_FUNCTIONS.items():
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"missing Phase 3 SUT source: {relative}")
            continue
        try:
            tree = ast.parse(
                path.read_text(encoding="utf-8"),
                filename=relative,
            )
        except (SyntaxError, UnicodeError) as error:
            errors.append(
                f"cannot audit Phase 3 SUT source {relative}: {error}"
            )
            continue
        trees[relative] = tree
        definitions = _function_definitions(tree)
        for name in sorted(expected_names):
            matches = definitions.get(name, [])
            if len(matches) != 1:
                errors.append(
                    f"expected exactly one audited function "
                    f"{relative}:{name}, found {len(matches)}"
                )
                continue
            errors.extend(
                _hot_node_errors(relative, name, matches[0])
            )
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _call_leaf(node) in {
                "cat",
                "repeat_interleave",
                "repeat_kv",
            }:
                errors.append(
                    f"forbidden GQA/cache operation in SUT source: "
                    f"{relative}:{getattr(node, 'lineno', 0)}"
                )
            if isinstance(node, ast.Name) and node.id == "DynamicCache":
                errors.append(
                    f"DynamicCache in SUT source: {relative}:"
                    f"{getattr(node, 'lineno', 0)}"
                )
            if _contains_forbidden_expand_reshape(node):
                errors.append(
                    f"expand-plus-reshape in SUT source: {relative}:"
                    f"{getattr(node, 'lineno', 0)}"
                )
    timing_path = ROOT / "src/kvbench/runtime/timing.py"
    if not timing_path.is_file():
        errors.append("missing Phase 3 timing source")
    else:
        try:
            timing_tree = ast.parse(
                timing_path.read_text(encoding="utf-8"),
                filename="src/kvbench/runtime/timing.py",
            )
        except (SyntaxError, UnicodeError) as error:
            errors.append(f"cannot audit Phase 3 timing source: {error}")
        else:
            definitions = _function_definitions(timing_tree)
            loop_specs = {
                "measure_fixed_batches": "count",
                "measure_growing_trajectory": "steps",
            }
            for name, range_name in loop_specs.items():
                matches = definitions.get(name, [])
                if len(matches) != 1:
                    errors.append(
                        f"expected exactly one timing function: {name}"
                    )
                    continue
                function = matches[0]
                synchronize_calls = sum(
                    1
                    for node in ast.walk(function)
                    if isinstance(node, ast.Call)
                    and _call_leaf(node) == "synchronize"
                )
                if synchronize_calls != 2:
                    errors.append(
                        f"{name} must contain exactly start/end "
                        f"synchronization boundaries"
                    )
                operation_loops = [
                    node
                    for node in ast.walk(function)
                    if isinstance(node, ast.For)
                    and isinstance(node.iter, ast.Call)
                    and _call_leaf(node.iter) == "range"
                    and len(node.iter.args) == 1
                    and isinstance(node.iter.args[0], ast.Name)
                    and node.iter.args[0].id == range_name
                ]
                if len(operation_loops) != 1:
                    errors.append(
                        f"{name} must retain one exact measured "
                        f"operation loop"
                    )
                    continue
                loop_wrapper = ast.Module(
                    body=operation_loops[0].body,
                    type_ignores=[],
                )
                errors.extend(
                    _hot_node_errors(
                        "src/kvbench/runtime/timing.py",
                        f"{name}:measured_loop",
                        loop_wrapper,
                    )
                )
    return report(
        "hot-path",
        errors,
        note="measured decode and GQA/cache AST audit",
    )


def module_name(path: Path) -> str:
    parts = list(path.relative_to(SRC).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def callable_annotation_errors(
    qualified_name: str,
    value: Callable[..., Any],
) -> list[str]:
    errors: list[str] = []
    try:
        signature = inspect.signature(value)
    except (TypeError, ValueError) as error:
        return [f"cannot inspect {qualified_name}: {error}"]
    for parameter in signature.parameters.values():
        if (
            parameter.name not in {"self", "cls"}
            and parameter.annotation is inspect.Signature.empty
        ):
            errors.append(
                f"missing parameter annotation: "
                f"{qualified_name}.{parameter.name}"
            )
    if signature.return_annotation is inspect.Signature.empty:
        errors.append(f"missing return annotation: {qualified_name}")
    try:
        typing.get_type_hints(value, include_extras=True)
    except Exception as error:  # annotation resolution is the check itself
        errors.append(
            f"unresolvable annotations: {qualified_name}: "
            f"{type(error).__name__}"
        )
    return errors


def check_annotations() -> int:
    errors: list[str] = []
    if not SRC.is_dir():
        return report("annotations", ["src/kvbench package is missing"])
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    phase3_child = os.environ.get(
        "KVBENCH_PHASE3_ANNOTATION_CHILD"
    ) == "1"
    modules: list[Any] = []
    for path in sorted(SRC.rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        is_phase3_external = relative in PHASE3_EXTERNAL_IMPORTS
        if phase3_child != is_phase3_external:
            continue
        name = module_name(path)
        if not name:
            continue
        try:
            modules.append(importlib.import_module(name))
        except Exception as error:
            errors.append(
                f"module import failed: {name}: {type(error).__name__}"
            )
    for module in modules:
        try:
            typing.get_type_hints(module, include_extras=True)
        except Exception as error:
            errors.append(
                f"module annotations do not resolve: {module.__name__}: "
                f"{type(error).__name__}"
            )
        for name, value in vars(module).items():
            if inspect.isfunction(value) and value.__module__ == module.__name__:
                errors.extend(
                    callable_annotation_errors(
                        f"{module.__name__}.{name}", value
                    )
                )
            if not inspect.isclass(value) or value.__module__ != module.__name__:
                continue
            try:
                typing.get_type_hints(value, include_extras=True)
            except Exception as error:
                errors.append(
                    f"class annotations do not resolve: "
                    f"{module.__name__}.{name}: {type(error).__name__}"
                )
            for member_name, member in vars(value).items():
                if member_name.startswith("__") and member_name.endswith("__"):
                    continue
                if isinstance(member, (staticmethod, classmethod)):
                    member = member.__func__
                elif isinstance(member, property):
                    for accessor in (member.fget, member.fset, member.fdel):
                        if accessor is not None:
                            errors.extend(
                                callable_annotation_errors(
                                    f"{module.__name__}.{name}."
                                    f"{accessor.__name__}",
                                    accessor,
                                )
                            )
                    continue
                if (
                    inspect.isfunction(member)
                    and member.__module__ == module.__name__
                ):
                    errors.extend(
                        callable_annotation_errors(
                            f"{module.__name__}.{name}.{member_name}",
                            member,
                        )
                    )
    return report(
        "annotations",
        errors,
        note=(
            "runtime annotation resolution, not third-party static "
            "type analysis"
        ),
    )


def check_phase3_annotations() -> int:
    errors: list[str] = []
    python = ROOT / ".venv" / "bin" / "python"
    site = ROOT / ".phase3" / "site-packages"
    if not python.is_file():
        errors.append("certified Phase 3 base interpreter is missing")
    if not site.is_dir():
        errors.append("isolated Phase 3 dependency target is missing")
    if errors:
        return report("phase3-annotations", errors)
    environment = {
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_OFFLINE": "1",
        "KVBENCH_PHASE3_ANNOTATION_CHILD": "1",
        "LANG": "C",
        "LC_ALL": "C",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": f"{site}:{SRC}",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
    }
    result = run(
        (str(python), str(Path(__file__).resolve()), "annotations"),
        environment=environment,
    )
    if result.returncode != 0:
        errors.append(
            "Phase 3 runtime annotation resolution failed"
        )
        detail = (result.stderr or result.stdout).strip()
        if detail:
            errors.append(detail.replace("\n", " | "))
    return report(
        "phase3-annotations",
        errors,
        note="isolated runtime annotation resolution",
    )


def phase2_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(SRC)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def cli_config_error(relative: str) -> str | None:
    result = run(
        (
            "/usr/bin/python3",
            "-m",
            "kvbench",
            "validate-config",
            relative,
        ),
        environment=phase2_environment(),
    )
    if result.returncode != 0:
        return (
            f"CLI validation failed for {relative} "
            f"(exit {result.returncode})"
        )
    return None


def check_configs() -> int:
    errors: list[str] = []
    for relative in CONFIG_PATHS:
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"missing example configuration: {relative}")
            continue
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            errors.append(
                "configuration is not the declared JSON-compatible "
                f"YAML subset: {relative}"
            )
            continue
        error = cli_config_error(relative)
        if error:
            errors.append(error)
    return report("configs", errors)


def parse_ledger(path: Path) -> tuple[dict[str, str], list[str]]:
    entries: dict[str, str] = {}
    errors: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        return {}, [
            f"cannot read checksum ledger {path}: {type(error).__name__}"
        ]
    for line_number, line in enumerate(lines, start=1):
        if "  " not in line:
            errors.append(
                f"malformed checksum ledger line {line_number}: {path}"
            )
            continue
        digest, relative = line.split("  ", 1)
        pure = PurePosixPath(relative)
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            errors.append(
                f"malformed checksum digest at line {line_number}: {path}"
            )
        if (
            not relative
            or pure.is_absolute()
            or ".." in pure.parts
            or relative in entries
        ):
            errors.append(
                f"unsafe or duplicate ledger path at line "
                f"{line_number}: {path}"
            )
            continue
        entries[relative] = digest
    return entries, errors


def validate_e00_run(
    run_id: str,
    expected: dict[str, str],
) -> list[str]:
    errors: list[str] = []
    run_root = ROOT / "docs" / "evidence" / "e00" / run_id
    manifest_path = run_root / "manifest.json"
    ledger_path = run_root / "checksums.sha256"
    complete_path = run_root / "COMPLETE"
    for required in (manifest_path, ledger_path, complete_path):
        if not required.is_file() or required.is_symlink():
            errors.append(
                f"missing or unsafe E00 file: {required.relative_to(ROOT)}"
            )
    if errors:
        return errors
    if sha256(manifest_path) != expected["manifest_sha256"]:
        errors.append(f"immutable E00 manifest hash mismatch: {run_id}")
    if sha256(ledger_path) != expected["ledger_sha256"]:
        errors.append(f"immutable E00 ledger hash mismatch: {run_id}")
    entries, ledger_errors = parse_ledger(ledger_path)
    errors.extend(ledger_errors)
    actual_payloads: set[str] = set()
    write_bits = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    for path in sorted(run_root.rglob("*")):
        if path.is_symlink():
            errors.append(
                f"symlink found in E00 evidence: {path.relative_to(ROOT)}"
            )
            continue
        if path.is_file():
            relative = path.relative_to(run_root).as_posix()
            if relative not in {"checksums.sha256", "COMPLETE"}:
                actual_payloads.add(relative)
            if path.stat().st_mode & write_bits:
                errors.append(
                    f"writable E00 evidence file: {path.relative_to(ROOT)}"
                )
        elif path.is_dir() and path.stat().st_mode & write_bits:
            errors.append(
                f"writable E00 evidence directory: {path.relative_to(ROOT)}"
            )
    if run_root.stat().st_mode & write_bits:
        errors.append(
            f"writable E00 run directory: {run_root.relative_to(ROOT)}"
        )
    if set(entries) != actual_payloads:
        errors.append(f"checksum ledger coverage mismatch: {run_id}")
    for relative, expected_hash in entries.items():
        target = run_root / relative
        if not target.is_file() or target.is_symlink():
            errors.append(
                f"checksum target missing or unsafe: {run_id}/{relative}"
            )
        elif sha256(target) != expected_hash:
            errors.append(f"checksum mismatch: {run_id}/{relative}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        errors.append(
            f"invalid E00 JSON: {run_id}: {type(error).__name__}"
        )
        return errors
    if manifest.get("run", {}).get("id") != run_id:
        errors.append(f"manifest run ID mismatch: {run_id}")
    if manifest.get("run", {}).get("status") != expected["status"]:
        errors.append(f"manifest status mismatch: {run_id}")
    if manifest.get("run", {}).get("benchmark_timing_collected") is not False:
        errors.append(f"E00 manifest does not deny benchmark timing: {run_id}")
    expected_complete = {
        "run_id": run_id,
        "status": expected["status"],
        "manifest_sha256": expected["manifest_sha256"],
        "checksum_ledger_path": "checksums.sha256",
        "checksum_ledger_sha256": expected["ledger_sha256"],
        "written_last": True,
    }
    for key, value in expected_complete.items():
        if complete.get(key) != value:
            errors.append(
                f"completion marker field mismatch: {run_id}/{key}"
            )
    return errors


def commit_is_ancestor(commit: str) -> bool:
    result = git(("merge-base", "--is-ancestor", commit, "HEAD"))
    return result.returncode == 0


def freeze_markers() -> list[Path]:
    markers: list[Path] = []
    for base, directories, files in os.walk(ROOT):
        base_path = Path(base)
        directories[:] = [
            item
            for item in directories
            if item not in {".git", ".phase3", ".venv", "__pycache__"}
        ]
        markers.extend(
            base_path / item
            for item in files
            if item == "PERFORMANCE_DATA_FROZEN"
        )
    return markers


def check_provenance() -> int:
    errors: list[str] = []
    head = git_output(("rev-parse", "HEAD")).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        errors.append("current Git HEAD is not a full SHA-1 identity")
    for commit, label in (
        (QUALITY_COMMIT, "quality protocol"),
        (ENVIRONMENT_COMMIT, "environment lock"),
        (EVIDENCE_COMMIT, "successful E00 evidence"),
        (PLAN_COMMIT, "Phase 2 plan"),
    ):
        if not commit_is_ancestor(commit):
            errors.append(
                f"required {label} commit is not an ancestor of HEAD"
            )
    for relative, expected_hash in QUALITY_PROTOCOL_HASHES.items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected_hash:
            errors.append(
                f"quality protocol byte lock mismatch: {relative}"
            )
    for run_id, expected in E00_RUNS.items():
        errors.extend(validate_e00_run(run_id, expected))
    if freeze_markers():
        errors.append(
            "PERFORMANCE_DATA_FROZEN marker exists while quality is locked"
        )
    status_path = ROOT / "docs" / "status.md"
    if not status_path.is_file():
        errors.append("docs/status.md is missing")
    else:
        status_text = status_path.read_text(encoding="utf-8").lower()
        normalized_status = " ".join(status_text.split())
        if "quality execution: locked" not in status_text:
            errors.append(
                "status does not record quality execution as LOCKED"
            )
        if (
            "full scan remains closed" not in normalized_status
            and "full-scan admission: closed" not in normalized_status
        ):
            errors.append("status does not record the full scan as CLOSED")
    return report("provenance", errors)


def make_target_block(text: str, target: str) -> str | None:
    lines = text.splitlines()
    start: int | None = None
    target_pattern = re.compile(
        r"^[A-Za-z0-9_.-]+(?:\s+[A-Za-z0-9_.-]+)*\s*:"
    )
    for index, line in enumerate(lines):
        if line == f"{target}:":
            start = index
            break
    if start is None:
        return None
    end = len(lines)
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if (
            line
            and not line[0].isspace()
            and target_pattern.match(line)
        ):
            end = index
            break
    while end > start and not lines[end - 1]:
        end -= 1
    return "\n".join(lines[start:end]) + "\n"


def validate_phase3_campaign_and_report_roots(artifacts: Path) -> list[str]:
    errors: list[str] = []
    campaigns = artifacts / "phase3_campaigns"
    reports = artifacts / "phase3_reports"
    if campaigns.exists() or campaigns.is_symlink():
        if campaigns.is_symlink() or not campaigns.is_dir():
            errors.append("Phase 3 campaign root is unsafe")
        else:
            from kvbench.runtime.phase3_campaign import (
                validate_phase3_campaign_directory,
            )

            for child in sorted(campaigns.iterdir()):
                if child.is_symlink() or not child.is_dir():
                    errors.append(
                        f"unsafe Phase 3 campaign child: {child.name}"
                    )
                    continue
                validation = validate_phase3_campaign_directory(child)
                if not validation.get("valid"):
                    errors.append(
                        f"invalid Phase 3 campaign: {child.name}"
                    )
    if reports.exists() or reports.is_symlink():
        if reports.is_symlink() or not reports.is_dir():
            errors.append("Phase 3 report root is unsafe")
        else:
            from kvbench.runtime.phase3_report import (
                validate_phase3_g1_report_directory,
            )
            from kvbench.runtime.phase3_report_publication import (
                validate_failed_report_attempt,
            )

            for child in sorted(reports.iterdir()):
                if child.name == ".kvbench-report-staging":
                    if child.is_symlink() or not child.is_dir():
                        errors.append("unsafe Phase 3 report staging root")
                    elif any(child.iterdir()):
                        errors.append("nonempty Phase 3 report staging root")
                    continue
                if child.name == ".kvbench-report-reservations":
                    if child.is_symlink() or not child.is_dir():
                        errors.append("unsafe Phase 3 report reservation root")
                        continue
                    for reservation in sorted(child.iterdir()):
                        metadata = reservation.lstat()
                        if (
                            stat.S_ISLNK(metadata.st_mode)
                            or not stat.S_ISREG(metadata.st_mode)
                            or metadata.st_mode
                            & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
                        ):
                            errors.append(
                                f"unsafe Phase 3 report reservation: {reservation.name}"
                            )
                    continue
                if child.name == ".kvbench-report-failed":
                    if child.is_symlink() or not child.is_dir():
                        errors.append("unsafe Phase 3 failed-report root")
                        continue
                    for attempt in sorted(child.iterdir()):
                        if (
                            attempt.is_symlink()
                            or not attempt.is_dir()
                            or not validate_failed_report_attempt(attempt).get("valid")
                        ):
                            errors.append(
                                f"invalid Phase 3 failed report: {attempt.name}"
                            )
                    continue
                if child.is_symlink() or not child.is_dir():
                    errors.append(
                        f"unsafe Phase 3 report child: {child.name}"
                    )
                    continue
                validation = validate_phase3_g1_report_directory(child)
                if not validation.get("valid"):
                    errors.append(
                        f"invalid Phase 3 report: {child.name}"
                    )
    return errors


def validate_phase3_artifact_root() -> list[str]:
    errors: list[str] = []
    artifacts = ROOT / "artifacts"
    phase3 = artifacts / "phase3"
    forbidden = (
        artifacts / "quality",
        artifacts / "profiler",
        ROOT / "docs" / "evidence" / "quality",
        ROOT / "paper-results",
        ROOT / "paper_results",
        ROOT / "results",
    )
    for path in forbidden:
        if path.exists() or path.is_symlink():
            errors.append(
                f"forbidden Phase 3 output path exists: "
                f"{path.relative_to(ROOT)}"
            )
    if artifacts.is_symlink():
        return [*errors, "artifact root is a symlink"]
    if artifacts.exists():
        unexpected = sorted(
            path.name
            for path in artifacts.iterdir()
            if path.name
            not in {"README.md", "phase3", "phase3_campaigns", "phase3_reports"}
        )
        if unexpected:
            errors.append(
                f"unapproved artifact roots: {unexpected!r}"
            )
    errors.extend(validate_phase3_campaign_and_report_roots(artifacts))
    if not phase3.exists() and not phase3.is_symlink():
        return errors
    if phase3.is_symlink() or not phase3.is_dir():
        return [*errors, "Phase 3 artifact root is unsafe"]
    for control_name in (
        ".kvbench-staging",
        ".kvbench-reservations",
    ):
        control = phase3 / control_name
        if control.is_symlink() or (
            control.exists() and not control.is_dir()
        ):
            errors.append(
                f"Phase 3 control path is unsafe: {control_name}"
            )
    staging = phase3 / ".kvbench-staging"
    if staging.is_dir() and any(staging.iterdir()):
        errors.append("Phase 3 contains incomplete staging runs")
    for child in sorted(phase3.iterdir()):
        if child.name in {
            ".kvbench-staging",
            ".kvbench-reservations",
        }:
            continue
        if child.is_symlink() or not child.is_dir():
            errors.append(
                f"unsafe Phase 3 artifact child: {child.name}"
            )
            continue
        result = run(
            (
                "/usr/bin/python3",
                "-m",
                "kvbench",
                "validate-run",
                str(child),
            ),
            environment=phase2_environment(),
        )
        if result.returncode != 0:
            errors.append(
                f"invalid or incomplete Phase 3 run: {child.name}"
            )
    return errors


def check_scope() -> int:
    errors: list[str] = []
    if not commit_is_ancestor(PHASE2_FINAL_COMMIT):
        errors.append(
            "the accepted Phase 2 final commit is not an ancestor of HEAD"
        )
    historical = historical_phase2_paths()
    historical_unexpected = sorted(
        historical - PHASE2_ALLOWED_PATHS
    )
    if historical_unexpected:
        errors.append(
            "historical files outside the approved Phase 2 plan: "
            f"{historical_unexpected!r}"
        )
    changed = current_phase3_paths()
    unexpected = sorted(changed - PHASE3_ALLOWED_PATHS)
    if unexpected:
        errors.append(
            f"files outside the approved Phase 3 plan: {unexpected!r}"
        )
    for relative in sorted(changed):
        if relative.startswith("docs/evidence/e00/"):
            errors.append(f"immutable E00 evidence changed: {relative}")
        if relative in QUALITY_PROTOCOL_HASHES:
            errors.append(
                f"quality protocol changed during Phase 3: {relative}"
            )
        if Path(relative).suffix in RAW_RESULT_SUFFIXES:
            errors.append(
                f"forbidden binary, kernel, model, or profiler artifact "
                f"in Phase 3 scope: {relative}"
            )
        if relative.startswith(
            (
                "artifacts/profiler/",
                "artifacts/quality/",
                "paper-results/",
                "paper_results/",
                "results/",
            )
        ):
            errors.append(
                f"forbidden result tree in Phase 3 scope: {relative}"
            )
    e00_changes = git_paths(
        (
            "diff",
            "--name-only",
            "-z",
            PHASE2_FINAL_COMMIT,
            "--",
            *E00_PROTECTED_PATHS,
        )
    )
    if e00_changes:
        errors.append(
            f"certified E00 implementation changed: "
            f"{sorted(e00_changes)!r}"
        )
    entry_makefile = git(
        ("show", f"{PHASE2_FINAL_COMMIT}:Makefile")
    )
    current_path = ROOT / "Makefile"
    if entry_makefile.returncode != 0 or not current_path.is_file():
        errors.append("cannot compare certified Makefile targets")
    else:
        current = current_path.read_text(encoding="utf-8")
        for target in ("preflight", "preflight-unit"):
            if make_target_block(
                current, target
            ) != make_target_block(entry_makefile.stdout, target):
                errors.append(
                    f"certified Makefile target semantics changed: {target}"
                )
    errors.extend(validate_phase3_artifact_root())
    forbidden_modules = (
        "src/kvbench/adapters",
        "src/kvbench/methods/turboquant",
        "src/kvbench/methods/kivi",
        "src/kvbench/methods/kvquant",
    )
    for relative in forbidden_modules:
        if (ROOT / relative).exists():
            errors.append(
                f"Phase 4+ implementation exists in Phase 3 scope: {relative}"
            )
    return report("scope", errors)


def check_immutable() -> int:
    errors: list[str] = []
    diff = git(
        (
            "diff",
            "--quiet",
            "--no-ext-diff",
            PHASE2_ENTRY_COMMIT,
            "--",
            "docs/evidence/e00",
        )
    )
    if diff.returncode != 0:
        errors.append(
            "E00 evidence differs from the Phase 2 entry commit"
        )
    evidence_status = git_output(
        (
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            "docs/evidence/e00",
        )
    )
    if evidence_status:
        errors.append(
            "tracked or untracked changes exist under E00 evidence"
        )
    flags = git_output(
        ("ls-files", "-v", "--", "docs/evidence/e00")
    )
    if any(not line.startswith("H ") for line in flags.splitlines()):
        errors.append(
            "E00 evidence has a non-ordinary Git index flag"
        )
    for run_id, expected in E00_RUNS.items():
        errors.extend(validate_e00_run(run_id, expected))
    return report("immutable", errors)


def parse_requirements(
    path: Path,
) -> tuple[dict[str, str], list[str]]:
    locked: dict[str, str] = {}
    errors: list[str] = []
    directives = (
        "--index-url ",
        "--extra-index-url ",
        "--only-binary=",
    )
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("--"):
            if not stripped.startswith(directives):
                errors.append(
                    f"unexpected requirements directive at "
                    f"line {line_number}"
                )
            continue
        match = REQUIREMENT.fullmatch(stripped)
        if match is None:
            errors.append(
                f"unhashed or malformed requirement at line {line_number}"
            )
            continue
        name = normalize_distribution(match.group(1))
        if name in locked:
            errors.append(f"duplicate locked distribution: {name}")
        locked[name] = match.group(2)
    return locked, errors


def installed_venv() -> tuple[dict[str, str], str | None]:
    python = ROOT / ".venv" / "bin" / "python"
    if not python.is_file():
        return {}, "certified .venv interpreter is missing"
    code = (
        "import importlib.metadata as m,json,re;"
        "norm=lambda s:re.sub(r'[-_.]+','-',s).lower();"
        "print(json.dumps({norm(d.metadata['Name']):d.version "
        "for d in m.distributions()},sort_keys=True))"
    )
    result = run(
        (str(python), "-I", "-c", code),
        environment={
            "LC_ALL": "C",
            "LANG": "C",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    if result.returncode != 0:
        return {}, (
            f"cannot inspect certified .venv "
            f"(exit {result.returncode})"
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}, "certified .venv distribution inventory is not JSON"
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in payload.items()
    ):
        return {}, (
            "certified .venv distribution inventory has invalid types"
        )
    return payload, None


def git_blob(commit: str, relative: str) -> bytes | None:
    environment = dict(os.environ)
    environment.update(
        {
            "LC_ALL": "C",
            "LANG": "C",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
        }
    )
    result = subprocess.run(
        ["/usr/bin/git", "show", f"{commit}:{relative}"],
        cwd=ROOT,
        env=environment,
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return result.stdout if result.returncode == 0 else None


def check_package_lock() -> int:
    errors: list[str] = []
    requirements_path = (
        ROOT / "preflight" / "requirements-e00.txt"
    )
    system_lock_path = (
        ROOT / "preflight" / "system-packages.lock.json"
    )
    for relative in (
        "preflight/requirements-e00.txt",
        "preflight/system-packages.lock.json",
    ):
        path = ROOT / relative
        current = path.read_bytes() if path.is_file() else None
        if current is None or current != git_blob(
            ENVIRONMENT_COMMIT, relative
        ):
            errors.append(
                f"E00 environment-lock bytes changed: {relative}"
            )
    if not requirements_path.is_file():
        errors.append("E00 Python requirements lock is missing")
        locked: dict[str, str] = {}
    else:
        locked, parse_errors = parse_requirements(
            requirements_path
        )
        errors.extend(parse_errors)
        if len(locked) != 35:
            errors.append(
                "E00 Python lock is not the accepted "
                "35-distribution closure"
            )
    installed, installed_error = installed_venv()
    if installed_error:
        errors.append(installed_error)
    elif installed != locked:
        errors.append(
            "certified .venv distributions differ from the exact E00 lock"
        )
    pip_check = run(
        (
            str(ROOT / ".venv" / "bin" / "python"),
            "-I",
            "-m",
            "pip",
            "check",
        ),
        environment={
            "LC_ALL": "C",
            "LANG": "C",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    if pip_check.returncode != 0:
        errors.append(
            "pip check failed in the certified E00 environment"
        )
    try:
        system_lock = json.loads(
            system_lock_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        system_lock = {}
        errors.append("E00 system package lock is invalid JSON")
    if system_lock:
        if system_lock.get("schema_version") != 1:
            errors.append(
                "unexpected E00 system package lock schema version"
            )
        packages = system_lock.get("dpkg_packages")
        tools = system_lock.get("tools")
        if not isinstance(packages, list) or not isinstance(tools, list):
            errors.append(
                "E00 system package/tool lock entries are malformed"
            )
        else:
            package_names = [
                item.get("name")
                for item in packages
                if isinstance(item, dict)
            ]
            tool_names = [
                item.get("name")
                for item in tools
                if isinstance(item, dict)
            ]
            if len(package_names) != len(set(package_names)):
                errors.append(
                    "duplicate E00 system package lock name"
                )
            if len(tool_names) != len(set(tool_names)):
                errors.append("duplicate E00 tool lock name")
            for tool in tools:
                if not isinstance(tool, dict):
                    continue
                invocation = Path(
                    str(tool.get("invocation_path", ""))
                )
                try:
                    resolved = invocation.resolve(strict=True)
                except OSError:
                    errors.append(
                        f"locked E00 tool is missing: {tool.get('name')}"
                    )
                    continue
                if str(resolved) != tool.get("resolved_path"):
                    errors.append(
                        f"locked E00 tool path mismatch: "
                        f"{tool.get('name')}"
                    )
                if (
                    not resolved.is_file()
                    or sha256(resolved) != tool.get("sha256")
                ):
                    errors.append(
                        f"locked E00 tool hash mismatch: "
                        f"{tool.get('name')}"
                    )
            if package_names:
                dollar = chr(36)
                dpkg_format = (
                    dollar
                    + "{Package}\t"
                    + dollar
                    + "{Version}\t"
                    + dollar
                    + "{Architecture}\n"
                )
                query = run(
                    (
                        "/usr/bin/dpkg-query",
                        "-W",
                        f"-f={dpkg_format}",
                        *package_names,
                    ),
                    environment={"LC_ALL": "C", "LANG": "C"},
                )
                observed: dict[str, tuple[str, str]] = {}
                if query.returncode == 0:
                    for line in query.stdout.splitlines():
                        parts = line.split("\t")
                        if len(parts) == 3:
                            observed[parts[0]] = (
                                parts[1],
                                parts[2],
                            )
                else:
                    errors.append(
                        "cannot query locked E00 system packages"
                    )
                for package in packages:
                    expected = (
                        package.get("version"),
                        package.get("architecture"),
                    )
                    if observed.get(package.get("name")) != expected:
                        errors.append(
                            f"locked system package mismatch: "
                            f"{package.get('name')}"
                        )
    pyproject_path = ROOT / "pyproject.toml"
    if not pyproject_path.is_file():
        errors.append("pyproject.toml is missing")
    else:
        try:
            pyproject = tomllib.loads(
                pyproject_path.read_text(encoding="utf-8")
            )
        except (UnicodeError, tomllib.TOMLDecodeError):
            pyproject = {}
            errors.append("pyproject.toml is invalid")
        project = (
            pyproject.get("project", {})
            if isinstance(pyproject, dict)
            else {}
        )
        dependencies = (
            project.get("dependencies", [])
            if isinstance(project, dict)
            else []
        )
        optional = (
            project.get("optional-dependencies", {})
            if isinstance(project, dict)
            else {}
        )
        if dependencies:
            errors.append(
                "Phase 2 pyproject declares runtime dependencies"
            )
        if optional:
            errors.append(
                "Phase 2 pyproject declares optional dependency groups"
            )
    return report("package-lock", errors)


def check_phase3_package_lock() -> int:
    errors: list[str] = []
    environment = {
        "LANG": "C",
        "LC_ALL": "C",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "TZ": "UTC",
    }
    result = run(
        (
            str(ROOT / ".venv" / "bin" / "python"),
            "scripts/bootstrap_phase3.py",
            "verify",
        ),
        environment=environment,
    )
    payload: dict[str, Any] = {}
    if result.returncode != 0:
        errors.append(
            "Phase 3 dependency verification failed "
            f"(exit {result.returncode})"
        )
    else:
        try:
            parsed = json.loads(result.stdout)
        except json.JSONDecodeError:
            errors.append(
                "Phase 3 dependency verifier output is not JSON"
            )
        else:
            if isinstance(parsed, dict):
                payload = parsed
            else:
                errors.append(
                    "Phase 3 dependency verifier output is not an object"
                )
    if payload and payload.get("status") != "pass":
        errors.append(
            "Phase 3 dependency verifier did not report pass"
        )
    if payload and payload.get("target") != str(
        ROOT / ".phase3" / "site-packages"
    ):
        errors.append(
            "Phase 3 dependency target identity mismatch"
        )
    return report("phase3-package-lock", errors)


def check_method() -> int:
    method = os.environ.get("KVBENCH_METHOD", "")
    errors: list[str] = []
    if method not in METHOD_NAMES:
        errors.append(
            "METHOD must be one of bf16, turboquant, kivi, or kvquant"
        )
    else:
        error = cli_config_error(
            f"configs/methods/{method}.yaml"
        )
        if error:
            errors.append(error)
    return report("method", errors)


def check_run_id() -> int:
    run_id = os.environ.get("KVBENCH_RUN_ID", "")
    errors: list[str] = []
    if SAFE_IDENTIFIER.fullmatch(run_id) is None or ".." in run_id:
        errors.append("RUN_ID is missing or unsafe")
    return report("run-id", errors)


CHECKS: dict[str, Callable[[], int]] = {
    "format": check_format,
    "lint": check_lint,
    "hot-path": check_hot_path,
    "annotations": check_annotations,
    "phase3-annotations": check_phase3_annotations,
    "configs": check_configs,
    "provenance": check_provenance,
    "scope": check_scope,
    "immutable": check_immutable,
    "package-lock": check_package_lock,
    "phase3-package-lock": check_phase3_package_lock,
    "method": check_method,
    "run-id": check_run_id,
}


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("check", choices=(*CHECKS, "all"))
    return parser.parse_args(list(argv))


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(
        sys.argv[1:] if argv is None else argv
    )
    if arguments.check != "all":
        try:
            return CHECKS[arguments.check]()
        except ValidationFailure as error:
            return report(arguments.check, [str(error)])
    result = 0
    for name in (
        "format",
        "lint",
        "hot-path",
        "annotations",
        "phase3-annotations",
        "configs",
        "provenance",
        "scope",
        "immutable",
        "package-lock",
        "phase3-package-lock",
    ):
        try:
            result |= CHECKS[name]()
        except ValidationFailure as error:
            result |= report(name, [str(error)])
    return result


if __name__ == "__main__":
    raise SystemExit(main())
