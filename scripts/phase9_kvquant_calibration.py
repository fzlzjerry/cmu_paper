#!/usr/bin/env python3
"""Host orchestrator for the single immutable Phase 9 KVQuant calibration."""

from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from kvbench.runtime.artifacts import (  # noqa: E402
    phase9_calibration_artifact_store,
)
from kvbench.schema import RunStatus  # noqa: E402
from kvbench.schema.phase9 import (  # noqa: E402
    CALIBRATION_DOCKERFILE_SHA256,
    CALIBRATION_IMAGE_DIGEST,
    DATASET_CONTENT_SHA256,
    DATASET_CONVERSION_REVISION,
    DATASET_REPOSITORY,
    DATASET_REVISION,
    METHOD_IDENTIFIER,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SNAPSHOT_MANIFEST_SHA256,
    PATCH_SHA256,
    PATCHED_COMMIT,
    PATCHED_TREE,
    Phase9CalibrationManifest,
    UPSTREAM_BASE_COMMIT,
    UPSTREAM_BASE_TREE,
    UPSTREAM_REPOSITORY,
)
from scripts.r2_artifact import validate_local_artifact  # noqa: E402


GPU_UUID = "GPU-75bd273e-6b20-0d22-1b0b-5fbb6fb0025b"
DATASET_SIZE_BYTES = 6_357_543
NUMBER_OF_EXAMPLES = 16
SEQUENCE_LENGTH = 2048
BASE_SEED = 20260721
MODEL_MANIFEST_PATH = (
    REPOSITORY_ROOT / "docs/evidence/phase9/model-snapshot-manifest.json"
)
IMAGE_MANIFEST_PATH = REPOSITORY_ROOT / "docker/calibration-kvquant.image.json"
PYTHON_FREEZE_PATH = (
    REPOSITORY_ROOT / "docker/calibration-kvquant.python-freeze.txt"
)
WORKER_PATH = REPOSITORY_ROOT / "scripts/phase9_kvquant_worker.py"
PATCH_MANIFEST_PATH = (
    REPOSITORY_ROOT / "third_party/patches/kvquant/manifest.json"
)
PRIOR_PHASE9_REPORT = (
    REPOSITORY_ROOT / "docs/phase_reports/phase9-kvquant-calibration-blocked.md"
)
PHASE9P_REPORT = (
    REPOSITORY_ROOT / "docs/phase_reports/phase9p-kvquant-upstream-gqa-patch.md"
)
DECISION_0021 = (
    REPOSITORY_ROOT
    / "docs/decisions/0021-kvquant-patch-main-repository-custody.md"
)
PRIOR_PHASE9_REPORT_SHA256 = (
    "05bbc9d21fe4bff900bd141ddc7f6daec226848178f8c0b78b7ecdaba2c180b7"
)
PHASE9P_REPORT_SHA256 = (
    "812cce928d598f19948c1f87ff675a2c685ed8b24bef923edda305d367abc95e"
)
DECISION_0021_SHA256 = (
    "e09cb0f7c59c07eb04ec28319d6705c436c9c25d466bbe63e2f1859cf75d4daf"
)
PATCH_SIZE_BYTES = 289_239


class Phase9CalibrationError(RuntimeError):
    """Fail-closed Phase 9 host error."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Phase9CalibrationError(f"{path} must contain a JSON object")
    return payload


def _run_checked(
    command: Sequence[str],
    *,
    cwd: Path = REPOSITORY_ROOT,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise Phase9CalibrationError(
            f"command failed with exit code {completed.returncode}: "
            f"{' '.join(command[:3])}"
        )
    return completed


def _git(*arguments: str, cwd: Path = REPOSITORY_ROOT) -> str:
    return _run_checked(("git", *arguments), cwd=cwd).stdout.strip()


def _require_safe_nonsecret_path(path: Path, label: str) -> Path:
    candidate = path.resolve(strict=True)
    if any(
        part == ".env" or part.startswith(".env.")
        for part in candidate.parts
    ):
        raise Phase9CalibrationError(f"{label} resolves through a prohibited path")
    return candidate


def _source_command(source_root: Path) -> str:
    return (
        f"make KVQUANT_GQA_SOURCE_ROOT={source_root} "
        "validate-kvquant-gqa-patch"
    )


def _document_has_status(path: Path, status: str) -> bool:
    accepted = {
        f"Status: {status}",
        f"Status: **{status}**",
        f"- Status: {status}",
        f"- Status: **{status}**",
    }
    lines = path.read_text(encoding="utf-8").splitlines()
    return any(line.strip() in accepted for line in lines)


def _source_validation_is_exact(validation: object) -> bool:
    patch_manifest = _load_json(PATCH_MANIFEST_PATH)
    patch = patch_manifest["patch"]
    expected_paths = [
        record["path"] for record in patch_manifest["patched_files"]
    ]
    expected = {
        "status": "PASS",
        "changed_paths": expected_paths,
        "patch_path": patch["path"],
        "patch_sha256": PATCH_SHA256,
        "patch_size_bytes": PATCH_SIZE_BYTES,
        "reconstruction": {
            "base_commit": UPSTREAM_BASE_COMMIT,
            "base_tree": UPSTREAM_BASE_TREE,
            "applied_patch_sha256": PATCH_SHA256,
            "changed_file_count": len(expected_paths),
            "patched_tree": PATCHED_TREE,
        },
    }
    return validation == expected


def _validate_entry_inputs(arguments: argparse.Namespace) -> dict[str, object]:
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise Phase9CalibrationError("Phase 9 execution requires a clean Git tree")
    head = _git("rev-parse", "HEAD")
    origin_main = _git("rev-parse", "origin/main")
    if _run_checked(("git", "merge-base", "--is-ancestor", origin_main, head)).returncode:
        raise Phase9CalibrationError("HEAD is not a clean descendant of origin/main")
    if (REPOSITORY_ROOT / "PERFORMANCE_DATA_FROZEN").exists():
        raise Phase9CalibrationError("PERFORMANCE_DATA_FROZEN must remain absent")
    factory = (REPOSITORY_ROOT / "src/kvbench/adapters/factory.py").read_text(
        encoding="utf-8"
    )
    if '_DEFERRED_METHODS = frozenset({"kvquant"})' not in factory:
        raise Phase9CalibrationError("KVQuant factory is not fail-closed")

    expected_files = {
        PRIOR_PHASE9_REPORT: PRIOR_PHASE9_REPORT_SHA256,
        PHASE9P_REPORT: PHASE9P_REPORT_SHA256,
        DECISION_0021: DECISION_0021_SHA256,
        MODEL_MANIFEST_PATH: MODEL_SNAPSHOT_MANIFEST_SHA256,
        REPOSITORY_ROOT / "docker/calibration-kvquant.Dockerfile": (
            CALIBRATION_DOCKERFILE_SHA256
        ),
    }
    for path, digest in expected_files.items():
        if _sha256_file(path) != digest:
            raise Phase9CalibrationError(f"frozen entry evidence drifted: {path.name}")
    if not _document_has_status(DECISION_0021, "Accepted"):
        raise Phase9CalibrationError("Decision 0021 is not Accepted")
    if not _document_has_status(PHASE9P_REPORT, "PASS"):
        raise Phase9CalibrationError("Phase 9P report is not PASS")

    source_root = _require_safe_nonsecret_path(
        Path(arguments.source_root),
        "patched source",
    )
    if _git("rev-parse", "HEAD", cwd=source_root) != PATCHED_COMMIT:
        raise Phase9CalibrationError("reconstructed patched commit drifted")
    if _git("rev-parse", "HEAD^{tree}", cwd=source_root) != PATCHED_TREE:
        raise Phase9CalibrationError("reconstructed patched tree drifted")
    if _git("status", "--porcelain=v1", "--untracked-files=all", cwd=source_root):
        raise Phase9CalibrationError("reconstructed patched source is dirty")
    validator = _run_checked(
        (
            sys.executable,
            "scripts/validate_kvquant_gqa_patch.py",
            "--source-root",
            str(source_root),
        )
    )
    validation = json.loads(validator.stdout)
    if not _source_validation_is_exact(validation):
        raise Phase9CalibrationError("patched source reconstruction validation failed")

    tracked = _git("ls-files").splitlines()
    forbidden_tracked = [
        path
        for path in tracked
        if path.startswith(
            (
                "third_party_worktrees/",
                "third_party/kvquant/",
                ".reference/kvquant/",
            )
        )
    ]
    if forbidden_tracked:
        raise Phase9CalibrationError("a complete upstream checkout is tracked")

    dataset = _require_safe_nonsecret_path(
        Path(arguments.dataset_parquet),
        "WikiText-2 train parquet",
    )
    if (
        dataset.stat().st_size != DATASET_SIZE_BYTES
        or _sha256_file(dataset) != DATASET_CONTENT_SHA256
    ):
        raise Phase9CalibrationError("WikiText-2 train object identity drifted")
    model_cache = _require_safe_nonsecret_path(
        Path(arguments.model_cache),
        "model cache",
    )
    snapshot = (
        model_cache
        / "models--meta-llama--Llama-3.1-8B-Instruct"
        / "snapshots"
        / MODEL_REVISION
    )
    if not snapshot.is_dir():
        raise Phase9CalibrationError("exact offline model snapshot is absent")

    inspect = _run_checked(
        ("docker", "image", "inspect", CALIBRATION_IMAGE_DIGEST)
    )
    image_records = json.loads(inspect.stdout)
    if (
        not isinstance(image_records, list)
        or len(image_records) != 1
        or image_records[0].get("Id") != CALIBRATION_IMAGE_DIGEST
    ):
        raise Phase9CalibrationError("calibration image config digest drifted")

    return {
        "head": head,
        "origin_main": origin_main,
        "source_root": source_root,
        "source_validation": validation,
        "dataset": dataset,
        "model_cache": model_cache,
        "snapshot": snapshot,
        "image_inspect_id": image_records[0]["Id"],
    }


def _calibration_contract() -> dict[str, object]:
    return {
        "schema_version": "kvbench-phase9-calibration-contract-1.0.0",
        "authority": {
            "method_identifier": METHOD_IDENTIFIER,
            "upstream_repository": UPSTREAM_REPOSITORY,
            "upstream_base_commit": UPSTREAM_BASE_COMMIT,
            "upstream_base_tree": UPSTREAM_BASE_TREE,
            "patch_sha256": PATCH_SHA256,
            "patched_commit": PATCHED_COMMIT,
            "patched_tree": PATCHED_TREE,
            "decision": "0021",
        },
        "implementation": {
            "container_digest": CALIBRATION_IMAGE_DIGEST,
            "dockerfile_sha256": CALIBRATION_DOCKERFILE_SHA256,
            "host_driver_sha256": _sha256_file(Path(__file__)),
            "container_worker_sha256": _sha256_file(WORKER_PATH),
        },
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "tokenizer_id": MODEL_ID,
            "tokenizer_revision": MODEL_REVISION,
            "snapshot_manifest_sha256": MODEL_SNAPSHOT_MANIFEST_SHA256,
            "dtype": "bfloat16",
            "layers": 32,
            "query_heads": 32,
            "kv_heads": 8,
            "head_dimension": 128,
            "gqa_group_size": 4,
            "key_capture": "post_k_proj_pre_rope",
            "value_capture": "post_v_proj_native_hkv",
        },
        "dataset": {
            "repository": DATASET_REPOSITORY,
            "revision": DATASET_REVISION,
            "conversion_revision": DATASET_CONVERSION_REVISION,
            "config": "wikitext-2-raw-v1",
            "split": "train",
            "content_sha256": DATASET_CONTENT_SHA256,
            "number_of_examples": NUMBER_OF_EXAMPLES,
            "sequence_length": SEQUENCE_LENGTH,
            "seed": BASE_SEED,
        },
        "precision": {
            "model_forward": "bfloat16",
            "fisher": "float32",
            "fitting": "float16",
            "codebook_threshold": "float32",
        },
        "quantizers": ["kvq4", "kvq3", "kvq2"],
        "policies": {
            "sink_tokens": 5,
            "key_outlier_cap": 12,
            "value_outlier_cap": 12,
            "entries_per_tail": 6,
            "outlier_value_dtype": "float32",
            "outlier_index_dtype": "int32",
        },
        "replay_tolerance": {"rtol": 1e-5, "atol": 1e-8},
        "forbidden_execution": {
            "performance": True,
            "profiler": True,
            "quality": True,
            "phase10": True,
        },
    }


def calibration_identity() -> tuple[str, str, dict[str, object]]:
    contract = _calibration_contract()
    digest = _sha256_bytes(_canonical_json_bytes(contract))
    return f"kvqcal-{digest[:32]}", digest, contract


def _initial_manifest(
    *,
    run_id: str,
    contract_sha256: str,
    git_sha: str,
    source_command: str,
    created_at: str,
) -> Phase9CalibrationManifest:
    return Phase9CalibrationManifest(
        schema_version=Phase9CalibrationManifest.SCHEMA_VERSION,
        artifact_schema_version=Phase9CalibrationManifest.ARTIFACT_SCHEMA_VERSION,
        run_id=run_id,
        status=RunStatus.CREATED,
        created_at_utc=created_at,
        started_at_utc=None,
        finished_at_utc=None,
        phase="phase9_kvquant_calibration",
        run_kind="offline_calibration",
        claim_class="none",
        git_sha=git_sha,
        git_dirty=False,
        container_digest=CALIBRATION_IMAGE_DIGEST,
        dockerfile_sha256=CALIBRATION_DOCKERFILE_SHA256,
        method_identifier=METHOD_IDENTIFIER,
        upstream_repository=UPSTREAM_REPOSITORY,
        upstream_base_commit=UPSTREAM_BASE_COMMIT,
        upstream_base_tree=UPSTREAM_BASE_TREE,
        patch_sha256=PATCH_SHA256,
        patched_commit=PATCHED_COMMIT,
        patched_tree=PATCHED_TREE,
        decision="0021",
        source_reconstruction_command=source_command,
        source_reconstruction_passed=True,
        reconstructed_source_checksum_result="PASS",
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        tokenizer_id=MODEL_ID,
        tokenizer_revision=MODEL_REVISION,
        model_snapshot_manifest_sha256=MODEL_SNAPSHOT_MANIFEST_SHA256,
        dataset_repository=DATASET_REPOSITORY,
        dataset_revision=DATASET_REVISION,
        dataset_conversion_revision=DATASET_CONVERSION_REVISION,
        dataset_content_sha256=DATASET_CONTENT_SHA256,
        dataset_split="train",
        number_of_examples=NUMBER_OF_EXAMPLES,
        sequence_length=SEQUENCE_LENGTH,
        random_seed=BASE_SEED,
        attempt_sequence=1,
        calibration_contract_sha256=contract_sha256,
        command_argv=("make", "calibrate-kvquant"),
        performance_measurement=False,
        profiler_execution=False,
        quality_evaluation=False,
        quality_execution="locked",
        performance_data_frozen=False,
        full_scan_state="closed",
        phase10_started=False,
        inventory_path=None,
        failure_reason=None,
    )


def _mount(source: Path, destination: str, *, readonly: bool) -> str:
    value = f"type=bind,src={source},dst={destination}"
    return f"{value},readonly" if readonly else value


def _docker_command(
    entry: dict[str, object],
    stage: Path,
    worker_arguments: Sequence[str],
    *,
    extra_mounts: Sequence[tuple[Path, str, bool]] = (),
) -> list[str]:
    command = [
        "docker",
        "run",
        "--rm",
        "--network=none",
        "--read-only",
        "--pids-limit=2048",
        "--gpus",
        f"device={GPU_UUID}",
        "--tmpfs",
        "/tmp:rw,exec,nosuid,nodev,size=32g",
        "--mount",
        _mount(WORKER_PATH, "/opt/phase9/worker.py", readonly=True),
        "--mount",
        _mount(
            MODEL_MANIFEST_PATH,
            "/input/model-snapshot-manifest.json",
            readonly=True,
        ),
        "--mount",
        _mount(IMAGE_MANIFEST_PATH, "/input/image-manifest.json", readonly=True),
        "--mount",
        _mount(entry["source_root"], "/source", readonly=True),
        "--mount",
        _mount(entry["model_cache"], "/model-cache", readonly=True),
        "--mount",
        _mount(entry["dataset"], "/input/train.parquet", readonly=True),
        "--mount",
        _mount(stage, "/output", readonly=False),
    ]
    for source, destination, readonly in extra_mounts:
        command.extend(
            ("--mount", _mount(source, destination, readonly=readonly))
        )
    command.extend(
        (
            "--env",
            (
                "PYTHONPATH=/opt/kvbench/.phase3/site-packages:"
                "/source:/source/gradients:/source/quant"
            ),
            "--workdir",
            "/tmp",
            CALIBRATION_IMAGE_DIGEST,
            "python3",
            "/opt/phase9/worker.py",
            *worker_arguments,
        )
    )
    return command


def _run_container(
    *,
    label: str,
    entry: dict[str, object],
    stage: Path,
    worker_arguments: Sequence[str],
    commands: list[dict[str, object]],
    extra_mounts: Sequence[tuple[Path, str, bool]] = (),
) -> None:
    logs = stage / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout_path = logs / f"{label}.stdout.txt"
    stderr_path = logs / f"{label}.stderr.txt"
    command = _docker_command(
        entry,
        stage,
        worker_arguments,
        extra_mounts=extra_mounts,
    )
    commands.append(
        {
            "label": label,
            "argv": command,
            "container_digest": CALIBRATION_IMAGE_DIGEST,
            "r2_credentials_passed": False,
            "network": "none",
        }
    )
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            check=False,
            stdout=stdout,
            stderr=stderr,
        )
    if completed.returncode != 0:
        raise Phase9CalibrationError(
            f"container stage {label} failed with exit code {completed.returncode}"
        )


def _model_arguments() -> list[str]:
    return [
        "--model-artifact-manifest",
        "/input/model-snapshot-manifest.json",
        "--cache-dir",
        "/model-cache",
    ]


def _write_authority_payloads(
    run: Any,
    *,
    run_id: str,
    contract_sha256: str,
    contract: dict[str, object],
    source_command: str,
    entry: dict[str, object],
) -> None:
    patch_manifest = _load_json(PATCH_MANIFEST_PATH)
    run.write_json(
        "calibration_config.json",
        {
            "schema_version": "kvbench-phase9-calibration-config-1.0.0",
            "calibration_id": run_id,
            "calibration_contract_sha256": contract_sha256,
            "contract": contract,
            "execution_git_sha": entry["head"],
            "quality_or_performance_selection": False,
        },
    )
    run.write_json(
        "authority_manifest.json",
        {
            "schema_version": "kvbench-phase9-authority-manifest-1.0.0",
            "authority_wording": "KVQuant-GQA patched upstream",
            "method_identifier": METHOD_IDENTIFIER,
            "upstream_repository": UPSTREAM_REPOSITORY,
            "upstream_base_commit": UPSTREAM_BASE_COMMIT,
            "upstream_base_tree": UPSTREAM_BASE_TREE,
            "patch_sha256": PATCH_SHA256,
            "patched_commit": PATCHED_COMMIT,
            "patched_tree": PATCHED_TREE,
            "decision": "0021",
            "decision_path": (
                "docs/decisions/0021-kvquant-patch-main-repository-custody.md"
            ),
            "source_reconstruction_command": source_command,
            "source_reconstruction_result": "PASS",
            "reconstructed_source_checksum_result": "PASS",
            "reconstructed_source_validation": entry["source_validation"],
            "patched_files": patch_manifest["patched_files"],
            "official_author_gqa_support_claimed": False,
            "complete_upstream_checkout_committed": False,
            "complete_upstream_checkout_published_to_r2": False,
            "source_containing_image_published": False,
        },
    )
    run.write_json(
        "outlier_policy.json",
        {
            "schema_version": "kvbench-phase9-outlier-policy-1.0.0",
            "sink_tokens": 5,
            "sink_token_indices_per_sequence": [0, 1, 2, 3, 4],
            "sink_storage_dtype": "float16",
            "key_interaction": "key_quantization_is_pre_rope",
            "sink_excluded_from_dense_quantized_history": True,
            "sink_excluded_from_sparse_selection": True,
            "native_kv_vector_width": 1024,
            "key_outlier_cap": 12,
            "value_outlier_cap": 12,
            "lower_tail_entries": 6,
            "upper_tail_entries": 6,
            "slot_layout": {
                "lower_tail": [0, 1, 2, 3, 4, 5],
                "upper_tail": [6, 7, 8, 9, 10, 11],
            },
            "outlier_value_dtype": "float32",
            "outlier_index_dtype": "int32",
            "metadata_dtype": "float32",
            "tie_breaking": "value_then_ascending_native_index",
            "duplicate_or_overlapping_indices": False,
            "unused_value_slots": "zero_filled",
            "unused_index_slots": "zero_filled",
            "shared_across_bits": ["kvq4", "kvq3", "kvq2"],
            "quality_or_performance_used_for_selection": False,
        },
    )
    run.write_bytes(
        "environment/python-freeze.txt",
        PYTHON_FREEZE_PATH.read_bytes(),
    )


def _build_compact_inventory(stage: Path, run_id: str) -> dict[str, object]:
    fisher_manifest = _load_json(stage / "fisher_manifest.json")
    fisher_digest = fisher_manifest["file_sha256"]
    files: list[dict[str, object]] = []
    for path in sorted(stage.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(stage).as_posix()
        if (
            relative == "inventory.json"
            or relative == "manifest.initial.json"
            or relative.startswith("lifecycle/")
        ):
            continue
        digest = (
            fisher_digest
            if relative == "fisher/fisher.safetensors"
            else _sha256_file(path)
        )
        files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    return {
        "schema_version": "kvbench-phase9-calibration-inventory-1.0.0",
        "calibration_id": run_id,
        "files": files,
        "excluded_lifecycle_controls": [
            "manifest.initial.json",
            "lifecycle/",
            "inventory.json",
            "manifest.json",
            "artifact_inventory.json",
            "checksums.sha256",
            "COMPLETE",
        ],
    }


def _quantizer_regeneration(
    *,
    run: Any,
    entry: dict[str, object],
    commands: list[dict[str, object]],
    token_sha256: str,
    dataset_root: str,
    fisher_root: str,
) -> None:
    records: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(
        prefix="kvbench-phase9-quantizer-regeneration-"
    ) as temporary:
        regeneration = Path(temporary)
        for bit_width in (4, 3, 2):
            label = f"regenerate-kvq{bit_width}"
            _run_container(
                label=label,
                entry=entry,
                stage=run.stage,
                worker_arguments=[
                    "run-quantizer",
                    "--bit-width",
                    str(bit_width),
                    "--source-root",
                    "/source",
                    "--token-tensor",
                    "/output/tokens/input_ids.safetensors",
                    "--token-tensor-sha256",
                    token_sha256,
                    "--dataset-root",
                    dataset_root,
                    "--fisher",
                    "/output/fisher/fisher.safetensors",
                    "--fisher-root",
                    fisher_root,
                    "--output-safe",
                    f"/regen/kvq{bit_width}.safetensors",
                    "--output-manifest",
                    f"/regen/kvq{bit_width}.manifest.json",
                    *_model_arguments(),
                ],
                commands=commands,
                extra_mounts=((regeneration, "/regen", False),),
            )
            original = run.stage / f"quantizers/kvq{bit_width}.safetensors"
            rebuilt = regeneration / f"kvq{bit_width}.safetensors"
            original_manifest = _load_json(
                run.stage / f"quantizers/kvq{bit_width}.manifest.json"
            )
            rebuilt_manifest = _load_json(
                regeneration / f"kvq{bit_width}.manifest.json"
            )
            byte_equal = (
                original.stat().st_size == rebuilt.stat().st_size
                and _sha256_file(original) == _sha256_file(rebuilt)
            )
            record = {
                "variant_id": f"kvq{bit_width}",
                "fresh_process": True,
                "safe_serialization_byte_equal": byte_equal,
                "original_safe_sha256": _sha256_file(original),
                "regenerated_safe_sha256": _sha256_file(rebuilt),
                "original_source_native_pickle_sha256": original_manifest[
                    "source_native_pickle_sha256"
                ],
                "regenerated_source_native_pickle_sha256": rebuilt_manifest[
                    "source_native_pickle_sha256"
                ],
            }
            records.append(record)
            if not byte_equal:
                failure = run.stage / "reproducibility/regeneration-mismatch"
                failure.mkdir(parents=True, exist_ok=True)
                shutil.copy2(
                    rebuilt,
                    failure / f"kvq{bit_width}.safetensors",
                )
                shutil.copy2(
                    regeneration / f"kvq{bit_width}.manifest.json",
                    failure / f"kvq{bit_width}.manifest.json",
                )
                raise Phase9CalibrationError(
                    f"kvq{bit_width} safe regeneration is not byte-identical"
                )
    run.write_json(
        "reproducibility/quantizer_regeneration.json",
        {
            "schema_version": "kvbench-phase9-quantizer-regeneration-1.0.0",
            "status": "PASS",
            "serialization": "canonical_safetensors",
            "exact_byte_identity_required": True,
            "records": records,
        },
    )


def command_run(arguments: argparse.Namespace) -> int:
    entry = _validate_entry_inputs(arguments)
    run_id, contract_sha256, contract = calibration_identity()
    source_command = _source_command(entry["source_root"])
    store = phase9_calibration_artifact_store(REPOSITORY_ROOT)
    root = REPOSITORY_ROOT / "calibration/kvquant"
    if (root / run_id).exists():
        raise Phase9CalibrationError(
            f"completed calibration ID already exists: {run_id}"
        )
    created_at = _utc_now()
    initial = _initial_manifest(
        run_id=run_id,
        contract_sha256=contract_sha256,
        git_sha=entry["head"],
        source_command=source_command,
        created_at=created_at,
    )
    run = store.create(run_id, initial)
    run.start()
    started_at = _utc_now()
    commands: list[dict[str, object]] = []
    current_stage = "authority"
    try:
        _write_authority_payloads(
            run,
            run_id=run_id,
            contract_sha256=contract_sha256,
            contract=contract,
            source_command=source_command,
            entry=entry,
        )

        current_stage = "freeze-dataset"
        _run_container(
            label=current_stage,
            entry=entry,
            stage=run.stage,
            worker_arguments=[
                "freeze-dataset",
                "--dataset-parquet",
                "/input/train.parquet",
                "--image-manifest",
                "/input/image-manifest.json",
                "--output-root",
                "/output",
                *_model_arguments(),
            ],
            commands=commands,
        )
        dataset_manifest = _load_json(run.stage / "dataset_manifest.json")
        token_sha256 = dataset_manifest["token_tensor"]["file_sha256"]
        dataset_root = dataset_manifest["dataset_root_sha256"]

        current_stage = "full-fisher"
        _run_container(
            label=current_stage,
            entry=entry,
            stage=run.stage,
            worker_arguments=[
                "run-fisher",
                "--source-root",
                "/source",
                "--token-tensor",
                "/output/tokens/input_ids.safetensors",
                "--output-dir",
                "/output/fisher",
                *_model_arguments(),
            ],
            commands=commands,
        )
        current_stage = "fisher-manifest"
        _run_container(
            label=current_stage,
            entry=entry,
            stage=run.stage,
            worker_arguments=[
                "fisher-manifest",
                "--fisher",
                "/output/fisher/fisher.safetensors",
                "--output",
                "/output/fisher_manifest.json",
            ],
            commands=commands,
        )
        fisher_manifest = _load_json(run.stage / "fisher_manifest.json")
        fisher_root = fisher_manifest["file_sha256"]

        for bit_width in (4, 3, 2):
            current_stage = f"generate-kvq{bit_width}"
            _run_container(
                label=current_stage,
                entry=entry,
                stage=run.stage,
                worker_arguments=[
                    "run-quantizer",
                    "--bit-width",
                    str(bit_width),
                    "--source-root",
                    "/source",
                    "--token-tensor",
                    "/output/tokens/input_ids.safetensors",
                    "--token-tensor-sha256",
                    token_sha256,
                    "--dataset-root",
                    dataset_root,
                    "--fisher",
                    "/output/fisher/fisher.safetensors",
                    "--fisher-root",
                    fisher_root,
                    "--output-safe",
                    f"/output/quantizers/kvq{bit_width}.safetensors",
                    "--output-manifest",
                    f"/output/quantizers/kvq{bit_width}.manifest.json",
                    *_model_arguments(),
                ],
                commands=commands,
            )

        current_stage = "layer-stats"
        _run_container(
            label=current_stage,
            entry=entry,
            stage=run.stage,
            worker_arguments=[
                "layer-stats",
                "--token-tensor",
                "/output/tokens/input_ids.safetensors",
                "--fisher-manifest",
                "/output/fisher_manifest.json",
                "--kvq4",
                "/output/quantizers/kvq4.safetensors",
                "--kvq3",
                "/output/quantizers/kvq3.safetensors",
                "--kvq2",
                "/output/quantizers/kvq2.safetensors",
                "--output",
                "/output/layer_stats.parquet",
                *_model_arguments(),
            ],
            commands=commands,
        )

        current_stage = "token-reconstruction"
        _run_container(
            label=current_stage,
            entry=entry,
            stage=run.stage,
            worker_arguments=[
                "reconstruct-tokens",
                "--dataset-parquet",
                "/input/train.parquet",
                "--expected-dataset-manifest",
                "/output/dataset_manifest.json",
                "--expected-token-tensor",
                "/output/tokens/input_ids.safetensors",
                "--output",
                "/output/reproducibility/token_reconstruction.json",
                *_model_arguments(),
            ],
            commands=commands,
        )

        current_stage = "fisher-replay"
        _run_container(
            label=current_stage,
            entry=entry,
            stage=run.stage,
            worker_arguments=[
                "replay-fisher",
                "--token-tensor",
                "/output/tokens/input_ids.safetensors",
                "--fisher",
                "/output/fisher/fisher.safetensors",
                "--output",
                "/output/reproducibility/fisher_replay.json",
                *_model_arguments(),
            ],
            commands=commands,
        )

        current_stage = "outlier-policy"
        _run_container(
            label=current_stage,
            entry=entry,
            stage=run.stage,
            worker_arguments=[
                "policy-check",
                "--output",
                "/output/reproducibility/outlier_policy.json",
            ],
            commands=commands,
        )

        current_stage = "quantizer-regeneration"
        _quantizer_regeneration(
            run=run,
            entry=entry,
            commands=commands,
            token_sha256=token_sha256,
            dataset_root=dataset_root,
            fisher_root=fisher_root,
        )

        current_stage = "payload-validation"
        _run_container(
            label=current_stage,
            entry=entry,
            stage=run.stage,
            worker_arguments=[
                "validate-payloads",
                "--bundle",
                "/output",
                "--output",
                "/output/validation/calibration.json",
            ],
            commands=commands,
        )
        run.write_json(
            "commands.json",
            {
                "schema_version": "kvbench-phase9-command-log-1.0.0",
                "source_reconstruction_command": source_command,
                "commands": commands,
                "performance_timing_collected": False,
            },
        )
        run.write_json(
            "inventory.json",
            _build_compact_inventory(run.stage, run_id),
        )
        completed = dataclasses.replace(
            initial,
            status=RunStatus.COMPLETED,
            started_at_utc=started_at,
            finished_at_utc=_utc_now(),
            inventory_path="artifact_inventory.json",
        )
        final = run.finalize(completed)
    except BaseException as error:
        try:
            if run.state == "running":
                if not (run.stage / "commands.json").exists():
                    run.write_json(
                        "commands.json",
                        {
                            "schema_version": "kvbench-phase9-command-log-1.0.0",
                            "source_reconstruction_command": source_command,
                            "commands": commands,
                            "performance_timing_collected": False,
                        },
                    )
                run.write_json(
                    "failure.json",
                    {
                        "schema_version": "kvbench-phase9-calibration-failure-1.0.0",
                        "stage": current_stage,
                        "error_type": type(error).__name__,
                        "source_patch_changed": False,
                        "samples_changed": False,
                        "seed_changed": False,
                        "partial_fisher_used_for_quantizers": False,
                    },
                )
                failed = dataclasses.replace(
                    initial,
                    status=RunStatus.RUNTIME_FAILED,
                    started_at_utc=started_at,
                    finished_at_utc=_utc_now(),
                    inventory_path="artifact_inventory.json",
                    failure_reason=(
                        "Phase 9 calibration failed; see failure.json and logs"
                    ),
                )
                run.finalize(failed)
        except BaseException:
            pass
        raise

    validated = validate_local_artifact(final)
    print(
        json.dumps(
            {
                "status": "PASS",
                "calibration_id": run_id,
                "artifact": str(final),
                "calibration_contract_sha256": contract_sha256,
                "root_sha256": validated.root_sha256,
                "object_count": len(validated.files),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _resolve_artifact(argument: Path | None) -> Path:
    if argument is not None:
        return argument.resolve(strict=True)
    config = _load_json(REPOSITORY_ROOT / "configs/methods/kvquant.yaml")
    calibration = config.get("calibration")
    if not isinstance(calibration, dict):
        raise Phase9CalibrationError(
            "KVQuant config lacks a finalized calibration reference"
        )
    relative = calibration.get("local_bundle_path")
    if not isinstance(relative, str):
        raise Phase9CalibrationError("KVQuant calibration path is invalid")
    return (REPOSITORY_ROOT / relative).resolve(strict=True)


def _validate_payloads_in_container(artifact: Path) -> None:
    command = [
        "docker",
        "run",
        "--rm",
        "--network=none",
        "--read-only",
        "--pids-limit=512",
        "--tmpfs",
        "/tmp:rw,exec,nosuid,nodev,size=1g",
        "--mount",
        _mount(WORKER_PATH, "/opt/phase9/worker.py", readonly=True),
        "--mount",
        _mount(artifact, "/bundle", readonly=True),
        "--env",
        "PYTHONPATH=/opt/kvbench/.phase3/site-packages",
        CALIBRATION_IMAGE_DIGEST,
        "python3",
        "/opt/phase9/worker.py",
        "validate-payloads",
        "--bundle",
        "/bundle",
    ]
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise Phase9CalibrationError("container payload validation failed")


def command_validate(arguments: argparse.Namespace) -> int:
    artifact = _resolve_artifact(arguments.artifact)
    local = validate_local_artifact(artifact)
    _validate_payloads_in_container(artifact)
    config = _load_json(artifact / "calibration_config.json")
    run_id, contract_sha256, contract = calibration_identity()
    if (
        artifact.name != run_id
        or config.get("calibration_id") != run_id
        or config.get("calibration_contract_sha256") != contract_sha256
        or config.get("contract") != contract
    ):
        raise Phase9CalibrationError("calibration contract identity drifted")
    print(
        json.dumps(
            {
                "status": "PASS",
                "artifact": str(artifact),
                "calibration_id": run_id,
                "root_sha256": local.root_sha256,
                "object_count": len(local.files),
                "complete": True,
                "payload_validation": "PASS",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--source-root", required=True)
    run.add_argument("--dataset-parquet", required=True)
    run.add_argument("--model-cache", required=True)
    run.set_defaults(function=command_run)
    validate = commands.add_parser("validate")
    validate.add_argument("--artifact", type=Path)
    validate.set_defaults(function=command_validate)
    identity = commands.add_parser("identity")
    identity.set_defaults(
        function=lambda _arguments: (
            print(
                json.dumps(
                    {
                        "calibration_id": calibration_identity()[0],
                        "calibration_contract_sha256": calibration_identity()[1],
                    },
                    sort_keys=True,
                )
            )
            or 0
        )
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        return int(arguments.function(arguments))
    except (
        FileExistsError,
        OSError,
        Phase9CalibrationError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
