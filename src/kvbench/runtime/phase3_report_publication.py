"""Append-only, COMPLETE-last publication for Phase 3 campaign reports."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
import os
from pathlib import Path, PurePosixPath
import secrets
import stat
from typing import Any

from kvbench.errors import ArtifactConflictError
from kvbench.runtime.artifacts import (
    _ensure_real_directory,
    _fsync_directory,
    _rename_noreplace,
    _write_exclusive,
)
from kvbench.schema import Phase3G1AdmissionReport, canonical_json_bytes, sha256_hex


REPORT_COMPLETION_V2 = "kvbench-phase3-g1-completion-2.0.0"
SOURCE_INDEX_V2 = "kvbench-phase3-report-source-index-2.0.0"
SOURCE_CAMPAIGNS_V2 = "kvbench-phase3-report-sources-2.0.0"
REPORT_INVENTORY_V2 = "kvbench-phase3-report-inventory-2.0.0"
WRITE_RESULT_V2 = "kvbench-phase3-g1-write-result-2.0.0"
_WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
_CONTROL_DIRECTORIES = (
    ".kvbench-report-staging",
    ".kvbench-report-reservations",
    ".kvbench-report-failed",
)
_FAILED_CONTROL_FILES = frozenset(
    {"failed_inventory.json", "failed_checksums.sha256"}
)
_FAILURE_MARKERS = (
    "FAILED",
    "PROMOTION_FAILED",
    "POST_PROMOTION_FAILED",
)
_EXCLUSIVE_NONCE_ATTEMPTS = 16
_QUARANTINE_REFERENCE_FILE = "quarantined_owned_stage_reference.json"


class ReportPublicationError(RuntimeError):
    """A report could not be published without weakening evidence integrity."""


def _resolve_unaliased_path(
    value: str | Path,
    *,
    strict: bool,
    label: str,
) -> Path:
    """Resolve a path while rejecting a symlink in any supplied component."""

    supplied = Path(value).expanduser()
    lexical = Path(os.path.abspath(os.fspath(supplied)))
    try:
        resolved = supplied.resolve(strict=strict)
    except OSError as error:
        raise ReportPublicationError(f"{label} cannot be resolved safely") from error
    if lexical != resolved:
        raise ReportPublicationError(f"{label} contains a symlink alias")
    if strict:
        metadata = resolved.lstat()
        if not stat.S_ISDIR(metadata.st_mode):
            raise ReportPublicationError(f"{label} is not a real directory")
    return resolved


def _json_bytes(value: Any) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _strict_json(path: Path) -> dict[str, Any]:
    from kvbench.runtime.phase3_report import _strict_json_object

    return _strict_json_object(path, canonical=True)


def _file_digest(path: Path) -> str:
    return sha256_hex(path.read_bytes())


def _validated_tree_members(
    directory: Path,
    *,
    require_immutable: bool,
) -> tuple[set[str], set[str]]:
    metadata = directory.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ReportPublicationError("evidence root is not a real directory")
    if require_immutable and metadata.st_mode & _WRITE_BITS:
        raise ReportPublicationError("evidence root is writable")
    files: set[str] = set()
    directories: set[str] = set()
    for target in sorted(directory.rglob("*")):
        item = target.lstat()
        relative = target.relative_to(directory).as_posix()
        if stat.S_ISLNK(item.st_mode):
            raise ReportPublicationError("evidence tree contains a symlink")
        if require_immutable and item.st_mode & _WRITE_BITS:
            raise ReportPublicationError("evidence tree contains writable content")
        if stat.S_ISDIR(item.st_mode):
            directories.add(relative)
        elif stat.S_ISREG(item.st_mode):
            if item.st_nlink != 1:
                raise ReportPublicationError(
                    "evidence tree contains a multiply-linked file"
                )
            files.add(relative)
        else:
            raise ReportPublicationError(
                "evidence tree contains non-regular content"
            )
    return files, directories


def _parent_directories(files: set[str]) -> set[str]:
    directories: set[str] = set()
    for relative in files:
        for parent in PurePosixPath(relative).parents:
            if parent != PurePosixPath("."):
                directories.add(parent.as_posix())
    return directories


def _immutable_bundle_snapshot(directory: Path) -> dict[str, Any]:
    files, directories = _validated_tree_members(
        directory,
        require_immutable=True,
    )
    records = [
        {
            "path": relative,
            "size_bytes": (directory / relative).stat().st_size,
            "sha256": _file_digest(directory / relative),
        }
        for relative in sorted(files)
    ]
    topology = {
        "directories": sorted(directories),
        "files": records,
    }
    return {
        "schema_version": "kvbench-immutable-report-tree-1.0.0",
        "file_count": len(records),
        "directory_count": len(directories),
        "tree_sha256": sha256_hex(canonical_json_bytes(topology)),
        **topology,
        "immutable_mode_valid": True,
    }


def _strict_failed_validation_payload(value: object) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "valid",
        "report_sha256",
        "errors",
    }:
        return False
    report_sha = value.get("report_sha256")
    errors = value.get("errors")
    return (
        value.get("schema_version")
        == "kvbench-phase3-g1-validation-2.0.0"
        and value.get("valid") is False
        and isinstance(report_sha, str)
        and (
            report_sha == ""
            or (
                len(report_sha) == 64
                and all(
                    character in "0123456789abcdef"
                    for character in report_sha
                )
            )
        )
        and isinstance(errors, list)
        and bool(errors)
        and all(isinstance(error, str) and bool(error) for error in errors)
        and len(errors) == len(set(errors))
    )
def _lstat_identity(path: Path) -> dict[str, Any]:
    """Capture one path entry without following it or traversing children."""

    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return {"kind": "absent"}
    if stat.S_ISDIR(metadata.st_mode):
        kind = "directory"
    elif stat.S_ISREG(metadata.st_mode):
        kind = "regular_file"
    elif stat.S_ISLNK(metadata.st_mode):
        kind = "symlink"
    elif stat.S_ISFIFO(metadata.st_mode):
        kind = "fifo"
    elif stat.S_ISSOCK(metadata.st_mode):
        kind = "socket"
    elif stat.S_ISCHR(metadata.st_mode):
        kind = "character_device"
    elif stat.S_ISBLK(metadata.st_mode):
        kind = "block_device"
    else:
        kind = "other"
    return {
        "kind": kind,
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": metadata.st_mode,
        "link_count": metadata.st_nlink,
    }


def _quarantine_reference_payload(
    *,
    stage: Path,
    report_id: str,
    phase: str,
    original_error: BaseException,
    topology_error: BaseException,
) -> dict[str, Any]:
    return {
        "schema_version": "kvbench-phase3-quarantined-stage-reference-1.0.0",
        "report_id": report_id,
        "stage_directory": str(stage),
        "stage_name": stage.name,
        "stage_root_lstat": _lstat_identity(stage),
        "complete_entry_lstat": _lstat_identity(stage / "COMPLETE"),
        "original_failure_phase": phase,
        "original_failure_type": type(original_error).__name__,
        "topology_error_type": type(topology_error).__name__,
        "topology_error_message": str(topology_error),
        "quarantine_reason": "unsafe_owned_stage_not_admitted",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _validate_quarantine_reference(
    reference: Mapping[str, Any],
    *,
    report_root: Path,
    report_id: str,
) -> None:
    expected_keys = {
        "schema_version",
        "report_id",
        "stage_directory",
        "stage_name",
        "stage_root_lstat",
        "complete_entry_lstat",
        "original_failure_phase",
        "original_failure_type",
        "topology_error_type",
        "topology_error_message",
        "quarantine_reason",
        "captured_at_utc",
    }
    stage_name = reference.get("stage_name")
    if not isinstance(stage_name, str):
        raise ReportPublicationError("quarantine stage name is absent")
    prefix = f"{report_id}."
    suffix = ".staging"
    nonce = (
        stage_name[len(prefix) : -len(suffix)]
        if stage_name.startswith(prefix) and stage_name.endswith(suffix)
        else ""
    )
    expected_stage = report_root / ".kvbench-report-staging" / stage_name
    captured_at = reference.get("captured_at_utc")
    try:
        parsed_timestamp = (
            datetime.fromisoformat(captured_at)
            if isinstance(captured_at, str)
            else None
        )
    except ValueError as error:
        raise ReportPublicationError(
            "quarantine capture timestamp is malformed"
        ) from error
    if (
        set(reference) != expected_keys
        or reference.get("schema_version")
        != "kvbench-phase3-quarantined-stage-reference-1.0.0"
        or reference.get("report_id") != report_id
        or len(nonce) != 12
        or any(character not in "0123456789abcdef" for character in nonce)
        or reference.get("stage_directory") != str(expected_stage)
        or reference.get("stage_root_lstat") != _lstat_identity(expected_stage)
        or reference.get("complete_entry_lstat")
        != _lstat_identity(expected_stage / "COMPLETE")
        or reference.get("quarantine_reason")
        != "unsafe_owned_stage_not_admitted"
        or not (
            isinstance(reference.get("original_failure_phase"), str)
            and reference.get("original_failure_phase")
        )
        or not (
            isinstance(reference.get("original_failure_type"), str)
            and reference.get("original_failure_type")
        )
        or not (
            isinstance(reference.get("topology_error_type"), str)
            and reference.get("topology_error_type")
        )
        or not (
            isinstance(reference.get("topology_error_message"), str)
            and reference.get("topology_error_message")
        )
        or parsed_timestamp is None
        or parsed_timestamp.tzinfo is None
        or reference.get("stage_root_lstat", {}).get("kind") == "absent"
    ):
        raise ReportPublicationError("quarantine reference is malformed")


def _tree_record(directory: Path, *, require_immutable: bool) -> dict[str, Any]:
    file_names, directory_names = _validated_tree_members(
        directory,
        require_immutable=require_immutable,
    )
    file_records = [
        {
            "path": relative,
            "size_bytes": (directory / relative).stat().st_size,
            "sha256": _file_digest(directory / relative),
        }
        for relative in sorted(file_names)
    ]
    if not file_records:
        raise ReportPublicationError("source evidence directory is empty")
    topology = {
        "directories": sorted(directory_names),
        "files": file_records,
    }
    return {
        "file_count": len(file_records),
        "directory_count": len(directory_names),
        "tree_sha256": sha256_hex(canonical_json_bytes(topology)),
        **topology,
        "immutable_mode_valid": require_immutable,
    }


def capture_phase3_source_index(
    repository_root: str | Path,
    fixed_campaign_id: str,
    growing_campaign_id: str,
) -> dict[str, Any]:
    """Validate and hash every byte of the exact 20-run source selection."""

    from kvbench.runtime.phase3_report import (
        load_phase3_campaign_evidence,
    )

    repository = _resolve_unaliased_path(
        repository_root, strict=True, label="repository root"
    )
    runs = load_phase3_campaign_evidence(
        fixed_campaign_id,
        growing_campaign_id,
        repository_root=repository,
    )
    run_records: list[dict[str, Any]] = []
    for run in runs:
        tree = _tree_record(run.run_dir, require_immutable=True)
        run_records.append(
            {
                "run_id": run.run_id,
                "point_id": run.point_id,
                "status": run.manifest.status.value,
                "relative_directory": run.run_dir.relative_to(repository).as_posix(),
                "manifest_sha256": run.completion.manifest_sha256,
                "artifact_inventory_sha256": (
                    run.completion.artifact_inventory_sha256
                ),
                "checksum_ledger_sha256": run.completion.checksum_ledger_sha256,
                "completion_sha256": _file_digest(run.run_dir / "COMPLETE"),
                **tree,
            }
        )
    campaign_records: list[dict[str, Any]] = []
    for campaign_id in (fixed_campaign_id, growing_campaign_id):
        directory = repository / "artifacts" / "phase3_campaigns" / campaign_id
        tree = _tree_record(directory, require_immutable=True)
        campaign_records.append(
            {
                "campaign_id": campaign_id,
                "relative_directory": directory.relative_to(repository).as_posix(),
                "preregistration_sha256": _file_digest(
                    directory / "preregistered.json"
                ),
                "result_sha256": _file_digest(directory / "result.json"),
                "completion_sha256": _file_digest(directory / "COMPLETE"),
                **tree,
            }
        )
    payload = {
        "schema_version": SOURCE_INDEX_V2,
        "fixed_campaign_id": fixed_campaign_id,
        "growing_campaign_id": growing_campaign_id,
        "run_count": len(run_records),
        "runs": run_records,
        "campaigns": campaign_records,
        "all_checksums_valid": True,
        "all_sources_immutable": True,
    }
    if len(run_records) != 20:
        raise ReportPublicationError("source index does not contain exactly 20 runs")
    return payload


def _payload_files(root: Path, excluded: set[str]) -> list[Path]:
    files, _ = _validated_tree_members(root, require_immutable=False)
    return [
        root / relative
        for relative in sorted(files)
        if relative not in excluded
    ]


def _inventory_for_exclusions(
    stage: Path,
    report_id: str,
    exclusions: set[str] | frozenset[str],
) -> dict[str, Any]:
    file_names, directories = _validated_tree_members(
        stage,
        require_immutable=False,
    )
    files = [
        {
            "path": relative,
            "size_bytes": (stage / relative).stat().st_size,
            "sha256": _file_digest(stage / relative),
        }
        for relative in sorted(file_names)
        if relative not in exclusions
    ]
    return {
        "schema_version": REPORT_INVENTORY_V2,
        "report_id": report_id,
        "files": files,
        "directories": sorted(directories),
        "excluded_control_files": sorted(exclusions),
    }


def _inventory(stage: Path, report_id: str) -> dict[str, Any]:
    return _inventory_for_exclusions(
        stage,
        report_id,
        {"artifact_inventory.json", "checksums.sha256", "COMPLETE"},
    )


def _failed_inventory(stage: Path, report_id: str) -> dict[str, Any]:
    """Independently inventory every preserved byte except failure controls."""

    return _inventory_for_exclusions(
        stage,
        report_id,
        _FAILED_CONTROL_FILES,
    )


def _logical_report_id(directory: Path) -> str:
    name = directory.name
    if name.endswith(".staging"):
        parts = name.rsplit(".", 2)
        if len(parts) != 3 or not parts[0]:
            raise ReportPublicationError("staging report name is malformed")
        return parts[0]
    return name


def _failed_report_id(directory: Path) -> str:
    from kvbench.runtime.phase3_report import _REPORT_ID

    suffix = ".failed"
    if not directory.name.endswith(suffix):
        raise ReportPublicationError("failed report directory name is malformed")
    stem = directory.name[: -len(suffix)]
    report_id, separator, nonce = stem.rpartition(".")
    if (
        not separator
        or not _REPORT_ID.fullmatch(report_id)
        or len(nonce) != 12
        or any(character not in "0123456789abcdef" for character in nonce)
    ):
        raise ReportPublicationError("failed report directory name is malformed")
    return report_id


def _ledger(stage: Path) -> bytes:
    return b"".join(
        f"{_file_digest(path)}  {path.relative_to(stage).as_posix()}\n".encode()
        for path in _payload_files(stage, {"checksums.sha256", "COMPLETE"})
    )


def _parse_ledger(data: bytes) -> dict[str, str]:
    entries: dict[str, str] = {}
    if data and not data.endswith(b"\n"):
        raise ReportPublicationError("report ledger lacks a trailing newline")
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeError as error:
        raise ReportPublicationError("report ledger is not UTF-8") from error
    for line in lines:
        digest, separator, relative = line.partition("  ")
        path = PurePosixPath(relative)
        if (
            not separator
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != relative
            or relative in entries
        ):
            raise ReportPublicationError("report ledger is malformed")
        entries[relative] = digest
    if list(entries) != sorted(entries):
        raise ReportPublicationError("report ledger is not sorted")
    return entries


def _write(
    stage: Path,
    relative: str,
    data: bytes,
    event_hook: Callable[[str], None] | None,
) -> None:
    _write_exclusive(stage, relative, data)
    if event_hook is not None:
        event_hook(relative)


def _validate_payloads(
    directory: Path,
    *,
    repository: Path,
    require_complete: bool,
    require_immutable: bool,
) -> dict[str, Any]:
    """Independently rederive source and report bytes for a v2 bundle."""

    from kvbench.runtime.phase3_report import (
        Phase3ReportError,
        build_phase3_g1_report,
    )

    errors: list[str] = []
    report_sha = ""
    try:
        actual_files, actual_directories = _validated_tree_members(
            directory,
            require_immutable=require_immutable,
        )
        sources = _strict_json(directory / "source_campaigns.json")
        fixed = sources.get("fixed_campaign_id")
        growing = sources.get("growing_campaign_id")
        if (
            sources.get("schema_version") != SOURCE_CAMPAIGNS_V2
            or not isinstance(fixed, str)
            or not isinstance(growing, str)
            or sources.get("explicit_selection") is not True
        ):
            raise ReportPublicationError("report source selection is invalid")
        source_index = _strict_json(directory / "source_runs.json")
        observed_index = capture_phase3_source_index(repository, fixed, growing)
        if source_index != observed_index:
            errors.append("source-run index differs from current immutable sources")
        report_payload = _strict_json(directory / "report.json")
        report = Phase3G1AdmissionReport.from_dict(report_payload)
        derivation = _strict_json(directory / "derivation.json")
        provenance = derivation.get("report_git_provenance")
        generator_sha = (
            provenance.get("report_generator_git_sha")
            if isinstance(provenance, Mapping)
            else None
        )
        if not isinstance(generator_sha, str):
            raise ReportPublicationError("report generator provenance is absent")
        expected_report, expected_stability, expected_derivation = (
            build_phase3_g1_report(
                fixed,
                growing,
                repository_root=repository,
                generated_at_utc=report.generated_at_utc,
                report_generator_git_sha=generator_sha,
                recorded_report_git_provenance=provenance,
            )
        )
        if report_payload != expected_report.to_dict():
            errors.append("report differs from independently rederived evidence")
        if derivation != expected_derivation:
            errors.append("derivation differs from independently rederived evidence")
        expected_files = {
            "source_runs.json",
            "source_campaigns.json",
            "derivation.json",
            "report.json",
            "artifact_inventory.json",
            "checksums.sha256",
            *expected_stability.keys(),
        }
        if require_complete:
            expected_files.add("COMPLETE")
        if actual_files != expected_files:
            errors.append("report exact file set differs")
        if actual_directories != _parent_directories(expected_files):
            errors.append("report exact directory topology differs")
        for relative, payload in expected_stability.items():
            if _strict_json(directory / relative) != payload:
                errors.append(f"stability derivation differs: {relative}")
        inventory = _strict_json(directory / "artifact_inventory.json")
        logical_report_id = _logical_report_id(directory)
        expected_inventory = _inventory(directory, logical_report_id)
        if inventory != expected_inventory:
            errors.append("report inventory differs")
        ledger_bytes = (directory / "checksums.sha256").read_bytes()
        entries = _parse_ledger(ledger_bytes)
        actual_ledger_files = {
            path.relative_to(directory).as_posix()
            for path in _payload_files(directory, {"checksums.sha256", "COMPLETE"})
        }
        if set(entries) != actual_ledger_files:
            errors.append("report ledger coverage differs")
        for relative, digest in entries.items():
            if _file_digest(directory / relative) != digest:
                errors.append(f"checksum mismatch: {relative}")
        report_sha = _file_digest(directory / "report.json")
        if require_complete:
            completion = _strict_json(directory / "COMPLETE")
            if (
                completion.get("schema_version") != REPORT_COMPLETION_V2
                or completion.get("report_id") != logical_report_id
                or completion.get("status") != report.status.value
                or completion.get("report_sha256") != report_sha
                or completion.get("source_index_sha256")
                != _file_digest(directory / "source_runs.json")
                or completion.get("artifact_inventory_sha256")
                != _file_digest(directory / "artifact_inventory.json")
                or completion.get("checksum_ledger_sha256")
                != sha256_hex(ledger_bytes)
                or completion.get("written_last") is not True
            ):
                errors.append("report completion marker differs")
    except (
        OSError,
        UnicodeError,
        ValueError,
        Phase3ReportError,
        ReportPublicationError,
    ) as error:
        errors.append(f"report validation failed closed: {type(error).__name__}")
    return {
        "schema_version": "kvbench-phase3-g1-validation-2.0.0",
        "valid": not errors,
        "report_sha256": report_sha,
        "errors": errors,
    }


def validate_phase3_g1_report_directory_v2(
    path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    try:
        directory = _resolve_unaliased_path(
            path, strict=True, label="report validation path"
        )
        repository = (
            directory.parents[2]
            if repository_root is None
            else _resolve_unaliased_path(
                repository_root, strict=True, label="repository root"
            )
        )
        return _validate_payloads(
            directory,
            repository=repository,
            require_complete=True,
            require_immutable=True,
        )
    except (OSError, ValueError, ReportPublicationError) as error:
        return {
            "schema_version": "kvbench-phase3-g1-validation-2.0.0",
            "valid": False,
            "report_sha256": "",
            "errors": [
                f"report validation failed closed: {type(error).__name__}"
            ],
        }


def validate_failed_report_attempt(path: str | Path) -> dict[str, Any]:
    """Validate one preserved, immutable report-publication failure."""

    errors: list[str] = []
    try:
        directory = _resolve_unaliased_path(
            path, strict=True, label="failed-report validation path"
        )
        directory_report_id = _failed_report_id(directory)
        actual_files, actual_directories = _validated_tree_members(
            directory,
            require_immutable=True,
        )
        markers = [
            name for name in _FAILURE_MARKERS if name in actual_files
        ]
        if len(markers) != 1:
            raise ReportPublicationError("failed report marker is absent or ambiguous")
        marker_name = markers[0]
        marker = _strict_json(directory / marker_name)
        complete_expected = marker_name != "FAILED"
        if (
            set(marker)
            != {
                "schema_version",
                "report_id",
                "failure_phase",
                "failure_type",
                "complete_written",
            }
            or marker.get("schema_version")
            != "kvbench-phase3-report-failure-1.0.0"
            or marker.get("report_id") != directory_report_id
            or not (
                isinstance(marker.get("failure_phase"), str)
                and marker.get("failure_phase")
            )
            or not (
                isinstance(marker.get("failure_type"), str)
                and marker.get("failure_type")
            )
            or marker.get("complete_written") is not complete_expected
        ):
            raise ReportPublicationError("failed report marker is malformed")
        local_complete = "COMPLETE" in actual_files
        reserved_directory_names = {
            "COMPLETE",
            *_FAILURE_MARKERS,
            *_FAILED_CONTROL_FILES,
        }
        if reserved_directory_names & actual_directories:
            raise ReportPublicationError("failure control path is a directory")
        if (marker_name == "PROMOTION_FAILED") is not local_complete:
            raise ReportPublicationError(
                "failure marker and local COMPLETE semantics differ"
            )
        if local_complete:
            local_completion = _strict_json(directory / "COMPLETE")
            if (
                local_completion.get("schema_version") != REPORT_COMPLETION_V2
                or local_completion.get("report_id") != directory_report_id
            ):
                raise ReportPublicationError(
                    "preserved COMPLETE does not identify the failed report"
                )
        reference_path = directory / "promoted_bundle_reference.json"
        if marker_name == "POST_PROMOTION_FAILED":
            reference = _strict_json(reference_path)
            validation = reference.get("validation")
            expected_final = directory.parent.parent / directory_report_id
            if (
                set(reference)
                != {
                    "schema_version",
                    "report_id",
                    "report_directory",
                    "bundle_snapshot",
                    "validation",
                }
                or reference.get("schema_version")
                != "kvbench-phase3-promoted-report-reference-1.0.0"
                or reference.get("report_id") != directory_report_id
                or reference.get("report_directory") != str(expected_final)
                or not _strict_failed_validation_payload(validation)
            ):
                raise ReportPublicationError(
                    "post-promotion failure reference is malformed"
                )
            final_metadata = expected_final.lstat()
            if stat.S_ISLNK(final_metadata.st_mode) or not stat.S_ISDIR(
                final_metadata.st_mode
            ):
                raise ReportPublicationError(
                    "promoted report locator is not a real directory"
                )
            observed_snapshot = _immutable_bundle_snapshot(expected_final)
            if reference.get("bundle_snapshot") != observed_snapshot:
                raise ReportPublicationError(
                    "promoted report tree differs from recorded snapshot"
                )
            promoted_completion = _strict_json(expected_final / "COMPLETE")
            if (
                promoted_completion.get("schema_version")
                != REPORT_COMPLETION_V2
                or promoted_completion.get("report_id") != directory_report_id
            ):
                raise ReportPublicationError(
                    "promoted COMPLETE does not identify the report"
                )
        elif "promoted_bundle_reference.json" in actual_files:
            raise ReportPublicationError(
                "pre-promotion failure has a promoted-bundle reference"
            )
        if _QUARANTINE_REFERENCE_FILE in actual_files:
            if marker_name != "FAILED":
                raise ReportPublicationError(
                    "quarantine reference requires a FAILED marker"
                )
            _validate_quarantine_reference(
                _strict_json(directory / _QUARANTINE_REFERENCE_FILE),
                report_root=directory.parent.parent,
                report_id=directory_report_id,
            )
        inventory = _strict_json(directory / "failed_inventory.json")
        expected_inventory = _failed_inventory(directory, directory_report_id)
        if inventory != expected_inventory:
            errors.append("failed report inventory differs")
        ledger_path = directory / "failed_checksums.sha256"
        entries = _parse_ledger(ledger_path.read_bytes())
        actual = actual_files - {"failed_checksums.sha256"}
        if set(entries) != actual:
            errors.append("failed report ledger coverage differs")
        for relative, digest in entries.items():
            if _file_digest(directory / relative) != digest:
                errors.append(f"failed report checksum mismatch: {relative}")
    except (OSError, UnicodeError, ValueError, ReportPublicationError) as error:
        errors.append(f"failed report validation failed closed: {type(error).__name__}")
    return {
        "schema_version": "kvbench-phase3-report-failure-validation-1.0.0",
        "valid": not errors,
        "errors": errors,
    }


def _freeze_tree(directory: Path) -> None:
    for path in sorted(directory.rglob("*"), reverse=True):
        path.chmod(0o444 if path.is_file() else 0o555)
    directory.chmod(0o555)
    _fsync_directory(directory)


def _open_owned_stage_for_failure_append(stage: Path) -> None:
    _, directories = _validated_tree_members(
        stage,
        require_immutable=False,
    )
    stage.chmod(0o700)
    for relative in sorted(directories):
        (stage / relative).chmod(0o700)


def _create_owned_stage(staging_root: Path, report_id: str) -> Path:
    for _ in range(_EXCLUSIVE_NONCE_ATTEMPTS):
        candidate = staging_root / f"{report_id}.{secrets.token_hex(6)}.staging"
        try:
            os.mkdir(candidate, 0o700)
        except FileExistsError:
            continue
        return candidate
    raise ReportPublicationError(
        "could not create a unique owned report stage"
    )


def _failure_marker(
    *,
    report_id: str,
    phase: str,
    error: BaseException,
    complete_written: bool,
) -> bytes:
    return _json_bytes(
        {
            "schema_version": "kvbench-phase3-report-failure-1.0.0",
            "report_id": report_id,
            "failure_phase": phase,
            "failure_type": type(error).__name__,
            "complete_written": complete_written,
        }
    )


def _validate_failure_stage_for_promotion(

    stage: Path,
    *,
    failed_root: Path,
    report_id: str,
    marker_name: str,
    phase: str,
    error: BaseException,
) -> None:
    """Prove a frozen failure bundle is safe before no-replace promotion."""

    actual_files, actual_directories = _validated_tree_members(
        stage,
        require_immutable=True,
    )
    reserved_directory_names = {
        "COMPLETE",
        *_FAILURE_MARKERS,
        *_FAILED_CONTROL_FILES,
    }
    if reserved_directory_names & actual_directories:
        raise ReportPublicationError("failure control path is a directory")
    markers = [name for name in _FAILURE_MARKERS if name in actual_files]
    if markers != [marker_name]:
        raise ReportPublicationError("failure marker set changed before promotion")
    expected_marker = _failure_marker(
        report_id=report_id,
        phase=phase,
        error=error,
        complete_written=marker_name != "FAILED",
    )
    if (stage / marker_name).read_bytes() != expected_marker:
        raise ReportPublicationError("failure marker changed before promotion")
    local_complete = "COMPLETE" in actual_files
    if (marker_name == "PROMOTION_FAILED") is not local_complete:
        raise ReportPublicationError("failure marker and COMPLETE semantics differ")
    if local_complete:
        completion = _strict_json(stage / "COMPLETE")
        if (
            completion.get("schema_version") != REPORT_COMPLETION_V2
            or completion.get("report_id") != report_id
        ):
            raise ReportPublicationError("preserved COMPLETE identifies another report")
    reference_name = "promoted_bundle_reference.json"
    if marker_name == "POST_PROMOTION_FAILED":
        reference = _strict_json(stage / reference_name)
        validation = reference.get("validation")
        expected_final = failed_root.parent / report_id
        if (
            set(reference)
            != {
                "schema_version",
                "report_id",
                "report_directory",
                "bundle_snapshot",
                "validation",
            }
            or reference.get("schema_version")
            != "kvbench-phase3-promoted-report-reference-1.0.0"
            or reference.get("report_id") != report_id
            or reference.get("report_directory") != str(expected_final)
            or not _strict_failed_validation_payload(validation)
            or reference.get("bundle_snapshot")
            != _immutable_bundle_snapshot(expected_final)
        ):
            raise ReportPublicationError(
                "post-promotion failure reference changed before promotion"
            )
    elif reference_name in actual_files:
        raise ReportPublicationError(
            "pre-promotion failure contains a promoted-bundle reference"
        )
    if _QUARANTINE_REFERENCE_FILE in actual_files:
        if marker_name != "FAILED":
            raise ReportPublicationError(
                "quarantine reference requires a FAILED marker"
            )
        _validate_quarantine_reference(
            _strict_json(stage / _QUARANTINE_REFERENCE_FILE),
            report_root=failed_root.parent,
            report_id=report_id,
        )
    inventory = _strict_json(stage / "failed_inventory.json")
    if inventory != _failed_inventory(stage, report_id):
        raise ReportPublicationError("failed report inventory changed before promotion")
    ledger_path = stage / "failed_checksums.sha256"
    entries = _parse_ledger(ledger_path.read_bytes())
    expected_ledger_files = actual_files - {"failed_checksums.sha256"}
    if set(entries) != expected_ledger_files:
        raise ReportPublicationError("failed report ledger coverage changed")
    for relative, digest in entries.items():
        if _file_digest(stage / relative) != digest:
            raise ReportPublicationError("failed report checksum changed")


def _finalize_failure_evidence(
    stage: Path,
    failed_root: Path,
    report_id: str,
    marker_name: str,
    phase: str,
    error: BaseException,
) -> Path:
    if marker_name not in _FAILURE_MARKERS:
        raise ReportPublicationError("failure marker kind is invalid")
    _open_owned_stage_for_failure_append(stage)
    complete_written = marker_name != "FAILED"
    _write_exclusive(
        stage,
        marker_name,
        _failure_marker(
            report_id=report_id,
            phase=phase,
            error=error,
            complete_written=complete_written,
        ),
    )
    _write_exclusive(
        stage,
        "failed_inventory.json",
        _json_bytes(_failed_inventory(stage, report_id)),
    )
    failed_ledger = b"".join(
        f"{_file_digest(path)}  {path.relative_to(stage).as_posix()}\n".encode()
        for path in _payload_files(stage, {"failed_checksums.sha256"})
    )
    _write_exclusive(stage, "failed_checksums.sha256", failed_ledger)
    _freeze_tree(stage)
    _validate_failure_stage_for_promotion(
        stage,
        failed_root=failed_root,
        report_id=report_id,
        marker_name=marker_name,
        phase=phase,
        error=error,
    )
    for _ in range(_EXCLUSIVE_NONCE_ATTEMPTS):
        target = failed_root / f"{report_id}.{secrets.token_hex(6)}.failed"
        try:
            _rename_noreplace(stage, target)
        except ArtifactConflictError:
            continue
        _fsync_directory(failed_root)
        return target
    raise ReportPublicationError(
        "could not preserve failure under a unique evidence ID"
    )


def _preserve_quarantined_owned_stage(
    *,
    stage: Path,
    failed_root: Path,
    report_id: str,
    phase: str,
    original_error: BaseException,
    topology_error: BaseException,
) -> Path:
    quarantine_stage = _create_owned_stage(stage.parent, report_id)
    _fsync_directory(stage.parent)
    _write_exclusive(
        quarantine_stage,
        _QUARANTINE_REFERENCE_FILE,
        _json_bytes(
            _quarantine_reference_payload(
                stage=stage,
                report_id=report_id,
                phase=phase,
                original_error=original_error,
                topology_error=topology_error,
            )
        ),
    )
    return _finalize_failure_evidence(
        quarantine_stage,
        failed_root,
        report_id,
        "FAILED",
        f"quarantine_after_{phase}",
        topology_error,
    )


def _preserve_failure(
    stage: Path,
    failed_root: Path,
    report_id: str,
    phase: str,
    error: BaseException,
) -> Path:
    complete_written = (stage / "COMPLETE").is_file()
    try:
        return _finalize_failure_evidence(
            stage,
            failed_root,
            report_id,
            "PROMOTION_FAILED" if complete_written else "FAILED",
            phase,
            error,
        )
    except BaseException as topology_error:
        if _lstat_identity(stage).get("kind") == "absent":
            raise
        return _preserve_quarantined_owned_stage(
            stage=stage,
            failed_root=failed_root,
            report_id=report_id,
            phase=phase,
            original_error=error,
            topology_error=topology_error,
        )


def _preserve_post_promotion_failure(
    *,
    staging_root: Path,
    failed_root: Path,
    report_id: str,
    final: Path,
    phase: str,
    error: BaseException,
    bundle_snapshot: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> Path:
    failure_stage = _create_owned_stage(staging_root, report_id)
    _fsync_directory(staging_root)
    _write_exclusive(
        failure_stage,
        "promoted_bundle_reference.json",
        _json_bytes(
            {
                "schema_version": (
                    "kvbench-phase3-promoted-report-reference-1.0.0"
                ),
                "report_id": report_id,
                "report_directory": str(final),
                "bundle_snapshot": dict(bundle_snapshot),
                "validation": dict(validation),
            }
        ),
    )
    return _finalize_failure_evidence(
        failure_stage,
        failed_root,
        report_id,
        "POST_PROMOTION_FAILED",
        phase,
        error,
    )


def _select_report_id(
    repository: Path,
    report_id: str | None,
    timestamp: str,
) -> str:
    from kvbench.runtime.phase3_report import _REPORT_ID, _git_command

    selected_id = report_id
    if selected_id is None:
        head = _git_command(repository, ["rev-parse", "HEAD"], text=True)
        git_sha = head.stdout.strip() if isinstance(head.stdout, str) else ""
        if (
            head.returncode != 0
            or len(git_sha) != 40
            or any(character not in "0123456789abcdef" for character in git_sha)
        ):
            raise ReportPublicationError("cannot resolve report ID Git identity")
        selected_id = (
            f"phase3-g1-{timestamp}-{git_sha[:8]}-{secrets.token_hex(3)}"
        )
    if not _REPORT_ID.fullmatch(selected_id):
        raise ReportPublicationError("report ID is invalid")
    return selected_id


def publish_phase3_g1_report(
    fixed_campaign_id: str,
    growing_campaign_id: str,
    *,
    repository_root: str | Path,
    report_root: str | Path | None = None,
    report_id: str | None = None,
    event_hook: Callable[[str], None] | None = None,
    before_complete_hook: Callable[[Path], None] | None = None,
) -> dict[str, Any]:
    """Publish one report by staging, validating, completing, and no-replace rename."""

    from kvbench.runtime.phase3_report import (
        Phase3ReportError,
        build_phase3_g1_report,
    )

    repository = _resolve_unaliased_path(
        repository_root, strict=True, label="repository root"
    )
    root = (
        repository / "artifacts" / "phase3_reports"
        if report_root is None
        else _resolve_unaliased_path(
            report_root, strict=False, label="Phase 3 report root"
        )
    )
    root.mkdir(mode=0o755, parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise ReportPublicationError("Phase 3 report root is unsafe")
    controls = [root / name for name in _CONTROL_DIRECTORIES]
    for control in controls:
        _ensure_real_directory(control, mode=0o700)
    staging_root, reservation_root, failed_root = controls
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%S%fZ").lower()
    selected_id = _select_report_id(repository, report_id, timestamp)
    phase = "reservation"
    stage: Path | None = None
    promoted: Path | None = None
    bundle_snapshot: Mapping[str, Any] = {}
    final_validation: Mapping[str, Any] = {
        "schema_version": "kvbench-phase3-g1-validation-2.0.0",
        "valid": False,
        "report_sha256": "",
        "errors": ["post-promotion validation did not complete"],
    }
    try:
        _write_exclusive(
            reservation_root,
            f"{selected_id}.json",
            _json_bytes(
                {
                    "schema_version": "kvbench-phase3-report-reservation-1.0.0",
                    "report_id": selected_id,
                    "reserved_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            ),
        )
        (reservation_root / f"{selected_id}.json").chmod(0o444)
        _fsync_directory(reservation_root)
        stage = _create_owned_stage(staging_root, selected_id)
        _fsync_directory(staging_root)
        if (root / selected_id).exists():
            raise ReportPublicationError("final report ID already exists")
        phase = "source_index"
        source_index = capture_phase3_source_index(
            repository, fixed_campaign_id, growing_campaign_id
        )
        _write(stage, "source_runs.json", _json_bytes(source_index), event_hook)
        _write(
            stage,
            "source_campaigns.json",
            _json_bytes(
                {
                    "schema_version": SOURCE_CAMPAIGNS_V2,
                    "fixed_campaign_id": fixed_campaign_id,
                    "growing_campaign_id": growing_campaign_id,
                    "explicit_selection": True,
                }
            ),
            event_hook,
        )
        phase = "build"
        report, stability, derivation = build_phase3_g1_report(
            fixed_campaign_id,
            growing_campaign_id,
            repository_root=repository,
        )
        if stability:
            os.mkdir(stage / "stability", 0o700)
            _fsync_directory(stage)
        for relative, payload in sorted(stability.items()):
            _write(stage, relative, _json_bytes(payload), event_hook)
        _write(stage, "derivation.json", _json_bytes(derivation), event_hook)
        _write(stage, "report.json", _json_bytes(report), event_hook)
        Phase3G1AdmissionReport.from_dict(_strict_json(stage / "report.json"))
        phase = "source_revalidation"
        if capture_phase3_source_index(
            repository, fixed_campaign_id, growing_campaign_id
        ) != source_index:
            raise ReportPublicationError("source evidence mutated during staging")
        inventory = _inventory(stage, selected_id)
        _write(
            stage,
            "artifact_inventory.json",
            _json_bytes(inventory),
            event_hook,
        )
        ledger = _ledger(stage)
        _write(stage, "checksums.sha256", ledger, event_hook)
        if before_complete_hook is not None:
            before_complete_hook(stage)
        if capture_phase3_source_index(
            repository, fixed_campaign_id, growing_campaign_id
        ) != source_index:
            raise ReportPublicationError("source evidence mutated before completion")
        phase = "pre_complete_validation"
        validation = _validate_payloads(
            stage,
            repository=repository,
            require_complete=False,
            require_immutable=False,
        )
        if not validation["valid"]:
            raise ReportPublicationError("staged report failed validation")
        phase = "complete"
        completion = {
            "schema_version": REPORT_COMPLETION_V2,
            "report_id": selected_id,
            "status": report.status.value,
            "report_sha256": _file_digest(stage / "report.json"),
            "source_index_sha256": _file_digest(stage / "source_runs.json"),
            "artifact_inventory_sha256": _file_digest(
                stage / "artifact_inventory.json"
            ),
            "checksum_ledger_sha256": sha256_hex(ledger),
            "written_last": True,
        }
        _write(stage, "COMPLETE", _json_bytes(completion), event_hook)
        complete_validation = _validate_payloads(
            stage,
            repository=repository,
            require_complete=True,
            require_immutable=False,
        )
        if not complete_validation["valid"]:
            raise ReportPublicationError("completed stage failed validation")
        phase = "promotion_freeze"
        _freeze_tree(stage)
        bundle_snapshot = _immutable_bundle_snapshot(stage)
        final = root / selected_id
        phase = "promotion"
        _rename_noreplace(stage, final)
        promoted = final
        stage = None
        phase = "promotion_fsync"
        _fsync_directory(root)
        phase = "post_promotion_validation"
        final_validation = validate_phase3_g1_report_directory_v2(
            final,
            repository_root=repository,
        )
        if not final_validation["valid"]:
            raise ReportPublicationError("published report failed validation")
        return {
            "schema_version": WRITE_RESULT_V2,
            "ok": True,
            "report_id": selected_id,
            "status": report.status.value,
            "report_dir": final.relative_to(root.parent.parent).as_posix()
            if report_root is None
            else str(final),
            "report_sha256": final_validation["report_sha256"],
            "execution_attempted": False,
            "timing_collected": False,
            "performance_claim_eligible": False,
        }
    except BaseException as error:
        preservation_error: BaseException | None = None
        try:
            if stage is not None:
                _preserve_failure(stage, failed_root, selected_id, phase, error)
            elif promoted is not None:
                _preserve_post_promotion_failure(
                    staging_root=staging_root,
                    failed_root=failed_root,
                    report_id=selected_id,
                    final=promoted,
                    phase=phase,
                    error=error,
                    bundle_snapshot=bundle_snapshot,
                    validation=final_validation,
                )
        except BaseException as preservation_failure:
            preservation_error = preservation_failure
        if preservation_error is not None:
            preservation_exception = ReportPublicationError(
                "report publication failed and failure evidence preservation failed"
            )
            preservation_exception.add_note(
                f"original publication failure: {type(error).__name__}"
            )
            raise preservation_exception from preservation_error
        if isinstance(error, (Phase3ReportError, ReportPublicationError)):
            raise
        raise ReportPublicationError("report publication failed closed") from error
