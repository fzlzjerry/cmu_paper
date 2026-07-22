"""Append-only, COMPLETE-last publication for Phase 3 campaign reports."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
import os
from pathlib import Path, PurePosixPath
import secrets
import stat
from typing import Any

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


class ReportPublicationError(RuntimeError):
    """A report could not be published without weakening evidence integrity."""


def _json_bytes(value: Any) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _strict_json(path: Path) -> dict[str, Any]:
    from kvbench.runtime.phase3_report import _strict_json_object

    return _strict_json_object(path, canonical=True)


def _file_digest(path: Path) -> str:
    return sha256_hex(path.read_bytes())


def _tree_record(directory: Path, *, require_immutable: bool) -> dict[str, Any]:
    metadata = directory.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ReportPublicationError("source evidence directory is unsafe")
    if require_immutable and metadata.st_mode & _WRITE_BITS:
        raise ReportPublicationError("source evidence directory is writable")
    files: list[dict[str, Any]] = []
    for target in sorted(directory.rglob("*")):
        item = target.lstat()
        if stat.S_ISLNK(item.st_mode):
            raise ReportPublicationError("source evidence contains a symlink")
        if require_immutable and item.st_mode & _WRITE_BITS:
            raise ReportPublicationError("source evidence contains writable content")
        if target.is_dir():
            continue
        if not target.is_file() or item.st_nlink != 1:
            raise ReportPublicationError("source evidence contains unsafe content")
        files.append(
            {
                "path": target.relative_to(directory).as_posix(),
                "size_bytes": item.st_size,
                "sha256": _file_digest(target),
            }
        )
    if not files:
        raise ReportPublicationError("source evidence directory is empty")
    return {
        "file_count": len(files),
        "tree_sha256": sha256_hex(canonical_json_bytes(files)),
        "files": files,
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

    repository = Path(repository_root).resolve(strict=True)
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
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() not in excluded
    )


def _inventory(stage: Path, report_id: str) -> dict[str, Any]:
    exclusions = {"artifact_inventory.json", "checksums.sha256", "COMPLETE"}
    files = [
        {
            "path": path.relative_to(stage).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _file_digest(path),
        }
        for path in _payload_files(stage, exclusions)
    ]
    return {
        "schema_version": REPORT_INVENTORY_V2,
        "report_id": report_id,
        "files": files,
        "excluded_control_files": sorted(exclusions),
    }


def _logical_report_id(directory: Path) -> str:
    name = directory.name
    if name.endswith(".staging"):
        parts = name.rsplit(".", 2)
        if len(parts) != 3 or not parts[0]:
            raise ReportPublicationError("staging report name is malformed")
        return parts[0]
    return name


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
        root_metadata = directory.lstat()
        if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(
            root_metadata.st_mode
        ):
            raise ReportPublicationError("report directory is unsafe")
        for target in (directory, *sorted(directory.rglob("*"))):
            metadata = target.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ReportPublicationError("report bundle contains a symlink")
            if require_immutable and metadata.st_mode & _WRITE_BITS:
                raise ReportPublicationError("report bundle is writable")
            if target.is_file() and metadata.st_nlink != 1:
                raise ReportPublicationError("report file has multiple links")
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
        actual_files = {
            path.relative_to(directory).as_posix()
            for path in directory.rglob("*")
            if path.is_file()
        }
        if actual_files != expected_files:
            errors.append("report exact file set differs")
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
    directory = Path(path).resolve(strict=True)
    repository = (
        directory.parents[2]
        if repository_root is None
        else Path(repository_root).resolve(strict=True)
    )
    return _validate_payloads(
        directory,
        repository=repository,
        require_complete=True,
        require_immutable=True,
    )


def validate_failed_report_attempt(path: str | Path) -> dict[str, Any]:
    """Validate one preserved, immutable report-publication failure."""

    directory = Path(path).resolve(strict=True)
    errors: list[str] = []
    try:
        for target in (directory, *sorted(directory.rglob("*"))):
            metadata = target.lstat()
            if stat.S_ISLNK(metadata.st_mode) or metadata.st_mode & _WRITE_BITS:
                raise ReportPublicationError("failed report evidence is unsafe")
            if target.is_file() and metadata.st_nlink != 1:
                raise ReportPublicationError("failed report file has multiple links")
        markers = [name for name in ("FAILED", "PROMOTION_FAILED") if (directory / name).is_file()]
        if len(markers) != 1:
            raise ReportPublicationError("failed report marker is absent or ambiguous")
        marker = _strict_json(directory / markers[0])
        if (
            marker.get("schema_version")
            != "kvbench-phase3-report-failure-1.0.0"
            or not isinstance(marker.get("report_id"), str)
            or not isinstance(marker.get("failure_phase"), str)
            or not isinstance(marker.get("failure_type"), str)
            or marker.get("complete_written") is not (markers[0] == "PROMOTION_FAILED")
        ):
            raise ReportPublicationError("failed report marker is malformed")
        inventory = _strict_json(directory / "failed_inventory.json")
        if (
            inventory.get("schema_version") != REPORT_INVENTORY_V2
            or inventory.get("report_id") != marker.get("report_id")
            or not isinstance(inventory.get("files"), list)
        ):
            raise ReportPublicationError("failed report inventory is malformed")
        ledger_path = directory / "failed_checksums.sha256"
        entries = _parse_ledger(ledger_path.read_bytes())
        actual = {
            item.relative_to(directory).as_posix()
            for item in directory.rglob("*")
            if item.is_file() and item.name != "failed_checksums.sha256"
        }
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


def _preserve_failure(
    stage: Path,
    failed_root: Path,
    report_id: str,
    phase: str,
    error: BaseException,
) -> None:
    if not stage.exists():
        return
    marker = "PROMOTION_FAILED" if (stage / "COMPLETE").exists() else "FAILED"
    try:
        _write_exclusive(
            stage,
            marker,
            _json_bytes(
                {
                    "schema_version": "kvbench-phase3-report-failure-1.0.0",
                    "report_id": report_id,
                    "failure_phase": phase,
                    "failure_type": type(error).__name__,
                    "complete_written": (stage / "COMPLETE").exists(),
                }
            ),
        )
        failed_inventory = _inventory(stage, report_id)
        _write_exclusive(
            stage,
            "failed_inventory.json",
            _json_bytes(failed_inventory),
        )
        failed_ledger = b"".join(
            f"{_file_digest(path)}  {path.relative_to(stage).as_posix()}\n".encode()
            for path in _payload_files(
                stage,
                {"failed_checksums.sha256"},
            )
        )
        _write_exclusive(stage, "failed_checksums.sha256", failed_ledger)
    except BaseException:
        pass
    _freeze_tree(stage)
    target = failed_root / f"{report_id}.{secrets.token_hex(6)}.failed"
    _rename_noreplace(stage, target)
    _fsync_directory(failed_root)


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

    repository = Path(repository_root).resolve(strict=True)
    root = (
        repository / "artifacts" / "phase3_reports"
        if report_root is None
        else Path(report_root).resolve()
    )
    root.mkdir(mode=0o755, parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise ReportPublicationError("Phase 3 report root is unsafe")
    controls = [root / name for name in _CONTROL_DIRECTORIES]
    for control in controls:
        _ensure_real_directory(control, mode=0o700)
    staging_root, reservation_root, failed_root = controls
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%S%fZ").lower()
    phase = "build"
    stage: Path | None = None
    try:
        report, stability, derivation = build_phase3_g1_report(
            fixed_campaign_id,
            growing_campaign_id,
            repository_root=repository,
        )
        selected_id = report_id or (
            f"phase3-g1-{timestamp}-{report.git_sha[:8]}-{secrets.token_hex(3)}"
        )
        from kvbench.runtime.phase3_report import _REPORT_ID

        if not _REPORT_ID.fullmatch(selected_id):
            raise ReportPublicationError("report ID is invalid")
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
        stage = staging_root / f"{selected_id}.{secrets.token_hex(6)}.staging"
        os.mkdir(stage, 0o700)
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
        phase = "pre_complete_validation"
        validation = _validate_payloads(
            stage,
            repository=repository,
            require_complete=False,
            require_immutable=False,
        )
        if not validation["valid"]:
            raise ReportPublicationError("staged report failed validation")
        if before_complete_hook is not None:
            before_complete_hook(stage)
        if capture_phase3_source_index(
            repository, fixed_campaign_id, growing_campaign_id
        ) != source_index:
            raise ReportPublicationError("source evidence mutated before completion")
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
        phase = "promotion"
        _freeze_tree(stage)
        final = root / selected_id
        _rename_noreplace(stage, final)
        stage = None
        _fsync_directory(root)
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
        if stage is not None:
            try:
                _preserve_failure(stage, failed_root, report_id or "unassigned", phase, error)
            except BaseException:
                pass
        if isinstance(error, (Phase3ReportError, ReportPublicationError)):
            raise
        raise ReportPublicationError("report publication failed closed") from error
