"""Lifecycle and realistic 20-run tests for Phase 3 report publication."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from kvbench.config import REPOSITORY_ROOT
import kvbench.runtime.phase3_report_publication as report_publication
from kvbench.runtime.phase3_report import Phase3ReportError
from kvbench.runtime.phase3_report_publication import (
    ReportPublicationError,
    capture_phase3_source_index,
    publish_phase3_g1_report,
    validate_failed_report_attempt,
    validate_phase3_g1_report_directory_v2,
)
from tests.unit.test_phase3_report import ZERO_GIT_SHA, blocked_report


FIXED_CAMPAIGN_ID = "phase3-20260722t112917207390z-457123b1-36731e"
GROWING_CAMPAIGN_ID = "phase3-20260722t113532869819z-457123b1-694228"
PHASE3_EXECUTION_SHA = "457123b12220aa4a724968c1b4dd04340cf34a54"


def _make_writable(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted(root.rglob("*"), reverse=True):
        try:
            path.chmod(0o755 if path.is_dir() else 0o644)
        except OSError:
            pass
    try:
        root.chmod(0o755)
    except OSError:
        pass


def _make_immutable(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o444 if path.is_file() else 0o555)
    root.chmod(0o555)


def _rewrite_failed_controls(root: Path) -> None:
    report_id = report_publication._failed_report_id(root)
    (root / "failed_inventory.json").write_bytes(
        report_publication._json_bytes(
            report_publication._failed_inventory(root, report_id)
        )
    )
    ledger = b"".join(
        f"{report_publication._file_digest(path)}  {path.relative_to(root).as_posix()}\n".encode()
        for path in report_publication._payload_files(
            root,
            {"failed_checksums.sha256"},
        )
    )
    (root / "failed_checksums.sha256").write_bytes(ledger)
    _make_immutable(root)


def _mock_derivation() -> dict[str, object]:
    return {
        "schema_version": "kvbench-phase3-g1-derivation-1.0.0",
        "report_git_provenance": {
            "schema_version": "kvbench-phase3-report-git-provenance-1.0.0",
            "source_execution_git_sha": ZERO_GIT_SHA,
            "report_generator_git_sha": ZERO_GIT_SHA,
            "execution_to_generator_changed_paths": [],
            "source_execution_is_ancestor": True,
            "reporting_only_descendant": True,
        },
        "selected_run_ids": [],
    }


def _mock_source_index() -> dict[str, object]:
    return {
        "schema_version": "kvbench-phase3-report-source-index-2.0.0",
        "fixed_campaign_id": FIXED_CAMPAIGN_ID,
        "growing_campaign_id": GROWING_CAMPAIGN_ID,
        "run_count": 20,
        "runs": [{"run_id": f"fixture-run-{index}"} for index in range(20)],
        "campaigns": [],
        "all_checksums_valid": True,
        "all_sources_immutable": True,
    }


class ReportPublicationLifecycleTests(unittest.TestCase):
    def test_complete_is_last_and_final_is_absent_before_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            report_root = Path(raw_root) / "reports"
            write_order: list[str] = []

            def before_complete(_: Path) -> None:
                visible = [
                    child
                    for child in report_root.iterdir()
                    if not child.name.startswith(".kvbench-")
                ]
                self.assertEqual(visible, [])

            try:
                with mock.patch(
                    "kvbench.runtime.phase3_report.build_phase3_g1_report",
                    return_value=(blocked_report(), {}, _mock_derivation()),
                ) as report_builder, mock.patch(
                    "kvbench.runtime.phase3_report_publication.capture_phase3_source_index",
                    return_value=_mock_source_index(),
                ):
                    result = publish_phase3_g1_report(
                        FIXED_CAMPAIGN_ID,
                        GROWING_CAMPAIGN_ID,
                        repository_root=REPOSITORY_ROOT,
                        report_root=report_root,
                        event_hook=write_order.append,
                        before_complete_hook=before_complete,
                    )
                    final = Path(result["report_dir"])
                    validation = validate_phase3_g1_report_directory_v2(
                        final,
                        repository_root=REPOSITORY_ROOT,
                    )
                self.assertTrue(validation["valid"])
                self.assertTrue(
                    any(
                        call.kwargs.get("recorded_report_git_provenance")
                        == _mock_derivation()["report_git_provenance"]
                        for call in report_builder.call_args_list
                    )
                )
                self.assertEqual(write_order[-1], "COMPLETE")
                self.assertEqual(write_order.count("COMPLETE"), 1)
                completion = json.loads((final / "COMPLETE").read_text())
                self.assertTrue(completion["written_last"])
            finally:
                _make_writable(report_root)

    def test_source_change_before_complete_is_preserved_without_complete(self) -> None:
        first = _mock_source_index()
        changed = {**first, "all_sources_immutable": False}
        with tempfile.TemporaryDirectory() as raw_root:
            report_root = Path(raw_root) / "reports"
            try:
                with mock.patch(
                    "kvbench.runtime.phase3_report.build_phase3_g1_report",
                    return_value=(blocked_report(), {}, _mock_derivation()),
                ), mock.patch(
                    "kvbench.runtime.phase3_report_publication.capture_phase3_source_index",
                    side_effect=(first, first, first, changed),
                ):
                    with self.assertRaises(ReportPublicationError):
                        publish_phase3_g1_report(
                            FIXED_CAMPAIGN_ID,
                            GROWING_CAMPAIGN_ID,
                            repository_root=REPOSITORY_ROOT,
                            report_root=report_root,
                        )
                attempts = list(
                    (report_root / ".kvbench-report-failed").iterdir()
                )
                self.assertEqual(len(attempts), 1)
                self.assertTrue((attempts[0] / "FAILED").is_file())
                self.assertFalse((attempts[0] / "COMPLETE").exists())
                self.assertTrue(validate_failed_report_attempt(attempts[0])["valid"])
            finally:
                _make_writable(report_root)

    def test_hook_mutation_and_added_file_fail_before_complete(self) -> None:
        for mutation in ("payload", "added_file"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as raw_root:
                report_root = Path(raw_root) / "reports"
                report_id = (
                    "phase3-g1-20260723t010100000000z-457123b1-"
                    + ("a10001" if mutation == "payload" else "a10002")
                )

                def mutate(stage: Path) -> None:
                    if mutation == "payload":
                        report_path = stage / "report.json"
                        report_path.write_bytes(report_path.read_bytes() + b" ")
                    else:
                        (stage / "unexpected.json").write_bytes(b"{}\n")

                try:
                    with mock.patch(
                        "kvbench.runtime.phase3_report.build_phase3_g1_report",
                        return_value=(blocked_report(), {}, _mock_derivation()),
                    ), mock.patch(
                        "kvbench.runtime.phase3_report_publication.capture_phase3_source_index",
                        return_value=_mock_source_index(),
                    ):
                        with self.assertRaises(ReportPublicationError):
                            publish_phase3_g1_report(
                                FIXED_CAMPAIGN_ID,
                                GROWING_CAMPAIGN_ID,
                                repository_root=REPOSITORY_ROOT,
                                report_root=report_root,
                                report_id=report_id,
                                before_complete_hook=mutate,
                            )
                    attempts = list(
                        (report_root / ".kvbench-report-failed").iterdir()
                    )
                    self.assertEqual(len(attempts), 1)
                    attempt = attempts[0]
                    marker = json.loads((attempt / "FAILED").read_text())
                    self.assertEqual(
                        marker["failure_phase"],
                        "pre_complete_validation",
                    )
                    self.assertFalse((attempt / "COMPLETE").exists())
                    self.assertTrue(
                        validate_failed_report_attempt(attempt)["valid"]
                    )
                finally:
                    _make_writable(report_root)

    def test_pre_build_failure_preserves_owned_stage_and_source_index(self) -> None:
        report_id = "phase3-g1-20260723t010101000000z-457123b1-a00001"
        with tempfile.TemporaryDirectory() as raw_root:
            report_root = Path(raw_root) / "reports"
            write_order: list[str] = []
            try:
                with mock.patch(
                    "kvbench.runtime.phase3_report_publication.capture_phase3_source_index",
                    return_value=_mock_source_index(),
                ), mock.patch(
                    "kvbench.runtime.phase3_report.build_phase3_g1_report",
                    side_effect=Phase3ReportError("pre-build failure"),
                ):
                    with self.assertRaises(Phase3ReportError):
                        publish_phase3_g1_report(
                            FIXED_CAMPAIGN_ID,
                            GROWING_CAMPAIGN_ID,
                            repository_root=REPOSITORY_ROOT,
                            report_root=report_root,
                            report_id=report_id,
                            event_hook=write_order.append,
                        )
                attempts = list(
                    (report_root / ".kvbench-report-failed").iterdir()
                )
                self.assertEqual(len(attempts), 1)
                attempt = attempts[0]
                marker = json.loads((attempt / "FAILED").read_text())
                self.assertEqual(marker["report_id"], report_id)
                self.assertEqual(marker["failure_phase"], "build")
                self.assertTrue((attempt / "source_runs.json").is_file())
                self.assertTrue((attempt / "source_campaigns.json").is_file())
                self.assertFalse((attempt / "report.json").exists())
                self.assertEqual(
                    write_order,
                    ["source_runs.json", "source_campaigns.json"],
                )
                self.assertTrue(validate_failed_report_attempt(attempt)["valid"])
            finally:
                _make_writable(report_root)

    def test_auto_report_id_links_reservation_stage_and_failure(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            report_root = Path(raw_root) / "reports"
            try:
                with mock.patch(
                    "kvbench.runtime.phase3_report_publication.capture_phase3_source_index",
                    return_value=_mock_source_index(),
                ), mock.patch(
                    "kvbench.runtime.phase3_report.build_phase3_g1_report",
                    side_effect=Phase3ReportError("pre-build failure"),
                ):
                    with self.assertRaises(Phase3ReportError):
                        publish_phase3_g1_report(
                            FIXED_CAMPAIGN_ID,
                            GROWING_CAMPAIGN_ID,
                            repository_root=REPOSITORY_ROOT,
                            report_root=report_root,
                        )
                reservations = list(
                    (report_root / ".kvbench-report-reservations").iterdir()
                )
                attempts = list(
                    (report_root / ".kvbench-report-failed").iterdir()
                )
                self.assertEqual(len(reservations), 1)
                self.assertEqual(len(attempts), 1)
                reservation = json.loads(reservations[0].read_text())
                marker = json.loads((attempts[0] / "FAILED").read_text())
                selected_id = reservation["report_id"]
                self.assertEqual(reservations[0].name, f"{selected_id}.json")
                self.assertTrue(attempts[0].name.startswith(f"{selected_id}."))
                self.assertEqual(marker["report_id"], selected_id)
                self.assertTrue(validate_failed_report_attempt(attempts[0])["valid"])
            finally:
                _make_writable(report_root)

    def test_staging_collision_is_not_adopted_or_preserved(self) -> None:
        report_id = "phase3-g1-20260723t010102000000z-457123b1-a00002"
        with tempfile.TemporaryDirectory() as raw_root:
            report_root = Path(raw_root) / "reports"
            report_root.mkdir()
            for control in report_publication._CONTROL_DIRECTORIES:
                (report_root / control).mkdir(mode=0o700)
            collision = (
                report_root
                / ".kvbench-report-staging"
                / f"{report_id}.aaaaaaaaaaaa.staging"
            )
            collision.mkdir(mode=0o700)
            sentinel = collision / "foreign-sentinel"
            sentinel.write_bytes(b"foreign\n")
            try:
                with mock.patch(
                    "kvbench.runtime.phase3_report_publication.secrets.token_hex",
                    return_value="aaaaaaaaaaaa",
                ), mock.patch(
                    "kvbench.runtime.phase3_report.build_phase3_g1_report"
                ) as report_builder:
                    with self.assertRaises(ReportPublicationError):
                        publish_phase3_g1_report(
                            FIXED_CAMPAIGN_ID,
                            GROWING_CAMPAIGN_ID,
                            repository_root=REPOSITORY_ROOT,
                            report_root=report_root,
                            report_id=report_id,
                        )
                report_builder.assert_not_called()
                self.assertEqual(sentinel.read_bytes(), b"foreign\n")
                self.assertEqual(
                    list((report_root / ".kvbench-report-failed").iterdir()),
                    [],
                )
            finally:
                _make_writable(report_root)

    def test_staging_collision_retries_without_adopting_foreign_directory(self) -> None:
        report_id = "phase3-g1-20260723t010106000000z-457123b1-a00006"
        with tempfile.TemporaryDirectory() as raw_root:
            report_root = Path(raw_root) / "reports"
            report_root.mkdir()
            for control in report_publication._CONTROL_DIRECTORIES:
                (report_root / control).mkdir(mode=0o700)
            collision = (
                report_root
                / ".kvbench-report-staging"
                / f"{report_id}.aaaaaaaaaaaa.staging"
            )
            collision.mkdir(mode=0o700)
            sentinel = collision / "foreign-sentinel"
            sentinel.write_bytes(b"foreign\n")
            try:
                with mock.patch(
                    "kvbench.runtime.phase3_report_publication.secrets.token_hex",
                    side_effect=("aaaaaaaaaaaa", "bbbbbbbbbbbb"),
                ), mock.patch(
                    "kvbench.runtime.phase3_report.build_phase3_g1_report",
                    return_value=(blocked_report(), {}, _mock_derivation()),
                ), mock.patch(
                    "kvbench.runtime.phase3_report_publication.capture_phase3_source_index",
                    return_value=_mock_source_index(),
                ):
                    result = publish_phase3_g1_report(
                        FIXED_CAMPAIGN_ID,
                        GROWING_CAMPAIGN_ID,
                        repository_root=REPOSITORY_ROOT,
                        report_root=report_root,
                        report_id=report_id,
                    )
                self.assertEqual(sentinel.read_bytes(), b"foreign\n")
                with mock.patch(
                    "kvbench.runtime.phase3_report.build_phase3_g1_report",
                    return_value=(blocked_report(), {}, _mock_derivation()),
                ), mock.patch(
                    "kvbench.runtime.phase3_report_publication.capture_phase3_source_index",
                    return_value=_mock_source_index(),
                ):
                    self.assertTrue(
                        validate_phase3_g1_report_directory_v2(
                            Path(result["report_dir"]),
                            repository_root=REPOSITORY_ROOT,
                        )["valid"]
                    )
            finally:
                _make_writable(report_root)

    def test_failed_destination_collision_retries_without_adoption(self) -> None:
        report_id = "phase3-g1-20260723t010107000000z-457123b1-a00007"
        with tempfile.TemporaryDirectory() as raw_root:
            report_root = Path(raw_root) / "reports"
            report_root.mkdir()
            for control in report_publication._CONTROL_DIRECTORIES:
                (report_root / control).mkdir(mode=0o700)
            collision = (
                report_root
                / ".kvbench-report-failed"
                / f"{report_id}.bbbbbbbbbbbb.failed"
            )
            collision.mkdir(mode=0o700)
            sentinel = collision / "foreign-sentinel"
            sentinel.write_bytes(b"foreign\n")
            try:
                with mock.patch(
                    "kvbench.runtime.phase3_report_publication.secrets.token_hex",
                    side_effect=(
                        "aaaaaaaaaaaa",
                        "bbbbbbbbbbbb",
                        "cccccccccccc",
                    ),
                ), mock.patch(
                    "kvbench.runtime.phase3_report_publication.capture_phase3_source_index",
                    return_value=_mock_source_index(),
                ), mock.patch(
                    "kvbench.runtime.phase3_report.build_phase3_g1_report",
                    side_effect=Phase3ReportError("pre-build failure"),
                ):
                    with self.assertRaises(Phase3ReportError):
                        publish_phase3_g1_report(
                            FIXED_CAMPAIGN_ID,
                            GROWING_CAMPAIGN_ID,
                            repository_root=REPOSITORY_ROOT,
                            report_root=report_root,
                            report_id=report_id,
                        )
                self.assertEqual(sentinel.read_bytes(), b"foreign\n")
                attempts = [
                    item
                    for item in (
                        report_root / ".kvbench-report-failed"
                    ).iterdir()
                    if item != collision
                ]
                self.assertEqual(len(attempts), 1)
                self.assertEqual(
                    attempts[0].name,
                    f"{report_id}.cccccccccccc.failed",
                )
                self.assertTrue(
                    validate_failed_report_attempt(attempts[0])["valid"]
                )
            finally:
                _make_writable(report_root)

    def test_published_validator_rejects_aliases_and_unsafe_topology(self) -> None:
        report_id = "phase3-g1-20260723t010108000000z-457123b1-a00008"
        with tempfile.TemporaryDirectory() as raw_root:
            report_root = Path(raw_root) / "reports"
            try:
                with mock.patch(
                    "kvbench.runtime.phase3_report.build_phase3_g1_report",
                    return_value=(blocked_report(), {}, _mock_derivation()),
                ), mock.patch(
                    "kvbench.runtime.phase3_report_publication.capture_phase3_source_index",
                    return_value=_mock_source_index(),
                ):
                    result = publish_phase3_g1_report(
                        FIXED_CAMPAIGN_ID,
                        GROWING_CAMPAIGN_ID,
                        repository_root=REPOSITORY_ROOT,
                        report_root=report_root,
                        report_id=report_id,
                    )
                final = Path(result["report_dir"])
                alias = report_root / "report-alias"
                alias.symlink_to(final, target_is_directory=True)
                self.assertFalse(
                    validate_phase3_g1_report_directory_v2(
                        alias,
                        repository_root=REPOSITORY_ROOT,
                    )["valid"]
                )
                alias.unlink()

                for mutation in (
                    "empty_directory",
                    "fifo",
                    "hardlink",
                    "internal_symlink",
                ):
                    with self.subTest(mutation=mutation):
                        _make_writable(final)
                        target = final / f"unsafe-{mutation}"
                        if mutation == "empty_directory":
                            target.mkdir()
                        elif mutation == "fifo":
                            os.mkfifo(target)
                        elif mutation == "hardlink":
                            os.link(final / "report.json", target)
                        else:
                            target.symlink_to(final / "report.json")
                        _make_immutable(final)
                        self.assertFalse(
                            validate_phase3_g1_report_directory_v2(
                                final,
                                repository_root=REPOSITORY_ROOT,
                            )["valid"]
                        )
                        _make_writable(final)
                        if target.is_dir() and not target.is_symlink():
                            target.rmdir()
                        else:
                            target.unlink()
                        _make_immutable(final)
                        with mock.patch(
                            "kvbench.runtime.phase3_report.build_phase3_g1_report",
                            return_value=(blocked_report(), {}, _mock_derivation()),
                        ), mock.patch(
                            "kvbench.runtime.phase3_report_publication.capture_phase3_source_index",
                            return_value=_mock_source_index(),
                        ):
                            self.assertTrue(
                                validate_phase3_g1_report_directory_v2(
                                    final,
                                    repository_root=REPOSITORY_ROOT,
                                )["valid"]
                            )
            finally:
                _make_writable(report_root)

    def test_failed_validator_rejects_aliases_unsafe_topology_and_complete_mismatch(self) -> None:
        report_id = "phase3-g1-20260723t010109000000z-457123b1-a00009"
        with tempfile.TemporaryDirectory() as raw_root:
            report_root = Path(raw_root) / "reports"
            try:
                with mock.patch(
                    "kvbench.runtime.phase3_report_publication.capture_phase3_source_index",
                    return_value=_mock_source_index(),
                ), mock.patch(
                    "kvbench.runtime.phase3_report.build_phase3_g1_report",
                    side_effect=Phase3ReportError("pre-build failure"),
                ):
                    with self.assertRaises(Phase3ReportError):
                        publish_phase3_g1_report(
                            FIXED_CAMPAIGN_ID,
                            GROWING_CAMPAIGN_ID,
                            repository_root=REPOSITORY_ROOT,
                            report_root=report_root,
                            report_id=report_id,
                        )
                attempt = next(
                    (report_root / ".kvbench-report-failed").iterdir()
                )
                alias = report_root / "failed-alias"
                alias.symlink_to(attempt, target_is_directory=True)
                self.assertFalse(validate_failed_report_attempt(alias)["valid"])
                alias.unlink()

                for mutation in (
                    "empty_directory",
                    "fifo",
                    "hardlink",
                    "internal_symlink",
                ):
                    with self.subTest(mutation=mutation):
                        _make_writable(attempt)
                        target = attempt / f"unsafe-{mutation}"
                        if mutation == "empty_directory":
                            target.mkdir()
                        elif mutation == "fifo":
                            os.mkfifo(target)
                        elif mutation == "hardlink":
                            os.link(attempt / "FAILED", target)
                        else:
                            target.symlink_to(attempt / "FAILED")
                        _make_immutable(attempt)
                        self.assertFalse(
                            validate_failed_report_attempt(attempt)["valid"]
                        )
                        _make_writable(attempt)
                        if target.is_dir() and not target.is_symlink():
                            target.rmdir()
                        else:
                            target.unlink()
                        _make_immutable(attempt)
                        self.assertTrue(
                            validate_failed_report_attempt(attempt)["valid"]
                        )

                _make_writable(attempt)
                (attempt / "COMPLETE").write_bytes(
                    report_publication._json_bytes(
                        {
                            "schema_version": report_publication.REPORT_COMPLETION_V2,
                            "report_id": report_id,
                        }
                    )
                )
                _rewrite_failed_controls(attempt)
                self.assertFalse(
                    validate_failed_report_attempt(attempt)["valid"]
                )
                _make_writable(attempt)
                (attempt / "COMPLETE").unlink()
                _rewrite_failed_controls(attempt)
                self.assertTrue(
                    validate_failed_report_attempt(attempt)["valid"]
                )
            finally:
                _make_writable(report_root)

    def test_promotion_failure_after_complete_is_preserved(self) -> None:
        report_id = "phase3-g1-20260723t010103000000z-457123b1-a00003"
        with tempfile.TemporaryDirectory() as raw_root:
            report_root = Path(raw_root) / "reports"
            original_rename = report_publication._rename_noreplace

            def reject_final(source: Path, target: Path) -> None:
                if target == report_root / report_id:
                    raise OSError("injected promotion failure")
                original_rename(source, target)

            try:
                with mock.patch(
                    "kvbench.runtime.phase3_report.build_phase3_g1_report",
                    return_value=(blocked_report(), {}, _mock_derivation()),
                ), mock.patch(
                    "kvbench.runtime.phase3_report_publication.capture_phase3_source_index",
                    return_value=_mock_source_index(),
                ), mock.patch(
                    "kvbench.runtime.phase3_report_publication._rename_noreplace",
                    side_effect=reject_final,
                ):
                    with self.assertRaises(ReportPublicationError):
                        publish_phase3_g1_report(
                            FIXED_CAMPAIGN_ID,
                            GROWING_CAMPAIGN_ID,
                            repository_root=REPOSITORY_ROOT,
                            report_root=report_root,
                            report_id=report_id,
                        )
                self.assertFalse((report_root / report_id).exists())
                attempts = list(
                    (report_root / ".kvbench-report-failed").iterdir()
                )
                self.assertEqual(len(attempts), 1)
                attempt = attempts[0]
                marker = json.loads((attempt / "PROMOTION_FAILED").read_text())
                self.assertEqual(marker["report_id"], report_id)
                self.assertEqual(marker["failure_phase"], "promotion")
                self.assertTrue(marker["complete_written"])
                self.assertTrue((attempt / "COMPLETE").is_file())
                self.assertTrue(validate_failed_report_attempt(attempt)["valid"])
                _make_writable(attempt)
                (attempt / "COMPLETE").unlink()
                _rewrite_failed_controls(attempt)
                self.assertFalse(
                    validate_failed_report_attempt(attempt)["valid"]
                )
            finally:
                _make_writable(report_root)

    def test_post_promotion_validation_failure_records_separate_evidence(self) -> None:
        report_id = "phase3-g1-20260723t010104000000z-457123b1-a00004"
        with tempfile.TemporaryDirectory() as raw_root:
            report_root = Path(raw_root) / "reports"
            promoted_bytes: dict[str, bytes] = {}

            def reject_published(path: Path, **_: object) -> dict[str, object]:
                promoted_bytes.update(
                    {
                        item.relative_to(path).as_posix(): item.read_bytes()
                        for item in path.rglob("*")
                        if item.is_file()
                    }
                )
                return {
                    "schema_version": "kvbench-phase3-g1-validation-2.0.0",
                    "valid": False,
                    "report_sha256": "",
                    "errors": ["injected post-promotion failure"],
                }

            try:
                with mock.patch(
                    "kvbench.runtime.phase3_report.build_phase3_g1_report",
                    return_value=(blocked_report(), {}, _mock_derivation()),
                ), mock.patch(
                    "kvbench.runtime.phase3_report_publication.capture_phase3_source_index",
                    return_value=_mock_source_index(),
                ), mock.patch(
                    "kvbench.runtime.phase3_report_publication.validate_phase3_g1_report_directory_v2",
                    side_effect=reject_published,
                ):
                    with self.assertRaises(ReportPublicationError):
                        publish_phase3_g1_report(
                            FIXED_CAMPAIGN_ID,
                            GROWING_CAMPAIGN_ID,
                            repository_root=REPOSITORY_ROOT,
                            report_root=report_root,
                            report_id=report_id,
                        )
                final = report_root / report_id
                current_bytes = {
                    item.relative_to(final).as_posix(): item.read_bytes()
                    for item in final.rglob("*")
                    if item.is_file()
                }
                self.assertEqual(current_bytes, promoted_bytes)
                attempts = list(
                    (report_root / ".kvbench-report-failed").iterdir()
                )
                self.assertEqual(len(attempts), 1)
                attempt = attempts[0]
                marker = json.loads(
                    (attempt / "POST_PROMOTION_FAILED").read_text()
                )
                reference = json.loads(
                    (attempt / "promoted_bundle_reference.json").read_text()
                )
                self.assertEqual(marker["report_id"], report_id)
                self.assertEqual(
                    marker["failure_phase"], "post_promotion_validation"
                )
                self.assertEqual(reference["report_directory"], str(final))
                self.assertFalse(reference["validation"]["valid"])
                self.assertTrue(validate_failed_report_attempt(attempt)["valid"])

                reference_path = attempt / "promoted_bundle_reference.json"
                original_reference = reference_path.read_bytes()
                for mutation in (
                    "report_directory",
                    "report_id",
                    "bundle_snapshot",
                    "validation_payload",
                ):
                    with self.subTest(mutation=mutation):
                        _make_writable(attempt)
                        tampered = json.loads(original_reference)
                        if mutation == "report_directory":
                            tampered["report_directory"] = str(
                                report_root / "different-report"
                            )
                        elif mutation == "report_id":
                            tampered["report_id"] = (
                                "phase3-g1-20260723t010104000000z-"
                                "457123b1-dead00"
                            )
                        elif mutation == "bundle_snapshot":
                            tampered["bundle_snapshot"]["tree_sha256"] = (
                                "0" * 64
                            )
                        else:
                            tampered["validation"]["unexpected"] = True
                        reference_path.write_bytes(
                            report_publication._json_bytes(tampered)
                        )
                        _rewrite_failed_controls(attempt)
                        self.assertFalse(
                            validate_failed_report_attempt(attempt)["valid"]
                        )
                        _make_writable(attempt)
                        reference_path.write_bytes(original_reference)
                        _rewrite_failed_controls(attempt)
                        self.assertTrue(
                            validate_failed_report_attempt(attempt)["valid"]
                        )

                report_path = final / "report.json"
                original_report = report_path.read_bytes()
                report_path.chmod(0o644)
                report_path.write_bytes(original_report + b" ")
                report_path.chmod(0o444)
                self.assertFalse(
                    validate_failed_report_attempt(attempt)["valid"]
                )
                report_path.chmod(0o644)
                report_path.write_bytes(original_report)
                report_path.chmod(0o444)
                self.assertTrue(
                    validate_failed_report_attempt(attempt)["valid"]
                )
            finally:
                _make_writable(report_root)

    def test_failed_inventory_tamper_is_rejected_after_ledger_repair(self) -> None:
        report_id = "phase3-g1-20260723t010105000000z-457123b1-a00005"
        with tempfile.TemporaryDirectory() as raw_root:
            report_root = Path(raw_root) / "reports"
            try:
                with mock.patch(
                    "kvbench.runtime.phase3_report_publication.capture_phase3_source_index",
                    return_value=_mock_source_index(),
                ), mock.patch(
                    "kvbench.runtime.phase3_report.build_phase3_g1_report",
                    side_effect=Phase3ReportError("pre-build failure"),
                ):
                    with self.assertRaises(Phase3ReportError):
                        publish_phase3_g1_report(
                            FIXED_CAMPAIGN_ID,
                            GROWING_CAMPAIGN_ID,
                            repository_root=REPOSITORY_ROOT,
                            report_root=report_root,
                            report_id=report_id,
                        )
                attempt = next(
                    (report_root / ".kvbench-report-failed").iterdir()
                )
                _make_writable(attempt)
                inventory_path = attempt / "failed_inventory.json"
                inventory = json.loads(inventory_path.read_text())
                inventory["files"] = []
                inventory_path.write_bytes(report_publication._json_bytes(inventory))
                ledger = b"".join(
                    f"{report_publication._file_digest(path)}  {path.relative_to(attempt).as_posix()}\n".encode()
                    for path in report_publication._payload_files(
                        attempt,
                        {"failed_checksums.sha256"},
                    )
                )
                (attempt / "failed_checksums.sha256").write_bytes(ledger)
                _make_immutable(attempt)
                validation = validate_failed_report_attempt(attempt)
                self.assertFalse(validation["valid"])
                self.assertIn("failed report inventory differs", validation["errors"])
                self.assertFalse(
                    any("checksum mismatch" in error for error in validation["errors"])
                )
            finally:
                _make_writable(report_root)


    def test_first_stability_write_failure_preserves_empty_directory(self) -> None:
        report_id = "phase3-g1-20260723t010110000000z-457123b1-a00010"
        with tempfile.TemporaryDirectory() as raw_root:
            report_root = Path(raw_root) / "reports"
            original_write = report_publication._write

            def fail_first_stability_write(
                stage: Path,
                relative: str,
                data: bytes,
                event_hook: object,
            ) -> None:
                if relative == "stability/first.json":
                    raise OSError("injected first stability write failure")
                original_write(stage, relative, data, event_hook)  # type: ignore[arg-type]

            try:
                with mock.patch(
                    "kvbench.runtime.phase3_report.build_phase3_g1_report",
                    return_value=(
                        blocked_report(),
                        {"stability/first.json": {"fixture": True}},
                        _mock_derivation(),
                    ),
                ), mock.patch(
                    "kvbench.runtime.phase3_report_publication.capture_phase3_source_index",
                    return_value=_mock_source_index(),
                ), mock.patch.object(
                    report_publication,
                    "_write",
                    side_effect=fail_first_stability_write,
                ):
                    with self.assertRaises(ReportPublicationError):
                        publish_phase3_g1_report(
                            FIXED_CAMPAIGN_ID,
                            GROWING_CAMPAIGN_ID,
                            repository_root=REPOSITORY_ROOT,
                            report_root=report_root,
                            report_id=report_id,
                        )
                attempt = next(
                    (report_root / ".kvbench-report-failed").iterdir()
                )
                self.assertTrue((attempt / "stability").is_dir())
                self.assertEqual(list((attempt / "stability").iterdir()), [])
                inventory = json.loads(
                    (attempt / "failed_inventory.json").read_text()
                )
                self.assertIn("stability", inventory["directories"])
                self.assertTrue(validate_failed_report_attempt(attempt)["valid"])
            finally:
                _make_writable(report_root)

    def test_unsafe_failed_stage_gets_immutable_quarantine_reference(self) -> None:
        for mutation in ("fifo", "hardlink"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as raw_root:
                report_root = Path(raw_root) / "reports"
                report_id = (
                    "phase3-g1-20260723t010111000000z-457123b1-"
                    + ("a00011" if mutation == "fifo" else "a00012")
                )

                def inject_unsafe_member(stage: Path) -> None:
                    target = stage / f"injected-{mutation}"
                    if mutation == "fifo":
                        os.mkfifo(target)
                    else:
                        os.link(stage / "report.json", target)

                try:
                    with mock.patch(
                        "kvbench.runtime.phase3_report.build_phase3_g1_report",
                        return_value=(blocked_report(), {}, _mock_derivation()),
                    ), mock.patch(
                        "kvbench.runtime.phase3_report_publication.capture_phase3_source_index",
                        return_value=_mock_source_index(),
                    ):
                        with self.assertRaises(ReportPublicationError):
                            publish_phase3_g1_report(
                                FIXED_CAMPAIGN_ID,
                                GROWING_CAMPAIGN_ID,
                                repository_root=REPOSITORY_ROOT,
                                report_root=report_root,
                                report_id=report_id,
                                before_complete_hook=inject_unsafe_member,
                            )
                    failed_root = report_root / ".kvbench-report-failed"
                    attempts = list(failed_root.iterdir())
                    self.assertEqual(len(attempts), 1)
                    stages = list(
                        (report_root / ".kvbench-report-staging").iterdir()
                    )
                    self.assertEqual(len(stages), 1)
                    self.assertTrue((stages[0] / f"injected-{mutation}").exists())
                    self.assertFalse((stages[0] / "COMPLETE").exists())
                    reference_path = (
                        attempts[0]
                        / report_publication._QUARANTINE_REFERENCE_FILE
                    )
                    reference = json.loads(reference_path.read_text())
                    self.assertEqual(reference["stage_directory"], str(stages[0]))
                    self.assertEqual(reference["stage_name"], stages[0].name)
                    self.assertEqual(
                        reference["quarantine_reason"],
                        "unsafe_owned_stage_not_admitted",
                    )
                    self.assertTrue(
                        validate_failed_report_attempt(attempts[0])["valid"]
                    )
                    reservations = list(
                        (report_root / ".kvbench-report-reservations").iterdir()
                    )
                    self.assertEqual(len(reservations), 1)

                    _make_writable(attempts[0])
                    reference["stage_root_lstat"]["inode"] += 1
                    reference_path.write_bytes(
                        report_publication._json_bytes(reference)
                    )
                    _rewrite_failed_controls(attempts[0])
                    self.assertFalse(
                        validate_failed_report_attempt(attempts[0])["valid"]
                    )
                finally:
                    _make_writable(report_root)

    def test_post_mkdir_staging_fsync_failure_preserves_owned_stage(self) -> None:
        report_id = "phase3-g1-20260723t010112000000z-457123b1-a00013"
        with tempfile.TemporaryDirectory() as raw_root:
            report_root = Path(raw_root) / "reports"
            original_fsync = report_publication._fsync_directory
            injected = False

            def fail_first_staging_fsync(path: Path) -> None:
                nonlocal injected
                if path.name == ".kvbench-report-staging" and not injected:
                    injected = True
                    raise OSError("injected post-mkdir staging fsync failure")
                original_fsync(path)

            try:
                with mock.patch.object(
                    report_publication,
                    "_fsync_directory",
                    side_effect=fail_first_staging_fsync,
                ):
                    with self.assertRaises(ReportPublicationError):
                        publish_phase3_g1_report(
                            FIXED_CAMPAIGN_ID,
                            GROWING_CAMPAIGN_ID,
                            repository_root=REPOSITORY_ROOT,
                            report_root=report_root,
                            report_id=report_id,
                        )
                self.assertTrue(injected)
                self.assertEqual(
                    list((report_root / ".kvbench-report-staging").iterdir()),
                    [],
                )
                attempts = list(
                    (report_root / ".kvbench-report-failed").iterdir()
                )
                self.assertEqual(len(attempts), 1)
                marker = json.loads((attempts[0] / "FAILED").read_text())
                self.assertEqual(marker["failure_phase"], "reservation")
                self.assertEqual(marker["failure_type"], "OSError")
                self.assertTrue(
                    validate_failed_report_attempt(attempts[0])["valid"]
                )
            finally:
                _make_writable(report_root)

    def test_report_root_and_validator_ancestor_aliases_are_rejected(self) -> None:
        success_id = "phase3-g1-20260723t010113000000z-457123b1-a00014"
        failure_id = "phase3-g1-20260723t010114000000z-457123b1-a00015"
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            report_root = root / "reports"
            alias_root = root / "reports-alias"
            try:
                with mock.patch(
                    "kvbench.runtime.phase3_report.build_phase3_g1_report",
                    return_value=(blocked_report(), {}, _mock_derivation()),
                ), mock.patch(
                    "kvbench.runtime.phase3_report_publication.capture_phase3_source_index",
                    return_value=_mock_source_index(),
                ):
                    result = publish_phase3_g1_report(
                        FIXED_CAMPAIGN_ID,
                        GROWING_CAMPAIGN_ID,
                        repository_root=REPOSITORY_ROOT,
                        report_root=report_root,
                        report_id=success_id,
                    )
                with mock.patch(
                    "kvbench.runtime.phase3_report_publication.capture_phase3_source_index",
                    return_value=_mock_source_index(),
                ), mock.patch(
                    "kvbench.runtime.phase3_report.build_phase3_g1_report",
                    side_effect=Phase3ReportError("fixture failure"),
                ):
                    with self.assertRaises(Phase3ReportError):
                        publish_phase3_g1_report(
                            FIXED_CAMPAIGN_ID,
                            GROWING_CAMPAIGN_ID,
                            repository_root=REPOSITORY_ROOT,
                            report_root=report_root,
                            report_id=failure_id,
                        )
                attempt = next(
                    (report_root / ".kvbench-report-failed").iterdir()
                )
                alias_root.symlink_to(report_root, target_is_directory=True)
                self.assertFalse(
                    validate_phase3_g1_report_directory_v2(
                        alias_root / success_id,
                        repository_root=REPOSITORY_ROOT,
                    )["valid"]
                )
                self.assertFalse(
                    validate_failed_report_attempt(
                        alias_root / ".kvbench-report-failed" / attempt.name
                    )["valid"]
                )
                with mock.patch(
                    "kvbench.runtime.phase3_report.build_phase3_g1_report"
                ) as report_builder:
                    with self.assertRaises(ReportPublicationError):
                        publish_phase3_g1_report(
                            FIXED_CAMPAIGN_ID,
                            GROWING_CAMPAIGN_ID,
                            repository_root=REPOSITORY_ROOT,
                            report_root=alias_root,
                            report_id=(
                                "phase3-g1-20260723t010115000000z-"
                                "457123b1-a00016"
                            ),
                        )
                    report_builder.assert_not_called()
                self.assertEqual(Path(result["report_dir"]), report_root / success_id)
            finally:
                if alias_root.is_symlink():
                    alias_root.unlink()
                _make_writable(report_root)

    def test_source_tree_digest_includes_empty_directory_topology(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            source = Path(raw_root) / "source"
            source.mkdir()
            (source / "payload.json").write_bytes(b"{}\n")
            _make_immutable(source)
            try:
                before = report_publication._tree_record(
                    source,
                    require_immutable=True,
                )
                _make_writable(source)
                (source / "new-empty-directory").mkdir()
                _make_immutable(source)
                after = report_publication._tree_record(
                    source,
                    require_immutable=True,
                )
                self.assertNotEqual(before["tree_sha256"], after["tree_sha256"])
                self.assertEqual(before["directories"], [])
                self.assertEqual(after["directories"], ["new-empty-directory"])
                self.assertEqual(after["directory_count"], 1)
            finally:
                _make_writable(source)
class RealTwentyRunPublicationTests(unittest.TestCase):
    def test_real_19_failed_one_aborted_fixture_and_tamper_controls(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            clone = root / "source-repository"
            report_root = root / "published-reports"
            try:
                subprocess.run(
                    (
                        "git",
                        "clone",
                        "--quiet",
                        "--no-hardlinks",
                        str(REPOSITORY_ROOT),
                        str(clone),
                    ),
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                subprocess.run(
                    ("git", "checkout", "--quiet", PHASE3_EXECUTION_SHA),
                    cwd=clone,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                for name in ("phase3", "phase3_campaigns"):
                    shutil.copytree(
                        REPOSITORY_ROOT / "artifacts" / name,
                        clone / "artifacts" / name,
                        dirs_exist_ok=True,
                        copy_function=shutil.copy2,
                    )

                before = capture_phase3_source_index(
                    clone,
                    FIXED_CAMPAIGN_ID,
                    GROWING_CAMPAIGN_ID,
                )
                statuses = [item["status"] for item in before["runs"]]
                self.assertEqual(statuses.count("aborted"), 1)
                self.assertEqual(
                    statuses.count("gqa_materialization_detected"),
                    19,
                )
                self.assertEqual(len(statuses), 20)
                self.assertEqual(
                    set(statuses),
                    {"gqa_materialization_detected", "aborted"},
                )

                write_order: list[str] = []
                result = publish_phase3_g1_report(
                    FIXED_CAMPAIGN_ID,
                    GROWING_CAMPAIGN_ID,
                    repository_root=clone,
                    report_root=report_root,
                    event_hook=write_order.append,
                )
                final = Path(result["report_dir"])
                after = capture_phase3_source_index(
                    clone,
                    FIXED_CAMPAIGN_ID,
                    GROWING_CAMPAIGN_ID,
                )
                self.assertEqual(before, after)
                self.assertEqual(write_order[-1], "COMPLETE")
                self.assertTrue(
                    validate_phase3_g1_report_directory_v2(
                        final,
                        repository_root=clone,
                    )["valid"]
                )

                original_complete = (final / "COMPLETE").read_bytes()
                with self.assertRaises((ReportPublicationError, Phase3ReportError)):
                    publish_phase3_g1_report(
                        FIXED_CAMPAIGN_ID,
                        GROWING_CAMPAIGN_ID,
                        repository_root=clone,
                        report_root=report_root,
                        report_id=result["report_id"],
                    )
                self.assertEqual((final / "COMPLETE").read_bytes(), original_complete)

                report_path = final / "report.json"
                report_path.chmod(0o644)
                report_path.write_bytes(report_path.read_bytes() + b" ")
                report_path.chmod(0o444)
                self.assertFalse(
                    validate_phase3_g1_report_directory_v2(
                        final,
                        repository_root=clone,
                    )["valid"]
                )

                tamper_target = (
                    clone
                    / "artifacts"
                    / "phase3"
                    / f"{FIXED_CAMPAIGN_ID}-fixed_l-b1-l128-eager-r1"
                    / "logs"
                    / "worker.stderr.txt"
                )

                def mutate_source(_: Path) -> None:
                    tamper_target.chmod(0o644)
                    tamper_target.write_bytes(tamper_target.read_bytes() + b"tamper")

                with self.assertRaises((ReportPublicationError, Phase3ReportError)):
                    publish_phase3_g1_report(
                        FIXED_CAMPAIGN_ID,
                        GROWING_CAMPAIGN_ID,
                        repository_root=clone,
                        report_root=report_root,
                        report_id="phase3-g1-20260722t235959000000z-457123b1-abcdef",
                        before_complete_hook=mutate_source,
                    )
                failed = list(
                    (report_root / ".kvbench-report-failed").iterdir()
                )
                self.assertEqual(len(failed), 1)
                self.assertTrue((failed[0] / "FAILED").is_file())
                self.assertFalse((failed[0] / "COMPLETE").exists())
                self.assertTrue(validate_failed_report_attempt(failed[0])["valid"])
            finally:
                _make_writable(report_root)
                _make_writable(clone / "artifacts")


if __name__ == "__main__":
    unittest.main()
