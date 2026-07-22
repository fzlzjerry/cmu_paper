"""Lifecycle and realistic 20-run tests for Phase 3 report publication."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from kvbench.config import REPOSITORY_ROOT
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
                self.assertEqual(len(statuses) - statuses.count("aborted"), 19)

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
