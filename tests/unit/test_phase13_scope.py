"""Focused scope tests for pre-existing Phase 3 backup custody roots."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import validate_phase2


EXPECTED_PHASE3_BACKUP_ROOTS = frozenset(
    {
        "cloud_backup_catalog",
        "historical_failure_custody",
        "phase3_r2_custody",
        "phase3_r2_outer",
        "residual_evidence_custody",
    }
)

EXPECTED_PHASE13_PATHS = frozenset(
    {
        "Makefile",
        "docs/evidence/phase13/pilot_qc.json",
        "docs/evidence/phase13/r2-publication.json",
        "docs/phase_reports/phase13-pilot-scan.md",
        "docs/plans/phase13-pilot-scan.md",
        "docs/plans/phase13-pilot-execution-order.json",
        "docs/risk_register.md",
        "docs/status.md",
        "docs/tasks.md",
        "scripts/phase13_pilot.py",
        "scripts/validate_phase2.py",
        "tests/unit/test_phase13_pilot.py",
        "tests/unit/test_phase13_scope.py",
    }
)


class Phase13ScopeTests(unittest.TestCase):
    def test_phase13_segment_and_allowlist_are_exact(self) -> None:
        self.assertEqual(
            validate_phase2.PHASE13_ENTRY_COMMIT,
            "7379e808ff687b10bf18c56364ae1c545cd00fe4",
        )
        self.assertEqual(
            validate_phase2.PHASE13_ALLOWED_PATHS,
            EXPECTED_PHASE13_PATHS,
        )
        self.assertEqual(
            validate_phase2.PHASE13_APPROVED_ARTIFACT_ROOT_NAMES,
            frozenset({"phase13"}),
        )

    def test_phase13_paths_use_exact_canonical_matching(self) -> None:
        for relative in EXPECTED_PHASE13_PATHS:
            self.assertTrue(validate_phase2.phase13_path_is_allowed(relative))
        for relative in (
            "docs/plans/phase13-pilot-scan.md.backup",
            "artifacts/phase13",
            "../docs/plans/phase13-pilot-scan.md",
            "/docs/plans/phase13-pilot-scan.md",
            "docs\\plans\\phase13-pilot-scan.md",
        ):
            self.assertFalse(
                validate_phase2.phase13_path_is_allowed(relative)
            )

    def test_phase3_backup_root_allowlist_is_exact(self) -> None:
        self.assertEqual(
            validate_phase2.PHASE3_BACKUP_ARTIFACT_ROOT_NAMES,
            EXPECTED_PHASE3_BACKUP_ROOTS,
        )
        self.assertFalse(
            any("*" in name for name in EXPECTED_PHASE3_BACKUP_ROOTS)
        )

    def test_exact_phase3_backup_roots_are_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            for name in EXPECTED_PHASE3_BACKUP_ROOTS:
                (artifacts / name).mkdir()
            with mock.patch.object(validate_phase2, "ROOT", root):
                errors = validate_phase2.validate_phase3_artifact_root()
        self.assertEqual(errors, [])

    def test_near_miss_phase3_backup_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            (artifacts / "phase3_r2_outer_copy").mkdir()
            with mock.patch.object(validate_phase2, "ROOT", root):
                errors = validate_phase2.validate_phase3_artifact_root()
        self.assertIn(
            "unapproved artifact roots: ['phase3_r2_outer_copy']",
            errors,
        )

    def test_symlinked_phase3_backup_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = root / "artifacts"
            target = root / "backup-target"
            artifacts.mkdir()
            target.mkdir()
            (artifacts / "phase3_r2_outer").symlink_to(
                target,
                target_is_directory=True,
            )
            with mock.patch.object(validate_phase2, "ROOT", root):
                errors = validate_phase2.validate_phase3_artifact_root()
        self.assertIn(
            "unsafe Phase 3 backup artifact root: phase3_r2_outer",
            errors,
        )


if __name__ == "__main__":
    unittest.main()
