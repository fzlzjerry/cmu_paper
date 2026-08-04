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
        "scripts/phase12_unified_admission.py",
        "scripts/phase13_pilot.py",
        "scripts/validate_phase2.py",
        "tests/unit/test_phase12_unified_admission.py",
        "tests/unit/test_phase13_pilot.py",
        "tests/unit/test_phase13_scope.py",
    }
)
EXPECTED_PHASE13R_PATHS = frozenset(
    {
        "Makefile",
        "docs/evidence/phase13r/pilot_qc.json",
        "docs/evidence/phase13r/r2-publication.json",
        "docs/phase_reports/phase13r-pilot-scan.md",
        "docs/plans/phase13-pilot-scan.md",
        "docs/risk_register.md",
        "docs/status.md",
        "docs/tasks.md",
        "scripts/phase13_pilot.py",
        "scripts/validate_phase2.py",
        "tests/unit/test_phase13_pilot.py",
        "tests/unit/test_phase13_scope.py",
    }
)
EXPECTED_PHASE13F_PATHS = frozenset(
    {
        "Makefile",
        "docs/blockers.md",
        "docs/decisions/0031-phase13-end-to-end-prefix-feasibility.md",
        "docs/evidence/phase13f/feasibility-summary.json",
        "docs/evidence/phase13f/r2-publication.json",
        "docs/phase_reports/phase13f-prefix-feasibility.md",
        "docs/plans/phase13f-prefix-feasibility.md",
        "docs/risk_register.md",
        "docs/status.md",
        "docs/tasks.md",
        "scripts/phase13_pilot.py",
        "scripts/phase13f_feasibility.py",
        "scripts/validate_phase2.py",
        "tests/unit/test_phase13_pilot.py",
        "tests/unit/test_phase13_scope.py",
        "tests/unit/test_phase13f_feasibility.py",
    }
)
EXPECTED_PHASE13R2_PATHS = frozenset(
    {
        "docs/blockers.md",
        "docs/evidence/phase13r2/pilot_qc.json",
        "docs/evidence/phase13r2/r2-publication.json",
        "docs/phase_reports/phase13r2-pilot-scan.md",
        "docs/plans/phase13-pilot-scan.md",
        "docs/risk_register.md",
        "docs/status.md",
        "docs/tasks.md",
        "scripts/validate_phase2.py",
        "tests/unit/test_phase13_scope.py",
    }
)
EXPECTED_PHASE13T_PATHS = frozenset(
    {
        "Makefile",
        "docs/blockers.md",
        "docs/decisions/0032-phase13-stage-aware-supervision.md",
        "docs/evidence/phase13t/r2-publication.json",
        "docs/evidence/phase13t/timeout-validation.json",
        "docs/evidence/phase13tr/pilot_qc.json",
        "docs/evidence/phase13tr/r2-publication.json",
        "docs/phase_reports/phase13t-supervisor-timeout.md",
        "docs/phase_reports/phase13tr-pilot-scan.md",
        "docs/plans/phase13t-supervisor-timeout.md",
        "docs/risk_register.md",
        "docs/status.md",
        "docs/tasks.md",
        "scripts/phase13_pilot.py",
        "scripts/phase13t_timeout.py",
        "scripts/validate_phase2.py",
        "src/kvbench/runtime/process_supervision.py",
        "tests/unit/test_phase13_scope.py",
        "tests/unit/test_phase13t_timeout.py",
    }
)
EXPECTED_PHASE13PR_PATHS = frozenset(
    {
        "Makefile",
        "docs/blockers.md",
        "docs/decisions/0033-phase13-untimed-prefix-state-reuse.md",
        "docs/evidence/phase13/pilot_qc.json",
        "docs/evidence/phase13/r2-publication.json",
        "docs/phase_reports/phase13-pilot-scan.md",
        "docs/plans/phase13-pilot-scan.md",
        "docs/risk_register.md",
        "docs/status.md",
        "docs/tasks.md",
        "scripts/phase13_pilot.py",
        "scripts/phase13_prefix_state.py",
        "scripts/validate_phase2.py",
        "tests/unit/test_phase13_scope.py",
        "tests/unit/test_phase13pr_prefix_state.py",
    }
)
EXPECTED_PHASE13PB_PATHS = frozenset(
    {
        "docs/decisions/0034-phase13-batch-exact-prefix-state.md",
        "docs/evidence/phase13pb/prefix-equivalence.json",
        "docs/evidence/phase13pb/r2-publication.json",
        "docs/phase_reports/phase13pb-batch-exact-prefix-state.md",
        "scripts/phase13_pilot.py",
        "scripts/phase13_prefix_state.py",
        "scripts/phase13pb_prefix_remediation.py",
        "scripts/validate_phase2.py",
        "tests/unit/test_phase13_scope.py",
        "tests/unit/test_phase13pr_prefix_state.py",
    }
)
EXPECTED_PHASE13PC_PATHS = frozenset(
    {
        "docs/blockers.md",
        "docs/decisions/0035-phase13-equivalence-cuda-context-isolation.md",
        "docs/evidence/phase13pc/isolation-validation.json",
        "docs/evidence/phase13pc/r2-publication.json",
        "docs/phase_reports/phase13pc-cuda-context-isolation.md",
        "docs/risk_register.md",
        "docs/status.md",
        "docs/tasks.md",
        "scripts/phase13_pilot.py",
        "scripts/phase13pc_cuda_context_isolation.py",
        "scripts/validate_phase2.py",
        "tests/unit/test_phase13_scope.py",
        "tests/unit/test_phase13pc_cuda_context_isolation.py",
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
            "scripts/phase12_unified_admission.py.backup",
            "tests/unit/test_phase12_unified_admission.py.backup",
            "artifacts/phase13",
            "../docs/plans/phase13-pilot-scan.md",
            "/docs/plans/phase13-pilot-scan.md",
            "docs\\plans\\phase13-pilot-scan.md",
        ):
            self.assertFalse(
                validate_phase2.phase13_path_is_allowed(relative)
            )

    def test_phase13r_segment_and_allowlist_are_exact(self) -> None:
        self.assertEqual(
            validate_phase2.PHASE13R_ENTRY_COMMIT,
            "3dcd075d987db2452793408f8dd2b8f97f87530b",
        )
        self.assertEqual(
            validate_phase2.PHASE13R_ALLOWED_PATHS,
            EXPECTED_PHASE13R_PATHS,
        )
        for relative in EXPECTED_PHASE13R_PATHS:
            self.assertTrue(validate_phase2.phase13r_path_is_allowed(relative))

    def test_phase13r_near_miss_and_broad_paths_are_rejected(self) -> None:
        for relative in (
            "docs/evidence/phase13r",
            "docs/evidence/phase13r/pilot_qc.json.backup",
            "scripts/phase13_pilot.py.backup",
            "artifacts/phase13",
            "artifacts/phase13/*",
            "../scripts/phase13_pilot.py",
            "/scripts/phase13_pilot.py",
            "scripts\\phase13_pilot.py",
        ):
            self.assertFalse(
                validate_phase2.phase13r_path_is_allowed(relative)
            )

    def test_phase13f_segment_and_allowlist_are_exact(self) -> None:
        self.assertEqual(
            validate_phase2.PHASE13F_ENTRY_COMMIT,
            "50d7820e1feb986acecd78d1b58a3fff6e0bd713",
        )
        self.assertEqual(
            validate_phase2.PHASE13F_ALLOWED_PATHS,
            EXPECTED_PHASE13F_PATHS,
        )
        for relative in EXPECTED_PHASE13F_PATHS:
            self.assertTrue(validate_phase2.phase13f_path_is_allowed(relative))
        self.assertEqual(
            validate_phase2.PHASE13F_APPROVED_ARTIFACT_ROOT_NAMES,
            frozenset({"phase13f"}),
        )

    def test_phase13f_near_miss_and_broad_paths_are_rejected(self) -> None:
        for relative in (
            "docs/evidence/phase13f",
            "docs/evidence/phase13f/feasibility-summary.json.backup",
            "scripts/phase13f_feasibility.py.backup",
            "artifacts/phase13f",
            "artifacts/phase13f/*",
            "../scripts/phase13f_feasibility.py",
            "/scripts/phase13f_feasibility.py",
            "scripts\\phase13f_feasibility.py",
        ):
            self.assertFalse(
                validate_phase2.phase13f_path_is_allowed(relative)
            )

    def test_exact_phase13f_artifact_root_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            (artifacts / "phase13f").mkdir()
            with mock.patch.object(validate_phase2, "ROOT", root):
                errors = validate_phase2.validate_phase3_artifact_root()
        self.assertEqual(errors, [])

    def test_phase13f_artifact_root_near_miss_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            (artifacts / "phase13f-copy").mkdir()
            with mock.patch.object(validate_phase2, "ROOT", root):
                errors = validate_phase2.validate_phase3_artifact_root()
        self.assertIn(
            "unapproved artifact roots: ['phase13f-copy']",
            errors,
        )

    def test_phase13r2_segment_and_allowlist_are_exact(self) -> None:
        self.assertEqual(
            validate_phase2.PHASE13R2_ENTRY_COMMIT,
            "bdfcb5ba41f150b1f56ddc8ac3515cd5a0b7bde3",
        )
        self.assertEqual(
            validate_phase2.PHASE13R2_ALLOWED_PATHS,
            EXPECTED_PHASE13R2_PATHS,
        )
        for relative in EXPECTED_PHASE13R2_PATHS:
            self.assertTrue(
                validate_phase2.phase13r2_path_is_allowed(relative)
            )

    def test_phase13r2_near_miss_and_broad_paths_are_rejected(self) -> None:
        for relative in (
            "docs/evidence/phase13r2",
            "docs/evidence/phase13r2/pilot_qc.json.backup",
            "artifacts/phase13",
            "artifacts/phase13/*",
            "scripts/phase13_pilot.py",
            "../docs/phase_reports/phase13r2-pilot-scan.md",
            "/docs/phase_reports/phase13r2-pilot-scan.md",
            "docs\\phase_reports\\phase13r2-pilot-scan.md",
        ):
            self.assertFalse(
                validate_phase2.phase13r2_path_is_allowed(relative)
            )

    def test_phase13t_segment_and_allowlist_are_exact(self) -> None:
        self.assertEqual(
            validate_phase2.PHASE13T_ENTRY_COMMIT,
            "fb6a1fedc0cc01abbfbd974005f7567fbd291fd7",
        )
        self.assertEqual(
            validate_phase2.PHASE13T_ALLOWED_PATHS,
            EXPECTED_PHASE13T_PATHS,
        )
        self.assertEqual(
            validate_phase2.PHASE13T_APPROVED_ARTIFACT_ROOT_NAMES,
            frozenset({"phase13t"}),
        )
        for relative in EXPECTED_PHASE13T_PATHS:
            self.assertTrue(
                validate_phase2.phase13t_path_is_allowed(relative)
            )

    def test_phase13t_near_miss_and_broad_paths_are_rejected(self) -> None:
        for relative in (
            "docs/evidence/phase13t",
            "docs/evidence/phase13t/timeout-validation.json.backup",
            "docs/evidence/phase13tr",
            "scripts/phase13t_timeout.py.backup",
            "artifacts/phase13",
            "artifacts/phase13t",
            "artifacts/phase13t/*",
            "../scripts/phase13_pilot.py",
            "/src/kvbench/runtime/process_supervision.py",
            "docs\\phase_reports\\phase13tr-pilot-scan.md",
        ):
            self.assertFalse(
                validate_phase2.phase13t_path_is_allowed(relative)
            )

    def test_phase13pr_segment_and_allowlist_are_exact(self) -> None:
        self.assertEqual(
            validate_phase2.PHASE13PR_ENTRY_COMMIT,
            "3e022662364fdbd129d4ca327e3eae8078a04ea1",
        )
        self.assertEqual(
            validate_phase2.PHASE13PR_ALLOWED_PATHS,
            EXPECTED_PHASE13PR_PATHS,
        )
        for relative in EXPECTED_PHASE13PR_PATHS:
            self.assertTrue(
                validate_phase2.phase13pr_path_is_allowed(relative)
            )

    def test_phase13pr_near_miss_and_broad_paths_are_rejected(self) -> None:
        for relative in (
            "docs/evidence/phase13",
            "docs/evidence/phase13/pilot_qc.json.backup",
            "docs/decisions/0033-phase13-untimed-prefix-state-reuse.md.old",
            "scripts/phase13_prefix_state.py.backup",
            "artifacts/phase13",
            "artifacts/phase13/*",
            "../scripts/phase13_prefix_state.py",
            "/scripts/phase13_prefix_state.py",
            "scripts\\phase13_prefix_state.py",
        ):
            self.assertFalse(
                validate_phase2.phase13pr_path_is_allowed(relative)
            )

    def test_phase13pb_segment_and_allowlist_are_exact(self) -> None:
        self.assertEqual(
            validate_phase2.PHASE13PB_ENTRY_COMMIT,
            "5ff41722edfcdabe0ec96dc226f0546b008c6438",
        )
        self.assertEqual(
            validate_phase2.PHASE13PB_ALLOWED_PATHS,
            EXPECTED_PHASE13PB_PATHS,
        )
        self.assertEqual(
            validate_phase2.PHASE13PB_APPROVED_ARTIFACT_ROOT_NAMES,
            frozenset({"phase13pb"}),
        )
        for relative in EXPECTED_PHASE13PB_PATHS:
            self.assertTrue(
                validate_phase2.phase13pb_path_is_allowed(relative)
            )

    def test_phase13pb_near_miss_and_broad_paths_are_rejected(self) -> None:
        for relative in (
            "docs/evidence/phase13pb",
            "docs/evidence/phase13pb/prefix-equivalence.json.backup",
            "scripts/phase13pb_prefix_remediation.py.backup",
            "artifacts/phase13pb",
            "artifacts/phase13pb/*",
            "../scripts/phase13_prefix_state.py",
            "/scripts/phase13_prefix_state.py",
            "scripts\\phase13_prefix_state.py",
        ):
            self.assertFalse(
                validate_phase2.phase13pb_path_is_allowed(relative)
            )

    def test_phase13pc_segment_and_allowlist_are_exact(self) -> None:
        self.assertEqual(
            validate_phase2.PHASE13PC_ENTRY_COMMIT,
            "91b984069e3305428db612a3515fdc483ec990ed",
        )
        self.assertEqual(
            validate_phase2.PHASE13PC_ALLOWED_PATHS,
            EXPECTED_PHASE13PC_PATHS,
        )
        self.assertEqual(
            validate_phase2.PHASE13PC_APPROVED_ARTIFACT_ROOT_NAMES,
            frozenset({"phase13pc"}),
        )
        for relative in EXPECTED_PHASE13PC_PATHS:
            self.assertTrue(
                validate_phase2.phase13pc_path_is_allowed(relative)
            )

    def test_phase13pc_near_miss_and_broad_paths_are_rejected(self) -> None:
        for relative in (
            "docs/evidence/phase13pc",
            "docs/evidence/phase13pc/r2-publication.json.backup",
            "scripts/phase13pc_cuda_context_isolation.py.backup",
            "artifacts/phase13pc",
            "artifacts/phase13pc/*",
            "../scripts/phase13_pilot.py",
            "/scripts/phase13_pilot.py",
            "scripts\\phase13_pilot.py",
        ):
            self.assertFalse(
                validate_phase2.phase13pc_path_is_allowed(relative)
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
