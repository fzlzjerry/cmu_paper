"""Focused Phase 6 scope and frozen-governance regressions."""

from __future__ import annotations

from pathlib import Path
import unittest

from scripts.validate_phase2 import (
    APPROVED_ARTIFACT_ROOT_NAMES,
    PHASE6_ALLOWED_PATHS,
    PHASE6_ENTRY_COMMIT,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class Phase6GovernanceTests(unittest.TestCase):
    def test_entry_and_allowlist_are_exact(self) -> None:
        self.assertEqual(
            PHASE6_ENTRY_COMMIT,
            "e06f638f4b913f9bd1be2975a478657f5bf2338e",
        )
        required = {
            "docs/plans/phase6-turboquant-measurement-adapter.md",
            "src/kvbench/adapters/turboquant.py",
            "src/kvbench/runtime/phase3_coordinator.py",
            "src/kvbench/runtime/turboquant_cache.py",
            "src/kvbench/runtime/turboquant_session.py",
            "tests/unit/test_process_supervision.py",
            "tests/unit/test_phase6_governance.py",
        }
        self.assertLessEqual(required, PHASE6_ALLOWED_PATHS)
        for rejected in (
            "src/kvbench/adapters/kivi.py",
            "src/kvbench/adapters/kvquant.py",
            "src/kvbench/plugins/turboquant.py",
            "scripts/phase7_kivi.py",
            "artifacts/quality/result.json",
            "results/turboquant.json",
        ):
            with self.subTest(rejected=rejected):
                self.assertNotIn(rejected, PHASE6_ALLOWED_PATHS)

    def test_artifact_root_allowlist_is_exact(self) -> None:
        self.assertEqual(
            APPROVED_ARTIFACT_ROOT_NAMES,
            frozenset(
                {
                    "README.md",
                    "phase3",
                    "phase3_campaigns",
                    "phase3_reports",
                    "phase4_smoke",
                    "phase6",
                    "phase6a",
                }
            ),
        )

    def test_admission_runtime_venv_stays_inside_ignored_directory(self) -> None:
        makefile = (REPOSITORY_ROOT / "Makefile").read_text(encoding="utf-8")
        self.assertIn(
            'mkdir "$$task_root/source/.venv"',
            makefile,
        )
        self.assertIn(
            'ln -s /opt/kvbench/.venv/bin '
            '"$$task_root/source/.venv/bin"',
            makefile,
        )
        self.assertIn(
            'ln -s /opt/kvbench/.venv/lib '
            '"$$task_root/source/.venv/lib"',
            makefile,
        )
        self.assertIn(
            'ln -s /opt/kvbench/.venv/pyvenv.cfg '
            '"$$task_root/source/.venv/pyvenv.cfg"',
            makefile,
        )
        self.assertNotIn(
            'ln -s /opt/kvbench/.venv "$$task_root/source/.venv"',
            makefile,
        )

    def test_admission_rehydrates_e00_immutable_modes_in_clone(self) -> None:
        makefile = (REPOSITORY_ROOT / "Makefile").read_text(encoding="utf-8")
        self.assertEqual(
            makefile.count(
                'chmod -R a-w "$$task_root/source/docs/evidence/e00"'
            ),
            1,
        )
        self.assertEqual(
            makefile.count(
                'find "$$task_root/source/docs/evidence/e00" '
                "-perm /222 -print -quit"
            ),
            1,
        )

    def test_admission_uses_only_the_locked_container_python(self) -> None:
        makefile = (REPOSITORY_ROOT / "Makefile").read_text(encoding="utf-8")
        command = (
            "make PHASE2_PYTHON=/opt/kvbench/.venv/bin/python "
            "PHASE3_PYTHON=/opt/kvbench/.venv/bin/python "
        )
        self.assertEqual(makefile.count(f"{command}test-cuda"), 1)
        self.assertEqual(makefile.count(f"{command}test-graph"), 1)

    def test_plan_freezes_tolerance_and_later_phases(self) -> None:
        plan = (
            REPOSITORY_ROOT
            / "docs"
            / "plans"
            / "phase6-turboquant-measurement-adapter.md"
        ).read_text(encoding="utf-8")
        self.assertIn("`atol=0.02, rtol=0.02`", plan)
        self.assertIn("Phase 7 is explicitly deferred", plan)
        self.assertIn("Full Scan remains closed", plan)
        self.assertIn("`r_hbm` null", plan)

    def test_quality_and_full_scan_remain_locked(self) -> None:
        status = (REPOSITORY_ROOT / "docs" / "status.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("Quality execution: LOCKED", status)
        self.assertIn("Full Scan remains CLOSED", status)
        self.assertFalse(any(REPOSITORY_ROOT.rglob("PERFORMANCE_DATA_FROZEN")))


if __name__ == "__main__":
    unittest.main()
