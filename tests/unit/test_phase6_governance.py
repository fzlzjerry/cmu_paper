"""Focused Phase 6 scope and frozen-governance regressions."""

from __future__ import annotations

from pathlib import Path
import unittest

from scripts.validate_phase2 import (
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
            "src/kvbench/runtime/turboquant_cache.py",
            "src/kvbench/runtime/turboquant_session.py",
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
