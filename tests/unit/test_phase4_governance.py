"""Focused Phase 4 lock and immutable-evidence preservation tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PHASE4_ENTRY_SHA = "a3a56e45354ac93ab3c25f82a82e8e6096b513b9"
PHASE3_REPORT = (
    REPOSITORY_ROOT
    / "artifacts/phase3_reports"
    / "phase3-g1-20260723t132609515797z-7f72c95f-f31ccb"
    / "report.json"
)
PHASE3_REPORT_SHA256 = (
    "c29aef1d9f22b328201599b3e6cdf9efe7c069e78abaf6b37bc3cb12931414c9"
)


class Phase4GovernanceTests(unittest.TestCase):
    def test_old_e00_and_phase3_paths_have_no_phase4_diff(self) -> None:
        result = subprocess.run(
            (
                "git",
                "diff",
                "--name-only",
                PHASE4_ENTRY_SHA,
                "--",
                "docs/evidence/e00",
                "artifacts/phase3",
                "artifacts/phase3_campaigns",
                "artifacts/phase3_reports",
            ),
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            hashlib.sha256(PHASE3_REPORT.read_bytes()).hexdigest(),
            PHASE3_REPORT_SHA256,
        )

    def test_quality_and_full_scan_remain_locked(self) -> None:
        status = (REPOSITORY_ROOT / "docs/status.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("Quality execution: LOCKED", status)
        self.assertIn("Full Scan remains CLOSED", status)
        self.assertFalse(
            any(REPOSITORY_ROOT.rglob("PERFORMANCE_DATA_FROZEN"))
        )
        self.assertFalse((REPOSITORY_ROOT / "artifacts/quality").exists())

    def test_no_quantized_implementation_or_plugin_framework_exists(self) -> None:
        for relative in (
            "src/kvbench/methods/turboquant",
            "src/kvbench/methods/kivi",
            "src/kvbench/methods/kvquant",
        ):
            self.assertFalse((REPOSITORY_ROOT / relative).exists())
        factory = (
            REPOSITORY_ROOT / "src/kvbench/adapters/factory.py"
        ).read_text(encoding="utf-8")
        for forbidden in (
            "entry_points",
            "importlib",
            "register_adapter",
            "plugin",
        ):
            self.assertNotIn(forbidden, factory)

    def test_functional_smokes_cannot_be_formal_timing_results(self) -> None:
        root = REPOSITORY_ROOT / "artifacts/phase4_smoke"
        if not root.exists():
            return
        for path in root.glob("*/smoke.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertFalse(payload["timing_collected"])
            self.assertFalse(payload["performance_claim_eligible"])
            self.assertEqual(payload["quality_status"], "unvalidated")
            self.assertEqual(payload["claim_eligibility"], "performance_only")
            self.assertEqual(
                payload["measurement_scope"],
                "native_host_admission",
            )
            self.assertNotIn("latency", payload)


if __name__ == "__main__":
    unittest.main()
