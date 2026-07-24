"""Focused Phase 6A scope, secret-safety, and preservation regressions."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import unittest

from scripts.validate_phase2 import (
    PHASE6A_ALLOWED_PATHS,
    PHASE6A_E00_ALLOWED_PATHS,
    PHASE6A_ENTRY_COMMIT,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_PHASE6_REPORT_SHA256 = (
    "18015eeda156eeb1718d551919605244b6"
    "ff21c819182fcd5d66fa316558ed74"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Phase6AGovernanceTests(unittest.TestCase):
    def test_allowlist_accepts_only_exact_phase6a_paths(self) -> None:
        required = {
            ".dockerignore",
            ".gitignore",
            "docker/measurement.Dockerfile",
            "docs/plans/phase6a-measurement-container-and-r2.md",
            "preflight/e00_manifest.schema.json",
            "preflight/run_preflight.py",
            "scripts/phase6a_bf16_parity.py",
            "scripts/r2_artifact.py",
            "scripts/validate_phase2.py",
            "tests/unit/test_measurement_container.py",
            "tests/unit/test_phase6a_bf16_parity.py",
            "tests/unit/test_phase6a_governance.py",
            "tests/unit/test_preflight_unit.py",
            "tests/unit/test_r2_artifact.py",
        }
        self.assertLessEqual(required, PHASE6A_ALLOWED_PATHS)
        self.assertEqual(
            PHASE6A_ENTRY_COMMIT,
            "a25a76a052a918428e8eb56cdfde63470cf6a152",
        )
        for rejected in (
            "docker/arbitrary.Dockerfile",
            "scripts/storage_framework.py",
            "docs/phase_reports/arbitrary.md",
            "docs/arbitrary.md",
            "tests/unit/test_arbitrary.py",
            "src/kvbench/methods/turboquant/adapter.py",
            "artifacts/phase6a/arbitrary/result.bin",
        ):
            with self.subTest(rejected=rejected):
                self.assertNotIn(rejected, PHASE6A_ALLOWED_PATHS)

    def test_e00_exception_is_narrow(self) -> None:
        self.assertEqual(
            PHASE6A_E00_ALLOWED_PATHS,
            {
                "preflight/e00_manifest.schema.json",
                "preflight/measurement-container-system-packages.expected.json",
                "preflight/measurement-container-system-packages.lock.json",
                "preflight/run_preflight.py",
                "tests/unit/test_preflight_unit.py",
            },
        )
        self.assertNotIn(
            "preflight/e00_cuda/xor_kernel.cu",
            PHASE6A_E00_ALLOWED_PATHS,
        )
        self.assertNotIn("scripts/preflight.sh", PHASE6A_E00_ALLOWED_PATHS)

    def test_env_is_untracked_ignored_and_excluded_from_context(self) -> None:
        tracked = subprocess.run(
            ("git", "ls-files", "--error-unmatch", ".env"),
            cwd=REPOSITORY_ROOT,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        ignored = subprocess.run(
            ("git", "check-ignore", "-q", ".env"),
            cwd=REPOSITORY_ROOT,
            check=False,
        )
        self.assertNotEqual(tracked.returncode, 0)
        self.assertEqual(ignored.returncode, 0)
        rules = {
            line.strip()
            for line in (REPOSITORY_ROOT / ".dockerignore")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertIn(".env", rules)
        self.assertIn(".env.*", rules)

    def test_historical_evidence_and_phase5_fixtures_are_unchanged(self) -> None:
        protected = (
            "docs/evidence",
            "artifacts/phase3",
            "artifacts/phase3_campaigns",
            "artifacts/phase3_reports",
            "artifacts/phase4_smoke",
            "reference/turboquant/fixtures",
            "docs/phase_reports/phase5-turboquant-reference.md",
            "docs/phase_reports/phase6-turboquant-measurement-blocked.md",
        )
        result = subprocess.run(
            (
                "git",
                "diff",
                "--name-only",
                PHASE6A_ENTRY_COMMIT,
                "--",
                *protected,
            ),
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            _sha256(
                REPOSITORY_ROOT
                / "docs"
                / "phase_reports"
                / "phase6-turboquant-measurement-blocked.md"
            ),
            EXPECTED_PHASE6_REPORT_SHA256,
        )

    def test_quality_and_later_work_remain_locked(self) -> None:
        status = (REPOSITORY_ROOT / "docs/status.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("Quality execution: LOCKED", status)
        self.assertIn("Full Scan remains CLOSED", status)
        self.assertFalse(any(REPOSITORY_ROOT.rglob("PERFORMANCE_DATA_FROZEN")))
        for relative in (
            "artifacts/quality",
            "artifacts/profiler",
            "results",
            "paper-results",
            "paper_results",
            "src/kvbench/methods/turboquant",
        ):
            self.assertFalse((REPOSITORY_ROOT / relative).exists(), relative)


if __name__ == "__main__":
    unittest.main()
