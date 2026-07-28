"""Focused Phase 9 preservation and boundary regressions."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import unittest

from scripts.validate_phase2 import (
    PHASE9_ALLOWED_PATHS,
    PHASE9_ENTRY_COMMIT,
    current_phase9_paths,
    make_target_block,
)


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Phase9GovernanceTests(unittest.TestCase):
    def test_entry_boundary_and_current_paths_are_exact(self) -> None:
        self.assertEqual(
            PHASE9_ENTRY_COMMIT,
            "b4d253724717076188a38032d6d6204fdf15e191",
        )
        self.assertLessEqual(current_phase9_paths(), PHASE9_ALLOWED_PATHS)
        for forbidden in (
            "docker/measurement.Dockerfile",
            "reference/kvquant/fixture.json",
            "src/kvbench/adapters/kvquant.py",
            "src/kvbench/runtime/kvquant.py",
            "artifacts/quality/kvquant.json",
            "artifacts/profiler/kvquant.ncu-rep",
            "calibration/kvquant/example/fisher.safetensors",
            ".env",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, PHASE9_ALLOWED_PATHS)

    def test_frozen_methods_container_and_historical_reports_are_unchanged(self) -> None:
        expected = {
            "configs/methods/bf16.yaml": (
                "fdffda79ca294ca7592f6ffc6033698b5875f3af9824970ccf08cf61af841fd8"
            ),
            "configs/methods/turboquant.yaml": (
                "a7e69050097820a455bb5086adf22d2adeb44068bec408504d8525f709260ec2"
            ),
            "configs/methods/kivi.yaml": (
                "5c48e8f0380f2c17750b25f91c721bbdbbb68385fa3d958b2b28cdd922716c81"
            ),
            "docker/measurement.Dockerfile": (
                "333a1e4264e8dc7798c5af06622fc97871371c3fc1e063f4a3b88cfb25389ace"
            ),
            "src/kvbench/adapters/factory.py": (
                "31e482c39f54319f2dbef814fdfa212283a8a3004dabda2248e14ea75cdf7672"
            ),
            "docs/phase_reports/phase8-kivi-measurement-adapter.md": (
                "63043780d7618ad684e1f64f28e785a941b578cd8b5fda59dd6e1b182b3a1dd2"
            ),
            "docs/evidence/phase9p/patch-manifest.json": (
                "c2390f52af2f4f6d4ef5731f64ed05b9f307e009391f7af7c79baee0209b5e5e"
            ),
            "docs/evidence/phase9p/test-report.json": (
                "4e4b94a50bba0c3bce73719205d62f7fefb20374f578817e151fef2f7d0517fd"
            ),
            "docs/phase_reports/phase9-kvquant-calibration-blocked.md": (
                "05bbc9d21fe4bff900bd141ddc7f6daec226848178f8c0b78b7ecdaba2c180b7"
            ),
        }
        for relative, digest in expected.items():
            with self.subTest(relative=relative):
                self.assertEqual(sha256(ROOT / relative), digest)

        for relative in (
            "configs/methods/bf16.yaml",
            "configs/methods/turboquant.yaml",
            "configs/methods/kivi.yaml",
            "docker/measurement.Dockerfile",
            "src/kvbench/adapters/factory.py",
            "docs/evidence/phase6",
            "docs/evidence/phase7",
            "docs/evidence/phase8",
            "docs/evidence/phase9p",
            "docs/phase_reports/phase9-kvquant-calibration-blocked.md",
        ):
            with self.subTest(relative=relative):
                result = subprocess.run(
                    [
                        "git",
                        "diff",
                        "--quiet",
                        PHASE9_ENTRY_COMMIT,
                        "--",
                        relative,
                    ],
                    cwd=ROOT,
                    check=False,
                )
                self.assertEqual(result.returncode, 0)

    def test_kvquant_execution_and_later_phases_remain_fail_closed(self) -> None:
        factory = (ROOT / "src/kvbench/adapters/factory.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            '_DEFERRED_METHODS = frozenset({"kvquant"})',
            factory,
        )
        self.assertFalse((ROOT / "reference/kvquant").exists())
        self.assertFalse((ROOT / "src/kvbench/adapters/kvquant.py").exists())
        self.assertFalse((ROOT / "src/kvbench/runtime/kvquant.py").exists())
        self.assertFalse((ROOT / "PERFORMANCE_DATA_FROZEN").exists())
        for forbidden in (
            ROOT / "artifacts/quality",
            ROOT / "artifacts/profiler",
            ROOT / "paper-results",
            ROOT / "paper_results",
            ROOT / "results",
        ):
            self.assertFalse(forbidden.exists())

    def test_calibration_container_is_separate_digest_pinned_and_secret_free(self) -> None:
        dockerfile = (
            ROOT / "docker/calibration-kvquant.Dockerfile"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "FROM --platform=linux/amd64 "
            "sha256:16ee632c5ac029deca5859f3da4c74f9e5e55f5080c10745a59653e95d8e5b44",
            dockerfile,
        )
        self.assertNotIn("docker/measurement.Dockerfile", dockerfile)
        for forbidden in (
            ".env",
            "HF_TOKEN",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "model.safetensors",
            "third_party/patches",
        ):
            self.assertNotIn(forbidden, dockerfile)
        ignore = (
            ROOT / "docker/calibration-kvquant.Dockerfile.dockerignore"
        ).read_text(encoding="utf-8")
        self.assertIn(".env", ignore)
        self.assertIn("**", ignore)

    def test_make_targets_are_narrow_and_do_not_start_later_work(self) -> None:
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        calibrate = make_target_block(makefile, "calibrate-kvquant")
        validate = make_target_block(
            makefile,
            "validate-calibration-kvquant",
        )
        self.assertIn("$(PHASE9_CALIBRATION)", calibrate)
        self.assertIn("validate-kvquant-gqa-patch", calibrate)
        self.assertIn("$(PHASE9_CALIBRATION)", validate)
        self.assertIn(
            "scripts/phase9_kvquant_calibration.py", makefile
        )
        for forbidden in (
            "pilot",
            "full-scan",
            "profile-subset",
            "fit",
            "figures",
            "quality",
            "LongBench",
            "ppl",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, calibrate)
                self.assertNotIn(forbidden, validate)

    def test_plan_defers_phase10_adapter_reference_and_quality(self) -> None:
        plan = (
            ROOT / "docs/plans/phase9-kvquant-calibration.md"
        ).read_text(encoding="utf-8")
        self.assertIn("Phase 10", plan)
        self.assertIn("remain deferred", plan)
        self.assertNotIn("Status: PASS", plan)


if __name__ == "__main__":
    unittest.main()
