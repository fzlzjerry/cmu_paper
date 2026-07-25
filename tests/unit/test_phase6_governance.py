"""Focused Phase 6 scope and frozen-governance regressions."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from kvbench.schema import MethodAdmissionReportV2
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

    def test_validation_target_imports_from_the_repository_root(self) -> None:
        makefile = (REPOSITORY_ROOT / "Makefile").read_text(encoding="utf-8")
        self.assertIn(
            "$(PHASE3_PYTHON) -m scripts.phase6_turboquant_admission "
            "--validate-only",
            makefile,
        )
        self.assertNotIn(
            "$(PHASE3_PYTHON) scripts/phase6_turboquant_admission.py "
            "--validate-only",
            makefile,
        )

    def test_sanitizer_probe_resets_only_its_isolated_cuda_context(
        self,
    ) -> None:
        probe = (
            REPOSITORY_ROOT
            / "tests"
            / "cuda"
            / "phase6_turboquant_sanitizer_probe.py"
        ).read_text(encoding="utf-8")
        self.assertEqual(probe.count('ctypes.CDLL("libcudart.so.13")'), 1)
        self.assertEqual(probe.count('ctypes.CDLL("libcublas.so.13")'), 1)
        self.assertEqual(probe.count("cudaDeviceReset"), 2)
        self.assertEqual(probe.count("cublasDestroy_v2"), 2)
        self.assertIn("torch._C._cuda_clearCublasWorkspaces()", probe)
        self.assertIn("torch._C._host_emptyCache()", probe)
        self.assertIn("_build_hadamard_cached.cache_clear()", probe)
        self.assertIn("os._exit(exit_code)", probe)

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

    def test_blocked_method_report_is_strict_and_evidence_backed(
        self,
    ) -> None:
        report_path = (
            REPOSITORY_ROOT
            / "docs"
            / "evidence"
            / "phase6"
            / "turboquant-method-admission.json"
        )
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        report = MethodAdmissionReportV2.from_dict(payload)
        self.assertEqual(report.status.value, "BLOCKED")
        self.assertEqual(report.blockers, ("B-018",))
        self.assertEqual(report.admitted_config_ids, ())
        self.assertFalse(report.performance_claim_eligible)
        publication = json.loads(
            (
                REPOSITORY_ROOT
                / "docs"
                / "evidence"
                / "phase6"
                / "r2-publication.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(publication["admission_status"], "BLOCKED")
        self.assertEqual(publication["clean_retrieval"]["result"], "PASS")
        self.assertFalse(publication["credential_values_recorded"])

    def test_quality_and_full_scan_remain_locked(self) -> None:
        status = (REPOSITORY_ROOT / "docs" / "status.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("Quality execution: LOCKED", status)
        self.assertIn("Full Scan remains CLOSED", status)
        self.assertFalse(any(REPOSITORY_ROOT.rglob("PERFORMANCE_DATA_FROZEN")))


if __name__ == "__main__":
    unittest.main()
