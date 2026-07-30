from __future__ import annotations

from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class Phase11RQ23MakeTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.makefile = (REPOSITORY_ROOT / "Makefile").read_text(
            encoding="utf-8"
        )

    def test_q23_targets_use_explicit_authority_profile(self) -> None:
        self.assertIn(
            "admit-kvquant-q23: override "
            "PHASE11_KVQUANT_ACTIVE_AUTHORITY_PROFILE := decision0029",
            self.makefile,
        )
        self.assertIn(
            "validate-admission-kvquant-q23: override "
            "PHASE11_KVQUANT_ACTIVE_AUTHORITY_PROFILE := decision0029",
            self.makefile,
        )
        self.assertIn(
            '--authority-profile "$(PHASE11_KVQUANT_ACTIVE_AUTHORITY_PROFILE)"',
            self.makefile,
        )

    def test_q23_target_binds_current_source_and_prior_cuda_evidence(self) -> None:
        for expected in (
            "$(PHASE11DQ23_KVQUANT_COMMIT)",
            "$(PHASE11DQ23_KVQUANT_TREE)",
            "$(PHASE11DQ23_KVQUANT_PATCH_SHA256)",
            "$(PHASE11DQ23_KVQUANT_EXTENSION_SHA256)",
            "scripts.validate_kvquant_q23_long_context_patch",
            "phase11dq23-launch.ssAO5M",
            "KVBENCH_PHASE11DQ23_EVIDENCE_ROOT=/opt/phase11dq23-evidence",
        ):
            self.assertIn(expected, self.makefile)

    def test_historical_target_remains_the_default_profile(self) -> None:
        self.assertIn(
            "override PHASE11_KVQUANT_ACTIVE_AUTHORITY_PROFILE := "
            "decision0027",
            self.makefile,
        )
        self.assertIn(
            "override PHASE11_KVQUANT_METHOD_ADMISSION_REPORT := "
            "docs/evidence/phase11/kvquant-method-admission.json",
            self.makefile,
        )
        self.assertIn(
            "override PHASE11RQ23_KVQUANT_METHOD_ADMISSION_REPORT := "
            "docs/evidence/phase11rq23/kvquant-method-admission.json",
            self.makefile,
        )


if __name__ == "__main__":
    unittest.main()
