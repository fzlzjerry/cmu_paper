"""Focused tests for the exact Phase 10 reference-lane scope."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from scripts import validate_phase2


ROOT = Path(__file__).resolve().parents[2]


class Phase10ScopeTests(unittest.TestCase):
    def test_phase10_allowlist_has_only_exact_reference_paths(self) -> None:
        expected = {
            "docs/decisions/0023-phase10-kvquant-source-faithful-sparse-fixture-semantics.md",
            "docs/evidence/phase10/blocked-report-custody.json",
            "docs/plans/phase10-kvquant-reference.md",
            "reference/kvquant/generate_fixtures.py",
            "reference/kvquant/validate_fixtures.py",
            "tests/unit/test_phase10_scope.py",
        }
        self.assertLessEqual(expected, validate_phase2.PHASE10_ALLOWED_PATHS)
        for rejected in (
            "docs/decisions/0023-kvquant-license.md",
            "docs/plans/phase10-reference-framework.md",
            "docker/measurement.Dockerfile",
            "reference/kvquant/framework.py",
            "reference/kvquant/fixtures/kvq4/no_outlier/fixture_manifest.json",
            "reference/kvquant/fixtures/kvq4/key_zero_value_fixed12/copy.safetensors",
            "scripts/r2_artifact_phase10.py",
            "src/kvbench/adapters/kvquant.py",
        ):
            with self.subTest(rejected=rejected):
                self.assertNotIn(
                    rejected,
                    validate_phase2.PHASE10_ALLOWED_PATHS,
                )

    def test_fixture_matrix_and_members_are_exact(self) -> None:
        self.assertEqual(
            validate_phase2.PHASE10_FIXTURE_FAMILIES,
            ("kvq4", "kvq3", "kvq2"),
        )
        self.assertEqual(
            validate_phase2.PHASE10_FIXTURE_CASES,
            (
                "key_zero_value_fixed12",
                "key_few_value_fixed12",
                "key_cap_value_fixed12",
            ),
        )
        expected_count = (
            len(validate_phase2.PHASE10_FIXTURE_FAMILIES)
            * len(validate_phase2.PHASE10_FIXTURE_CASES)
            * len(validate_phase2.PHASE10_FIXTURE_MEMBERS)
        )
        self.assertEqual(
            len(validate_phase2.PHASE10_FIXTURE_PATHS),
            expected_count,
        )
        self.assertFalse(
            any(
                "/no_outlier/" in path
                or "/few_outliers/" in path
                or "/cap_reached/" in path
                for path in validate_phase2.PHASE10_FIXTURE_PATHS
            )
        )

    def test_only_exact_fixture_safe_tensors_escape_raw_suffix_rejection(
        self,
    ) -> None:
        self.assertTrue(validate_phase2.PHASE10_SAFE_TENSOR_PATHS)
        self.assertTrue(
            all(
                path.endswith(".safetensors")
                and path in validate_phase2.PHASE10_FIXTURE_PATHS
                for path in validate_phase2.PHASE10_SAFE_TENSOR_PATHS
            )
        )
        self.assertNotIn(
            "reference/kvquant/fixtures/kvq4/key_zero_value_fixed12/debug.pt",
            validate_phase2.PHASE10_SAFE_TENSOR_PATHS,
        )

    def test_contract_decision_and_blocked_report_custody_are_exact(
        self,
    ) -> None:
        decision = (
            ROOT
            / "docs/decisions/0023-phase10-kvquant-source-faithful-sparse-fixture-semantics.md"
        ).read_text(encoding="utf-8")
        self.assertIn("Key path can deterministically produce counts 0, 6, and 12", decision)
        self.assertIn("six lowest entries", decision)
        self.assertIn("fixture-contract correction, not an algorithm change", decision)
        report = (
            ROOT / "docs/phase_reports/phase10-kvquant-reference-blocked.md"
        ).read_bytes()
        custody = json.loads(
            (
                ROOT / "docs/evidence/phase10/blocked-report-custody.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            hashlib.sha256(report).hexdigest(),
            "0362ac36f03ec8b92c0b154ad85fd4e810c3d45332f7c063b1da00d7adc94e4d",
        )
        self.assertEqual(
            custody["repository_storage"]["sha256"],
            hashlib.sha256(report).hexdigest(),
        )
        self.assertEqual(
            custody["source"]["sha256"],
            "ca8669e8a056a2f77428c3a31a2115407b63013e989fe93ebebef5af3af9921f",
        )
        self.assertTrue(custody["immutable"])


if __name__ == "__main__":
    unittest.main()
