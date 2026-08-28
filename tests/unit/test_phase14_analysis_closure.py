"""Focused tests for the immutable Phase 14 analysis closure."""

from __future__ import annotations

import copy
import unittest

from scripts import phase14_analysis_closure as closure


class Phase14AnalysisClosureTests(unittest.TestCase):
    def _eligible_effect(self) -> dict[str, object]:
        return {
            "eager_fit_status": "knee_observed",
            "graph_fit_status": "knee_observed",
            "material_floor_reduction": True,
            "slope_similar": True,
            "semantics_unchanged": True,
            "host_minus_device_proxy_lower": True,
        }

    def test_complete_criterion_requires_every_frozen_term(self) -> None:
        row = self._eligible_effect()
        self.assertTrue(closure.criterion_supported(row))
        for key in (
            "material_floor_reduction",
            "slope_similar",
            "semantics_unchanged",
            "host_minus_device_proxy_lower",
        ):
            rejected = copy.deepcopy(row)
            rejected[key] = False
            self.assertFalse(closure.criterion_supported(rejected), key)
        unstable = copy.deepcopy(row)
        unstable["eager_fit_status"] = "unstable_data"
        self.assertFalse(closure.criterion_supported(unstable))
        no_slope = copy.deepcopy(row)
        no_slope["eager_fit_status"] = "no_positive_slope"
        self.assertFalse(closure.criterion_supported(no_slope))

    def test_original_campaign_and_report_hashes_are_frozen(self) -> None:
        closure._validate_source_hashes()
        self.assertEqual(
            closure.SOURCE_CAMPAIGN_ROOT_SHA256,
            "22a613b07c1ee6d3e9a0a7fc81df6065ccc8a10bf783b1a69aded3c2eb8068f0",
        )
        self.assertEqual(
            closure.SOURCE_FILE_HASHES[
                "docs/phase_reports/phase14-graph-ab.md"
            ],
            "96210af393acfeaa00a92ed5e41e41f654bc9c7c67205db29abe81a1e9e5acb3",
        )

    def test_frozen_denominators_and_cv_threshold(self) -> None:
        self.assertEqual(closure.CV_THRESHOLD, 0.03)
        self.assertEqual(closure.EXPECTED_COMPLETED_RUNS, 660)
        self.assertEqual(closure.EXPECTED_STABLE_CONDITIONS, 105)
        self.assertEqual(closure.EXPECTED_UNSTABLE_CONDITIONS, 5)
        self.assertEqual(closure.EXPECTED_IDENTIFIABLE_COMPARISONS, 14)
        self.assertEqual(closure.EXPECTED_UNSTABLE_EAGER_COMPARISONS, 4)
        self.assertEqual(closure.EXPECTED_NO_POSITIVE_SLOPE_COMPARISONS, 2)

    def test_phase_states_are_fail_closed_in_payload_validator(self) -> None:
        summary = {
            "completed_mode_runs": 660,
            "stable_ab_condition_count": 105,
            "unstable_eager_conditions": [],
            "comparison_categories": {
                "fully_identifiable": 14,
                "unstable_eager": 4,
                "no_positive_eager_slope": 2,
                "other_non_identifiable": 0,
            },
            "fully_identifiable_comparisons": [],
            "unstable_eager_comparisons": [],
            "no_positive_eager_slope_comparisons": [],
            "launch_floor_only": {
                "support_count": 0,
                "eligible_denominator": 14,
                "floor_decreases": 14,
                "material_floor_reductions": 8,
                "slope_similar": 10,
                "semantics_unchanged": 14,
            },
            "knee": {"shift_counts": {}, "graph_knee_disappeared": 0},
            "host_minus_device_proxy": {
                "graph_lower_count": 23,
                "stable_pair_denominator": 105,
                "graph_not_lower_count": 82,
                "direct_launch_gap_measured": False,
            },
            "semantic_mismatches": 0,
            "fit_status_counts": {},
            "upstream_knee_tables": {},
        }
        payload = closure.build_closure_payload(
            summary,
            analysis_git_sha="a" * 40,
            recorded_at_utc="2026-08-28T00:00:00Z",
        )
        self.assertEqual(payload["gates"]["phase15"], "READY")
        self.assertEqual(payload["gates"]["full_scan"], "CLOSED")
        self.assertEqual(payload["gates"]["quality"], "LOCKED")
        tampered = copy.deepcopy(payload)
        tampered["gates"]["full_scan"] = "OPEN"
        with self.assertRaises(closure.Phase14ClosureError):
            closure.validate_closure_payload(tampered)


if __name__ == "__main__":
    unittest.main()
