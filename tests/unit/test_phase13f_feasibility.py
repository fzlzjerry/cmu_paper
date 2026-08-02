"""Focused tests for Decision 0031's end-to-end feasibility contract."""

from __future__ import annotations

import copy
from pathlib import Path
import unittest
from unittest import mock

from scripts import phase13_pilot, phase13f_feasibility


ROOT = Path(__file__).resolve().parents[2]


class Phase13FFeasibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records, cls.summary = phase13f_feasibility.recompute()

    def _point(self, configuration: str, batch: int, context: int) -> dict:
        matches = [
            record
            for record in self.records
            if record["method_config_id"] == configuration
            and record["batch_size"] == batch
            and record["context_label"] == context
            and record["replicate_index"] == 0
        ]
        self.assertEqual(len(matches), 1)
        return matches[0]

    def test_exact_810_record_classification_and_boundary(self) -> None:
        self.assertEqual(len(self.records), 810)
        self.assertEqual(
            self.summary["status_counts"],
            {"capacity_infeasible": 126, "feasible": 684},
        )
        self.assertEqual(
            self.summary["unique_point_status_counts"],
            {"capacity_infeasible": 42, "feasible": 228},
        )
        self.assertEqual(
            self._point("tq_3bit_nc", 8, 49152)["status"], "feasible"
        )
        self.assertEqual(
            self._point("tq_3bit_nc", 8, 65536)["status"],
            "capacity_infeasible",
        )
        self.assertEqual(
            self._point("tq_3bit_nc", 4, 98304)["status"], "feasible"
        )

    def test_old_formula_underestimated_former_failed_point(self) -> None:
        point = self._point("tq_3bit_nc", 8, 98304)
        old_required = (
            point["model_weight_bytes"]
            + point["cache_allocated_bytes"]
            + point["graph_reserve_reference_bytes"]
            * 8
            * point["capacity"]
            // phase13_pilot.REFERENCE_CAPACITY
        )
        self.assertLess(old_required, point["limit_bytes"])
        self.assertGreater(point["predicted_required_bytes"], point["limit_bytes"])
        self.assertEqual(point["hidden_bf16_bytes"], 6 * 1024**3)
        self.assertEqual(point["status"], "capacity_infeasible")

    def test_component_sum_and_source_peak_are_exact(self) -> None:
        point = self._point("tq_3bit_nc", 8, 98304)
        expected = (
            point["model_weight_bytes"]
            + point["cache_allocated_bytes"]
            + point["endpoint_workspace_bytes"]
            + point["prefix_control_tensor_bytes"]
            + point["prefix_compute_peak_bytes"]
            + point["graph_pool_or_capture_reserve_bytes"]
        )
        self.assertEqual(expected, point["predicted_required_bytes"])
        self.assertEqual(
            point["mlp_peak_bytes"],
            2 * point["hidden_bf16_bytes"]
            + 3 * point["intermediate_bf16_bytes"],
        )
        self.assertGreater(
            point["mlp_peak_bytes"],
            point["attention_output_projection_peak_bytes"],
        )
        self.assertEqual(point["max_memory_fraction"], 0.88)

    def test_recomputation_is_deterministic(self) -> None:
        second_records, second_summary = phase13f_feasibility.recompute()
        self.assertEqual(self.records, second_records)
        self.assertEqual(self.summary, second_summary)
        by_point: dict[tuple[str, int, int], set[tuple[str, int]]] = {}
        for record in self.records:
            key = (
                record["method_config_id"],
                record["batch_size"],
                record["context_label"],
            )
            by_point.setdefault(key, set()).add(
                (record["status"], record["predicted_required_bytes"])
            )
        self.assertTrue(all(len(values) == 1 for values in by_point.values()))

    def test_order_and_fingerprint_tampering_remain_rejected(self) -> None:
        order = phase13_pilot.derive_execution_order()
        tampered = copy.deepcopy(order)
        tampered["records"][0]["method_config_fingerprint"] = "0" * 64
        with self.assertRaisesRegex(
            phase13_pilot.Phase13PilotError, "execution order differs"
        ):
            phase13_pilot.build_feasibility_records(tampered)

    def test_historical_or_source_tampering_fails_closed(self) -> None:
        original = phase13f_feasibility.sha256_file

        def altered(path: Path) -> str:
            if Path(path).name == "worker.stderr.txt":
                return "0" * 64
            return original(path)

        with mock.patch.object(phase13f_feasibility, "sha256_file", altered):
            with self.assertRaisesRegex(
                phase13f_feasibility.Phase13FFeasibilityError,
                "failed point evidence differs",
            ):
                phase13f_feasibility.former_point_proof(self.records)

        with mock.patch.object(
            phase13f_feasibility,
            "SOURCE_HASHES",
            {"src/kvbench/runtime/bf16_endpoint.py": "0" * 64},
        ):
            with self.assertRaisesRegex(
                phase13f_feasibility.Phase13FFeasibilityError,
                "authority differs",
            ):
                phase13f_feasibility._source_authority()

    def test_decision_and_stopped_campaign_custody(self) -> None:
        decision = (
            ROOT
            / "docs/decisions/0031-phase13-end-to-end-prefix-feasibility.md"
        ).read_text(encoding="utf-8")
        self.assertIn("Status: Accepted", decision)
        self.assertIn("0.88", decision)
        custody = phase13f_feasibility.validate_historical_custody()
        self.assertEqual(len(custody), 2)
        self.assertTrue(all(record["changed"] is False for record in custody))
        self.assertTrue(all(record["clean_retrieval"] == "PASS" for record in custody))


if __name__ == "__main__":
    unittest.main()
