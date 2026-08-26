"""Focused tests for the preregistered Phase 14 Graph OFF/ON contract."""

from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path
import unittest

from scripts import phase14_graph_ab as phase14


ROOT = Path(__file__).resolve().parents[2]


class Phase14PreregistrationTests(unittest.TestCase):
    def test_exact_grid_and_committed_order(self) -> None:
        self.assertEqual(
            phase14.CONFIGURATIONS,
            (
                "bf16",
                "tq_4bit_nc",
                "tq_k3v4_nc",
                "tq_3bit_nc",
                "k4v4",
                "k2v4",
                "k2v2",
                "kvq4",
                "kvq3",
                "kvq2",
            ),
        )
        self.assertEqual(phase14.BATCH_SIZES, (1, 4))
        self.assertEqual(
            phase14.CONTEXT_LABELS,
            (4096, 16384, 24576, 32768, 65536, 131072),
        )
        self.assertEqual(phase14.GRAPH_MODES, ("eager", "cuda_graph"))
        self.assertEqual(phase14.SEEDS, (20260826, 20260827, 20260828))
        self.assertEqual(phase14.WARMUP_STEPS, 64)
        self.assertEqual(phase14.MEASURED_STEPS, 128)
        self.assertEqual(phase14.REPLICATES, 3)
        self.assertEqual(phase14.PLANNED_PAIR_RECORDS, 360)
        self.assertEqual(phase14.PLANNED_RUN_RECORDS, 720)
        committed = json.loads(phase14.ORDER_PATH.read_text(encoding="utf-8"))
        phase14.validate_execution_order(committed)
        self.assertEqual(committed, phase14.derive_execution_order())

    def test_pair_members_are_adjacent_and_mode_first_is_randomized(self) -> None:
        records, _ = phase14._execution_records()
        first_modes: set[str] = set()
        for index in range(0, len(records), 2):
            first, second = records[index : index + 2]
            self.assertEqual(first["pair_key"], second["pair_key"])
            self.assertEqual(
                {first["graph_mode"], second["graph_mode"]},
                {"eager", "cuda_graph"},
            )
            self.assertEqual(first["pair_member_index"], 0)
            self.assertEqual(second["pair_member_index"], 1)
            self.assertEqual(first["order_index"] + 1, second["order_index"])
            first_modes.add(str(first["graph_mode"]))
        self.assertEqual(first_modes, {"eager", "cuda_graph"})

    def test_execution_order_tampering_fails_closed(self) -> None:
        value = phase14.derive_execution_order()
        tampered = copy.deepcopy(value)
        tampered["pair_orders"][0]["pairs"][0]["mode_order"].reverse()
        with self.assertRaises(phase14.Phase14Error):
            phase14.validate_execution_order(tampered)

    def test_top_context_mapping_never_exceeds_model_limit(self) -> None:
        self.assertEqual(phase14.actual_historical_context(131072), 131071)
        self.assertEqual(phase14.actual_historical_context(65536), 65536)
        with self.assertRaises(phase14.Phase14Error):
            phase14.actual_historical_context(131073)

    def test_pair_feasibility_is_symmetric_and_complete(self) -> None:
        rows = phase14.build_feasibility_records(phase14.derive_execution_order())
        self.assertEqual(len(rows), 720)
        self.assertEqual(
            sum(row["status"] == "pair_feasible" for row in rows), 660
        )
        self.assertEqual(
            sum(row["status"] == "pair_capacity_infeasible" for row in rows),
            60,
        )
        by_pair: dict[str, list[dict[str, object]]] = {}
        for row in rows:
            by_pair.setdefault(str(row["pair_key"]), []).append(row)
        self.assertEqual(len(by_pair), 360)
        for pair in by_pair.values():
            self.assertEqual(len(pair), 2)
            self.assertEqual(len({str(row["status"]) for row in pair}), 1)
        top_b4 = [
            row
            for row in rows
            if row["batch_size"] == 4 and row["context_label"] == 131072
        ]
        self.assertEqual(len(top_b4), 60)
        self.assertTrue(
            all(row["status"] == "pair_capacity_infeasible" for row in top_b4)
        )

    def test_prefix_catalog_selects_only_exact_phase14_feasible_states(self) -> None:
        index = phase14._source_prefix_index()
        self.assertEqual(len(index), 110)
        self.assertTrue(
            all(batch in (1, 4) and label in phase14.CONTEXT_LABELS for _, batch, label in index)
        )
        self.assertTrue(
            all((configuration, 4, 131072) not in index for configuration in phase14.CONFIGURATIONS)
        )

    def test_plan_freezes_mechanism_only_claim_boundaries(self) -> None:
        text = phase14.PLAN_PATH.read_text(encoding="utf-8")
        for required in (
            "20260826",
            "20260827",
            "20260828",
            "CV <= 0.03",
            "quality_status = unvalidated",
            "performance_claim_eligible = false",
            "Full Scan remains CLOSED",
            "Phase 15 is deferred",
        ):
            self.assertIn(required, text)


class Phase14MechanismTests(unittest.TestCase):
    def test_eager_untimed_audit_retains_inference_backend_context(self) -> None:
        source = inspect.getsource(phase14._run_worker)
        self.assertIn(
            "with torch.inference_mode(), forced_flash_execution():",
            source,
        )
        self.assertIn("operation_callable = eager_operation", source)
        self.assertIn('run_artifact_root / "setup-audit.json"', source)

    def test_eager_passthrough_executes_without_graph_claim(self) -> None:
        calls: list[int] = []
        eager = phase14._EagerPassthrough(lambda: calls.append(1) or 7)
        self.assertEqual(eager.replay(), 7)
        self.assertEqual(calls, [1])
        self.assertEqual(
            eager.to_dict(),
            {
                "captured": False,
                "fallback": False,
                "warmup_steps": 0,
                "graph_mode": "eager",
                "phase14_eager_passthrough": True,
            },
        )

    @staticmethod
    def _stable_inputs() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        summaries: list[dict[str, object]] = []
        records: list[dict[str, object]] = []
        for configuration in phase14.CONFIGURATIONS:
            for batch in phase14.BATCH_SIZES:
                for label in phase14.CONTEXT_LABELS:
                    for mode in phase14.GRAPH_MODES:
                        wall = 10.0 if mode == "eager" else 8.0
                        cuda = 9.0 if mode == "eager" else 7.0
                        summaries.append(
                            {
                                "method_config_id": configuration,
                                "batch_size": batch,
                                "context_label": label,
                                "graph_mode": mode,
                                "disposition": "stable",
                                "median_ms": wall,
                                "cuda_median_ms": cuda,
                            }
                        )
                        for replicate in range(3):
                            records.append(
                                {
                                    "method_config_id": configuration,
                                    "batch_size": batch,
                                    "context_label": label,
                                    "graph_mode": mode,
                                    "replicate_index": replicate,
                                    "status": "completed",
                                    "output_checksum": "a" * 64,
                                    "semantic_kernel_path_fingerprint": "b" * 64,
                                    "cache_identity_fingerprint": "c" * 64,
                                }
                            )
        return summaries, records

    def test_ab_aggregation_requires_cross_mode_semantic_agreement(self) -> None:
        summaries, records = self._stable_inputs()
        pairs = phase14._pair_records(summaries, records)
        self.assertEqual(len(pairs), 120)
        self.assertTrue(all(row["pair_status"] == "stable" for row in pairs))
        self.assertTrue(all(row["graph_ratio"] == 1.25 for row in pairs))
        self.assertTrue(all(row["quality_status"] == "unvalidated" for row in pairs))
        self.assertTrue(
            all(row["performance_claim_eligible"] is False for row in pairs)
        )
        records[3]["output_checksum"] = "d" * 64
        failed = phase14._pair_records(summaries, records)
        self.assertEqual(failed[0]["pair_status"], "failed")
        self.assertIsNone(failed[0]["graph_ratio"])

    def test_graph_effect_interpretation_is_threshold_bound_and_nonclaiming(self) -> None:
        fits: list[dict[str, object]] = []
        pairs: list[dict[str, object]] = []
        for configuration in phase14.CONFIGURATIONS:
            for batch in phase14.BATCH_SIZES:
                fits.extend(
                    (
                        {
                            "method_config_id": configuration,
                            "batch_size": batch,
                            "graph_mode": "eager",
                            "fit_status": "knee_observed",
                            "tau": 12.0,
                            "s": 1.0,
                            "L_star": 20000.0,
                        },
                        {
                            "method_config_id": configuration,
                            "batch_size": batch,
                            "graph_mode": "cuda_graph",
                            "fit_status": "knee_observed",
                            "tau": 10.0,
                            "s": 1.0,
                            "L_star": 18000.0,
                        },
                    )
                )
                pairs.append(
                    {
                        "method_config_id": configuration,
                        "batch_size": batch,
                        "pair_status": "stable",
                        "wall_minus_gpu_eager_ms": 2.0,
                        "wall_minus_gpu_graph_ms": 1.0,
                        "output_agreement": True,
                        "kernel_path_agreement": True,
                        "cache_identity_agreement": True,
                    }
                )
        effects = phase14._graph_effects(fits, pairs)
        self.assertEqual(len(effects), 20)
        self.assertTrue(
            all(row["launch_floor_interpretation_supported"] for row in effects)
        )
        self.assertTrue(all(row["direct_launch_gap_measured"] is False for row in effects))
        self.assertTrue(
            all(row["performance_claim_eligible"] is False for row in effects)
        )

    def test_cv_threshold_and_unstable_classification_are_frozen(self) -> None:
        stable = phase14.pilot.point_statistics([1.0, 1.0, 1.0])
        unstable = phase14.pilot.point_statistics([1.0, 1.0, 1.2])
        self.assertEqual(
            phase14.pilot.classify_point(
                statistics_record=stable, agreements=True
            ),
            "stable",
        )
        self.assertEqual(
            phase14.pilot.classify_point(
                statistics_record=unstable, agreements=True
            ),
            "unstable",
        )


if __name__ == "__main__":
    unittest.main()
