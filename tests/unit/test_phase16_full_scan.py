from __future__ import annotations

import json
from pathlib import Path
import unittest

from scripts import phase16_full_scan as phase16


class Phase16FullScanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.points = phase16.logical_points()
        cls.feasibility = phase16.feasibility_records()
        cls.orders = phase16.derive_execution_orders()

    def test_exact_configuration_batch_and_grid_counts(self) -> None:
        self.assertEqual(len(phase16.CONFIGURATIONS), 10)
        self.assertEqual(phase16.BATCH_SIZES, (1, 2, 4, 8, 16))
        self.assertEqual(len([p for p in self.points if p["grid_source"] == "base"]), 450)
        self.assertEqual(
            len([p for p in self.points if p["grid_source"] == "phase13d_adaptive"]),
            84,
        )
        self.assertEqual(len(self.points), 534)
        self.assertEqual(534 * 5, 2670)

    def test_top_context_mapping(self) -> None:
        self.assertEqual(phase16.actual_historical_context(131072), 131071)
        self.assertEqual(phase16.actual_historical_context(65536), 65536)
        with self.assertRaises(phase16.Phase16FullScanError):
            phase16.actual_historical_context(True)
        with self.assertRaises(phase16.Phase16FullScanError):
            phase16.actual_historical_context(131073)

    def test_worker_context_scope_accepts_exact_adaptive_label_and_restores(self) -> None:
        original = phase16.phase13.actual_historical_context
        with phase16._worker_context_label_override(4480):
            self.assertEqual(phase16.phase13.actual_historical_context(4480), 4480)
            with self.assertRaises(phase16.Phase16FullScanError):
                phase16.phase13.actual_historical_context(4096)
        self.assertIs(phase16.phase13.actual_historical_context, original)

    def test_worker_context_scope_preserves_top_context_mapping(self) -> None:
        with phase16._worker_context_label_override(131072):
            self.assertEqual(
                phase16.phase13.actual_historical_context(131072), 131071
            )
        with self.assertRaises(phase16.Phase16FullScanError):
            with phase16._worker_context_label_override(4097):
                pass

    def test_feasibility_is_exact_and_r_hbm_is_not_populated(self) -> None:
        self.assertEqual(sum(r["status"] == "feasible" for r in self.feasibility), 441)
        self.assertEqual(
            sum(r["status"] == "capacity_infeasible" for r in self.feasibility), 93
        )
        self.assertTrue(all(r.get("r_hbm") is None for r in self.feasibility))
        self.assertTrue(all(r["max_memory_fraction"] == 0.88 for r in self.feasibility))

    def test_execution_orders_are_deterministic_and_complete(self) -> None:
        self.assertEqual(self.orders, phase16.derive_execution_orders())
        self.assertEqual(self.orders["seeds"], list(phase16.SEEDS))
        self.assertEqual(len(self.orders["segments"]), 5)
        self.assertTrue(all(len(s["records"]) == 534 for s in self.orders["segments"]))
        for segment in self.orders["segments"]:
            self.assertEqual(
                [r["order_index"] for r in segment["records"]], list(range(534))
            )

    def test_prefix_sources_are_exact_and_missing_points_are_direct(self) -> None:
        index = phase16.prefix_index(container_paths=False)
        self.assertEqual(len(index), 50)
        direct = sum(
            record["status"] == "feasible"
            and (
                record["method_config_id"],
                record["batch_size"],
                record["context_label"],
            )
            not in index
            for record in self.feasibility
        )
        self.assertEqual(direct, 391)
        self.assertTrue(all(item["kind"] == "snapshot" for item in index.values()))
        self.assertTrue(
            all(
                configuration == "bf16" or item["source"] == "phase16g"
                for (configuration, _, _), item in index.items()
            )
        )

    def test_legacy_compressed_prefixes_fail_layout_source_match(self) -> None:
        entries = phase16._prefix_catalog_entries(
            phase16.PHASE13_BASE_PREFIX_CATALOG
        )
        by_family = {entry["method_family"]: entry for entry in entries}
        self.assertTrue(
            phase16._legacy_prefix_layout_source_matches_current(by_family["bf16"])
        )
        for family in ("turboquant", "kivi", "kvquant"):
            self.assertFalse(
                phase16._legacy_prefix_layout_source_matches_current(
                    by_family[family]
                )
            )

    def test_container_prefix_reads_use_mounted_roots(self) -> None:
        source = Path("scripts/phase16_full_scan.py").read_text(encoding="utf-8")
        self.assertIn("read_root = mounted_root if container_paths else host_root", source)
        self.assertIn("geometry_root.glob", source)

    def test_failure_classification_does_not_reclassify_correctness(self) -> None:
        self.assertEqual(
            phase16._worker_failure_status("CUDA out of memory", None)[0],
            "allocation_failed",
        )
        self.assertEqual(
            phase16._worker_failure_status("output mismatch", None)[0],
            "output_mismatch",
        )
        self.assertEqual(
            phase16._worker_failure_status("anything", "measurement")[0],
            "infrastructure_failed",
        )

    def test_phase16_timeout_scaling_accepts_new_and_adaptive_geometry(self) -> None:
        b2 = phase16._phase16_stage_timeout_contract(batch=2, historical=131071)
        b16 = phase16._phase16_stage_timeout_contract(batch=16, historical=24576)
        adaptive = phase16._phase16_stage_timeout_contract(batch=1, historical=4480)
        self.assertEqual(b2["prefix_construction"], 67336.0)
        self.assertEqual(b16["graph_capture"], 21461.0)
        self.assertEqual(adaptive["warmup_and_audit"], 10800.0)
        with self.assertRaises(phase16.Phase16FullScanError):
            phase16._phase16_stage_timeout_contract(batch=3, historical=4096)

    def test_phase16g_admits_every_requested_geometry(self) -> None:
        authority = phase16.load_phase16g_authority()["report"]
        for configuration in phase16.CONFIGURATIONS:
            for batch in phase16.BATCH_SIZES:
                key = phase16.require_admitted_geometry(
                    authority, configuration=configuration, batch_size=batch
                )
                self.assertEqual(key, f"{configuration}/B{batch}")

    def test_order_json_round_trip_is_canonical(self) -> None:
        rendered = json.loads(
            json.dumps(self.orders, sort_keys=True, separators=(",", ":"))
        )
        phase16.validate_execution_orders(rendered)

    def test_timing_critical_hash_manifest_is_complete(self) -> None:
        authority = phase16.timing_critical_hashes()
        self.assertEqual(set(authority["files"]), set(phase16.TIMING_CRITICAL_PATHS))
        self.assertEqual(len(authority["files_sha256"]), 64)
        self.assertEqual(
            authority["authorized_container_digest"], phase16.PHASE16G_CONTAINER_DIGEST
        )

    def test_replacement_link_is_explicit_and_r_hbm_null(self) -> None:
        record = dict(self.orders["segments"][0]["records"][0])
        manifest = phase16._run_manifest(
            family_id="phase16-20260830t000000000000z-12345678-abcdef",
            segment_id="segment",
            run_id="replacement",
            record=record,
            status="infrastructure_failed",
            reason="test",
            result_path=None,
            replacement_of="original",
        )
        self.assertEqual(manifest["replacement_of"], "original")
        self.assertIsNone(manifest["r_hbm"])
        self.assertFalse(manifest["selective_rerun"])

    def test_phase15_features_are_scoped_not_extrapolated(self) -> None:
        source = Path("scripts/phase16_full_scan.py").read_text(encoding="utf-8")
        self.assertIn('"feature_scope": "phase15_common_point_only"', source)
        self.assertIn('"extrapolation_permitted": False', source)


if __name__ == "__main__":
    unittest.main()
