from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

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

    def test_prefix_sources_are_strict_but_logical_inputs_are_complete(self) -> None:
        index = phase16.prefix_index(container_paths=False)
        incompatible = phase16.restore_incompatible_prefix_index(
            container_paths=False
        )
        self.assertEqual(len(index), 50)
        self.assertEqual(len(incompatible), 282)
        self.assertEqual(len(phase16.logical_prefix_specs()), 59)
        self.assertTrue(all(item["kind"] == "snapshot" for item in index.values()))
        self.assertTrue(
            all(
                configuration == "bf16" or item["source"] == "phase16g"
                for (configuration, _, _), item in index.items()
            )
        )
        self.assertTrue(
            all(
                item["readability"] == "readable_historical_layout"
                and item["restore_compatibility"] == "restore_incompatible"
                for item in incompatible.values()
            )
        )

    def test_snapshot_restore_policy_preserves_strict_layout_identity(self) -> None:
        mode, exact = phase16.snapshot_restore_policy(
            current={"snapshot_root": "/exact"}, legacy_incompatible=None
        )
        self.assertEqual(mode, "optional_exact_snapshot")
        self.assertEqual(exact["restore_compatibility"], "restore_allowed")
        mode, legacy = phase16.snapshot_restore_policy(
            current=None,
            legacy_incompatible={
                "snapshot_root": "/legacy",
                "readability": "readable_historical_layout",
                "restore_compatibility": "restore_incompatible",
            },
        )
        self.assertEqual(mode, "logical_reconstruct")
        self.assertEqual(legacy["restore_compatibility"], "restore_incompatible")
        with self.assertRaises(phase16.Phase16FullScanError):
            phase16.snapshot_restore_policy(
                current={"snapshot_root": "/exact"},
                legacy_incompatible={"snapshot_root": "/legacy"},
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
        preservation = phase16.preserved_timing_semantics_hashes()
        self.assertTrue(preservation["unchanged"])
        self.assertTrue(
            all(item["unchanged"] for item in preservation["files"].values())
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

    def test_verified_remote_promotion_retains_index_before_eviction(self) -> None:
        temporary = Path(tempfile.mkdtemp(prefix="phase16-promotion-test."))
        family = temporary / "phase16-20260831t000000000000z-12345678-abcdef"
        segment = family / "segments" / "replicate-0"
        try:
            segment.mkdir(parents=True)
            (family / "retained-segments").mkdir()
            (family / "publication").mkdir()
            (family / "family-reservation.json").write_text(
                json.dumps({"execution_git_sha": "1" * 40}), encoding="utf-8"
            )
            for name in (
                "segment_manifest.json",
                "segment-result.json",
                "inventory.json",
                "run_index.parquet",
                "point_records.parquet",
                "exclusions.parquet",
                "manifest.json",
                "artifact_inventory.json",
                "checksums.sha256",
                "COMPLETE",
            ):
                (segment / name).write_bytes(name.encode("utf-8"))
            (segment / "raw").mkdir()
            (segment / "raw" / "payload").write_bytes(b"raw")
            publication = {
                "root_sha256": "a" * 64,
                "r2_uri": "r2://bucket/segment/",
                "object_count": 12,
            }
            with (
                mock.patch.object(
                    phase16, "_publication_record", return_value=publication
                ),
                mock.patch.object(
                    phase16,
                    "validate_local_artifact",
                    return_value=SimpleNamespace(root_sha256="a" * 64),
                ),
                mock.patch.object(
                    phase16,
                    "_segment_records",
                    return_value=[{"r_hbm": None}] * phase16.LOGICAL_POINTS,
                ),
                mock.patch.object(
                    phase16,
                    "_read_parquet",
                    return_value=[{}] * phase16.LOGICAL_POINTS,
                ),
            ):
                receipt = phase16.promote_segment_remote(
                    family_root=family, replicate=0
                )
            self.assertTrue(receipt["local_raw_staging_evicted"])
            self.assertFalse(segment.exists())
            retained = family / "retained-segments" / "replicate-0"
            self.assertTrue((retained / "run_index.parquet").is_file())
            self.assertTrue((retained / "remote-authoritative.json").is_file())
        finally:
            for path in sorted(temporary.rglob("*"), reverse=True):
                try:
                    path.chmod(0o755 if path.is_dir() else 0o644)
                except FileNotFoundError:
                    pass
            shutil.rmtree(temporary)


if __name__ == "__main__":
    unittest.main()
