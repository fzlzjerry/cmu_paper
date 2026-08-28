"""Focused Phase 15 profiler-subset tests."""

from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from scripts import phase15_profiler_subset as phase15


ROOT = Path(__file__).resolve().parents[2]


class Phase15SelectionTests(unittest.TestCase):
    def test_frozen_selection_is_exact_and_complete(self) -> None:
        value = json.loads(phase15.SELECTION_PATH.read_text(encoding="utf-8"))
        phase15.validate_selection(value)
        self.assertEqual(value, phase15.derive_selection())
        self.assertEqual(len(value["nsys_profiles"]), 32)
        self.assertEqual(len(value["ncu_profiles"]), 22)
        self.assertEqual(
            [row["method_config_id"] for row in value["anchors"]],
            ["bf16", "tq_4bit_nc", "k4v4", "kvq4"],
        )

    def test_common_point_is_largest_all_ten_b1_intersection(self) -> None:
        common = phase15.derive_selection()["common_same_work"]
        self.assertEqual(common["batch_size"], 1)
        self.assertEqual(common["context_label"], 131072)
        self.assertEqual(common["historical_context"], 131071)
        self.assertEqual(common["total_attended_context"], 131072)
        common_ncu = [
            row
            for row in phase15.derive_selection()["ncu_profiles"]
            if row["roles"] == ["common_long"]
        ]
        self.assertEqual(len(common_ncu), 10)
        self.assertEqual(
            {row["method_config_id"] for row in common_ncu},
            set(phase15.CONFIGURATIONS),
        )

    def test_regime_fallback_is_explicit(self) -> None:
        anchors = {
            row["method_config_id"]: row
            for row in phase15.derive_selection()["anchors"]
        }
        self.assertEqual(
            anchors["bf16"]["regime_selection"],
            "fallback_no_identifiable_knee",
        )
        self.assertEqual(
            anchors["tq_4bit_nc"]["regime_selection"],
            "fallback_no_identifiable_knee",
        )
        self.assertEqual(
            anchors["k4v4"]["regime_selection"], "refined_knee_relative"
        )
        self.assertEqual(
            anchors["kvq4"]["regime_selection"], "refined_knee_relative"
        )

    def test_selection_tampering_fails_closed(self) -> None:
        value = phase15.derive_selection()
        tampered = copy.deepcopy(value)
        tampered["common_same_work"]["context_label"] = 98304
        with self.assertRaises(phase15.Phase15Error):
            phase15.validate_selection(tampered)


class Phase15MetricTests(unittest.TestCase):
    @staticmethod
    def _metric_text() -> str:
        return "\n".join(candidate[0] for candidate in phase15._METRIC_CANDIDATES.values())

    def test_metric_map_is_live_query_derived(self) -> None:
        value = phase15.resolve_metric_map(self._metric_text(), "SpeedOfLight\n")
        semantics = {row["semantic"] for row in value["selected_metrics"]}
        self.assertIn("dram_read_bytes", semantics)
        self.assertIn("dram_write_bytes", semantics)
        self.assertIn("l2_read_sectors", semantics)
        self.assertEqual(value["selection_source"], "live_ncu_query_metrics_and_sections")
        with self.assertRaises(phase15.Phase15Error):
            phase15.resolve_metric_map("gpu__time_duration.sum", "sections")

    def test_current_ncu_section_option_transition_is_narrow(self) -> None:
        source = inspect.getsource(phase15._query_ncu)
        self.assertIn("--query-metrics-mode=all", source)
        self.assertIn("--query-sections", source)
        self.assertIn("--list-sections", source)
        self.assertIn("section-query-transition.json", source)
        self.assertIn("section_inventory_only", source)
        self.assertIn("result.stdout + result.stderr", source)

    def test_ncu_csv_parser_and_byte_aggregation(self) -> None:
        metric_map = phase15.resolve_metric_map(self._metric_text(), "sections")
        header = "ID,Kernel Name,Context,Stream,Metric Name,Metric Unit,Metric Value"
        rows = [header]
        values = {
            "dram__bytes_op_read.sum": 100,
            "dram__bytes_op_write.sum": 20,
            "lts__t_sectors_op_read.sum": 4,
            "lts__t_sectors_op_write.sum": 1,
            "gpu__time_duration.sum": 50,
        }
        for metric, value in values.items():
            rows.append(f"1,flash_fwd,1,7,{metric},unit,{value}")
        parsed = phase15.parse_ncu_csv("\n".join(rows), metric_map)
        kernels, summary = phase15.aggregate_kernel_metrics(parsed, configuration="bf16")
        self.assertEqual(len(kernels), 1)
        self.assertEqual(kernels[0]["kernel_role"], "dense_cache_attention")
        self.assertEqual(summary["total_decode_dram_bytes"], 120.0)
        self.assertEqual(summary["cache_path_dram_bytes"], 120.0)
        self.assertEqual(summary["total_decode_l2_bytes"], 160.0)

    def test_kernel_classification_preserves_unknown(self) -> None:
        self.assertEqual(
            phase15.classify_kernel("totally_ambiguous_kernel", configuration="kvq4")[0],
            "unknown",
        )
        self.assertEqual(
            phase15.classify_kernel("select_fixed_outliers", configuration="kvq4")[0],
            "kvquant_sparse_selection",
        )
        self.assertEqual(
            phase15.classify_kernel("select_fixed_outliers", configuration="bf16")[0],
            "unknown",
        )

    def test_same_work_hbm_and_traffic_amplification(self) -> None:
        bf16 = {"cache_path_dram_bytes": 1000.0, "total_decode_dram_bytes": 2000.0}
        method = {"cache_path_dram_bytes": 400.0, "total_decode_dram_bytes": 1000.0}
        ratio = phase15.hbm_ratio(bf16=bf16, method=method, same_work=True)
        self.assertEqual(ratio["r_hbm"], 2.5)
        self.assertEqual(ratio["rho_hbm_cache"], 0.4)
        self.assertEqual(
            phase15.traffic_amplification(rho_hbm=0.4, rho_alloc=0.2), 2.0
        )
        self.assertIsNone(
            phase15.hbm_ratio(bf16=bf16, method=method, same_work=False)["r_hbm"]
        )


class Phase15NsysTests(unittest.TestCase):
    def test_nsys_capture_accepts_dynamic_pytorch_nvtx_string(self) -> None:
        command = phase15._profiler_command(
            record={"run_kind": "nsys"},
            worker=["python", "worker.py"],
            raw_base=Path("/tmp/raw"),
            metric_map={},
        )
        self.assertIn("--env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0", command)
        self.assertIn("--cuda-graph-trace=node", command)

    def test_nsys_parser_separates_submission_idle_and_sync(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.sqlite"
            connection = sqlite3.connect(path)
            connection.executescript(
                """
                CREATE TABLE StringIds(id INTEGER, value TEXT);
                CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, textId INTEGER);
                CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(
                    start INTEGER, end INTEGER, nameId INTEGER,
                    correlationId INTEGER, globalTid INTEGER
                );
                CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(
                    start INTEGER, end INTEGER, demangledName INTEGER,
                    correlationId INTEGER, streamId INTEGER, deviceId INTEGER
                );
                """
            )
            connection.executemany(
                "INSERT INTO StringIds VALUES (?,?)",
                [(1, phase15.NVTX_RANGE), (2, "cudaLaunchKernel"), (3, "cudaDeviceSynchronize"), (4, "flash_fwd")],
            )
            connection.execute("INSERT INTO NVTX_EVENTS VALUES (100,1000,1)")
            connection.executemany(
                "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?)",
                [(120, 130, 2, 10, 1), (400, 410, 2, 11, 1), (900, 950, 3, 12, 1)],
            )
            connection.executemany(
                "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?)",
                [(150, 300, 4, 10, 7, 0), (450, 700, 4, 11, 7, 0)],
            )
            connection.commit()
            connection.close()
            events, summary = phase15.parse_nsys_sqlite(path, configuration="bf16")
        self.assertGreater(len(events), 0)
        self.assertEqual(summary["cpu_cuda_submission_call_count"], 2)
        self.assertEqual(summary["cpu_submission_interval_ns"], 280.0)
        self.assertEqual(summary["gpu_inter_kernel_idle_total_ns"], 150)
        self.assertEqual(summary["synchronization_time_ns"], 50)
        self.assertEqual(summary["kernel_count"], 2)

    def test_pair_effect_denominators_are_explicit(self) -> None:
        base = {
            "cpu_submission_interval_ns": 10,
            "api_to_gpu_start_mean_ns": 8,
            "gpu_inter_kernel_idle_total_ns": 7,
            "synchronization_time_ns": 6,
            "kernel_count": 4,
            "graph_launch_count": 0,
            "kernel_order_sha256": "a",
            "overlap_total_ns": 1,
        }
        graph = {**base, "cpu_submission_interval_ns": 3, "kernel_count": 5, "graph_launch_count": 1}
        effect = phase15.nsys_pair_effect(base, graph)
        self.assertEqual(effect["delta_cpu_submission_interval"], 7.0)
        self.assertEqual(effect["kernel_count_change"], 1)
        self.assertEqual(effect["graph_launch_change"], 1)


class Phase15GovernanceTests(unittest.TestCase):
    def test_profiler_annotations_are_worker_only(self) -> None:
        source = inspect.getsource(phase15._run_profile_worker)
        self.assertIn("_install_profiler_adapter_nvtx_annotations", source)
        self.assertIn("replayed_adapter_fingerprint", source)
        self.assertIn("session.adapter_config_fingerprint", source)
        self.assertNotIn(
            "session.adapter_config_fingerprint != CONFIG_FINGERPRINTS", source
        )
        self.assertNotIn("run_fixed_l", source)
        normal = (ROOT / "src/kvbench/runtime/timing.py").read_text(encoding="utf-8")
        self.assertNotIn("phase15_decode", normal)
        self.assertEqual(phase15.RUN_KINDS, ("nsys", "ncu"))

    def test_plan_preserves_claim_boundaries(self) -> None:
        text = phase15.PLAN_PATH.read_text(encoding="utf-8")
        for value in (
            "mechanism-only",
            "never enter normal timing",
            "Full Scan",
            "quality execution",
            "Phase 16 is deferred",
            "r_hbm",
        ):
            self.assertIn(value, text)


if __name__ == "__main__":
    unittest.main()
