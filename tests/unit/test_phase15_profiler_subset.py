"""Focused Phase 15 profiler-subset tests."""

from __future__ import annotations

import copy
import csv
import inspect
import io
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

    def test_ncu_wide_csv_parser_uses_exact_selected_columns(self) -> None:
        metric_map = phase15.resolve_metric_map(self._metric_text(), "sections")
        metrics = metric_map["metric_names"]
        header = ["ID", "Kernel Name", "Context", "Stream", *metrics]
        units = ["", "", "", "", *(
            "byte" if "bytes" in metric else "sector" if "sectors" in metric else "ns"
            for metric in metrics
        )]
        values = ["7", "flash_fwd", "1", "9", *("4" for _ in metrics)]
        stream = io.StringIO()
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerows((header, units, values))
        parsed = phase15.parse_ncu_csv(stream.getvalue(), metric_map)
        self.assertEqual(len(parsed), len(metrics))
        self.assertEqual({row["kernel_id"] for row in parsed}, {"7"})
        self.assertEqual({row["kernel_name"] for row in parsed}, {"flash_fwd"})
        self.assertEqual({row["metric_name"] for row in parsed}, set(metrics))

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
        self.assertEqual(
            phase15.classify_kernel(
                "at::reduce_kernel<MeanOps<float>>", configuration="bf16"
            )[0],
            "other_model",
        )
        self.assertEqual(
            phase15.classify_kernel(
                "indexSelectSmallIndex", configuration="bf16"
            )[0],
            "other_model",
        )

    def test_method_authority_kernel_classification_is_family_scoped(self) -> None:
        cases = (
            ("_tq_decode_stage1", "tq_4bit_nc", "dense_cache_attention"),
            ("_fwd_kernel_stage2", "tq_4bit_nc", "output_merge"),
            ("_tq_fused_store_mse", "tq_4bit_nc", "quantize"),
            ("bgemv4_kernel_outer_dim", "k4v4", "dense_cache_attention"),
            (
                "VecQuant4MatMulKernelNUQPerChannelTransposedMHABatchedFusedOptDeterministicTiles",
                "kvq4",
                "dense_cache_attention",
            ),
            (
                "VecQuant4MatMulKernelNUQPerChannelTransposedMHABatchedFusedOptDeterministicReduce",
                "kvq4",
                "output_merge",
            ),
            ("VecQuant4AppendVecKSparse", "kvq4", "cache_append"),
            (
                "SelectFixedOutliers1024Cap12Kernel",
                "kvq4",
                "kvquant_sparse_selection",
            ),
        )
        for symbol, configuration, expected in cases:
            with self.subTest(symbol=symbol, configuration=configuration):
                self.assertEqual(
                    phase15.classify_kernel(symbol, configuration=configuration)[0],
                    expected,
                )
        self.assertEqual(
            phase15.classify_kernel("_tq_decode_stage1", configuration="bf16")[0],
            "unknown",
        )

    def test_ncu_reclassification_conserves_recorded_totals(self) -> None:
        rows = [
            {
                "kernel_id": "1",
                "kernel_name": "_tq_decode_stage1",
                "kernel_role": "unknown",
                "classification_basis": "insufficient_unambiguous_evidence",
                "dram_read_bytes": 100.0,
                "dram_write_bytes": 20.0,
                "l2_read_bytes": 64.0,
                "l2_write_bytes": 32.0,
            }
        ]
        recorded = {
            "total_decode_dram_read_bytes": 100.0,
            "total_decode_dram_write_bytes": 20.0,
            "total_decode_dram_bytes": 120.0,
            "total_decode_l2_read_bytes": 64.0,
            "total_decode_l2_write_bytes": 32.0,
            "total_decode_l2_bytes": 96.0,
        }
        events, summary, roles = phase15.reclassify_kernel_events(
            rows, configuration="tq_4bit_nc", recorded_summary=recorded
        )
        self.assertEqual(events[0]["recorded_kernel_role"], "unknown")
        self.assertEqual(events[0]["kernel_role"], "dense_cache_attention")
        self.assertEqual(summary["cache_path_dram_bytes"], 120.0)
        self.assertEqual(summary["unclassified_dram_bytes"], 0.0)
        self.assertEqual(roles, {"dense_cache_attention": 120.0})
        tampered = {**recorded, "total_decode_dram_bytes": 121.0}
        with self.assertRaises(phase15.Phase15Error):
            phase15.reclassify_kernel_events(
                rows, configuration="tq_4bit_nc", recorded_summary=tampered
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

        ncu = phase15._profiler_command(
            record={"run_kind": "ncu"},
            worker=["python", "worker.py"],
            raw_base=Path("/tmp/raw"),
            metric_map={"metric_names": ["dram__bytes_op_read.sum"]},
        )
        self.assertIn(f"--nvtx-include={phase15.NVTX_RANGE}/", ncu)
        self.assertIn("--disable-extra-suffixes", ncu)

    def test_ncu_launch_failure_is_kernel_replay_failure(self) -> None:
        self.assertEqual(
            phase15._classify_profiler_failure(
                run_kind="ncu",
                returncode=9,
                stderr="counter library: LaunchFailed",
                raw_exists=True,
            ),
            "kernel_replay_failed",
        )

    def test_ncu_replay_pass_count_is_explicit_and_consistent(self) -> None:
        self.assertEqual(
            phase15.parse_ncu_replay_pass_count(
                '==PROF== Profiling "kernel" - 0: 100% - 10 passes\n'
                '==PROF== Profiling "other" - 1: 100% - 10 passes\n'
            ),
            10,
        )
        for text in ("no pass evidence", "- 9 passes\n- 10 passes\n"):
            with self.subTest(text=text):
                with self.assertRaises(phase15.Phase15Error):
                    phase15.parse_ncu_replay_pass_count(text)

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
            "cpu_cuda_submission_call_count": 8,
            "synchronization_call_count": 1,
            "kernel_order_sha256": "a",
            "overlap_total_ns": 1,
        }
        graph = {
            **base,
            "cpu_submission_interval_ns": 3,
            "kernel_count": 5,
            "graph_launch_count": 1,
            "cpu_cuda_submission_call_count": 1,
        }
        effect = phase15.nsys_pair_effect(base, graph)
        self.assertEqual(effect["delta_cpu_submission_interval"], 7.0)
        self.assertEqual(effect["kernel_count_change"], 1)
        self.assertEqual(effect["graph_launch_change"], 1)
        self.assertEqual(effect["cpu_cuda_submission_call_reduction"], 7)


class Phase15GovernanceTests(unittest.TestCase):
    def test_dependency_free_plots_are_complete_and_append_only(self) -> None:
        pair = {
            "method_config_id": "bf16",
            "context_label": 4096,
            "delta_cpu_submission_interval": 1.0,
            "delta_gpu_inter_kernel_idle_total": 2.0,
            "delta_synchronization_time": -3.0,
        }
        traffic = {
            "method_config_id": "bf16",
            "common_same_work": True,
            "cache_path_dram_bytes": 10.0,
            "total_decode_dram_bytes": 20.0,
            "l2_hit_rate": 50.0,
            "sm_activity": 40.0,
            "achieved_occupancy": 30.0,
            "dram_bytes_by_role": {"dense_cache_attention": 10.0},
        }
        amplification = {
            "method_config_id": "bf16",
            "rho_alloc": 1.0,
            "rho_hbm": 1.0,
            "A_traffic": 1.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "plots").mkdir()
            phase15._plot_outputs_svg(
                root,
                nsys_pairs=[pair],
                traffic=[traffic],
                amplifications=[amplification],
            )
            plots = sorted((root / "plots").glob("*.svg"))
            self.assertEqual(len(plots), 10)
            original = plots[0].read_bytes()
            phase15._plot_outputs_svg(
                root,
                nsys_pairs=[pair],
                traffic=[traffic],
                amplifications=[amplification],
            )
            self.assertEqual(plots[0].read_bytes(), original)
            plots[0].write_text("tampered", encoding="utf-8")
            with self.assertRaises(phase15.Phase15Error):
                phase15._plot_outputs_svg(
                    root,
                    nsys_pairs=[pair],
                    traffic=[traffic],
                    amplifications=[amplification],
                )

    def test_ncu_continuation_attempts_are_append_only_and_contiguous(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory)
            profile_id = "p15-ncu-bf16-b1-l4096-graph"
            (stage / "runs").mkdir()
            self.assertEqual(phase15.next_profile_attempt(stage, profile_id), 0)
            incomplete = stage / "runs" / f"{profile_id}-attempt0"
            incomplete.mkdir()
            (incomplete / "request.json").write_text("{}\n", encoding="utf-8")
            self.assertEqual(phase15.next_profile_attempt(stage, profile_id), 1)
            for attempt in (0, 1):
                run = stage / "runs" / f"{profile_id}-attempt{attempt}"
                run.mkdir(exist_ok=True)
                (run / "manifest.json").write_text(
                    json.dumps({"profile_id": profile_id, "attempt": attempt}),
                    encoding="utf-8",
                )
                self.assertEqual(
                    phase15.next_profile_attempt(stage, profile_id), attempt + 1
                )

            gap = stage / "runs" / f"{profile_id}-attempt3"
            gap.mkdir()
            (gap / "manifest.json").write_text(
                json.dumps({"profile_id": profile_id, "attempt": 3}),
                encoding="utf-8",
            )
            with self.assertRaises(phase15.Phase15Error):
                phase15.next_profile_attempt(stage, profile_id)

    def test_ncu_continuation_allowlist_is_profiler_only(self) -> None:
        self.assertEqual(
            phase15._CONTINUATION_ALLOWED_SOURCE_CHANGES,
            {
                "scripts/phase15_profiler_subset.py",
                "tests/unit/test_phase15_profiler_subset.py",
            },
        )

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
