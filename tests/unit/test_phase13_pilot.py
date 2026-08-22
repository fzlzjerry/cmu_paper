"""Focused tests for the preregistered Phase 13 Pilot contract."""

from __future__ import annotations

import copy
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from scripts import phase13_pilot


ROOT = Path(__file__).resolve().parents[2]


class Phase13PilotTests(unittest.TestCase):
    def test_key_chunk_packing_matches_independent_scalar_control(self) -> None:
        import torch

        valid = 2
        for bits in (4, 3, 2):
            levels = 1 << bits
            packed_rows = bits * 128 // 32
            codes = (
                torch.arange(valid * 8 * 128, dtype=torch.int64)
                .reshape(valid, 8 * 128)
                .remainder(levels)
            )
            workspace = SimpleNamespace(
                key_codes=codes.clone(),
                key_packed_long=torch.empty(
                    (valid, 8, packed_rows), dtype=torch.int64
                ),
                key_shifted=torch.empty(
                    (valid, 8, packed_rows), dtype=torch.int64
                ),
                packed_words=torch.empty(
                    (8, packed_rows, valid), dtype=torch.int32
                ),
            )
            phase13_pilot._pack_kvquant_key_codes_out(
                workspace, valid=valid, bits=bits
            )
            first = workspace.packed_words.clone()
            expected = torch.zeros(
                (valid, 8, packed_rows), dtype=torch.int64
            )
            source = codes.view(valid, 8, 128)
            for row in range(valid):
                for head in range(8):
                    for native_index in range(128):
                        value = int(source[row, head, native_index])
                        bit_offset = native_index * bits
                        word = bit_offset // 32
                        shift = bit_offset % 32
                        expected[row, head, word] += value << shift
                        if shift + bits > 32:
                            expected[row, head, word + 1] += value >> (32 - shift)
            expected_i32 = expected.permute(1, 2, 0).to(torch.int32)
            self.assertTrue(torch.equal(first, expected_i32))
            phase13_pilot._pack_kvquant_key_codes_out(
                workspace, valid=valid, bits=bits
            )
            self.assertTrue(torch.equal(first, workspace.packed_words))

    def test_kvquant_prefix_chunks_are_fixed_bounded_and_fully_accounted(self) -> None:
        expected_bytes = {
            "kvq4": 11_369_600,
            "kvq3": 7_089_280,
            "kvq2": 4_908_160,
        }
        for configuration, expected in expected_bytes.items():
            specification = phase13_pilot.kvquant_prefix_chunk_workspace_spec(
                configuration
            )
            self.assertEqual(specification["chunk_tokens"], 128)
            self.assertEqual(specification["workspace_bytes"], expected)
            self.assertEqual(
                sum(specification["components"].values()), expected
            )
            self.assertFalse(specification["batch_scaled"])
            self.assertFalse(specification["context_scaled"])
            self.assertFalse(specification["full_context_fp32_copy"])
        source = inspect.getsource(
            phase13_pilot._kvquant_chunked_store_prefill
        )
        self.assertIn("range(0, quantized_tokens, chunk)", source)
        self.assertNotIn("range(cache.sink_tokens, tokens)", source)
        self.assertNotIn(".float().contiguous()", source)
        self.assertNotIn("appendvecKsparseParallel", source)

    def test_prefix_allocator_is_exact_child_only_and_fail_closed(self) -> None:
        base = {"BASE": "preserved"}
        with mock.patch.object(
            phase13_pilot.phase12,
            "_child_environment",
            return_value=dict(base),
        ):
            child = phase13_pilot._prefix_builder_child_environment()
        self.assertEqual(base, {"BASE": "preserved"})
        self.assertEqual(child["BASE"], "preserved")
        self.assertEqual(
            child[
                phase13_pilot.PREFIX_BUILDER_ALLOCATOR_ENVIRONMENT_VARIABLE
            ],
            phase13_pilot.PREFIX_BUILDER_ALLOCATOR_CONFIGURATION,
        )

        variable = phase13_pilot.PREFIX_BUILDER_ALLOCATOR_ENVIRONMENT_VARIABLE
        with mock.patch.dict(
            os.environ,
            {
                variable: phase13_pilot.PREFIX_BUILDER_ALLOCATOR_CONFIGURATION
            },
            clear=False,
        ):
            phase13_pilot._require_prefix_builder_allocator()
        with mock.patch.dict(os.environ, {variable: "wrong"}, clear=False):
            with self.assertRaisesRegex(
                phase13_pilot.Phase13PilotError,
                "allocator authority differs",
            ):
                phase13_pilot._require_prefix_builder_allocator()

    def test_formal_timing_worker_does_not_inherit_prefix_allocator(self) -> None:
        builder_source = inspect.getsource(
            phase13_pilot._run_prefix_builder_process
        )
        timing_source = inspect.getsource(phase13_pilot._run_one_process)
        self.assertIn("_prefix_builder_child_environment()", builder_source)
        self.assertIn("phase12._child_environment()", timing_source)
        self.assertNotIn("_prefix_builder_child_environment()", timing_source)

    def test_feasibility_includes_exact_kvquant_prefix_chunk_bytes(self) -> None:
        record = phase13_pilot.feasibility_record(
            {
                "method_config_id": "kvq4",
                "batch_size": 8,
                "historical_context": 16384,
                "context_label": 16384,
            }
        )
        specification = phase13_pilot.kvquant_prefix_chunk_workspace_spec(
            "kvq4"
        )
        self.assertEqual(
            record["kvquant_prefix_chunk_workspace"], specification
        )
        self.assertEqual(
            record["kvquant_prefix_chunk_workspace_bytes"],
            specification["workspace_bytes"],
        )
        recomposed = (
            record["model_weight_bytes"]
            + record["cache_allocated_bytes"]
            + record["endpoint_workspace_bytes"]
            + record["prefix_control_tensor_bytes"]
            + record["prefix_compute_peak_bytes"]
            + record["kvquant_prefix_chunk_workspace_bytes"]
            + record["graph_pool_or_capture_reserve_bytes"]
        )
        self.assertEqual(record["predicted_required_bytes"], recomposed)
        self.assertLessEqual(recomposed, record["limit_bytes"])

    def test_seed_materialization_is_exact_append_only_and_hardlinked(self) -> None:
        entry = {
            "snapshot_id": "prefix-kvq4-b8-l16384",
            "state_file_sha256": "a" * 64,
            "state_file_bytes": 5,
            "source_layout_fingerprint": "b" * 64,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seed = root / "seed"
            state = seed / "states" / entry["snapshot_id"]
            state.mkdir(parents=True)
            (state / "state.safetensors").write_bytes(b"state")
            (state / "manifest.json").write_bytes(b"{}\n")
            (state / "COMPLETE").write_bytes(b"c" * 64 + b"\n")
            destination = root / "campaign"
            with mock.patch.object(
                phase13_pilot,
                "validate_persistent_prefix_seed",
                return_value={"entries": [entry]},
            ):
                result = phase13_pilot.materialize_persistent_prefix_seed(
                    seed_root=seed,
                    destination=destination,
                    git_sha="d" * 40,
                )
            target = (
                destination
                / "catalog"
                / "states"
                / entry["snapshot_id"]
                / "state.safetensors"
            )
            self.assertEqual(result["reused_snapshot_count"], 1)
            self.assertFalse(result["timing_samples_reused"])
            self.assertEqual(
                target.stat().st_ino,
                (state / "state.safetensors").stat().st_ino,
            )
            with mock.patch.object(
                phase13_pilot,
                "validate_persistent_prefix_seed",
                return_value={"entries": [entry]},
            ):
                with self.assertRaisesRegex(
                    phase13_pilot.Phase13PilotError, "already exists"
                ):
                    phase13_pilot.materialize_persistent_prefix_seed(
                        seed_root=seed,
                        destination=destination,
                        git_sha="d" * 40,
                    )

    def test_persistent_seed_control_tampering_fails_closed(self) -> None:
        snapshot_id = "prefix-kvq4-b8-l16384"
        plan = {
            "snapshot_id": snapshot_id,
            "method_config_id": "kvq4",
            "method_family": "kvquant",
            "source_batch": 8,
            "batch_size": 8,
            "historical_context": 16384,
            "context_label": 16384,
            "method_config_fingerprint": "f" * 64,
        }
        manifest = {
            "state_file_sha256": "a" * 64,
            "state_file_bytes": 5,
            "source_layout_fingerprint": "b" * 64,
        }
        entry = {
            **plan,
            **manifest,
            "source_execution_git_sha": (
                phase13_pilot.PERSISTENT_PREFIX_SEED_EXECUTION_GIT_SHA
            ),
            "state_bytes_verified": True,
        }
        payload = {
            "schema_version": "kvbench-phase13-prefix-seed-1.0.0",
            "seed_id": phase13_pilot.PERSISTENT_PREFIX_SEED_ID,
            "source_campaign_id": "stopped",
            "source_campaign_status": "stopped_non_claim_bearing",
            "source_execution_git_sha": (
                phase13_pilot.PERSISTENT_PREFIX_SEED_EXECUTION_GIT_SHA
            ),
            "authorized_container_digest": (
                phase13_pilot.AUTHORIZED_CONTAINER_DIGEST
            ),
            "snapshot_count": 1,
            "state_file_bytes": 5,
            "full_state_bytes_verified": True,
            "verified_at": "2026-08-22T00:00:00Z",
            "timing_samples_reusable": False,
            "prefix_states_reusable": True,
            "reuse_policy": "exact_configuration_batch_context_only",
            "recompute_allowed": False,
            "entries": [entry],
        }
        with tempfile.TemporaryDirectory() as temporary:
            seed = Path(temporary) / "seed"
            state = seed / "states" / snapshot_id
            state.mkdir(parents=True)
            (state / "state.safetensors").write_bytes(b"state")
            catalog = seed / "seed-catalog.json"
            catalog.write_bytes(phase13_pilot.json_bytes(payload))
            catalog_sha = phase13_pilot.sha256_file(catalog)
            ledger = seed / "checksums.sha256"
            ledger.write_text(
                f"{catalog_sha}  seed-catalog.json\n", encoding="ascii"
            )
            ledger_sha = phase13_pilot.sha256_file(ledger)
            complete = seed / "COMPLETE"
            complete.write_text(
                phase13_pilot._prefix_seed_complete_digest(
                    catalog_sha256=catalog_sha,
                    ledger_sha256=ledger_sha,
                )
                + "\n",
                encoding="ascii",
            )
            patches = (
                mock.patch.object(
                    phase13_pilot,
                    "PERSISTENT_PREFIX_SEED_EXPECTED_COUNT",
                    1,
                ),
                mock.patch.object(
                    phase13_pilot,
                    "PERSISTENT_PREFIX_SEED_MANIFEST_SHA256",
                    catalog_sha,
                ),
                mock.patch.object(
                    phase13_pilot,
                    "_prefix_seed_expected_plan",
                    return_value={snapshot_id: plan},
                ),
                mock.patch.object(
                    phase13_pilot,
                    "validate_prefix_state",
                    return_value=manifest,
                ),
            )
            with patches[0], patches[1], patches[2], patches[3]:
                self.assertEqual(
                    phase13_pilot.validate_persistent_prefix_seed(seed)[
                        "snapshot_count"
                    ],
                    1,
                )
                ledger.write_bytes(b"0" * 64 + b"  seed-catalog.json\n")
                with self.assertRaisesRegex(
                    phase13_pilot.Phase13PilotError, "ledger differs"
                ):
                    phase13_pilot.validate_persistent_prefix_seed(seed)
                ledger.write_text(
                    f"{catalog_sha}  seed-catalog.json\n", encoding="ascii"
                )
                complete.write_bytes(b"0" * 64 + b"\n")
                with self.assertRaisesRegex(
                    phase13_pilot.Phase13PilotError, "COMPLETE differs"
                ):
                    phase13_pilot.validate_persistent_prefix_seed(seed)
                complete.write_text(
                    phase13_pilot._prefix_seed_complete_digest(
                        catalog_sha256=catalog_sha,
                        ledger_sha256=phase13_pilot.sha256_file(ledger),
                    )
                    + "\n",
                    encoding="ascii",
                )
                catalog.write_bytes(phase13_pilot.json_bytes({**payload, "x": 1}))
                with self.assertRaisesRegex(
                    phase13_pilot.Phase13PilotError, "manifest differs"
                ):
                    phase13_pilot.validate_persistent_prefix_seed(seed)

    def test_prefix_builder_validates_and_fsyncs_before_finalization(self) -> None:
        source = inspect.getsource(phase13_pilot._build_prefix_state_worker)
        validated = source.index("validated = validate_prefix_state(")
        fsynced = source.index("_fsync_prefix_builder_snapshot(output)")
        completed = source.index('recorder.record("finalization", "completed")')
        self.assertLess(validated, fsynced)
        self.assertLess(fsynced, completed)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "snapshot"
            root.mkdir()
            (root / "state.safetensors").write_bytes(b"state")
            (root / "manifest.json").write_bytes(b"{}\n")
            (root / "COMPLETE").write_bytes(b"a" * 64 + b"\n")
            with mock.patch.object(
                phase13_pilot.os, "fsync", wraps=os.fsync
            ) as fsync:
                phase13_pilot._fsync_prefix_builder_snapshot(root)
            self.assertEqual(fsync.call_count, 5)

    def test_prefix_builder_success_flushes_result_then_uses_os_exit(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        payload = {"snapshot_id": "prefix-bf16-b4-l49152", "status": "PASS"}
        with mock.patch.object(
            phase13_pilot.sys, "stdout", stdout
        ), mock.patch.object(
            phase13_pilot.sys, "stderr", stderr
        ), mock.patch.object(
            phase13_pilot.os, "_exit", side_effect=RuntimeError("exit sentinel")
        ) as exit_call:
            with self.assertRaisesRegex(RuntimeError, "exit sentinel"):
                phase13_pilot._emit_prefix_builder_result_and_exit(payload)
        exit_call.assert_called_once_with(0)
        self.assertEqual(
            stdout.getvalue(),
            phase13_pilot.PREFIX_BUILDER_PREFIX
            + json.dumps(payload, sort_keys=True, separators=(",", ":"))
            + "\n",
        )

    def test_prefix_builder_os_exit_is_reaped_with_exact_result(self) -> None:
        payload = {"snapshot_id": "unit-prefix", "status": "PASS"}
        command = (
            sys.executable,
            "-c",
            "from scripts.phase13_pilot import "
            "_emit_prefix_builder_result_and_exit as emit; "
            f"emit({payload!r})",
        )
        completed = subprocess.run(
            command,
            cwd=phase13_pilot.REPOSITORY_ROOT,
            env=dict(os.environ),
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(
            completed.stdout,
            phase13_pilot.PREFIX_BUILDER_PREFIX
            + json.dumps(payload, sort_keys=True, separators=(",", ":"))
            + "\n",
        )

    def test_prefix_builder_flush_failure_never_reports_success(self) -> None:
        class FailingFlush(io.StringIO):
            def flush(self) -> None:
                raise OSError("injected flush failure")

        with mock.patch.object(
            phase13_pilot.sys, "stdout", FailingFlush()
        ), mock.patch.object(
            phase13_pilot.sys, "stderr", io.StringIO()
        ), mock.patch.object(phase13_pilot.os, "_exit") as exit_call:
            with self.assertRaisesRegex(OSError, "injected flush failure"):
                phase13_pilot._emit_prefix_builder_result_and_exit(
                    {"status": "PASS"}
                )
        exit_call.assert_not_called()

    def test_prefix_builder_worker_failure_exits_nonzero_normally(self) -> None:
        arguments = [
            "--build-prefix-state",
            "--snapshot-id",
            "prefix-bf16-b4-l49152",
            "--configuration",
            "bf16",
            "--source-batch",
            "4",
            "--context-label",
            "49152",
            "--git-sha",
            "a" * 40,
            "--build-root",
            "/tmp/build",
            "--output",
            "/tmp/output",
        ]
        with mock.patch.object(
            phase13_pilot,
            "_build_prefix_state_worker",
            side_effect=phase13_pilot.Phase13PilotError(
                "injected builder failure"
            ),
        ), mock.patch.object(
            phase13_pilot, "_emit_prefix_builder_result_and_exit"
        ) as emit:
            with self.assertRaisesRegex(
                phase13_pilot.Phase13PilotError, "injected builder failure"
            ):
                phase13_pilot.main(arguments)
        emit.assert_not_called()

    def test_exact_grid_configuration_set_and_top_context(self) -> None:
        self.assertEqual(len(phase13_pilot.CONFIGURATIONS), 10)
        self.assertEqual(phase13_pilot.BATCH_SIZES, (1, 4, 8))
        self.assertEqual(len(phase13_pilot.CONTEXT_LABELS), 9)
        self.assertEqual(phase13_pilot.actual_historical_context(131072), 131071)
        self.assertEqual(phase13_pilot.actual_historical_context(98304), 98304)
        self.assertNotIn("turboquant_k8v4", phase13_pilot.CONFIGURATIONS)
        self.assertNotIn("k4v2", phase13_pilot.CONFIGURATIONS)

    def test_execution_order_is_deterministic_complete_and_tamper_evident(self) -> None:
        first = phase13_pilot.derive_execution_order()
        second = phase13_pilot.derive_execution_order()
        self.assertEqual(first, second)
        self.assertEqual(len(first["records"]), 810)
        self.assertEqual(first["seeds"], [20260805, 20260806, 20260807])
        phase13_pilot.validate_execution_order(first)
        tampered = copy.deepcopy(first)
        tampered["records"][0]["batch_size"] = 1
        with self.assertRaisesRegex(
            phase13_pilot.Phase13PilotError, "execution order differs"
        ):
            phase13_pilot.validate_execution_order(tampered)

    def test_committed_order_is_exact(self) -> None:
        path = ROOT / phase13_pilot.ORDER_PATH
        payload = json.loads(path.read_text(encoding="utf-8"))
        phase13_pilot.validate_execution_order(payload)
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "d64fc06cda5d6f594ea82247eca5b3a400b39dc32605b93f308461591641d35a",
        )

    def test_feasibility_has_every_record_and_never_masks_geometry(self) -> None:
        records = phase13_pilot.build_feasibility_records(
            phase13_pilot.derive_execution_order()
        )
        self.assertEqual(len(records), 810)
        self.assertEqual(sum(item["status"] == "feasible" for item in records), 684)
        self.assertEqual(
            sum(item["status"] == "capacity_infeasible" for item in records), 126
        )
        unsupported = [
            item for item in records if not item["adapter_geometry_supported_at_entry"]
        ]
        self.assertEqual(unsupported, [])
        self.assertTrue(
            all(
                item["unsupported_geometry_is_not_reclassified_as_capacity"]
                for item in records
            )
        )
        target = [
            item
            for item in records
            if item["method_config_id"] == "tq_3bit_nc"
            and item["batch_size"] == 8
            and item["context_label"] == 98304
        ]
        self.assertEqual(len(target), 3)
        self.assertTrue(
            all(item["status"] == "capacity_infeasible" for item in target)
        )

    def test_phase13b_successors_and_allocation_formulas_are_exact(self) -> None:
        authority = phase13_pilot.validate_phase13b_entry()
        self.assertEqual(authority["decision"], "0030")
        self.assertEqual(authority["clean_retrieval"], "PASS")
        self.assertEqual(set(authority["families"]), {"turboquant", "kivi", "kvquant"})
        q4 = authority["families"]["kvquant"]["q4_successor"]
        self.assertEqual(q4["decision"], "0036")
        self.assertEqual(
            q4["report_sha256"],
            "75605637f460a309081e1e0a4065e90e8e14194d365250cb513092616ef89ec7",
        )
        self.assertEqual(q4["clean_retrieval"], "PASS")
        matrix = json.loads(
            (ROOT / "docs/evidence/phase13b/cuda-validation.json").read_text(
                encoding="utf-8"
            )
        )
        for record in matrix["records"]:
            configuration = record["configuration"]
            batch = record["batch_size"]
            current_allocated = phase13_pilot.cache_allocated_bytes(
                configuration, batch, 129
            )
            historical_allocated = record["accounting"]["allocated_bytes"]
            if configuration == "kvq4":
                historical_fixed_workspace = batch * 32 * 32 * 128 * 4
                current_capacity_workspace = (
                    phase13_pilot.kvquant_q4_value_decode_workspace_bytes(
                        batch_size=batch,
                        total_attended_capacity=129,
                    )
                )
                self.assertEqual(
                    current_allocated,
                    historical_allocated
                    - historical_fixed_workspace
                    + current_capacity_workspace,
                    f"{configuration}/B{batch}",
                )
            else:
                self.assertEqual(
                    current_allocated,
                    historical_allocated,
                    f"{configuration}/B{batch}",
                )
            family = record["method_family"]
            geometry_key = f"{configuration}/B{batch}"
            successor = authority["families"][family]
            self.assertEqual(
                successor["adapter_config_fingerprints_l128"][geometry_key],
                record["adapter_config_fingerprint"],
            )
            self.assertEqual(
                successor["cache_layout_fingerprints_l128"][geometry_key],
                record["cache_layout_fingerprint"],
            )

    def test_pilot_mounts_exact_successor_bundles_read_only(self) -> None:
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        bindings = (
            (
                "PHASE13B",
                "phase13b_bundle",
                "artifacts/phase13b/"
                "phase13b-20260801t143138050263z-b862af64-batch-admission",
                "f1c96eaacbbace1c23b249d1afe8d892aa26c3f6b8d04e07f373a2becafba1fe",
            ),
            (
                "PHASE13RQ4",
                "phase13rq4_bundle",
                "artifacts/phase13rq4/"
                "phase13rq4-20260820t094629495794z-ab4e0b84-b8c7bd",
                "9f027d64424844d0d62311daad5740e2b76960d1d5e7b8be26aa1ccba100a8db",
            ),
        )
        for authority, variable, bundle, root in bindings:
            self.assertIn(
                f"override {authority}_LOCAL_BUNDLE := $(CURDIR)/{bundle}",
                makefile,
            )
            self.assertIn(
                f"override {authority}_LOCAL_ROOT_SHA256 := {root}",
                makefile,
            )
            self.assertIn(
                f"--mount \"type=bind,src=$${variable},"
                f"dst=/home/rockrock/cmu_paper/{bundle},readonly\"",
                makefile,
            )
            self.assertIn(
                f'mkdir -p "$$task_root/repository/{bundle}"',
                makefile,
            )
            self.assertNotIn(
                f"--mount \"type=bind,src=$${variable},"
                f"dst=/home/rockrock/cmu_paper/{bundle}\"",
                makefile,
            )
        self.assertIn(
            'validate_local_artifact(sys.argv[1], environ={}).root_sha256',
            makefile,
        )

    def test_cv_uses_sample_standard_deviation_and_frozen_boundary(self) -> None:
        equal = phase13_pilot.point_statistics((1.0, 1.0, 1.0))
        self.assertEqual(equal["cv"], 0.0)
        boundary = {"cv": 0.03}
        self.assertEqual(
            phase13_pilot.classify_point(
                statistics_record=boundary, agreements=True
            ),
            "stable",
        )
        self.assertEqual(
            phase13_pilot.classify_point(
                statistics_record={"cv": 0.0300001}, agreements=True
            ),
            "unstable",
        )
        self.assertEqual(
            phase13_pilot.classify_point(
                statistics_record=boundary, agreements=False
            ),
            "failed",
        )
        with self.assertRaisesRegex(
            phase13_pilot.Phase13PilotError, "exactly three"
        ):
            phase13_pilot.point_statistics((1.0, 1.0))

    def test_fit_and_density_fail_closed(self) -> None:
        insufficient = phase13_pilot.provisional_knee_fit(
            ((4096.0, 1.0), (8192.0, 1.1), (16384.0, 1.2))
        )
        self.assertEqual(insufficient["fit_status"], "insufficient_feasible_span")
        fitted = phase13_pilot.provisional_knee_fit(
            (
                (4096.0, 1.0),
                (8192.0, 1.0),
                (16384.0, 1.2),
                (32768.0, 1.8),
                (65536.0, 3.0),
            )
        )
        self.assertIn(
            fitted["fit_status"],
            {"knee_observed", "knee_below_range", "knee_above_range"},
        )
        self.assertIn("residuals", fitted["knee_model"])
        density = phase13_pilot.knee_density((4096, 8192, 16384), 32768.0)
        self.assertFalse(density["sufficient"])
        self.assertIsNotNone(density["missing_interval"])

    def test_bootstrap_uses_process_medians_and_is_deterministic(self) -> None:
        summaries = [
            {
                "context_label": context,
                "process_medians_ms": [value, value * 1.001, value * 0.999],
            }
            for context, value in (
                (4096, 1.0),
                (8192, 1.0),
                (16384, 1.2),
                (32768, 1.8),
                (65536, 3.0),
            )
        ]
        first = phase13_pilot._session_bootstrap_knee(
            summaries, seed=20260801, draws=100
        )
        second = phase13_pilot._session_bootstrap_knee(
            summaries, seed=20260801, draws=100
        )
        self.assertEqual(first, second)
        self.assertTrue(first["estimable"])
        self.assertEqual(first["valid_draws"], 100)

    def test_svg_plot_contains_data_and_claim_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "plot.svg"
            phase13_pilot._svg_line_plot(
                path,
                title="Pilot",
                y_label="milliseconds",
                series={"bf16": [(4096.0, 1.0), (8192.0, 2.0)]},
            )
            rendered = path.read_text(encoding="utf-8")
        self.assertIn("<path", rendered)
        self.assertIn("bf16", rendered)
        self.assertIn("quality unvalidated", rendered)
        self.assertNotIn("No eligible", rendered)

    def test_governance_and_historical_phase12_evidence_are_unchanged(self) -> None:
        config = json.loads((ROOT / "configs/plans/pilot.yaml").read_text())
        self.assertEqual(config["admission"]["full_scan_state"], "closed")
        self.assertEqual(config["quality"]["quality_execution"], "locked")
        self.assertFalse(config["quality"]["performance_data_frozen"])
        self.assertEqual(config["measurement"]["seed"], 20260721)
        self.assertEqual(
            hashlib.sha256((ROOT / "configs/plans/pilot.yaml").read_bytes()).hexdigest(),
            "6eb8ee48a9a569d0378879e40a6c4c965ad568e0dbf025b4ed9ca69f7ab39ea1",
        )
        self.assertEqual(
            hashlib.sha256(
                (ROOT / "docs/evidence/phase12/unified-admission.json").read_bytes()
            ).hexdigest(),
            "3e337e883baaf055d307e97f25a001fb0c9a5b8a8bc14dab6230fd3d8823b4bb",
        )


if __name__ == "__main__":
    unittest.main()
