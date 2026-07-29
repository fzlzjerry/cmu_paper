"""CPU-only tests for the narrow Phase 11 KVQuant admission driver."""

from __future__ import annotations

import dataclasses
import inspect
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from kvbench.errors import SchemaValidationError
from kvbench.runtime.allocation_attribution import (
    AllocationGeometry,
    instantiate_decision_0013_direct_compressed_rules,
    instantiate_decision_0013_phase8_kivi_rules,
)
from kvbench.runtime.artifacts import (
    AppendOnlyArtifactStore,
    ArtifactStateError,
    validate_run_directory,
)
from kvbench.schema import GraphMode, RunnerKind, RunStatus
from kvbench.schema.phase11 import (
    PHASE11_AUTHORIZED_CONTAINER_DIGEST,
    PHASE11_BOUNDED_POINT_SIGNATURES,
    PHASE11_EXTENSION_SHA256,
    PHASE11_FIXTURE_ROOT,
    Phase11RunManifest,
)
from scripts import phase11_kvquant_admission as driver


_REQUIRED_COMPLETED_PAYLOADS = (
    "accounting/contexts.json",
    "allocation/audit.json",
    "config/authority.json",
    "environment/container_identity.json",
    "execution-path/audit.json",
    "gqa/audit.json",
    "numerical/fixture-conformance.json",
    "validation/admission-candidate.json",
    "validation/bounded-grid.json",
    "validation/cuda-graph.json",
    "validation/sanitizer.json",
)


class Phase11KVQuantAdmissionDriverTests(unittest.TestCase):
    def test_grid_is_exactly_the_ordered_nine_point_contract(self) -> None:
        observed = tuple(
            (
                configuration,
                runner.value,
                graph.value,
                context,
                output_steps,
            )
            for configuration, runner, graph, context, output_steps in (
                PHASE11_BOUNDED_POINT_SIGNATURES
            )
        )
        self.assertEqual(
            observed,
            (
                ("kvq4", "fixed_l", "eager", 128, 1),
                ("kvq4", "fixed_l", "cuda_graph", 128, 1),
                ("kvq3", "fixed_l", "eager", 128, 1),
                ("kvq3", "fixed_l", "cuda_graph", 128, 1),
                ("kvq2", "fixed_l", "eager", 128, 1),
                ("kvq2", "fixed_l", "cuda_graph", 128, 1),
                ("kvq4", "fixed_l", "eager", 4096, 1),
                ("kvq4", "fixed_l", "cuda_graph", 4096, 1),
                ("kvq4", "growing_context", "eager", 17, 4),
            ),
        )

    def test_manifest_binds_authority_and_remains_a_nonclaim(self) -> None:
        manifest = driver._manifest(
            run_id="phase11-driver-unit-manifest",
            git_sha="1" * 40,
            created_at_utc="2026-07-30T00:00:00Z",
        )
        self.assertIs(manifest.status, RunStatus.CREATED)
        self.assertEqual(
            manifest.authority.authorized_container_digest,
            PHASE11_AUTHORIZED_CONTAINER_DIGEST,
        )
        self.assertEqual(
            manifest.authority.extension_sha256,
            PHASE11_EXTENSION_SHA256,
        )
        self.assertEqual(
            manifest.authority.fixture_root,
            PHASE11_FIXTURE_ROOT,
        )
        self.assertFalse(manifest.git_dirty)
        self.assertFalse(manifest.performance_claim_eligible)
        self.assertFalse(manifest.performance_data_frozen)
        self.assertFalse(manifest.quality_benchmark_executed)
        self.assertFalse(manifest.speedup_calculated)
        self.assertIsNone(manifest.r_hbm)
        self.assertEqual(
            Phase11RunManifest.from_dict(manifest.to_dict()),
            manifest,
        )

    def test_manifest_rejects_dirty_git_or_authority_drift(self) -> None:
        manifest = driver._manifest(
            run_id="phase11-driver-unit-rejection",
            git_sha="2" * 40,
            created_at_utc="2026-07-30T00:00:00Z",
        )
        with self.assertRaises(SchemaValidationError):
            Phase11RunManifest.from_dict(
                {**manifest.to_dict(), "git_dirty": True}
            )
        authority = manifest.authority.to_dict()
        authority["fixture_root"] = "f" * 64
        with self.assertRaises(SchemaValidationError):
            Phase11RunManifest.from_dict(
                {**manifest.to_dict(), "authority": authority}
            )

    def test_completed_bundle_requires_the_exact_evidence_payloads(self) -> None:
        initial = driver._manifest(
            run_id="phase11-driver-unit-artifact",
            git_sha="3" * 40,
            created_at_utc="2026-07-30T00:00:00Z",
        )
        with tempfile.TemporaryDirectory() as directory:
            store = AppendOnlyArtifactStore(Path(directory) / "phase11")
            run = store.create(initial.run_id, initial)
            run.start()
            incomplete = dataclasses.replace(
                initial,
                status=RunStatus.COMPLETED,
                started_at_utc="2026-07-30T00:00:01Z",
                finished_at_utc="2026-07-30T00:00:02Z",
                inventory_path="artifact_inventory.json",
            )
            with self.assertRaisesRegex(
                ArtifactStateError,
                "required evidence",
            ):
                run.finalize(incomplete)

        with tempfile.TemporaryDirectory() as directory:
            store = AppendOnlyArtifactStore(Path(directory) / "phase11")
            run = store.create(initial.run_id, initial)
            run.start()
            for relative in _REQUIRED_COMPLETED_PAYLOADS:
                run.write_json(relative, {"unit_test": True})
            final = run.finalize(incomplete)
            validation = validate_run_directory(final)
            self.assertTrue(validation.valid)
            self.assertTrue(validation.complete)
            self.assertEqual(validation.status, "completed")

    def test_child_environment_is_allowlisted_and_secret_free(self) -> None:
        values = {
            name: "credential-placeholder"
            for name in driver._HOST_ONLY_R2_ENVIRONMENT
        }
        values.update(
            {
                "KVBENCH_KVQUANT_EXTENSION": "/authority/quant.so",
                "KVBENCH_KVQUANT_FRESH_BUILD_EXTENSION": "/fresh/quant.so",
                "UNRELATED_HOST_VALUE": "must-not-be-inherited",
            }
        )
        with mock.patch.dict(os.environ, values, clear=False):
            child = driver._child_environment()
        self.assertTrue(
            all(name not in child for name in driver._HOST_ONLY_R2_ENVIRONMENT)
        )
        self.assertNotIn("UNRELATED_HOST_VALUE", child)
        self.assertEqual(
            child["KVBENCH_KVQUANT_EXTENSION"],
            "/authority/quant.so",
        )
        self.assertEqual(
            child["KVBENCH_KVQUANT_FRESH_BUILD_EXTENSION"],
            "/fresh/quant.so",
        )

    def test_memcheck_requires_zero_errors_and_zero_leaks(self) -> None:
        clean = (
            b"========= LEAK SUMMARY: 0 bytes leaked in 0 allocations\n"
            b"========= ERROR SUMMARY: 0 errors\n"
        )
        self.assertTrue(driver._zero_memcheck(clean, b""))
        self.assertFalse(
            driver._zero_memcheck(
                b"========= ERROR SUMMARY: 1 error\n",
                b"",
            )
        )
        self.assertFalse(
            driver._zero_memcheck(
                b"========= LEAK SUMMARY: 8 bytes leaked in 1 allocation\n"
                b"========= ERROR SUMMARY: 0 errors\n",
                b"",
            )
        )

    def test_sanitizer_probe_matches_result_mode_not_top_level(self) -> None:
        stdout = (
            b"========= COMPUTE-SANITIZER\n"
            b'{"results":[{"mode":"kvq4-cap"}],"status":"PASS"}\n'
            b"========= LEAK SUMMARY: 0 bytes leaked in 0 allocations\n"
            b"========= ERROR SUMMARY: 0 errors\n"
        )
        probe = driver._last_json(stdout)
        self.assertTrue(driver._zero_memcheck(stdout, b""))
        self.assertTrue(
            driver._sanitizer_probe_matches(probe, "kvq4-cap")
        )
        self.assertFalse(
            driver._sanitizer_probe_matches(probe, "kvq3-distinct")
        )
        self.assertFalse(
            driver._sanitizer_probe_matches(
                {
                    "status": "PASS",
                    "mode": "kvq4-cap",
                    "results": [],
                },
                "kvq4-cap",
            )
        )
        self.assertFalse(
            driver._sanitizer_probe_matches(
                {
                    "status": "PASS",
                    "results": [
                        {"mode": "kvq4-cap"},
                        {"mode": "kvq4-cap"},
                    ],
                },
                "kvq4-cap",
            )
        )
        self.assertFalse(
            driver._sanitizer_probe_matches(
                {"status": "PASS", "results": "kvq4-cap"},
                "kvq4-cap",
            )
        )

    def test_code_object_check_requires_sm120_cubin_and_ptx(self) -> None:
        extension = Path("/tmp/unit-quant.so")
        results = (
            mock.Mock(
                returncode=0,
                stdout=b"ELF file .sm_120.cubin",
                stderr=b"",
            ),
            mock.Mock(
                returncode=0,
                stdout=b".target sm_120",
                stderr=b"",
            ),
        )
        with (
            mock.patch.object(Path, "is_file", return_value=True),
            mock.patch.object(
                driver.subprocess,
                "run",
                side_effect=results,
            ),
        ):
            evidence = driver._validate_sm120_code_objects(extension)
        self.assertTrue(evidence["native_sm120"])
        self.assertTrue(evidence["sm_120_cubin"])
        self.assertTrue(evidence["compute_120_ptx"])

    def test_accounting_includes_exact_endpoint_rope_scratch(self) -> None:
        breakdown = {
            "dense_k_payload": 10,
            "dense_v_payload": 10,
            "key_metadata": 10,
            "value_metadata": 10,
            "key_sparse_values": 10,
            "key_sparse_indices": 10,
            "value_sparse_values": 10,
            "value_sparse_indices": 10,
            "active_count_mask": 10,
            "sink_k": 10,
            "sink_v": 10,
            "staging": 10,
            "padding_alignment": 0,
            "persistent_workspace": 10,
        }
        cache_bytes = sum(breakdown.values())
        cache = SimpleNamespace(
            capacity=17,
            byte_breakdown=lambda: dict(breakdown),
            accounting=lambda: SimpleNamespace(
                allocated_bytes=cache_bytes,
                predicted_tensor_bytes=cache_bytes,
                temporary_peak_bytes=0,
            ),
            logical_bf16_storage_bytes=1_000_000,
            active_storage_bytes=(
                lambda active_context, key_active_entries: cache_bytes
            ),
            active_logical_bf16_bytes=lambda active_context: 500_000,
        )
        accounting = driver._byte_accounting(
            cache,
            configuration="kvq4",
            active_context=17,
            endpoint_rope_scratch_bytes=163_840,
        )
        self.assertEqual(
            accounting.allocated_bytes,
            cache_bytes + 163_840,
        )
        self.assertEqual(
            accounting.breakdown.staging,
            breakdown["staging"] + 163_840,
        )
        self.assertEqual(
            accounting.breakdown.total,
            accounting.allocated_bytes,
        )
        self.assertEqual(accounting.predicted_relative_error, 0.0)

    def test_grid_setup_and_audit_use_forced_flash_context(self) -> None:
        source = inspect.getsource(driver._execute_grid_point)
        context = source.index("with forced_flash_execution():")
        build = source.index("build_kvquant_endpoint_session(")
        audit = source.index(
            "_collect_phase11_allocation_attribution("
        )
        admit = source.index("session.admit(")
        runner = source.index("run_fixed_l(")
        self.assertLess(context, build)
        self.assertLess(build, audit)
        self.assertLess(audit, admit)
        self.assertLess(admit, runner)
        self.assertNotIn("audit_cuda_allocations(", source)
        self.assertNotIn("allocation_event_count == 0", source)

    def test_eager_audit_attributes_model_ephemerals_but_forbids_growth(
        self,
    ) -> None:
        source = inspect.getsource(
            driver._evaluate_phase11_eager_allocations
        )
        self.assertIn("permitted_allocation_policies", source)
        self.assertIn("_FORBIDDEN_ALLOCATION_CLASSES", source)
        self.assertIn("persistent_allocated_delta_nonzero", source)
        self.assertIn("persistent_reserved_delta_nonzero", source)
        self.assertIn("context_dependent_allocation_detected", source)
        self.assertNotIn("allocation_event_count == 0", source)

    def test_decision0013_rules_are_method_neutral_without_kivi_drift(
        self,
    ) -> None:
        geometry = AllocationGeometry(
            batch=1,
            query_heads=32,
            kv_heads=8,
            context=129,
            head_dim=128,
            dtype_bytes=2,
            operation_output_width=128_256,
            operation_output_dtype_bytes=4,
        )
        arguments = {
            "geometry": geometry,
            "backend_identity": "unit-backend",
            "composition_binding_sha256": "a" * 64,
        }
        neutral = instantiate_decision_0013_direct_compressed_rules(
            **arguments
        )
        kivi = instantiate_decision_0013_phase8_kivi_rules(**arguments)
        self.assertEqual(
            neutral.policy_authority,
            "decision_0013_direct_compressed_composition",
        )
        self.assertEqual(
            kivi.policy_authority,
            "decision_0013_phase8_kivi_composition",
        )
        neutral_payload = neutral.to_dict()
        kivi_payload = kivi.to_dict()
        neutral_payload.pop("policy_authority")
        kivi_payload.pop("policy_authority")
        self.assertEqual(neutral_payload, kivi_payload)

    def test_sanitizer_commands_bind_the_exact_container_digest(self) -> None:
        source = inspect.getsource(driver._run_sanitizer)
        self.assertIn('"--image-config-digest"', source)
        self.assertIn("PHASE11_AUTHORIZED_CONTAINER_DIGEST", source)

    def test_static_path_binds_split_fstring_kernel_suffixes(self) -> None:
        paths = driver._static_execution_path()
        self.assertEqual(
            tuple(item.configuration for item in paths),
            ("kvq4", "kvq3", "kvq2"),
        )
        self.assertTrue(all(item.direct_compressed_decode for item in paths))
        self.assertTrue(all(item.no_backend_fallback for item in paths))

    def test_static_path_rejects_unrelated_kernel_name_decoy(self) -> None:
        real_getsource = inspect.getsource
        expected = (
            "matmul_nuq_perchannel_transposed_"
            "rope_mha_batched_fused_opt2"
        )

        def source(value: object) -> str:
            observed = real_getsource(value)
            if value is driver.KVQuantMethodAdapter._decode_compressed:
                observed = observed.replace(
                    '"rope_mha_batched_fused_opt2"',
                    '"rope_mha_batched_fused_opt2_renamed"',
                    1,
                )
                observed = observed.replace(
                    "        torch = _torch()\n",
                    f'        "{expected}"\n        torch = _torch()\n',
                    1,
                )
            return observed

        with mock.patch.object(
            driver.inspect,
            "getsource",
            side_effect=source,
        ):
            with self.assertRaisesRegex(
                driver.Phase11KVQuantDriverError,
                "source binding is incomplete",
            ):
                driver._static_execution_path()

    def test_sanitizer_graph_probe_explicitly_releases_capture(self) -> None:
        probe = (
            Path(__file__).resolve().parents[1]
            / "cuda"
            / "phase11_kvquant_sanitizer_probe.py"
        ).read_text(encoding="utf-8")
        self.assertIn("graph.graph.reset()", probe)
        self.assertIn("finally:", probe)
        self.assertLess(
            probe.index("graph.graph.reset()"),
            probe.index("del first, first_copy, graph, second"),
        )

    def test_final_validation_cli_requires_outer_report_and_receipt(self) -> None:
        parsed = driver._parse_args(
            [
                "--validate-only",
                "--artifact",
                "/inner",
                "--outer-artifact",
                "/outer",
                "--method-admission-report",
                "/report",
                "--publication-receipt",
                "/receipt",
            ]
        )
        self.assertTrue(parsed.validate_only)
        self.assertEqual(parsed.outer_artifact, Path("/outer"))
        error = driver.main(
            [
                "--validate-only",
                "--artifact",
                "/inner",
                "--method-admission-report",
                "/report",
                "--publication-receipt",
                "/receipt",
            ]
        )
        self.assertEqual(error, 2)

    def test_admission_path_has_no_publication_or_campaign_execution(self) -> None:
        source = inspect.getsource(driver.run_admission)
        for forbidden in (
            "publish_artifact",
            "run_pilot",
            "run_full_scan",
            "collect_torch_profiler",
            "run_quality",
            "speedup =",
            "timing_replicates",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("load_frozen_model()", source)
        self.assertIn("PHASE11_BOUNDED_POINT_SIGNATURES", source)

    def test_point_run_ids_are_unique_and_valid_for_all_signatures(self) -> None:
        observed = {
            driver._point_run_id(
                "phase11-driver-unit-grid",
                index,
                signature,
            )
            for index, signature in enumerate(
                PHASE11_BOUNDED_POINT_SIGNATURES
            )
        }
        self.assertEqual(len(observed), 9)
        self.assertTrue(all(len(run_id) <= 128 for run_id in observed))
        self.assertIn(GraphMode.CUDA_GRAPH, {
            item[2] for item in PHASE11_BOUNDED_POINT_SIGNATURES
        })
        self.assertIn(RunnerKind.GROWING_CONTEXT, {
            item[1] for item in PHASE11_BOUNDED_POINT_SIGNATURES
        })


if __name__ == "__main__":
    unittest.main()
