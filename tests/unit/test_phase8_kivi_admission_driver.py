"""CPU-only contract tests for the narrow Phase 8 KIVI admission driver."""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from scripts import phase8_kivi_admission as driver
from kvbench.runtime.kivi_admission import PHASE8_ADMISSION_GRID
from kvbench.schema import GraphMode, RunnerKind, RunStatus
from kvbench.schema.phase8 import (
    PHASE8_AUTHORIZED_CONTAINER_DIGEST,
    Phase8ByteAccounting,
    Phase8ByteBreakdown,
)


def _accounting(
    capacity: int = 129,
    *,
    active_context: int | None = None,
) -> Phase8ByteAccounting:
    breakdown = Phase8ByteBreakdown(
        quantized_historical_k_payload=16,
        quantized_historical_v_payload=16,
        k_scales=4,
        k_zeros=4,
        v_scales=4,
        v_zeros=4,
        other_metadata=0,
        residual_k=16,
        residual_v=16,
        fp16_staging=16,
        quantization_staging=16,
        padding_alignment=0,
        persistent_workspace=16,
        value_rollover_shift_scratch=0,
        block_group_rounding=0,
    )
    allocated = breakdown.total
    logical = capacity * 1024
    active = capacity if active_context is None else active_context
    return Phase8ByteAccounting(
        capacity=capacity,
        active_context=active,
        allocated_bytes=allocated,
        predicted_allocated_bytes=allocated,
        active_storage_bytes=allocated,
        logical_bf16_allocated_bytes=logical,
        logical_bf16_active_bytes=active * 1024,
        rho_alloc=allocated / logical,
        r_alloc=logical / allocated,
        predicted_relative_error=0.0,
        temporary_peak_bytes=0,
        breakdown=breakdown,
        r_hbm=None,
    )


def _runner_result() -> dict[str, object]:
    return {
        "measurement_scope": "measurement_container_admission",
        "performance_claim_eligible": False,
        "timing": {
            "samples": [{"completed_operations": 1}],
            "sample_count": 1,
            "paper_claim_eligible": False,
            "measurement_scope": "native_host_admission",
        },
    }


def _supervision_evidence() -> dict[str, object]:
    return {
        "schema_version": (
            "kvbench-generic-supervised-command-result-1.0.0"
        ),
        "returncode": 0,
        "timeout": {"timed_out": False},
        "direct_child": {
            "verified": True,
            "parent_pid_verified": True,
            "start_time_ticks_verified": True,
            "process_handle_retained": True,
        },
        "final_reap": {"completed": True, "count": 1},
    }


class Phase8KIVIAdmissionDriverTests(unittest.TestCase):
    def test_grid_is_the_exact_ordered_ten_point_contract(self) -> None:
        observed = tuple(
            (
                item.configuration,
                item.runner_kind.value,
                item.graph_mode.value,
                item.context_length,
                item.output_steps,
            )
            for item in PHASE8_ADMISSION_GRID
        )
        self.assertEqual(
            observed,
            (
                ("k4v4", "fixed_l", "eager", 128, 1),
                ("k4v4", "fixed_l", "cuda_graph", 128, 1),
                ("k2v4", "fixed_l", "eager", 128, 1),
                ("k2v4", "fixed_l", "cuda_graph", 128, 1),
                ("k2v2", "fixed_l", "eager", 128, 1),
                ("k2v2", "fixed_l", "cuda_graph", 128, 1),
                ("k4v4", "fixed_l", "eager", 4096, 1),
                ("k4v4", "fixed_l", "cuda_graph", 4096, 1),
                ("k4v4", "growing_context", "eager", 31, 4),
                ("k4v2", "fixed_l", "eager", 128, 1),
            ),
        )

    def test_manifest_is_strict_non_claim_container_admission(self) -> None:
        manifest = driver._manifest(
            run_id="phase8-kivi-driver-unit-manifest",
            git_sha="8" * 40,
            configuration="k4v4",
            runner_kind=RunnerKind.FIXED_L,
            graph_mode=GraphMode.EAGER,
            context_length=128,
            output_steps=1,
            cache_layout_fingerprint="1" * 64,
            method_fingerprint="2" * 64,
            accounting=_accounting(active_context=128),
            created_at_utc="2026-07-27T00:00:00Z",
        )
        self.assertIs(manifest.status, RunStatus.CREATED)
        self.assertEqual(
            manifest.authorized_container_digest,
            PHASE8_AUTHORIZED_CONTAINER_DIGEST,
        )
        self.assertEqual(manifest.capacity, 129)
        self.assertEqual(manifest.accounting.active_context, 128)
        self.assertFalse(manifest.performance_claim_eligible)
        self.assertFalse(manifest.speedup_calculated)
        self.assertFalse(manifest.quality_benchmark_executed)
        self.assertIsNone(manifest.accounting.r_hbm)

    def test_method_identity_comes_from_canonical_config_and_roles(self) -> None:
        config = driver._load_kivi_method_config()
        self.assertEqual(config.source_revision, driver.KIVI_OFFICIAL_COMMIT)
        roles = {
            variant.variant_id: variant.role.value
            for variant in config.variants
        }
        self.assertEqual(
            roles,
            {
                "k4v4": "main",
                "k2v4": "main",
                "k2v2": "main",
                "k4v2": "held_out",
            },
        )
        self.assertEqual(
            len(driver._method_config_fingerprint("k4v2")),
            64,
        )
        method = driver._canonical_factory_method("k4v2")
        self.assertEqual(method.config_name, "k4v2")

    def test_cache_accounting_maps_every_physical_category_once(self) -> None:
        values = {
            "quantized_k_payload": 16,
            "quantized_v_payload": 16,
            "key_scales": 4,
            "key_zero_points": 4,
            "value_scales": 4,
            "value_zero_points": 4,
            "other_metadata": 0,
            "residual_k": 16,
            "residual_v": 16,
            "fp16_staging": 16,
            "quantization_staging": 16,
            "padding_alignment": 0,
            "persistent_workspace": 16,
            "value_rollover_shift_scratch": 0,
            "block_group_rounding_bytes": 0,
        }
        allocated = sum(values.values())
        cache = SimpleNamespace(
            capacity=129,
            logical_bf16_storage_bytes=129 * 1024,
            byte_breakdown=lambda: dict(values),
            accounting=lambda: SimpleNamespace(
                allocated_bytes=allocated,
                predicted_tensor_bytes=allocated,
                temporary_peak_bytes=0,
            ),
            active_storage_bytes=lambda active: allocated,
            active_logical_bf16_bytes=lambda active: active * 1024,
        )
        accounting = driver._phase8_byte_accounting(
            cache,
            active_context=129,
        )
        self.assertEqual(accounting.breakdown.total, allocated)
        self.assertEqual(accounting.allocated_bytes, allocated)
        self.assertLessEqual(
            abs(accounting.r_alloc * accounting.rho_alloc - 1.0),
            1e-9,
        )
        self.assertIsNone(accounting.r_hbm)

    def test_nested_legacy_timing_scope_is_asserted_then_normalized(self) -> None:
        raw = _runner_result()
        normalized = driver._normalize_runner_scope(raw)
        self.assertEqual(
            normalized["timing"]["measurement_scope"],
            "measurement_container_admission",
        )
        self.assertEqual(
            normalized["timing"]["scope_normalization"],
            {
                "legacy_value_observed": "native_host_admission",
                "canonical_value": "measurement_container_admission",
                "timing_semantics_changed": False,
            },
        )
        self.assertEqual(
            raw["timing"]["measurement_scope"],
            "native_host_admission",
        )
        self.assertFalse(normalized["speedup_calculated"])

    def test_unexpected_timing_scope_or_claim_flag_fails_closed(self) -> None:
        wrong_scope = _runner_result()
        wrong_scope["timing"]["measurement_scope"] = (
            "measurement_container_admission"
        )
        with self.assertRaisesRegex(
            driver.Phase8KIVIDriverError,
            "measurement-scope",
        ):
            driver._normalize_runner_scope(wrong_scope)
        wrong_claim = _runner_result()
        wrong_claim["timing"]["paper_claim_eligible"] = True
        with self.assertRaises(driver.Phase8KIVIDriverError):
            driver._normalize_runner_scope(wrong_claim)

    def test_common_runners_use_one_engineering_sample_only(self) -> None:
        fixed_result = mock.Mock()
        fixed_result.to_dict.return_value = _runner_result()
        growing_result = mock.Mock()
        growing_result.to_dict.return_value = _runner_result()
        with (
            mock.patch.object(
                driver, "run_fixed_l", return_value=fixed_result
            ) as fixed,
            mock.patch.object(
                driver,
                "run_growing_context",
                return_value=growing_result,
            ) as growing,
        ):
            driver._run_common_runner(object(), RunnerKind.FIXED_L)
            driver._run_common_runner(
                object(), RunnerKind.GROWING_CONTEXT
            )
        fixed.assert_called_once_with(
            mock.ANY,
            measured_steps=1,
            measured_batches=1,
        )
        growing.assert_called_once_with(mock.ANY, expected_steps=4)

    def test_allocation_admission_cannot_infer_attribution_from_aggregates(
        self,
    ) -> None:
        self.assertFalse(
            hasattr(driver, "_full_model_allocation_criterion")
        )
        source = inspect.getsource(driver._audit_session)
        self.assertIn("collect_kivi_allocation_attribution", source)
        self.assertIn("raw_evidence_sha256", source)
        self.assertNotIn("allocation_event_count == 898", source)
        self.assertNotIn("fully_attributed = passed", source)

    def test_memcheck_parser_requires_zero_errors_and_zero_leaks(self) -> None:
        clean = (
            b"========= LEAK SUMMARY: 0 bytes leaked in 0 allocations\n"
            b"========= ERROR SUMMARY: 0 errors\n"
        )
        self.assertTrue(driver._memcheck_summaries_pass(clean, b""))
        leaked = (
            b"========= LEAK SUMMARY: 1 bytes leaked in 1 allocation\n"
            b"========= ERROR SUMMARY: 0 errors\n"
        )
        self.assertFalse(driver._memcheck_summaries_pass(leaked, b""))

    def test_child_process_environment_removes_host_only_r2_names(
        self,
    ) -> None:
        dummy = {
            name: "unit-test-placeholder"
            for name in driver._HOST_ONLY_R2_ENVIRONMENT
        }
        dummy["UNRELATED_HOST_VALUE"] = "must-not-be-inherited"
        with mock.patch.dict(os.environ, dummy, clear=False):
            child = driver._child_environment()
        self.assertTrue(
            all(name not in child for name in driver._HOST_ONLY_R2_ENVIRONMENT)
        )
        self.assertNotIn("UNRELATED_HOST_VALUE", child)
        self.assertEqual(
            child["PYTHONPATH"].split(os.pathsep),
            [
                str(driver.CONTAINER_PHASE3_SITE),
                str(driver.REPOSITORY_ROOT / "src"),
                str(driver.REPOSITORY_ROOT),
            ],
        )
        self.assertEqual(
            set(child).difference(driver._CHILD_ENVIRONMENT_PASSTHROUGH),
            {
                "PYTHONPATH",
                "PYTHONIOENCODING",
                "PYTHONDONTWRITEBYTECODE",
                "LANG",
                "LC_ALL",
            },
        )

    def test_driver_has_no_r2_or_profiler_or_quality_execution_path(
        self,
    ) -> None:
        source = inspect.getsource(driver)
        self.assertNotIn("r2_artifact", source)
        self.assertNotIn("collect_torch_profiler_trace", source)
        self.assertNotIn("run_quality", source)
        self.assertNotIn("run_full_scan", source)
        self.assertNotIn("speedup =", source)
        self.assertNotIn("timing_replicates", source)
        self.assertEqual(
            Path(driver.__file__).name,
            "phase8_kivi_admission.py",
        )

    def test_candidate_gates_are_derived_from_records(self) -> None:
        points = []
        run_ids = []
        for index, spec in enumerate(PHASE8_ADMISSION_GRID):
            graph = spec.graph_mode is GraphMode.CUDA_GRAPH
            run_id = f"phase8-derived-candidate-{index:02d}"
            run_ids.append(run_id)
            points.append(
                {
                    "configuration": spec.configuration,
                    "runner_kind": spec.runner_kind.value,
                    "graph_mode": spec.graph_mode.value,
                    "context_length": spec.context_length,
                    "output_steps": spec.output_steps,
                    "passed": True,
                    "cache_pointers_stable": True,
                    "native_gqa": True,
                    "rollover_active_lengths": (
                        [31, 32, 33, 34]
                        if spec.runner_kind
                        is RunnerKind.GROWING_CONTEXT
                        else None
                    ),
                    "accounting": {
                        "predicted_relative_error": 0.0,
                    },
                    "byte_breakdown_sum": 100,
                    "allocated_bytes": 100,
                    "reciprocal_product_error": 0.0,
                    "r_hbm": None,
                    "allocation": {
                        "graph_passed": True,
                        "operation_allocations": [
                            {
                                "criterion": {
                                    "passed": True,
                                    "unknown_allocation_count": 0,
                                    "persistent_allocated_delta": 0,
                                    "persistent_reserved_delta": 0,
                                    "strict_graph_zero_events": (
                                        True if graph else None
                                    ),
                                }
                            }
                        ],
                    },
                }
            )
        execution_path = SimpleNamespace(
            passed=True,
            cache_growth_detected=False,
            measured_torch_cat_detected=False,
            two_bit_kernel_verified=True,
            four_bit_kernel_verified=True,
            full_prefix_dequantization_detected=False,
            full_prefix_temporary_detected=False,
            native_gqa_indexing_verified=True,
            gqa_materialization_detected=False,
            query_head_sized_kv_temporary_detected=False,
            backend_fallback_detected=False,
        )
        keyword = {
            "git_sha": "8" * 40,
            "fixture": {
                "passed": True,
                "process_supervision": _supervision_evidence(),
            },
            "graph": {
                "passed": True,
                "process_supervision": _supervision_evidence(),
            },
            "sanitizer": {
                "passed": True,
                "probe_passed": True,
                "rollover_covered": True,
                "process_supervision": _supervision_evidence(),
            },
            "execution_path": execution_path,
            "point_records": points,
            "run_ids": run_ids,
        }
        candidate = driver._derive_local_candidate(**keyword)
        self.assertEqual(
            candidate["status"],
            "LOCAL_CHECKS_PASS_PUBLICATION_PENDING",
        )
        self.assertTrue(candidate["bounded_admission_grid"])
        self.assertFalse(candidate["derivation"]["literal_gate_overrides"])
        failed = driver._derive_local_candidate(
            **{**keyword, "fixture": {"passed": False}}
        )
        self.assertEqual(failed["status"], "LOCAL_CHECKS_FAILED")
        self.assertFalse(failed["fixture_conformance"])
        unsupervised = driver._derive_local_candidate(
            **{
                **keyword,
                "sanitizer": {
                    **keyword["sanitizer"],
                    "process_supervision": {},
                },
            }
        )
        self.assertEqual(unsupervised["status"], "LOCAL_CHECKS_FAILED")
        self.assertFalse(unsupervised["child_process_supervision"])

    def test_supervision_gate_requires_returncode_child_and_reap(self) -> None:
        valid = {"process_supervision": _supervision_evidence()}
        self.assertTrue(driver._supervision_evidence_passed(valid))
        for field, bad_value in (
            ("returncode", 1),
            ("direct_child", {"verified": False}),
            ("final_reap", {"completed": False, "count": 0}),
        ):
            evidence = _supervision_evidence()
            evidence[field] = bad_value
            self.assertFalse(
                driver._supervision_evidence_passed(
                    {"process_supervision": evidence}
                )
            )

    def test_launcher_probe_uses_two_separate_untimed_sequences(self) -> None:
        import torch

        class Record:
            def to_dict(self) -> dict[str, object]:
                return {
                    "kernel_family": "bgemv4_kernel_outer_dim",
                    "bits": 4,
                    "input_shape": [32, 1, 128],
                    "packed_shape": [8, 16, 128],
                    "metadata_shape": [8, 4, 128],
                    "output_shape": [32, 1, 128],
                    "group_size": 32,
                    "num_query_heads": 32,
                    "num_kv_heads": 8,
                }

        class Launcher:
            def __init__(self) -> None:
                self.records: list[Record] | None = None

            def begin_observation(self) -> None:
                self.records = []

            def end_observation(self) -> tuple[Record, ...]:
                assert self.records is not None
                result = tuple(self.records)
                self.records = None
                return result

        launcher = Launcher()

        def operation() -> object:
            assert launcher.records is not None
            launcher.records.extend((Record(), Record()))
            return torch.zeros((1, 1), dtype=torch.bfloat16)

        session = SimpleNamespace(
            operation_keys=(
                SimpleNamespace(
                    configuration="k4v4",
                    runner_kind=RunnerKind.FIXED_L,
                    graph_mode=GraphMode.EAGER,
                ),
            ),
            _fixed_operation=operation,
            _growing_operations=(),
            method=SimpleNamespace(
                _runtime=lambda: (None, None, launcher)
            ),
            cache_device=torch.device("cpu"),
            prepare_audit_step=lambda step: None,
        )
        with (
            mock.patch("torch.cuda.synchronize"),
            mock.patch.object(
                driver,
                "tensor_sha256_untimed",
                return_value="a" * 64,
            ),
        ):
            result = driver._capture_launcher_probe(session)
        self.assertTrue(result["passed"])
        self.assertTrue(result["stable_post_warmup_sequence"])
        self.assertTrue(result["instrumented_audit_separate"])
        self.assertFalse(result["allocation_audit_instrumented"])
        self.assertFalse(result["normal_timing_instrumented"])


if __name__ == "__main__":
    unittest.main()
