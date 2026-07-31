"""Focused tests for strict Phase 12 G1-G4 evidence aggregation."""

from __future__ import annotations

import copy
import inspect
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from kvbench.runtime.artifacts import sha256_file as actual_sha256_file
from scripts import phase12_unified_admission as phase12
from scripts.phase12_unified_admission import (
    EXPECTED_CONFIG_FINGERPRINTS,
    EXPECTED_REPORT_SHA256S,
    GATE_EVIDENCE_REQUIREMENTS,
    HELD_OUT_CONFIG_IDS,
    MAIN_CONFIG_IDS,
    Phase12UnifiedAdmissionError,
    aggregate_g1_g4,
    load_and_validate_prior_admission_evidence,
)


REPOSITORY = Path(__file__).resolve().parents[2]

EXPECTED_MAIN_CONFIG_IDS = (
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
)

EXPECTED_HELD_OUT_CONFIG_IDS = (
    "turboquant_k8v4",
    "k4v2",
)

EXPECTED_FINGERPRINTS = {
    "bf16": "81ca6a0d74727a9c8a54f14d6d222dee502ee5e9a9ab6e8c9a03b4fed98f371b",
    "tq_4bit_nc": "5b56167c81ef2042be5fa45ed4e7f8ddc60670d5611283e19e848307076c27eb",
    "tq_k3v4_nc": "ff92cd334059888584564bb84999353a698baf3b991a602a827a53aab726908e",
    "tq_3bit_nc": "0d17950c2502fe7399cf5a896efa429861eb53194060ab95a7369287889dc49a",
    "k4v4": "97289ed9c875e27013ddcf7659fc6e849b3d438c58d0d86bd3dcac5d82eefb09",
    "k2v4": "568493e09cad122088716533c954beb6b25a01209fa28016f761b2ede4930a3f",
    "k2v2": "667395cefa882efc7c54f9088e3706dcdc3ba33c8734bdf8de9e0dd8ae1124b8",
    "kvq4": "8f3ea4f49056a5c4ada715a853ec506de4b6bcab262cfd88dc5796bacc032fa0",
    "kvq3": "2f0d1a99db2e6884745b6cd54c50eedfa17744b89a3e8b2ffd840986126bd802",
    "kvq2": "eb75d6cbf8ff27365cd2799c4e0232649c94d6f094cb4d041bbe8c3ac1cda5ee",
}

EXPECTED_REPORTS = {
    "bf16": "1362fd1817b8bb5706baaa09ed6e5115789fbc4d35d394f184d0b132a0e58d22",
    "turboquant": "388e8107b649a9093491699357c8b1ad1d8e12c8c75378bce658f8a09bf9ab2a",
    "kivi": "3a4b63b9da0eab12db9a916ebdc1cffd788ea6f93678d87964a8332ae7cec83a",
    "kvquant": "9cfed618cee9514a1071392d0a2dca327dcf6acd33d81ac72cc477c7880c09e2",
}

EXPECTED_GATE_EVIDENCE = {
    "bf16": {
        "G1": ("correctness", "execution_path"),
        "G2": ("byte_accounting",),
        "G3": ("execution_path",),
        "G4": ("graph",),
    },
    "turboquant": {
        "G1": (
            "fixture_conformance",
            "store_append_correctness",
            "decode_tolerance",
            "finite_output",
            "no_backend_fallback",
        ),
        "G2": (
            "byte_accounting",
            "static_cache_skip_policy",
            "no_cache_growth",
            "no_unknown_allocation",
        ),
        "G3": (
            "no_full_prefix_dequantization",
            "no_gqa_replication",
            "no_cache_growth",
            "no_unknown_allocation",
            "no_backend_fallback",
        ),
        "G4": (
            "graph_capture_replay",
            "graph_zero_replay_allocation",
            "no_backend_fallback",
        ),
    },
    "kivi": {
        "G1": (
            "fixture_conformance",
            "token_integrity",
            "no_backend_fallback",
        ),
        "G2": (
            "byte_accounting",
            "residual_rollover",
            "static_cache",
            "no_unknown_allocation",
        ),
        "G3": (
            "no_measured_torch_cat",
            "direct_compressed_decode",
            "native_gqa",
            "no_unknown_allocation",
            "no_backend_fallback",
        ),
        "G4": (
            "graph_capture_replay",
            "graph_zero_replay_allocation",
            "no_backend_fallback",
        ),
    },
    "kvquant": {
        "G1": (
            "fixture_conformance",
            "sparse_contract",
            "sink_storage",
            "store_append_correctness",
            "execution_path",
        ),
        "G2": (
            "byte_accounting",
            "no_dynamic_or_unknown_allocation",
        ),
        "G3": (
            "direct_compressed_decode",
            "native_gqa",
            "execution_path",
            "no_dynamic_or_unknown_allocation",
            "no_host_synchronization",
        ),
        "G4": (
            "graph_capture_replay",
            "graph_zero_replay_allocation",
        ),
    },
}


def _load() -> dict[str, object]:
    return load_and_validate_prior_admission_evidence(REPOSITORY)


class Phase12PriorAdmissionEvidenceTests(unittest.TestCase):
    def test_cuda_graph_path_normalization_preserves_kernel_topology(
        self,
    ) -> None:
        template = (
            "digraph dot {\n"
            "subgraph cluster_GRAPH_ID {\n"
            '\"graph_GRAPH_ID_node_0\"[shape=\"record\" '
            'label=\"{KERNEL|symbol_A|{node handle | POINTER}}\"];\n'
            "}\n"
            "}\n"
        )
        first = phase12._normalize_cuda_graph_debug_dot(
            template.replace("GRAPH_ID", "4")
            .replace("POINTER", "0xABC0")
            .encode()
        )
        second = phase12._normalize_cuda_graph_debug_dot(
            template.replace("GRAPH_ID", "9")
            .replace("POINTER", "0xDEF0")
            .encode()
        )
        changed = phase12._normalize_cuda_graph_debug_dot(
            template.replace("symbol_A", "symbol_B")
            .replace("GRAPH_ID", "4")
            .replace("POINTER", "0xABC0")
            .encode()
        )
        self.assertEqual(first, second)
        self.assertNotEqual(first[0], changed[0])
        self.assertEqual(first[1:], (1, 1, 0))

    def test_cuda_graph_path_normalization_rejects_non_graph_input(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            Phase12UnifiedAdmissionError,
            "unrecognized structure",
        ):
            phase12._normalize_cuda_graph_debug_dot(b"not a CUDA graph\n")

    def test_frozen_configuration_and_report_authorities_are_exact(self) -> None:
        self.assertEqual(MAIN_CONFIG_IDS, EXPECTED_MAIN_CONFIG_IDS)
        self.assertEqual(HELD_OUT_CONFIG_IDS, EXPECTED_HELD_OUT_CONFIG_IDS)
        self.assertEqual(EXPECTED_CONFIG_FINGERPRINTS, EXPECTED_FINGERPRINTS)
        self.assertEqual(EXPECTED_REPORT_SHA256S, EXPECTED_REPORTS)
        self.assertEqual(GATE_EVIDENCE_REQUIREMENTS, EXPECTED_GATE_EVIDENCE)

    def test_genuine_current_reports_and_references_validate(self) -> None:
        prior = _load()
        self.assertEqual(
            tuple(prior["main_configurations"]),
            EXPECTED_MAIN_CONFIG_IDS,
        )
        self.assertEqual(
            tuple(prior["held_out_configurations"]),
            EXPECTED_HELD_OUT_CONFIG_IDS,
        )
        self.assertEqual(prior["config_fingerprints"], EXPECTED_FINGERPRINTS)
        self.assertEqual(prior["report_sha256s"], EXPECTED_REPORTS)
        self.assertEqual(set(prior["methods"]), set(EXPECTED_REPORTS))
        for method in EXPECTED_REPORTS:
            self.assertTrue(prior["methods"][method]["all_references_valid"])
        self.assertEqual(
            {
                method: prior["methods"][method]["authority_bridge"]
                for method in ("bf16", "turboquant")
            },
            phase12._expected_historical_source_bridges(REPOSITORY),
        )

    def test_unrecognized_decision0026_transition_fails_closed(self) -> None:
        original = phase12._git_blob_authority

        def altered(
            root: Path,
            *,
            commit: str,
            path: str,
        ) -> dict[str, str]:
            record = original(root, commit=commit, path=path)
            if (
                path == "src/kvbench/runtime/bf16_endpoint.py"
                and commit
                != phase12.PHASE12_TURBOQUANT_EXECUTION_COMMIT
            ):
                return {**record, "sha256": "0" * 64}
            return record

        with mock.patch.object(
            phase12,
            "_git_blob_authority",
            side_effect=altered,
        ):
            with self.assertRaisesRegex(
                Phase12UnifiedAdmissionError,
                "endpoint transition",
            ):
                phase12._validate_decision0026_transition(
                    REPOSITORY,
                    execution_commit=(
                        phase12.PHASE12_TURBOQUANT_EXECUTION_COMMIT
                    ),
                    unchanged_paths={
                        "src/kvbench/adapters/turboquant.py": (
                            phase12.PHASE12_TURBOQUANT_ADAPTER_SHA256
                        ),
                        "src/kvbench/runtime/turboquant_session.py": (
                            phase12.PHASE12_TURBOQUANT_SESSION_SHA256
                        ),
                    },
                )

    def test_report_byte_tampering_fails_closed(self) -> None:
        source = REPOSITORY / "docs/evidence/phase6/turboquant-method-admission.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["status"] = "FAIL"
        with tempfile.TemporaryDirectory() as directory:
            tampered = Path(directory) / source.name
            tampered.write_text(
                json.dumps(payload, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                Phase12UnifiedAdmissionError,
                "TurboQuant.*SHA-256|SHA-256.*TurboQuant",
            ):
                load_and_validate_prior_admission_evidence(
                    REPOSITORY,
                    report_paths={"turboquant": tampered},
                )

    def test_config_fingerprint_tampering_fails_after_schema_parse(self) -> None:
        source = REPOSITORY / "docs/evidence/phase6/turboquant-method-admission.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["method_config_fingerprints"]["turboquant_4bit_nc"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            tampered = Path(directory) / source.name
            tampered.write_text(
                json.dumps(payload, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            real_sha = actual_sha256_file(tampered)

            def hash_with_authorized_report(path: Path) -> str:
                candidate = Path(path)
                if candidate == tampered:
                    return EXPECTED_REPORTS["turboquant"]
                return actual_sha256_file(candidate)

            with mock.patch(
                "scripts.phase12_unified_admission.sha256_file",
                side_effect=hash_with_authorized_report,
            ):
                with self.assertRaisesRegex(
                    Phase12UnifiedAdmissionError,
                    "fingerprint",
                ):
                    load_and_validate_prior_admission_evidence(
                        REPOSITORY,
                        report_paths={"turboquant": tampered},
                    )
            self.assertNotEqual(real_sha, EXPECTED_REPORTS["turboquant"])

    def test_missing_referenced_evidence_fails_closed(self) -> None:
        missing = (
            REPOSITORY
            / "artifacts/phase4_smoke"
            / "phase4-smoke-fixed-l-eager-20260723t184024203883z-0cf160ca-429f62"
            / "smoke.json"
        )

        def hash_with_missing_reference(path: Path) -> str:
            candidate = Path(path)
            if candidate == missing:
                raise FileNotFoundError(candidate)
            return actual_sha256_file(candidate)

        with mock.patch(
            "scripts.phase12_unified_admission.sha256_file",
            side_effect=hash_with_missing_reference,
        ):
            with self.assertRaisesRegex(
                Phase12UnifiedAdmissionError,
                "evidence.*missing|missing.*evidence",
            ):
                _load()

    def test_referenced_evidence_hash_tampering_fails_closed(self) -> None:
        tampered = (
            REPOSITORY
            / "artifacts/phase4_smoke"
            / "phase4-smoke-fixed-l-eager-20260723t184024203883z-0cf160ca-429f62"
            / "smoke.json"
        )

        def hash_with_tampered_reference(path: Path) -> str:
            candidate = Path(path)
            if candidate == tampered:
                return "f" * 64
            return actual_sha256_file(candidate)

        with mock.patch(
            "scripts.phase12_unified_admission.sha256_file",
            side_effect=hash_with_tampered_reference,
        ):
            with self.assertRaisesRegex(
                Phase12UnifiedAdmissionError,
                "evidence.*SHA-256|SHA-256.*evidence",
            ):
                _load()


class Phase12G1G4AggregationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prior = _load()

    def test_all_ten_configurations_pass_g1_through_g4(self) -> None:
        result = aggregate_g1_g4(self.prior)
        self.assertEqual(
            tuple(result["configurations"]),
            EXPECTED_MAIN_CONFIG_IDS,
        )
        self.assertEqual(
            result["global_gates"],
            {"G1": "PASS", "G2": "PASS", "G3": "PASS", "G4": "PASS"},
        )
        for config_id, expected_fingerprint in EXPECTED_FINGERPRINTS.items():
            config = result["configurations"][config_id]
            self.assertEqual(config["fingerprint"], expected_fingerprint)
            self.assertEqual(
                config["gates"],
                {"G1": "PASS", "G2": "PASS", "G3": "PASS", "G4": "PASS"},
            )
            expected = EXPECTED_GATE_EVIDENCE[config["method"]]
            self.assertEqual(
                {
                    gate: tuple(evidence)
                    for gate, evidence in config["evidence"].items()
                },
                expected,
            )

    def test_held_out_controls_are_not_aggregated(self) -> None:
        result = aggregate_g1_g4(self.prior)
        for config_id in EXPECTED_HELD_OUT_CONFIG_IDS:
            self.assertNotIn(config_id, result["configurations"])

    def test_missing_named_check_fails_closed(self) -> None:
        prior = copy.deepcopy(self.prior)
        del prior["methods"]["turboquant"]["checks"]["finite_output"]
        with self.assertRaisesRegex(
            Phase12UnifiedAdmissionError,
            "finite_output",
        ):
            aggregate_g1_g4(prior)

    def test_non_pass_named_check_fails_closed(self) -> None:
        prior = copy.deepcopy(self.prior)
        prior["methods"]["kivi"]["checks"]["native_gqa"] = "FAIL"
        with self.assertRaisesRegex(
            Phase12UnifiedAdmissionError,
            "native_gqa",
        ):
            aggregate_g1_g4(prior)

    def test_extra_main_configuration_fails_closed(self) -> None:
        prior = copy.deepcopy(self.prior)
        prior["main_configurations"] = (
            *prior["main_configurations"],
            "k4v2",
        )
        with self.assertRaisesRegex(
            Phase12UnifiedAdmissionError,
            "configuration",
        ):
            aggregate_g1_g4(prior)

    def test_fingerprint_mismatch_fails_closed(self) -> None:
        prior = copy.deepcopy(self.prior)
        prior["config_fingerprints"]["kvq3"] = "0" * 64
        with self.assertRaisesRegex(
            Phase12UnifiedAdmissionError,
            "fingerprint|configuration.*authority",
        ):
            aggregate_g1_g4(prior)


class Phase12CoordinatorBoundaryTests(unittest.TestCase):
    @staticmethod
    def _bf16_worker_accounting() -> dict[str, object]:
        breakdown = {
            "data_bytes": 10_000,
            "workspace_bytes": 20,
            "padding_bytes": 5,
            "scale_bytes": 0,
            "zero_point_bytes": 0,
            "metadata_bytes": 0,
        }
        allocated = sum(breakdown.values())
        return {
            "runner": {
                "cache_byte_breakdown": breakdown,
                "cache_accounting": {
                    "predicted_tensor_bytes": 10_000,
                    "measured_tensor_bytes": 10_005,
                    "allocated_bytes": allocated,
                    "padding_bytes": 5,
                    "workspace_bytes": 20,
                    "capacity": phase12.PHASE12_CONTEXT_LENGTH + 1,
                    "active_context": phase12.PHASE12_CONTEXT_LENGTH,
                    "model_baseline_allocated_bytes": 1_000,
                },
            }
        }

    def test_worker_accounting_replays_independent_prediction(self) -> None:
        worker = self._bf16_worker_accounting()
        accounting = phase12._phase12_byte_accounting("bf16", worker)
        self.assertEqual(accounting.predicted_allocated_bytes, 10_000)
        self.assertEqual(accounting.allocated_bytes, 10_025)
        self.assertAlmostEqual(
            accounting.predicted_relative_error,
            25 / 10_025,
        )
        self.assertIsNone(accounting.r_hbm)

    def test_worker_accounting_tampering_fails_closed(self) -> None:
        for field, value in (
            ("predicted_tensor_bytes", 1),
            ("measured_tensor_bytes", 1),
            ("allocated_bytes", 1),
            ("r_hbm", 123),
            ("rho_alloc", 1.0),
            ("r_alloc", 1.0),
        ):
            with self.subTest(field=field):
                worker = self._bf16_worker_accounting()
                worker["runner"]["cache_accounting"][field] = value
                with self.assertRaises(Phase12UnifiedAdmissionError):
                    phase12._phase12_byte_accounting("bf16", worker)

    def test_observed_execution_path_tampering_fails_closed(self) -> None:
        record = phase12.execution_path_audit_facade(
            backend_identity_verified=True,
            device_kernel_family_verified=True,
            allocation_categories_verified=True,
            temporary_tensor_shapes_verified=True,
            gqa_replication_detected=False,
            full_prefix_temporary_detected=False,
            host_synchronization_detected=False,
            backend_fallback_detected=False,
            full_prefix_dequantization="verified_false",
        ).to_dict()
        phase12._validate_execution_path_record(
            record,
            family="kvquant",
        )
        for field, value in (
            ("backend_fallback_detected", True),
            ("host_synchronization_detected", True),
            ("device_kernel_family_verified", False),
            ("passed", False),
        ):
            with self.subTest(field=field):
                tampered = dict(record)
                tampered[field] = value
                with self.assertRaises(Phase12UnifiedAdmissionError):
                    phase12._validate_execution_path_record(
                        tampered,
                        family="kvquant",
                    )

    def test_prior_g3_binding_is_exact_and_tamper_rejected(self) -> None:
        for family in ("bf16", "turboquant", "kivi", "kvquant"):
            with self.subTest(family=family):
                binding = phase12._expected_prior_g3_binding(family)
                phase12._validate_prior_g3_binding(
                    binding,
                    family=family,
                )
                for field, value in (
                    ("report_sha256", "0" * 64),
                    ("check_ids", []),
                    ("report_path", "docs/evidence/forged.json"),
                ):
                    tampered = dict(binding)
                    tampered[field] = value
                    with self.assertRaises(
                        Phase12UnifiedAdmissionError
                    ):
                        phase12._validate_prior_g3_binding(
                            tampered,
                            family=family,
                        )

    def test_runtime_adapter_fingerprint_is_not_method_fingerprint(
        self,
    ) -> None:
        runtime_fingerprint = "b" * 64

        class Cache:
            @staticmethod
            def layout_fingerprint() -> str:
                return "c" * 64

        class Method:
            @staticmethod
            def config_fingerprint(layout_fingerprint: str) -> str:
                self.assertEqual(layout_fingerprint, "c" * 64)
                return runtime_fingerprint

        self.assertNotEqual(
            runtime_fingerprint,
            phase12.EXPECTED_CONFIG_FINGERPRINTS["kvq4"],
        )
        self.assertEqual(
            phase12._validate_runtime_adapter_fingerprint(
                method=Method(),
                cache=Cache(),
                observed=runtime_fingerprint,
            ),
            runtime_fingerprint,
        )
        with self.assertRaises(Phase12UnifiedAdmissionError):
            phase12._validate_runtime_adapter_fingerprint(
                method=Method(),
                cache=Cache(),
                observed=phase12.EXPECTED_CONFIG_FINGERPRINTS["kvq4"],
            )

    def test_live_factory_identity_cannot_be_mislabeled(self) -> None:
        genuine_identities = {
            "bf16": {"name": "bf16"},
            "tq_4bit_nc": {
                "name": "turboquant",
                "config_name": "turboquant_4bit_nc",
            },
            "tq_k3v4_nc": {
                "name": "turboquant",
                "config_name": "turboquant_k3v4_nc",
            },
            "tq_3bit_nc": {
                "name": "turboquant",
                "config_name": "turboquant_3bit_nc",
            },
            "k4v4": {
                "name": "kivi",
                "config_name": "k4v4",
                "k_bits": 4,
                "v_bits": 4,
            },
            "k2v4": {
                "name": "kivi",
                "config_name": "k2v4",
                "k_bits": 2,
                "v_bits": 4,
            },
            "k2v2": {
                "name": "kivi",
                "config_name": "k2v2",
                "k_bits": 2,
                "v_bits": 2,
            },
            "kvq4": {"name": "kvquant", "config_name": "kvq4", "bits": 4},
            "kvq3": {"name": "kvquant", "config_name": "kvq3", "bits": 3},
            "kvq2": {"name": "kvquant", "config_name": "kvq2", "bits": 2},
        }
        for configuration, identity in genuine_identities.items():
            with self.subTest(configuration=configuration):
                genuine = type("Method", (), identity)()
                phase12._validate_live_method_identity(
                    genuine,
                    configuration,
                )
        for field, value in (
            ("name", "kivi"),
            ("config_name", "kvq2"),
            ("bits", 2),
        ):
            with self.subTest(field=field):
                tampered = type(
                    "KVQuantMethod",
                    (),
                    {
                        "name": "kvquant",
                        "config_name": "kvq3",
                        "bits": 3,
                    },
                )()
                setattr(tampered, field, value)
                with self.assertRaisesRegex(
                    Phase12UnifiedAdmissionError,
                    "factory identity",
                ):
                    phase12._validate_live_method_identity(
                        tampered,
                        "kvq3",
                    )

    def test_child_environment_is_locked_and_secret_free(self) -> None:
        parent = {
            "PATH": "/usr/local/cuda-13.0/bin:/usr/bin:/bin",
            "LD_LIBRARY_PATH": "/usr/local/cuda-13.0/lib64",
            "KVBENCH_AUTHORIZED_IMAGE_DIGEST": (
                phase12.PHASE12_AUTHORIZED_CONTAINER_DIGEST
            ),
            "KVBENCH_EXECUTION_ENVIRONMENT": "measurement_container",
        }
        with mock.patch.dict(os.environ, parent, clear=True):
            child = phase12._child_environment()
        self.assertEqual(
            child["PYTHONPATH"],
            os.pathsep.join(
                (
                    "/opt/kvbench/.phase3/site-packages",
                    str(REPOSITORY / "src"),
                    str(REPOSITORY),
                )
            ),
        )
        self.assertEqual(child["CUDA_HOME"], "/usr/local/cuda-13.0")
        self.assertEqual(child["CC"], "/usr/bin/gcc")
        self.assertEqual(child["CXX"], "/usr/bin/c++")
        self.assertEqual(child["CUDAARCHS"], "120")
        self.assertEqual(child["CMAKE_CUDA_ARCHITECTURES"], "120")
        self.assertEqual(child["HF_HUB_DISABLE_TELEMETRY"], "1")
        self.assertTrue(
            set(child).isdisjoint(phase12._FORBIDDEN_CHILD_ENVIRONMENT)
        )

    def test_container_runtime_requires_docker_and_read_only_source(
        self,
    ) -> None:
        environment = {
            "KVBENCH_AUTHORIZED_IMAGE_DIGEST": (
                phase12.PHASE12_AUTHORIZED_CONTAINER_DIGEST
            ),
            "KVBENCH_EXECUTION_ENVIRONMENT": "measurement_container",
        }
        read_only = mock.Mock(f_flag=os.ST_RDONLY)
        with (
            mock.patch.dict(os.environ, environment, clear=True),
            mock.patch.object(Path, "is_file", return_value=True),
            mock.patch.object(os, "statvfs", return_value=read_only),
        ):
            self.assertEqual(
                phase12._require_authorized_container_runtime(),
                phase12._expected_container_runtime_attestation(),
            )
        read_write = mock.Mock(f_flag=0)
        with (
            mock.patch.dict(os.environ, environment, clear=True),
            mock.patch.object(Path, "is_file", return_value=True),
            mock.patch.object(os, "statvfs", return_value=read_write),
            self.assertRaises(Phase12UnifiedAdmissionError),
        ):
            phase12._require_authorized_container_runtime()
        with (
            mock.patch.dict(os.environ, environment, clear=True),
            mock.patch.object(Path, "is_file", return_value=False),
            mock.patch.object(os, "statvfs", return_value=read_only),
            self.assertRaises(Phase12UnifiedAdmissionError),
        ):
            phase12._require_authorized_container_runtime()

    def test_original_timing_scope_must_be_the_known_runner_value(self) -> None:
        raw = {
            "measurement_scope": "measurement_container_admission",
            "performance_claim_eligible": False,
            "timing": {
                "paper_claim_eligible": False,
                "measurement_scope": "native_host_admission",
                "sample_count": phase12.PHASE12_MEASURED_BATCHES,
                "samples": [
                    {
                        "completed_operations": (
                            phase12.PHASE12_MEASURED_STEPS
                        ),
                        "failed_operations": 0,
                    }
                    for _ in range(phase12.PHASE12_MEASURED_BATCHES)
                ],
            },
        }
        normalized = phase12._normalize_runner_result(raw)
        self.assertEqual(
            normalized["timing"]["measurement_scope"],
            "measurement_container_admission",
        )
        raw["timing"]["measurement_scope"] = "measurement_container_admission"
        with self.assertRaisesRegex(
            Phase12UnifiedAdmissionError,
            "governance differs",
        ):
            phase12._normalize_runner_result(raw)

    def test_worker_setup_is_inside_forced_flash_context(self) -> None:
        source = inspect.getsource(phase12._run_g5_worker)
        context = (
            "with torch.inference_mode(), forced_flash_execution():"
        )
        self.assertIn(context, source)
        self.assertLess(source.index(context), source.index(
            "session = _build_phase12_session("
        ))

    def test_make_target_is_one_secret_free_exact_container(self) -> None:
        makefile = (REPOSITORY / "Makefile").read_text(encoding="utf-8")
        segment = makefile.split(
            "unified-admission: override ",
            maxsplit=1,
        )[1].split("\nvalidate-unified-admission:", maxsplit=1)[0]
        self.assertIn(
            (
                "override PHASE12_AUTHORIZED_IMAGE_CONFIG_DIGEST := "
                f"{phase12.PHASE12_AUTHORIZED_CONTAINER_DIGEST}"
            ),
            makefile,
        )
        for required in (
            "--read-only --network=none --pid=host",
            '--gpus "device=$(MEASUREMENT_GPU_UUID)"',
            (
                "src=$$task_root/repository,"
                "dst=/home/rockrock/cmu_paper,readonly"
            ),
            (
                "src=$$phase12_artifact_root,"
                "dst=/home/rockrock/cmu_paper/artifacts/phase12,readonly"
            ),
            "src=$$stage,dst=/home/rockrock/cmu_paper/$$stage_relative",
            "--run-campaign",
            "--finalize-staged-campaign",
            "--finalize-failed-campaign",
            "--validate-campaign",
            "$(PHASE11DQ23_KVQUANT_SOURCE_ROOT)",
            "$(PHASE11DQ23_KVQUANT_COMMIT)",
            "$(PHASE11DQ23_KVQUANT_TREE)",
            "scripts.validate_kvquant_q23_long_context_patch",
            "$(PHASE11DQ23_KVQUANT_PATCH_SHA256)",
            "$(PHASE11DQ23_KVQUANT_EXTENSION_SHA256)",
        ):
            self.assertIn(required, segment)
        for forbidden in (
            "--env R2_",
            "--env AWS_",
            "--env CLOUDFLARE_",
            "make pilot",
            "make full-scan",
            "make profile-subset",
            "$(PHASE11_KVQUANT_CORRECTED_COMMIT)",
            "scripts.validate_kvquant_long_context_patch",
            "docs/evidence/phase11/kvquant-method-admission.json",
            "59ef5bfc581a68cdc4d21c4c0a840f046e698633f7475f79906063c6e333ae6a",
            "KVBENCH_PHASE11DQ23_EVIDENCE_ROOT",
        ):
            self.assertNotIn(forbidden, segment)
        coordinator_source = inspect.getsource(phase12.run_campaign)
        self.assertIn(
            '_run_container_test(resolved_stage, "test-cuda")',
            coordinator_source,
        )
        self.assertIn(
            '_run_container_test(resolved_stage, "test-graph")',
            coordinator_source,
        )


if __name__ == "__main__":
    unittest.main()
