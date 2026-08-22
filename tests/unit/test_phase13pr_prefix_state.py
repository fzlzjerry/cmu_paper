"""Focused tests for Phase 13P-R untimed deterministic prefix restoration."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import tempfile
import unittest

import torch

from kvbench.runtime.static_cache import BF16StaticCache
from scripts import phase13_pilot as pilot
from scripts import phase13pb_prefix_remediation as remediation
from scripts.phase13_prefix_state import (
    Phase13PrefixStateError,
    restore_prefix_state,
    save_prefix_state,
    validate_prefix_state,
)


ROOT = Path(__file__).resolve().parents[2]
TEST_FINGERPRINT = "a" * 64


def _cache(batch: int) -> BF16StaticCache:
    cache = BF16StaticCache(
        num_layers=2,
        batch_size=batch,
        num_kv_heads=2,
        capacity=18,
        head_dim=4,
        device="cpu",
    )
    cache.initialize_deterministic()
    return cache


def _equivalence_session(pointer: int, configuration: str) -> dict[str, object]:
    null_labels = sorted(
        pilot._EXPECTED_EQUIVALENCE_NONALLOCATED_NULL_POINTER_LABELS[
            configuration
        ]
    )
    return {
        "cache_layout_fingerprint": "b" * 64,
        "cache_accounting": {"allocated_bytes": 1},
        "cache_byte_breakdown": {"data_bytes": 1},
        "pointer_labels": ["cache", *null_labels],
        "pointer_values": [*([0] * len(null_labels)), pointer],
        "pointers_stable": True,
        "pointer_count": 1 + len(null_labels),
        "pointers_unique": True,
        "raw_pointer_values_unique": len(null_labels) <= 1,
        "nonallocated_null_pointer_labels": null_labels,
        "null_pointer_tensor_contract_verified": True,
        "recognized_same_tensor_alias_groups": [],
        "unexpected_pointer_alias_groups": [],
        "output_checksum": "c" * 64,
        "kernel_path_fingerprint": "d" * 64,
        "kernel_count": 1,
        "graph_capture": True,
        "graph_fallback": False,
        "graph_replay_exact": True,
        "eager_graph_agreement": True,
    }


def _equivalence_payload() -> dict[str, object]:
    records = []
    for configuration_index, configuration in enumerate(pilot.CONFIGURATIONS):
        for batch in pilot.BATCH_SIZES:
            state_sha256 = (
                f"{configuration_index * len(pilot.BATCH_SIZES) + batch:064x}"
            )
            records.append(
                {
                    "case_id": f"{configuration}/B{batch}/L17",
                    "configuration": configuration,
                    "method_config_fingerprint": pilot.CONFIG_FINGERPRINTS[
                        configuration
                    ],
                    "source_batch": batch,
                    "target_batch": batch,
                    "historical_context": 17,
                    "batch_reuse_policy": "exact_target_batch_only",
                    "source_state_sha256": state_sha256,
                    "target_state_sha256": state_sha256,
                    "restored_state_sha256": state_sha256,
                    "state_bytes_exact": True,
                    "lifecycle_exact": True,
                    "layout_exact": True,
                    "allocation_exact": True,
                    "output_checksum_exact": True,
                    "kernel_path_exact": True,
                    "cuda_graph_result_exact": True,
                    "direct": _equivalence_session(
                        1000 + configuration_index * 100 + batch,
                        configuration,
                    ),
                    "restored": _equivalence_session(
                        2000 + configuration_index * 100 + batch,
                        configuration,
                    ),
                    "fresh_target_allocation": True,
                    "direct_and_restored_pointers_disjoint": True,
                    "restore_outside_timing": True,
                    "runtime_prefix_cache_sharing": False,
                    "cross_batch_control_rejected": (
                        None if batch == 1 else True
                    ),
                    "passed": True,
                }
            )
    rejections = [
        {
            "configuration": configuration,
            "source_batch": 1,
            "target_batch": batch,
            "historical_context": 17,
            "rejected_before_cache_mutation": True,
            "pointers_unchanged": True,
            "error": "prefix state exact batch differs",
        }
        for configuration in pilot.CONFIGURATIONS
        for batch in (4, 8)
    ]
    return {
        "schema_version": "kvbench-phase13-prefix-equivalence-2.0.0",
        "status": "PASS",
        "execution_git_sha": "e" * 40,
        "authorized_container_digest": pilot.AUTHORIZED_CONTAINER_DIGEST,
        "decision_id": "0034",
        "configuration_count": 10,
        "batch_sizes": [1, 4, 8],
        "historical_context": 17,
        "case_count": 30,
        "all_configurations_passed": True,
        "all_cases_passed": True,
        "cross_batch_restore_fail_closed": True,
        "cross_batch_rejection_count": 20,
        "fresh_target_allocation": True,
        "restore_outside_timing": True,
        "runtime_prefix_cache_sharing": False,
        "timing_collected": False,
        "cross_batch_rejections": rejections,
        "records": records,
    }


class PrefixStateTests(unittest.TestCase):
    def test_equivalence_pointer_alias_contract_accepts_only_same_tensor_pairs(
        self,
    ) -> None:
        allowed = pilot._ALLOWED_EQUIVALENCE_POINTER_ALIAS_GROUPS
        self.assertEqual(
            allowed,
            frozenset(
                {
                    frozenset({"keys_data_ptr", "keys_storage_ptr"}),
                    frozenset({"values_data_ptr", "values_storage_ptr"}),
                }
            ),
        )
        self.assertEqual(
            pilot._EXPECTED_EQUIVALENCE_NONALLOCATED_NULL_POINTER_LABELS[
                "bf16"
            ],
            frozenset(),
        )
        self.assertEqual(
            pilot._EXPECTED_EQUIVALENCE_NONALLOCATED_NULL_POINTER_LABELS[
                "tq_4bit_nc"
            ],
            frozenset({"reserved_workspace_data_ptr"}),
        )

        accepted = pilot._equivalence_pointer_alias_record(
            {
                "keys_data_ptr": 11,
                "keys_storage_ptr": 11,
                "values_data_ptr": 22,
                "values_storage_ptr": 22,
                "endpoint_query_rope_scratch_data_ptr": 33,
                "endpoint_key_rope_scratch_data_ptr": 44,
            }
        )
        self.assertTrue(accepted["pointers_unique"])
        self.assertFalse(accepted["raw_pointer_values_unique"])
        self.assertEqual(accepted["unexpected_pointer_alias_groups"], [])

        empty_workspace = pilot._equivalence_pointer_alias_record(
            {
                "keys_data_ptr": 11,
                "reserved_workspace_data_ptr": 0,
            },
            expected_nonallocated_null_pointer_labels=(
                "reserved_workspace_data_ptr",
            ),
        )
        self.assertEqual(
            empty_workspace["nonallocated_null_pointer_labels"],
            ["reserved_workspace_data_ptr"],
        )

        rejected = pilot._equivalence_pointer_alias_record(
            {
                "keys_data_ptr": 11,
                "keys_storage_ptr": 11,
                "values_data_ptr": 11,
                "values_storage_ptr": 22,
            }
        )
        self.assertFalse(rejected["pointers_unique"])
        self.assertEqual(
            rejected["unexpected_pointer_alias_groups"],
            [["keys_data_ptr", "keys_storage_ptr", "values_data_ptr"]],
        )

        with self.assertRaisesRegex(
            pilot.Phase13PilotError,
            "pointer evidence",
        ):
            pilot._equivalence_pointer_alias_record({"keys_data_ptr": 0})

    def test_equivalence_zero_pointer_contract_checks_zero_byte_backing(
        self,
    ) -> None:
        source = inspect.getsource(pilot._equivalence_null_pointer_tensor_verified)
        self.assertIn("tensor.numel() != 0", source)
        self.assertIn("q23_value_decode_workspace_data_ptr", source)
        self.assertIn("decode_logits.untyped_storage().data_ptr()", source)
        self.assertIn("storage.nbytes() == decode_logits.untyped_storage().nbytes()", source)

        record_source = inspect.getsource(pilot._equivalence_session_record)
        self.assertIn(
            "_EXPECTED_EQUIVALENCE_NONALLOCATED_NULL_POINTER_LABELS",
            record_source,
        )
        self.assertIn("null_pointer_tensor_contract_verified", record_source)

        matrix_source = inspect.getsource(pilot.run_prefix_equivalence)
        self.assertIn("direct_allocated_pointers", matrix_source)
        self.assertIn("if pointer > 0", matrix_source)

    def test_catalog_plan_is_deterministic_and_uses_228_exact_batch_states(
        self,
    ) -> None:
        feasibility = pilot.build_feasibility_records(pilot.derive_execution_order())
        first = pilot.derive_prefix_catalog_plan(feasibility)
        second = pilot.derive_prefix_catalog_plan(feasibility)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 228)
        self.assertEqual(
            len(
                {
                    (
                        item["method_config_id"],
                        item["batch_size"],
                        item["context_label"],
                    )
                    for item in first
                }
            ),
            228,
        )
        for item in first:
            self.assertEqual(item["source_batch"], item["batch_size"])
            self.assertEqual(item["target_batch"], item["batch_size"])
            self.assertEqual(
                item["batch_reuse_policy"],
                "exact_target_batch_only",
            )
            self.assertFalse(item["cross_batch_reuse"])

    def test_restore_copies_exact_batch_into_fresh_caller_owned_buffers(self) -> None:
        source = _cache(4)
        source.prepare_prefill(17)
        values = torch.arange(source.keys[:, :, :, :17, :].numel()).reshape(
            source.keys[:, :, :, :17, :].shape
        )
        source.keys[:, :, :, :17, :].copy_(values.to(torch.bfloat16))
        source.values[:, :, :, :17, :].copy_((values + 37).to(torch.bfloat16))
        source.complete_prefill()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "snapshot"
            manifest = save_prefix_state(
                cache=source,
                family="bf16",
                configuration="bf16",
                historical=17,
                source_batch=4,
                method_config_fingerprint=TEST_FINGERPRINT,
                output=root,
                authority={"test": True},
            )
            target = _cache(4)
            target_pointers = target.pointers()
            receipt = restore_prefix_state(
                cache=target,
                family="bf16",
                configuration="bf16",
                historical=17,
                root=root,
                expected_state_sha256=manifest["state_file_sha256"],
                expected_method_config_fingerprint=TEST_FINGERPRINT,
            )
            self.assertTrue(torch.equal(target.keys, source.keys))
            self.assertTrue(torch.equal(target.values, source.values))
            self.assertTrue(torch.count_nonzero(target.keys[:, :, :, 17:, :]) == 0)
            self.assertEqual(target.pointers(), target_pointers)
            self.assertTrue(set(target.pointers().values()).isdisjoint(source.pointers().values()))
            self.assertTrue(receipt["fresh_target_allocation"])
            self.assertFalse(receipt["runtime_prefix_sharing"])
            self.assertTrue(receipt["restore_outside_timing"])
            self.assertEqual(receipt["source_batch"], 4)
            self.assertEqual(receipt["target_batch"], 4)
            self.assertEqual(
                receipt["batch_reuse_policy"],
                "exact_target_batch_only",
            )

    def test_cross_batch_restore_fails_before_target_mutation(self) -> None:
        source = _cache(8)
        source.prepare_prefill(17)
        source.keys[:, :, :, :17, :].fill_(1)
        source.values[:, :, :, :17, :].fill_(2)
        source.complete_prefill()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "snapshot"
            manifest = save_prefix_state(
                cache=source,
                family="bf16",
                configuration="bf16",
                historical=17,
                source_batch=8,
                method_config_fingerprint=TEST_FINGERPRINT,
                output=root,
                authority={"test": True},
            )
            target = _cache(1)
            pointers = target.pointers()
            keys = target.keys.clone()
            values = target.values.clone()
            with self.assertRaisesRegex(
                Phase13PrefixStateError,
                "exact batch",
            ):
                restore_prefix_state(
                    cache=target,
                    family="bf16",
                    configuration="bf16",
                    historical=17,
                    root=root,
                    expected_state_sha256=manifest["state_file_sha256"],
                    expected_method_config_fingerprint=TEST_FINGERPRINT,
                )
            self.assertEqual(target.pointers(), pointers)
            self.assertTrue(torch.equal(target.keys, keys))
            self.assertTrue(torch.equal(target.values, values))
            self.assertEqual(target.active_context, 0)

    def test_state_and_manifest_tampering_fail_closed(self) -> None:
        source = _cache(1)
        source.prepare_prefill(17)
        source.keys[:, :, :, :17, :].fill_(1)
        source.values[:, :, :, :17, :].fill_(2)
        source.complete_prefill()
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first"
            save_prefix_state(
                cache=source,
                family="bf16",
                configuration="bf16",
                historical=17,
                source_batch=1,
                method_config_fingerprint=TEST_FINGERPRINT,
                output=first,
                authority={"test": True},
            )
            state = first / "state.safetensors"
            with state.open("r+b") as handle:
                handle.seek(-1, 2)
                original = handle.read(1)
                handle.seek(-1, 2)
                handle.write(bytes([original[0] ^ 1]))
            with self.assertRaisesRegex(Phase13PrefixStateError, "checksum"):
                validate_prefix_state(first, verify_state_bytes=True)

            second = Path(temporary) / "second"
            save_prefix_state(
                cache=source,
                family="bf16",
                configuration="bf16",
                historical=17,
                source_batch=1,
                method_config_fingerprint=TEST_FINGERPRINT,
                output=second,
                authority={"test": True},
            )
            manifest_path = second / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["source_batch"] = 8
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            target = _cache(1)
            with self.assertRaisesRegex(Phase13PrefixStateError, "exact batch"):
                restore_prefix_state(
                    cache=target,
                    family="bf16",
                    configuration="bf16",
                    historical=17,
                    root=second,
                    expected_state_sha256=manifest["state_file_sha256"],
                    expected_method_config_fingerprint=TEST_FINGERPRINT,
                )

    def test_identity_shape_layout_and_fingerprint_mismatches_fail_closed(
        self,
    ) -> None:
        source = _cache(1)
        source.prepare_prefill(17)
        source.complete_prefill()

        def snapshot(parent: Path, name: str) -> tuple[Path, dict[str, object]]:
            root = parent / name
            manifest = save_prefix_state(
                cache=source,
                family="bf16",
                configuration="bf16",
                historical=17,
                source_batch=1,
                method_config_fingerprint=TEST_FINGERPRINT,
                output=root,
                authority={"test": True},
            )
            return root, manifest

        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root, manifest = snapshot(parent, "identity")
            target = _cache(1)
            for keyword, value, message in (
                ("configuration", "other", "configuration"),
                ("historical_context", 16, "context"),
                ("method_config_fingerprint", "b" * 64, "fingerprint"),
                ("source_layout_fingerprint", "b" * 64, "geometry"),
            ):
                arguments = {
                    "cache": target,
                    "family": "bf16",
                    "configuration": "bf16",
                    "historical": 17,
                    "root": root,
                    "expected_state_sha256": manifest["state_file_sha256"],
                    "expected_method_config_fingerprint": TEST_FINGERPRINT,
                }
                if keyword == "configuration":
                    arguments["configuration"] = value
                elif keyword == "historical_context":
                    arguments["historical"] = value
                elif keyword == "method_config_fingerprint":
                    arguments["expected_method_config_fingerprint"] = value
                else:
                    manifest_path = root / "manifest.json"
                    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                    payload[keyword] = value
                    manifest_path.write_text(
                        json.dumps(payload, sort_keys=True, separators=(",", ":"))
                        + "\n",
                        encoding="utf-8",
                    )
                with self.assertRaisesRegex(Phase13PrefixStateError, message):
                    restore_prefix_state(**arguments)
                if keyword == "source_layout_fingerprint":
                    root, manifest = snapshot(parent, "shape")

            manifest_path = root / "manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["tensors"][0]["shape"][-1] += 1
            manifest_path.write_text(
                json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            shape_target = _cache(1)
            shape_pointers = shape_target.pointers()
            shape_keys = shape_target.keys.clone()
            shape_values = shape_target.values.clone()
            with self.assertRaisesRegex(
                Phase13PrefixStateError,
                "target tensor metadata",
            ):
                restore_prefix_state(
                    cache=shape_target,
                    family="bf16",
                    configuration="bf16",
                    historical=17,
                    root=root,
                    expected_state_sha256=manifest["state_file_sha256"],
                    expected_method_config_fingerprint=TEST_FINGERPRINT,
                )
            self.assertEqual(shape_target.pointers(), shape_pointers)
            self.assertTrue(torch.equal(shape_target.keys, shape_keys))
            self.assertTrue(torch.equal(shape_target.values, shape_values))
            self.assertEqual(shape_target.active_context, 0)

    def test_formal_worker_uses_restore_and_omits_full_per_point_audits(self) -> None:
        worker = inspect.getsource(pilot._run_worker)
        self.assertIn("_build_restored_session", worker)
        self.assertNotIn("execution_path_audit_facade", worker)
        self.assertNotIn("fixture", worker.lower())
        self.assertNotIn("sanitizer", worker.lower())
        self.assertIn("prefix_state_restore", worker)
        session_bridge = inspect.getsource(pilot._bind_session_prefix_witness)
        self.assertIn("current_historical_prefix_sha256", session_bridge)
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        self.assertIn(
            "src=$$prefix_campaign_root,dst=/opt/kvbench-prefix-states",
            makefile,
        )
        self.assertIn("--materialize-prefix-seed", makefile)

    def test_equivalence_loads_model_before_inference_mode(self) -> None:
        source = inspect.getsource(pilot.run_prefix_equivalence)
        self.assertLess(
            source.index("loaded = load_frozen_model"),
            source.index("with torch.inference_mode()"),
        )

    def test_equivalence_uses_existing_graph_witness_phase(self) -> None:
        source = inspect.getsource(pilot._equivalence_session_record)
        self.assertIn('phase="before"', source)
        self.assertNotIn('phase="equivalence"', source)

    def test_equivalence_uses_observable_graph_factory(self) -> None:
        source = inspect.getsource(pilot.run_prefix_equivalence)
        self.assertEqual(
            source.count("_observable_cuda_graph_factory("),
            2,
        )

    def test_equivalence_records_each_required_invariant(self) -> None:
        source = inspect.getsource(pilot._equivalence_session_record)
        for field in (
            "pointers_stable",
            "pointers_unique",
            "graph_capture",
            "graph_fallback",
            "graph_replay_exact",
            "eager_graph_agreement",
        ):
            self.assertIn(f'"{field}"', source)
        self.assertIn("unexpected_pointer_alias_groups", source)

    def test_equivalence_matrix_is_exactly_ten_by_three(self) -> None:
        source = inspect.getsource(pilot.run_prefix_equivalence)
        self.assertEqual(pilot.PREFIX_EQUIVALENCE_BATCHES, (1, 4, 8))
        self.assertIn("for batch in PREFIX_EQUIVALENCE_BATCHES", source)
        self.assertIn('"case_count": len(records)', source)
        self.assertIn('"cross_batch_restore_fail_closed"', source)
        self.assertNotIn("source-b8", source)

    def test_remediation_evidence_validator_accepts_exact_matrix(self) -> None:
        result = remediation.validate_equivalence(_equivalence_payload())
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["case_count"], 30)
        self.assertEqual(result["cross_batch_rejection_count"], 20)

    def test_remediation_evidence_tampering_fails_closed(self) -> None:
        for tamper in ("batch", "output", "rejection"):
            payload = _equivalence_payload()
            if tamper == "batch":
                payload["records"][0]["source_batch"] = 8
            elif tamper == "output":
                payload["records"][0]["restored"]["output_checksum"] = "f" * 64
            else:
                payload["cross_batch_rejections"][0][
                    "rejected_before_cache_mutation"
                ] = False
            with self.assertRaises(remediation.Phase13PBBatchExactError):
                remediation.validate_equivalence(payload)


if __name__ == "__main__":
    unittest.main()
