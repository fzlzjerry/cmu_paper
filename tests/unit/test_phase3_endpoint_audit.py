"""CPU-only tests for the concrete Phase 3 endpoint session."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from kvbench.runtime.phase3_audit_operation import Phase3AuditOperationKey
from kvbench.runtime.phase3_endpoint_audit import (
    PHASE3_ENDPOINT_WARMUP_OPERATIONS,
    Phase3EndpointAuditError,
    build_phase3_endpoint_session,
)
from kvbench.runtime.static_cache import layout_fingerprint_for_geometry
from kvbench.schema import GraphMode, RunnerKind
from kvbench.schema.phase3 import (
    PHASE3_FIXED_PLAN_PATH,
    PHASE3_GROWING_PLAN_PATH,
    PHASE3_PLAN_FINGERPRINTS,
    Phase3ProcessPoint,
)


_DIGEST = "a" * 64
_RECEIPT = "b" * 64
_WORKSPACE_BYTES = 32 * (32 + 8) * 64 * 2


class _Endpoint:
    instances: list[_Endpoint] = []

    def __init__(self, model: object, cache: object) -> None:
        del model
        self.cache = cache
        self.decode_calls = 0
        self.prefill_calls = 0
        self.__class__.instances.append(self)

    def prepare_position_embeddings(self, position_ids: object) -> tuple[object, object]:
        return position_ids, position_ids

    def prefill(self, input_ids: object) -> None:
        self.prefill_calls += 1
        self.cache.prepare_prefill(int(input_ids.shape[1]))
        self.cache.complete_prefill()

    def decode(
        self,
        input_ids: object,
        cache_position: object,
        position_embeddings: object,
    ) -> object:
        del cache_position, position_embeddings
        self.decode_calls += 1
        return torch.zeros(
            (int(input_ids.shape[0]), 1, 32),
            dtype=torch.bfloat16,
            device=input_ids.device,
        )


class _Graph:
    def __init__(self) -> None:
        self.output = torch.zeros((1, 1, 32), dtype=torch.bfloat16)
        self.replay_calls = 0

    def replay(self) -> object:
        self.replay_calls += 1
        return self.output


def _point(*, growing: bool, graph: bool = False) -> Phase3ProcessPoint:
    runner = RunnerKind.GROWING_CONTEXT if growing else RunnerKind.FIXED_L
    mode = GraphMode.CUDA_GRAPH if graph else GraphMode.EAGER
    return Phase3ProcessPoint(
        point_id=f"{runner.value}-b1-l128-{mode.value}-r1",
        runner_kind=runner,
        graph_mode=mode,
        batch_size=1,
        context_length=128,
        output_steps=16 if growing else 1,
        process_replicate=1,
        stability_member=False,
    )


def _keys(*, growing: bool, graph: bool = False) -> tuple[Phase3AuditOperationKey, ...]:
    point = _point(growing=growing, graph=graph)
    plan = PHASE3_GROWING_PLAN_PATH if growing else PHASE3_FIXED_PLAN_PATH
    layout = layout_fingerprint_for_geometry(
        num_layers=32,
        batch_size=1,
        num_kv_heads=8,
        capacity=point.context_length + point.output_steps,
        head_dim=128,
        device="cpu",
        workspace_bytes=_WORKSPACE_BYTES,
    )
    steps = range(16) if growing else range(1)
    return tuple(
        Phase3AuditOperationKey.from_point(
            run_id="phase3-endpoint-session-test",
            point=point,
            decode_step=step,
            cache_layout_fingerprint=layout,
            execution_git_sha="1" * 40,
            plan_fingerprint=PHASE3_PLAN_FINGERPRINTS[plan],
            hardware_identity_sha256="2" * 64,
            software_identity_sha256="3" * 64,
            model_identity_sha256="4" * 64,
            backend_identity_sha256="5" * 64,
            source_identity_sha256="6" * 64,
        )
        for step in steps
    )


def _loaded() -> object:
    return SimpleNamespace(
        model=object(),
        receipt=SimpleNamespace(receipt_sha256=_RECEIPT),
    )


class Phase3EndpointSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        _Endpoint.instances.clear()
        self.prefix = torch.zeros((1, 128), dtype=torch.long)

    def test_fixed_graph_warms_sixteen_then_captures_once(self) -> None:
        graph = _Graph()
        capture_counts: list[int] = []

        def capture(
            operation: object,
            *,
            warmup_steps: int,
            device: object,
        ) -> _Graph:
            del operation, device
            capture_counts.append(_Endpoint.instances[-1].decode_calls)
            self.assertEqual(warmup_steps, 0)
            return graph

        with (
            mock.patch(
                "kvbench.runtime.phase3_endpoint_audit.BF16DecodeEndpoint",
                _Endpoint,
            ),
            mock.patch(
                "kvbench.runtime.phase3_endpoint_audit."
                "validate_loaded_frozen_model_receipt"
            ),
            mock.patch(
                "kvbench.runtime.phase3_endpoint_audit._cache_pair_sha256",
                return_value=_DIGEST,
            ),
            mock.patch(
                "kvbench.runtime.phase3_endpoint_audit.capture_fixed_graph",
                side_effect=capture,
            ) as capture_mock,
        ):
            session = build_phase3_endpoint_session(
                loaded=_loaded(),
                operation_keys=_keys(growing=False, graph=True),
                prefix_input_ids=self.prefix,
                decode_input_ids=torch.zeros((1, 1), dtype=torch.long),
            )

        self.assertEqual(
            _Endpoint.instances[-1].decode_calls,
            PHASE3_ENDPOINT_WARMUP_OPERATIONS,
        )
        self.assertEqual(capture_counts, [PHASE3_ENDPOINT_WARMUP_OPERATIONS])
        capture_mock.assert_called_once()
        self.assertIs(session.graph, graph)
        backup = session._fixed_destination_backup
        self.assertIsNotNone(backup)
        assert backup is not None
        self.assertEqual(sum(int(item.numel()) for item in backup), 65_536)
        self.assertLess(
            sum(int(item.numel()) for item in backup),
            int(session.cache.keys.numel()),
        )
        with self.assertRaisesRegex(
            Phase3EndpointAuditError,
            "timing is not admitted",
        ):
            session.fixed_measurement_callable()

    def test_fixed_eager_admits_only_after_audit_and_restore(self) -> None:
        released: list[bool] = []
        with (
            mock.patch(
                "kvbench.runtime.phase3_endpoint_audit.BF16DecodeEndpoint",
                _Endpoint,
            ),
            mock.patch(
                "kvbench.runtime.phase3_endpoint_audit."
                "validate_loaded_frozen_model_receipt"
            ),
            mock.patch(
                "kvbench.runtime.phase3_endpoint_audit._cache_pair_sha256",
                return_value=_DIGEST,
            ),
            mock.patch(
                "kvbench.runtime.phase3_endpoint_audit.capture_fixed_graph"
            ) as capture_mock,
        ):
            session = build_phase3_endpoint_session(
                loaded=_loaded(),
                operation_keys=_keys(growing=False),
                prefix_input_ids=self.prefix,
                decode_input_ids=torch.zeros((1, 1), dtype=torch.long),
            )
            self.assertTrue(callable(session.audit_call(0).operation))
            session.record_audit(
                0,
                dispatch_audit_sha256="7" * 64,
                allocation_audit_sha256="8" * 64,
                destination_slot_sha256="9" * 64,
                locally_verified=True,
            )
            session.finish_audits(
                release_audit_buffers=lambda: released.append(True)
            )
            session.fixed_measurement_callable()()
            session.mark_measured()

        capture_mock.assert_not_called()
        self.assertEqual(released, [True])
        self.assertEqual(session.state, "measured")
        provenance = session.provenance_payload()
        self.assertEqual(provenance["receipt_sha256"], _RECEIPT)
        self.assertEqual(provenance["dispatch_audit_sha256"], ["7" * 64])
        self.assertEqual(provenance["allocation_audit_sha256"], ["8" * 64])
        self.assertNotEqual(
            provenance["history_chain_sha256"],
            provenance["prefix_sha256"],
        )

    def test_growing_uses_one_endpoint_cache_and_ordered_trajectory(self) -> None:
        with (
            mock.patch(
                "kvbench.runtime.phase3_endpoint_audit.BF16DecodeEndpoint",
                _Endpoint,
            ),
            mock.patch(
                "kvbench.runtime.phase3_endpoint_audit."
                "validate_loaded_frozen_model_receipt"
            ),
            mock.patch(
                "kvbench.runtime.phase3_endpoint_audit._cache_pair_sha256",
                return_value=_DIGEST,
            ),
        ):
            session = build_phase3_endpoint_session(
                loaded=_loaded(),
                operation_keys=_keys(growing=True),
                prefix_input_ids=self.prefix,
                decode_input_ids=torch.zeros((1, 16), dtype=torch.long),
            )
            endpoint = _Endpoint.instances[-1]
            self.assertEqual(len(_Endpoint.instances), 1)
            self.assertIs(endpoint.cache, session.cache)
            self.assertEqual(endpoint.decode_calls, 16)
            self.assertEqual(endpoint.prefill_calls, 2)
            with self.assertRaisesRegex(
                Phase3EndpointAuditError,
                "ordered and complete",
            ):
                session.audit_call(1)
            for step in range(16):
                session.audit_call(step)
                session.record_audit(
                    step,
                    dispatch_audit_sha256="7" * 64,
                    allocation_audit_sha256="8" * 64,
                    destination_slot_sha256=f"{step:064x}",
                    locally_verified=True,
                )
            session.finish_audits(release_audit_buffers=lambda: None)
            operations = session.growing_measurement_callables()

        self.assertEqual(len(operations), 16)
        self.assertEqual(session.cache.active_context, 128)


if __name__ == "__main__":
    unittest.main()
