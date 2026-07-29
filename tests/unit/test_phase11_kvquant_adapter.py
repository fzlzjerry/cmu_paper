"""Focused CPU controls for the Phase 11 adapter-boundary correction."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
import unittest

import torch

from kvbench.adapters import (
    BF16MethodAdapter,
    KIVIMethodAdapter,
    KVCacheMethod,
    TurboQuantMethodAdapter,
)
from kvbench.adapters.base import method_requires_pre_rope_key
from kvbench.runtime.bf16_endpoint import (
    BF16DecodeEndpoint,
    EndpointGeometryError,
    preserve_pre_rope_key_in_query_scratch,
)


class _Projection:
    def __init__(self, output: torch.Tensor) -> None:
        self.output = output

    def __call__(self, _hidden_states: torch.Tensor) -> torch.Tensor:
        return self.output.clone()


class _LegacyMethod:
    """Old boundary shape: it deliberately accepts no new keyword."""

    def __init__(self) -> None:
        self.appended_key: torch.Tensor | None = None

    def append_decode(
        self,
        cache_state: object,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del cache_state, layer_idx, cache_position
        self.appended_key = key_states
        return key_states, value_states

    store_prefill = append_decode

    def decode_attention(
        self,
        attention: object,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *,
        scaling: float,
    ) -> torch.Tensor:
        del attention, key_states, value_states, scaling
        return query_states


class _PreRoPEMethod(_LegacyMethod):
    requires_pre_rope_key = True

    def __init__(self) -> None:
        super().__init__()
        self.pre_rope_key: torch.Tensor | None = None

    def append_decode(
        self,
        cache_state: object,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_position: torch.Tensor,
        *,
        key_pre_rope_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.pre_rope_key = key_pre_rope_states
        return super().append_decode(
            cache_state,
            key_states,
            value_states,
            layer_idx,
            cache_position,
        )

    store_prefill = append_decode


def _endpoint(method: object) -> BF16DecodeEndpoint:
    endpoint = object.__new__(BF16DecodeEndpoint)
    endpoint.cache = object()
    endpoint.method = method
    endpoint.method_requires_pre_rope_key = method_requires_pre_rope_key(method)
    endpoint.num_kv_heads = 8
    endpoint.head_dim = 128
    endpoint.query_rope_scratch = torch.empty(
        (1, 1, 32, 1, 64),
        dtype=torch.bfloat16,
    )
    endpoint.key_rope_scratch = torch.empty(
        (1, 1, 8, 1, 64),
        dtype=torch.bfloat16,
    )
    return endpoint


def _attention_inputs() -> tuple[
    SimpleNamespace,
    torch.Tensor,
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor],
]:
    query_projection = torch.arange(
        4096,
        dtype=torch.bfloat16,
    ).reshape(1, 1, 4096)
    key_projection = (
        torch.arange(1024, dtype=torch.bfloat16).reshape(1, 1, 1024) + 3
    )
    value_projection = torch.full(
        (1, 1, 1024),
        7,
        dtype=torch.bfloat16,
    )
    attention = SimpleNamespace(
        q_proj=_Projection(query_projection),
        k_proj=_Projection(key_projection),
        v_proj=_Projection(value_projection),
        layer_idx=0,
        scaling=128**-0.5,
        o_proj=lambda output: output,
    )
    hidden = torch.zeros((1, 1, 4096), dtype=torch.bfloat16)
    cos = torch.zeros((1, 1, 128), dtype=torch.bfloat16)
    sin = torch.ones((1, 1, 128), dtype=torch.bfloat16)
    return attention, hidden, key_projection, (cos, sin)


class Phase11KVQuantBoundaryTests(unittest.TestCase):
    def test_protocol_pre_rope_input_is_optional_keyword_only(self) -> None:
        for operation in (
            KVCacheMethod.store_prefill,
            KVCacheMethod.append_decode,
        ):
            parameter = inspect.signature(operation).parameters[
                "key_pre_rope_states"
            ]
            self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
            self.assertIsNone(parameter.default)

    def test_existing_adapters_default_to_post_rope_only(self) -> None:
        for adapter_type in (
            BF16MethodAdapter,
            TurboQuantMethodAdapter,
            KIVIMethodAdapter,
        ):
            with self.subTest(adapter=adapter_type.__name__):
                adapter = object.__new__(adapter_type)
                self.assertFalse(method_requires_pre_rope_key(adapter))

    def test_capability_is_static_and_fails_closed(self) -> None:
        class _Invalid:
            requires_pre_rope_key = "yes"

        self.assertTrue(method_requires_pre_rope_key(_PreRoPEMethod()))
        with self.assertRaisesRegex(TypeError, "class-level bool"):
            method_requires_pre_rope_key(_Invalid())

    def test_legacy_method_receives_only_attention_ready_key(self) -> None:
        method = _LegacyMethod()
        endpoint = _endpoint(method)
        attention, hidden, original_key, positions = _attention_inputs()
        endpoint._attention(
            attention,
            hidden,
            positions,
            torch.tensor([0]),
            measured_decode=True,
        )
        self.assertIsNotNone(method.appended_key)
        expected_pre_rope = original_key.view(1, 1, 8, 128).transpose(1, 2)
        self.assertFalse(torch.equal(method.appended_key, expected_pre_rope))

    def test_declared_method_receives_pre_and_post_rope_keys(self) -> None:
        method = _PreRoPEMethod()
        endpoint = _endpoint(method)
        attention, hidden, original_key, positions = _attention_inputs()
        endpoint._attention(
            attention,
            hidden,
            positions,
            torch.tensor([0]),
            measured_decode=True,
        )
        expected_pre_rope = original_key.view(1, 1, 8, 128).transpose(1, 2)
        self.assertIsNotNone(method.pre_rope_key)
        self.assertTrue(torch.equal(method.pre_rope_key, expected_pre_rope))
        self.assertFalse(torch.equal(method.appended_key, method.pre_rope_key))
        self.assertEqual(tuple(method.pre_rope_key.shape), (1, 8, 1, 128))
        self.assertEqual(
            method.pre_rope_key.untyped_storage().data_ptr(),
            endpoint.query_rope_scratch[0].untyped_storage().data_ptr(),
        )

    def test_query_scratch_reuse_fails_closed_when_storage_is_insufficient(
        self,
    ) -> None:
        key = torch.empty((1, 8, 1, 128), dtype=torch.bfloat16)
        scratch = torch.empty((1, 7, 1, 64), dtype=torch.bfloat16)
        with self.assertRaisesRegex(EndpointGeometryError, "too small"):
            preserve_pre_rope_key_in_query_scratch(scratch, key)

    def test_boundary_helper_has_no_forbidden_measured_operations(self) -> None:
        source = inspect.getsource(preserve_pre_rope_key_in_query_scratch)
        for forbidden in (
            "torch.cat",
            "repeat_kv",
            "repeat_interleave",
            ".item(",
            ".tolist(",
            ".cpu(",
            ".numpy(",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
