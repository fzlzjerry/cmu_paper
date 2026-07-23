"""Stable Phase 4 cache-method adapter surface."""

from kvbench.adapters.base import KVCacheMethod, MethodRuntimeContext
from kvbench.adapters.bf16 import (
    BF16_ADAPTER_FINGERPRINT_SCHEMA_VERSION,
    BF16_ADAPTER_VERSION,
    BF16MethodAdapter,
    declared_bf16_runtime_context,
)
from kvbench.adapters.factory import build_method_adapter

__all__ = [
    "BF16_ADAPTER_FINGERPRINT_SCHEMA_VERSION",
    "BF16_ADAPTER_VERSION",
    "BF16MethodAdapter",
    "KVCacheMethod",
    "MethodRuntimeContext",
    "build_method_adapter",
    "declared_bf16_runtime_context",
]
