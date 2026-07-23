"""Explicit fail-closed cache-method adapter construction."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType

from kvbench.adapters.base import KVCacheMethod, MethodRuntimeContext
from kvbench.adapters.bf16 import BF16MethodAdapter
from kvbench.errors import ConfigLoadError, PhaseNotImplementedError
from kvbench.schema import MethodConfig, MethodName


AdapterBuilder = Callable[[MethodRuntimeContext], KVCacheMethod]


def _build_bf16(runtime_context: MethodRuntimeContext) -> KVCacheMethod:
    return BF16MethodAdapter(runtime_context)


_BUILDERS: Mapping[str, AdapterBuilder] = MappingProxyType(
    {"bf16": _build_bf16}
)
_DEFERRED_METHODS = frozenset({"turboquant", "kivi", "kvquant"})


def _method_name(method_config: MethodConfig | MethodName | str) -> str:
    if isinstance(method_config, MethodConfig):
        return method_config.method.value
    if isinstance(method_config, MethodName):
        return method_config.value
    if type(method_config) is str:
        return method_config
    raise ConfigLoadError("method adapter selection has an invalid type")


def build_method_adapter(
    method_config: MethodConfig | MethodName | str,
    runtime_context: MethodRuntimeContext,
) -> KVCacheMethod:
    """Construct one explicit adapter; later methods remain unimplemented."""

    name = _method_name(method_config)
    if name in _DEFERRED_METHODS:
        raise PhaseNotImplementedError(
            f"{name} method adapter is deferred beyond Phase 4"
        )
    try:
        builder = _BUILDERS[name]
    except KeyError as error:
        raise ConfigLoadError("unknown method adapter") from error
    if isinstance(method_config, MethodConfig):
        if (
            method_config.method is not MethodName.BF16
            or method_config.method_config_id != "bf16"
        ):
            raise ConfigLoadError("BF16 adapter requires the BF16 method config")
    return builder(runtime_context)
