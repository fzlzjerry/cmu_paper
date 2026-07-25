"""Explicit fail-closed cache-method adapter construction."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType

from kvbench.adapters.base import KVCacheMethod, MethodRuntimeContext
from kvbench.adapters.bf16 import BF16MethodAdapter
from kvbench.adapters.turboquant import (
    TURBOQUANT_SOURCE_COMMIT,
    TurboQuantMethodAdapter,
)
from kvbench.errors import ConfigLoadError, PhaseNotImplementedError
from kvbench.schema import MethodConfig, MethodName
from kvbench.schema.config import TurboQuantParameters, VariantRole


AdapterBuilder = Callable[[MethodRuntimeContext], KVCacheMethod]


def _build_bf16(runtime_context: MethodRuntimeContext) -> KVCacheMethod:
    return BF16MethodAdapter(runtime_context)


_BUILDERS: Mapping[str, AdapterBuilder] = MappingProxyType(
    {"bf16": _build_bf16}
)
_DEFERRED_METHODS = frozenset({"kivi", "kvquant"})
_TURBOQUANT_CONFIGS = frozenset(
    {
        "turboquant_4bit_nc",
        "turboquant_k3v4_nc",
        "turboquant_3bit_nc",
    }
)
_TURBOQUANT_RUNTIME_PARAMETERS = {
    "turboquant_4bit_nc": (4, 4, "mse", True),
    "turboquant_k3v4_nc": (3, 4, "mse", True),
    "turboquant_3bit_nc": (3, 3, "mse", True),
}


def _method_name(method_config: MethodConfig | MethodName | str) -> str:
    if isinstance(method_config, MethodConfig):
        return method_config.method.value
    if isinstance(method_config, MethodName):
        return method_config.value
    if type(method_config) is str:
        if method_config in _TURBOQUANT_CONFIGS:
            return MethodName.TURBOQUANT.value
        return method_config
    raise ConfigLoadError("method adapter selection has an invalid type")


def _turboquant_config_name(
    method_config: MethodConfig | MethodName | str,
    variant_id: str | None,
) -> str:
    inferred = (
        method_config
        if type(method_config) is str
        and method_config in _TURBOQUANT_CONFIGS
        else None
    )
    if inferred is not None and variant_id not in {None, inferred}:
        raise ConfigLoadError("TurboQuant preset selection is ambiguous")
    selected = inferred if inferred is not None else variant_id
    if selected not in _TURBOQUANT_CONFIGS:
        raise ConfigLoadError(
            "TurboQuant requires one explicit mandatory configuration"
        )
    if isinstance(method_config, MethodConfig):
        if (
            method_config.method is not MethodName.TURBOQUANT
            or method_config.method_config_id != "turboquant"
            or method_config.source_revision != TURBOQUANT_SOURCE_COMMIT
        ):
            raise ConfigLoadError(
                "TurboQuant adapter requires the pinned method config"
            )
        variant = next(
            (
                item
                for item in method_config.variants
                if item.variant_id == selected
            ),
            None,
        )
        if variant is None or variant.role is not VariantRole.MAIN:
            raise ConfigLoadError(
                "TurboQuant configuration is not a mandatory main variant"
            )
        parameters = variant.parameters
        if not isinstance(parameters, TurboQuantParameters):
            raise ConfigLoadError("TurboQuant parameters have the wrong type")
        observed = (
            parameters.key_bits,
            parameters.value_bits,
            parameters.key_path,
            parameters.norm_correction,
        )
        if observed != _TURBOQUANT_RUNTIME_PARAMETERS[selected]:
            raise ConfigLoadError(
                "TurboQuant parameters differ from the pinned preset"
            )
        optional_runtime = (
            parameters.cache_dtype_name,
            parameters.skipped_layers,
            parameters.block_size,
            parameters.decode_split_count,
        )
        expected_runtime = (selected, (0, 1, 30, 31), 16, 4)
        if any(value is not None for value in optional_runtime):
            if optional_runtime != expected_runtime:
                raise ConfigLoadError(
                    "TurboQuant runtime parameters differ from Phase 6"
                )
    return selected


def build_method_adapter(
    method_config: MethodConfig | MethodName | str,
    runtime_context: MethodRuntimeContext,
    *,
    variant_id: str | None = None,
) -> KVCacheMethod:
    """Construct one explicit adapter; unimplemented methods remain closed."""

    name = _method_name(method_config)
    if name in _DEFERRED_METHODS:
        raise PhaseNotImplementedError(
            f"{name} method adapter is deferred beyond Phase 6"
        )
    if name == MethodName.TURBOQUANT.value:
        selected = _turboquant_config_name(method_config, variant_id)
        return TurboQuantMethodAdapter(runtime_context, selected)
    if variant_id is not None:
        raise ConfigLoadError(
            "variant_id is accepted only for the TurboQuant adapter"
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
