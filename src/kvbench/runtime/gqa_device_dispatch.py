"""Untimed CUDA-device dispatch evidence for the Phase 3 GQA audit.

Chrome parsing and proof evaluation intentionally have no PyTorch dependency.
The collection helper imports PyTorch lazily and returns only raw-artifact
identity: profiler timestamps and durations never become benchmark timing.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Set
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any

from kvbench.runtime.gqa_taxonomy import classify_gqa_evidence
from kvbench.schema import GQAVerdict


FLASH_FORWARD_FAMILY = "pytorch_flash::flash_fwd_kernel"
FLASH_SPLIT_KV_FAMILY = "pytorch_flash::flash_fwd_splitkv"
DEVICE_EVENT_CATEGORIES = frozenset({"kernel", "gpu_memcpy", "gpu_memset"})
MATERIALIZATION_CLASSIFICATIONS = frozenset(
    {
        "device_copy",
        "repeat_materialization",
        "expand_materialization",
        "transpose_materialization",
        "copy_materialization",
    }
)
_SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")
_EVENT_CLASSIFICATIONS = frozenset(
    {
        "flash_attention",
        "device_copy",
        "device_memset",
        "repeat_materialization",
        "expand_materialization",
        "transpose_materialization",
        "copy_materialization",
        "unknown_kernel",
    }
)


class GQADeviceDispatchError(RuntimeError):
    """Device-dispatch evidence could not be collected or validated."""


class ChromeTraceValidationError(GQADeviceDispatchError):
    """A Chrome trace cannot support a fail-closed CUDA audit."""


def _strict_json_loads(raw: bytes) -> object:
    if not isinstance(raw, bytes) or not raw:
        raise ChromeTraceValidationError("Chrome trace bytes are absent")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ChromeTraceValidationError("Chrome trace is not UTF-8") from error

    def reject_constant(value: str) -> None:
        raise ChromeTraceValidationError(
            f"Chrome trace contains a non-finite constant: {value}"
        )

    try:
        return json.loads(text, parse_constant=reject_constant)
    except ChromeTraceValidationError:
        raise
    except (json.JSONDecodeError, RecursionError) as error:
        raise ChromeTraceValidationError("Chrome trace is malformed JSON") from error


def _required_trace_integer(
    args: Mapping[str, object],
    key: str,
    *,
    positive: bool,
) -> int:
    value = args.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ChromeTraceValidationError(
            "CUDA device event lacks integer " + repr(key)
        )
    minimum = 1 if positive else 0
    if value < minimum:
        raise ChromeTraceValidationError(
            "CUDA device event has invalid " + repr(key)
        )
    return value


def _optional_trace_integer(args: Mapping[str, object], key: str) -> int | None:
    value = args.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ChromeTraceValidationError(
            "CUDA device event has invalid optional " + repr(key)
        )
    return value


def normalize_kernel_family(name: str) -> str | None:
    """Return a frozen Flash forward family without matching templates."""

    if not isinstance(name, str) or not name.strip():
        raise ValueError("kernel name must be a non-empty string")
    if "pytorch_flash::flash_fwd_splitkv" in name:
        return FLASH_SPLIT_KV_FAMILY
    if FLASH_FORWARD_FAMILY in name:
        return FLASH_FORWARD_FAMILY
    return None


def classify_device_event(category: str, name: str) -> tuple[str, str | None]:
    """Classify a CUDA activity without high-level ATen child names."""

    if category not in DEVICE_EVENT_CATEGORIES:
        raise ValueError("unsupported CUDA device event category")
    family = normalize_kernel_family(name)
    if family is not None:
        return "flash_attention", family
    if category == "gpu_memcpy":
        return "device_copy", None
    if category == "gpu_memset":
        return "device_memset", None
    lowered = name.casefold().replace("-", "_")
    if "repeat_interleave" in lowered or "repeat_kv" in lowered:
        return "repeat_materialization", None
    if "repeat" in lowered:
        return "repeat_materialization", None
    if "expand" in lowered and (
        "copy" in lowered or "kernel" in lowered or "material" in lowered
    ):
        return "expand_materialization", None
    if "transpose" in lowered and ("copy" in lowered or "kernel" in lowered):
        return "transpose_materialization", None
    if "copy" in lowered or "memcpy" in lowered:
        return "copy_materialization", None
    return "unknown_kernel", None


@dataclass(frozen=True, slots=True)
class CUDADeviceEvent:
    """Parsed device event with ordering but no trace timing fields."""

    order: int
    category: str
    name: str
    stream: int
    correlation_id: int
    external_id: int
    device: int | None
    context: int | None
    classification: str
    kernel_family: str | None

    def __post_init__(self) -> None:
        if self.order < 0:
            raise ValueError("device event order must be nonnegative")
        if self.category not in DEVICE_EVENT_CATEGORIES:
            raise ValueError("device event category is unsupported")
        if not self.name.strip():
            raise ValueError("device event name must be non-empty")
        if self.stream < 0 or self.correlation_id <= 0 or self.external_id <= 0:
            raise ValueError("device event identifiers are invalid")
        if self.device is not None and self.device < 0:
            raise ValueError("device event device is invalid")
        if self.context is not None and self.context < 0:
            raise ValueError("device event context is invalid")
        if self.classification not in _EVENT_CLASSIFICATIONS:
            raise ValueError("device event classification is unsupported")
        classified_as_flash = self.classification == "flash_attention"
        if classified_as_flash != (self.kernel_family is not None):
            raise ValueError("Flash classification and kernel family differ")

    def to_dict(self) -> dict[str, object]:
        return {
            "order": self.order,
            "category": self.category,
            "name": self.name,
            "stream": self.stream,
            "correlation_id": self.correlation_id,
            "external_id": self.external_id,
            "device": self.device,
            "context": self.context,
            "classification": self.classification,
            "kernel_family": self.kernel_family,
        }


@dataclass(frozen=True, slots=True)
class _TraceEventCandidate:
    timestamp: float
    original_index: int
    category: str
    name: str
    stream: int
    correlation_id: int
    external_id: int
    device: int | None
    context: int | None
    classification: str
    kernel_family: str | None


def parse_chrome_cuda_events(raw: bytes) -> tuple[CUDADeviceEvent, ...]:
    """Strictly parse CUDA events from untouched Chrome JSON bytes.

    Device timestamps establish event order only. Neither timestamps nor
    durations are retained in the returned evidence.
    """

    payload = _strict_json_loads(raw)
    if not isinstance(payload, dict):
        raise ChromeTraceValidationError("Chrome trace root must be an object")
    trace_events = payload.get("traceEvents")
    if not isinstance(trace_events, list):
        raise ChromeTraceValidationError("Chrome trace lacks traceEvents")

    parsed: list[_TraceEventCandidate] = []
    for original_index, raw_event in enumerate(trace_events):
        if not isinstance(raw_event, dict):
            raise ChromeTraceValidationError("Chrome trace event must be an object")
        category = raw_event.get("cat")
        if category not in DEVICE_EVENT_CATEGORIES:
            continue
        if raw_event.get("ph") != "X":
            raise ChromeTraceValidationError(
                "CUDA device event must be a complete activity"
            )
        name = raw_event.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ChromeTraceValidationError("CUDA device event lacks a name")
        timestamp = raw_event.get("ts")
        if (
            not isinstance(timestamp, (int, float))
            or isinstance(timestamp, bool)
            or not math.isfinite(float(timestamp))
        ):
            raise ChromeTraceValidationError(
                "CUDA device event lacks a finite ordering timestamp"
            )
        args = raw_event.get("args")
        if not isinstance(args, dict):
            raise ChromeTraceValidationError("CUDA device event lacks arguments")
        classification, family = classify_device_event(category, name)
        parsed.append(
            _TraceEventCandidate(
                timestamp=float(timestamp),
                original_index=original_index,
                category=category,
                name=name,
                stream=_required_trace_integer(args, "stream", positive=False),
                correlation_id=_required_trace_integer(
                    args, "correlation", positive=True
                ),
                external_id=_required_trace_integer(
                    args, "External id", positive=True
                ),
                device=_optional_trace_integer(args, "device"),
                context=_optional_trace_integer(args, "context"),
                classification=classification,
                kernel_family=family,
            )
        )

    parsed.sort(key=lambda item: (item.timestamp, item.original_index))
    return tuple(
        CUDADeviceEvent(
            order=order,
            category=item.category,
            name=item.name,
            stream=item.stream,
            correlation_id=item.correlation_id,
            external_id=item.external_id,
            device=item.device,
            context=item.context,
            classification=item.classification,
            kernel_family=item.kernel_family,
        )
        for order, item in enumerate(parsed)
    )


def _positive_integer(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class KVByteEvidence:
    """Exact native and hypothetical query-head-expanded K/V byte counts."""

    batch_size: int
    num_query_heads: int
    num_kv_heads: int
    context_length: int
    head_dim: int
    dtype_bytes: int
    native_kv_bytes: int
    expanded_kv_bytes: int

    def to_dict(self) -> dict[str, int]:
        return {
            "batch_size": self.batch_size,
            "num_query_heads": self.num_query_heads,
            "num_kv_heads": self.num_kv_heads,
            "context_length": self.context_length,
            "head_dim": self.head_dim,
            "dtype_bytes": self.dtype_bytes,
            "native_kv_bytes": self.native_kv_bytes,
            "expanded_kv_bytes": self.expanded_kv_bytes,
        }


def calculate_kv_bytes(
    *,
    batch_size: int,
    num_query_heads: int,
    num_kv_heads: int,
    context_length: int,
    head_dim: int,
    dtype_bytes: int,
) -> KVByteEvidence:
    """Calculate both preregistered K+V formulas exactly."""

    values = {
        "batch_size": _positive_integer(batch_size, "batch_size"),
        "num_query_heads": _positive_integer(
            num_query_heads, "num_query_heads"
        ),
        "num_kv_heads": _positive_integer(num_kv_heads, "num_kv_heads"),
        "context_length": _positive_integer(context_length, "context_length"),
        "head_dim": _positive_integer(head_dim, "head_dim"),
        "dtype_bytes": _positive_integer(dtype_bytes, "dtype_bytes"),
    }
    if values["num_query_heads"] % values["num_kv_heads"] != 0:
        raise ValueError("query heads must be divisible by KV heads")
    common = (
        2
        * values["batch_size"]
        * values["context_length"]
        * values["head_dim"]
        * values["dtype_bytes"]
    )
    return KVByteEvidence(
        **values,
        native_kv_bytes=common * values["num_kv_heads"],
        expanded_kv_bytes=common * values["num_query_heads"],
    )


@dataclass(frozen=True, slots=True)
class RawTraceArtifact:
    """Identity of an untouched dispatch trace, with no timing metric."""

    relative_path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        path = PurePosixPath(self.relative_path)
        if (
            not self.relative_path
            or path.is_absolute()
            or ".." in path.parts
            or "." in path.parts
        ):
            raise ValueError("raw trace artifact path must be safe and relative")
        if not _SHA256_RE.fullmatch(self.sha256):
            raise ValueError("raw trace SHA-256 is invalid")
        if not isinstance(self.size_bytes, int) or self.size_bytes <= 0:
            raise ValueError("raw trace size must be positive")

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        relative_path: str,
    ) -> RawTraceArtifact:
        raw = path.read_bytes()
        return cls(
            relative_path=relative_path,
            sha256=hashlib.sha256(raw).hexdigest(),
            size_bytes=len(raw),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "run_kind": "dispatch_audit",
            "trace_mechanism": "torch.profiler_cpu_cuda",
            "activities": ["CPU", "CUDA"],
            "profiler_instrumented": True,
            "benchmark_timing_eligible": False,
            "raw_trace_untouched": True,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }
