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
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
from typing import Any
import warnings

from kvbench.runtime.gqa_taxonomy import classify_gqa_evidence
from kvbench.schema import GQAVerdict


FLASH_FORWARD_FAMILY = "pytorch_flash::flash_fwd_kernel"
FLASH_SPLIT_KV_FAMILY = "pytorch_flash::flash_fwd_splitkv"
DEVICE_EVENT_CATEGORIES = frozenset({"kernel", "gpu_memcpy", "gpu_memset"})
MATERIALIZATION_CLASSIFICATIONS = frozenset(
    {
        "repeat_materialization",
        "expand_materialization",
    }
)
COPY_CANDIDATE_CLASSIFICATIONS = frozenset(
    {"device_copy_candidate", "copy_candidate", "transpose_copy_candidate"}
)
FROZEN_CONTROL_BATCH_SIZE = 1
FROZEN_CONTROL_CONTEXT_LENGTH = 128
FROZEN_CONTROL_QUERY_LENGTH = 1
REQUIRED_SUT_SOURCES = (
    "src/kvbench/runtime/backend.py",
    "src/kvbench/runtime/bf16_endpoint.py",
    "src/kvbench/runtime/static_cache.py",
)
_SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")
_EVENT_CLASSIFICATIONS = frozenset(
    {
        "flash_attention",
        "device_copy_candidate",
        "device_memset",
        "repeat_materialization",
        "expand_materialization",
        "transpose_copy_candidate",
        "copy_candidate",
        "unknown_kernel",
    }
)
_FORBIDDEN_SOURCE_PATTERNS = (
    ("repeat_kv", re.compile(r"\brepeat_kv\b")),
    ("repeat_interleave", re.compile(r"\brepeat_interleave\b")),
    ("tensor_repeat", re.compile(r"\.repeat\s*\(")),
    ("tensor_expand", re.compile(r"\.expand\s*\(")),
    ("torch_cat", re.compile(r"\btorch\.cat\s*\(")),
    ("dynamic_cache", re.compile(r"\bDynamicCache\b")),
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
        return "device_copy_candidate", None
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
        return "transpose_copy_candidate", None
    if "copy" in lowered or "memcpy" in lowered:
        return "copy_candidate", None
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
    copy_bytes: int | None
    copy_direction: str | None

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
        is_copy = self.classification in COPY_CANDIDATE_CLASSIFICATIONS
        if self.copy_bytes is not None and (
            not is_copy
            or not isinstance(self.copy_bytes, int)
            or isinstance(self.copy_bytes, bool)
            or self.copy_bytes <= 0
        ):
            raise ValueError("device copy byte evidence is invalid")
        if self.copy_direction is not None and (
            not is_copy or not self.copy_direction.strip()
        ):
            raise ValueError("device copy direction evidence is invalid")

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
            "copy_bytes": self.copy_bytes,
            "copy_direction": self.copy_direction,
        }


@dataclass(frozen=True, slots=True)
class _TraceEventCandidate:
    timestamp: float
    duration: float
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
    copy_bytes: int | None
    copy_direction: str | None


def _trace_event_list(raw: bytes) -> list[dict[str, object]]:
    payload = _strict_json_loads(raw)
    if not isinstance(payload, dict):
        raise ChromeTraceValidationError("Chrome trace root must be an object")
    trace_events = payload.get("traceEvents")
    if not isinstance(trace_events, list):
        raise ChromeTraceValidationError("Chrome trace lacks traceEvents")
    if any(not isinstance(item, dict) for item in trace_events):
        raise ChromeTraceValidationError("Chrome trace event must be an object")
    return trace_events


def _finite_trace_number(event: Mapping[str, object], key: str) -> float:
    value = event.get(key)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ChromeTraceValidationError(
            f"Chrome trace event lacks finite {key!r}"
        )
    return float(value)


def _complete_interval(event: Mapping[str, object]) -> tuple[float, float]:
    if event.get("ph") != "X":
        raise ChromeTraceValidationError("scoped trace event must be complete")
    start = _finite_trace_number(event, "ts")
    duration = _finite_trace_number(event, "dur")
    if duration < 0.0:
        raise ChromeTraceValidationError("trace event duration must be nonnegative")
    return start, start + duration


def _event_args(event: Mapping[str, object]) -> Mapping[str, object]:
    args = event.get("args")
    if not isinstance(args, dict):
        raise ChromeTraceValidationError("Chrome trace event lacks arguments")
    return args


def _trace_integer(event: Mapping[str, object], key: str) -> int:
    value = event.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ChromeTraceValidationError(
            f"Chrome trace event lacks integer {key!r}"
        )
    return value


def _interval_contains(
    outer: tuple[float, float],
    inner: tuple[float, float],
) -> bool:
    tolerance = 0.01
    return bool(
        inner[0] + tolerance >= outer[0]
        and inner[1] <= outer[1] + tolerance
    )


def _copy_metadata(
    *,
    classification: str,
    name: str,
    args: Mapping[str, object],
) -> tuple[int | None, str | None]:
    if classification not in COPY_CANDIDATE_CLASSIFICATIONS:
        return None, None
    size_values: list[int] = []
    for key in ("bytes", "Bytes", "size", "Size"):
        value = args.get(key)
        if value is None:
            continue
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ChromeTraceValidationError("device copy byte count is invalid")
        size_values.append(value)
    if len(set(size_values)) > 1:
        raise ChromeTraceValidationError("device copy byte counts disagree")
    direction_values: list[str] = []
    for key in (
        "copy direction",
        "Copy direction",
        "copy kind",
        "Copy kind",
        "Memcpy type",
    ):
        value = args.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ChromeTraceValidationError("device copy direction is invalid")
        direction_values.append(value.strip())
    if len(set(direction_values)) > 1:
        raise ChromeTraceValidationError("device copy directions disagree")
    if direction_values:
        direction = direction_values[0]
    else:
        match = re.search(r"\b([DH])to([DH])\b", name, flags=re.IGNORECASE)
        direction = None if match is None else match.group(0)
    size = None if not size_values else size_values[0]
    return size, direction


def _parse_device_candidates(
    trace_events: list[dict[str, object]],
) -> tuple[_TraceEventCandidate, ...]:
    parsed: list[_TraceEventCandidate] = []
    for original_index, raw_event in enumerate(trace_events):
        category = raw_event.get("cat")
        if not isinstance(category, str) or category not in DEVICE_EVENT_CATEGORIES:
            continue
        interval = _complete_interval(raw_event)
        name = raw_event.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ChromeTraceValidationError("CUDA device event lacks a name")
        args = _event_args(raw_event)
        classification, family = classify_device_event(category, name)
        copy_bytes, copy_direction = _copy_metadata(
            classification=classification,
            name=name,
            args=args,
        )
        parsed.append(
            _TraceEventCandidate(
                timestamp=interval[0],
                duration=interval[1] - interval[0],
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
                copy_bytes=copy_bytes,
                copy_direction=copy_direction,
            )
        )
    return tuple(parsed)


def _canonical_device_events(
    parsed: tuple[_TraceEventCandidate, ...],
) -> tuple[CUDADeviceEvent, ...]:
    ordered = sorted(parsed, key=lambda item: (item.timestamp, item.original_index))
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
            copy_bytes=item.copy_bytes,
            copy_direction=item.copy_direction,
        )
        for order, item in enumerate(ordered)
    )


def parse_chrome_cuda_events(raw: bytes) -> tuple[CUDADeviceEvent, ...]:
    """Strictly parse CUDA events from untouched Chrome JSON bytes.

    Device timestamps establish event order only. Neither timestamps nor
    durations are retained in the returned evidence.
    """

    return _canonical_device_events(_parse_device_candidates(_trace_event_list(raw)))


@dataclass(frozen=True, slots=True)
class TraceScopeEvidence:
    """Marker-to-CPU-to-runtime-to-device linkage with timing removed."""

    marker: str
    marker_external_id: int
    cpu_process_id: int
    cpu_thread_id: int
    sdpa_external_id: int
    nested_cpu_external_ids: tuple[int, ...]
    runtime_correlations: tuple[int, ...]
    gpu_stream: int

    def __post_init__(self) -> None:
        if not self.marker.strip():
            raise ValueError("trace scope marker is absent")
        for name in (
            "marker_external_id",
            "cpu_process_id",
            "cpu_thread_id",
            "sdpa_external_id",
        ):
            _positive_integer(getattr(self, name), name)
        if self.gpu_stream < 0:
            raise ValueError("trace scope GPU stream is invalid")
        if (
            not self.nested_cpu_external_ids
            or self.sdpa_external_id not in self.nested_cpu_external_ids
            or self.nested_cpu_external_ids
            != tuple(sorted(set(self.nested_cpu_external_ids)))
        ):
            raise ValueError("trace scope CPU external IDs are not canonical")
        if (
            not self.runtime_correlations
            or self.runtime_correlations
            != tuple(sorted(set(self.runtime_correlations)))
            or any(value <= 0 for value in self.runtime_correlations)
        ):
            raise ValueError("trace scope correlations are not canonical")

    def to_dict(self) -> dict[str, object]:
        return {
            "marker": self.marker,
            "marker_external_id": self.marker_external_id,
            "cpu_process_id": self.cpu_process_id,
            "cpu_thread_id": self.cpu_thread_id,
            "sdpa_external_id": self.sdpa_external_id,
            "nested_cpu_external_ids": list(self.nested_cpu_external_ids),
            "runtime_correlations": list(self.runtime_correlations),
            "gpu_stream": self.gpu_stream,
            "timestamps_retained": False,
            "durations_retained": False,
        }


@dataclass(frozen=True, slots=True)
class ScopedCUDAActivities:
    scope: TraceScopeEvidence
    device_events: tuple[CUDADeviceEvent, ...]

    def __post_init__(self) -> None:
        if not self.device_events:
            raise ValueError("scoped CUDA activity is absent")
        if any(event.stream != self.scope.gpu_stream for event in self.device_events):
            raise ValueError("scoped CUDA event stream differs from marker")


def _is_device_producing_runtime(name: str) -> bool:
    lowered = name.casefold()
    return "launch" in lowered or "memcpy" in lowered or "memset" in lowered


def parse_scoped_chrome_cuda_events(
    raw: bytes,
    *,
    marker: str,
) -> ScopedCUDAActivities:
    """Require an unambiguous marker/CPU/runtime/device correlation chain."""

    if not isinstance(marker, str) or not marker.strip():
        raise ValueError("dispatch trace marker must be non-empty")
    trace_events = _trace_event_list(raw)
    host_markers = [
        event
        for event in trace_events
        if event.get("cat") == "user_annotation" and event.get("name") == marker
    ]
    gpu_markers = [
        event
        for event in trace_events
        if event.get("cat") == "gpu_user_annotation"
        and event.get("name") == marker
    ]
    if len(host_markers) != 1 or len(gpu_markers) != 1:
        raise ChromeTraceValidationError(
            "dispatch marker does not have one host/GPU annotation pair"
        )
    host_marker = host_markers[0]
    gpu_marker = gpu_markers[0]
    host_interval = _complete_interval(host_marker)
    gpu_interval = _complete_interval(gpu_marker)
    host_args = _event_args(host_marker)
    gpu_args = _event_args(gpu_marker)
    marker_external_id = _required_trace_integer(
        host_args, "External id", positive=True
    )
    if (
        _required_trace_integer(gpu_args, "External id", positive=True)
        != marker_external_id
    ):
        raise ChromeTraceValidationError("host/GPU marker external IDs differ")
    cpu_pid = _trace_integer(host_marker, "pid")
    cpu_tid = _trace_integer(host_marker, "tid")
    gpu_stream = _trace_integer(gpu_marker, "tid")
    root_sdpa = [
        event
        for event in trace_events
        if event.get("cat") == "cpu_op"
        and event.get("name") == "aten::scaled_dot_product_attention"
        and event.get("pid") == cpu_pid
        and event.get("tid") == cpu_tid
        and _interval_contains(host_interval, _complete_interval(event))
    ]
    if len(root_sdpa) != 1:
        raise ChromeTraceValidationError(
            "dispatch marker does not contain exactly one root SDPA CPU op"
        )
    sdpa_interval = _complete_interval(root_sdpa[0])
    sdpa_external_id = _required_trace_integer(
        _event_args(root_sdpa[0]), "External id", positive=True
    )
    nested_cpu_ids: list[int] = []
    for event in trace_events:
        if (
            event.get("cat") == "cpu_op"
            and event.get("pid") == cpu_pid
            and event.get("tid") == cpu_tid
            and _interval_contains(sdpa_interval, _complete_interval(event))
        ):
            nested_cpu_ids.append(
                _required_trace_integer(
                    _event_args(event), "External id", positive=True
                )
            )
    if len(nested_cpu_ids) != len(set(nested_cpu_ids)):
        raise ChromeTraceValidationError("nested CPU external IDs are ambiguous")
    nested_ids = frozenset(nested_cpu_ids)
    runtime_by_link: dict[tuple[int, int], list[Mapping[str, object]]] = {}
    producing_runtime_links: set[tuple[int, int]] = set()
    for event in trace_events:
        if (
            event.get("cat") != "cuda_runtime"
            or event.get("pid") != cpu_pid
            or event.get("tid") != cpu_tid
            or not _interval_contains(sdpa_interval, _complete_interval(event))
        ):
            continue
        args = _event_args(event)
        external_id = args.get("External id")
        if external_id not in nested_ids:
            continue
        if not isinstance(external_id, int) or isinstance(external_id, bool):
            raise ChromeTraceValidationError("runtime external ID is invalid")
        correlation = _required_trace_integer(args, "correlation", positive=True)
        link = (external_id, correlation)
        runtime_by_link.setdefault(link, []).append(event)
        name = event.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ChromeTraceValidationError("CUDA runtime event lacks a name")
        if _is_device_producing_runtime(name):
            producing_runtime_links.add(link)
    candidates = _parse_device_candidates(trace_events)
    selected: list[_TraceEventCandidate] = []
    for candidate in candidates:
        interval = (
            candidate.timestamp,
            candidate.timestamp + candidate.duration,
        )
        linked_cpu = candidate.external_id in nested_ids
        within_gpu_marker = _interval_contains(gpu_interval, interval)
        on_marker_stream = candidate.stream == gpu_stream
        link = (candidate.external_id, candidate.correlation_id)
        runtime_matches = runtime_by_link.get(link, [])
        related = linked_cpu or within_gpu_marker
        fully_linked = bool(
            linked_cpu
            and within_gpu_marker
            and on_marker_stream
            and len(runtime_matches) == 1
            and link in producing_runtime_links
        )
        if related and not fully_linked:
            raise ChromeTraceValidationError(
                "CUDA device event is ambiguously or incompletely marker-linked"
            )
        if fully_linked:
            selected.append(candidate)
    if not selected:
        raise ChromeTraceValidationError("marked SDPA has no linked CUDA activity")
    selected_links = {
        (candidate.external_id, candidate.correlation_id)
        for candidate in selected
    }
    if producing_runtime_links != selected_links:
        raise ChromeTraceValidationError(
            "marked SDPA runtime/device correlations are unmatched"
        )
    correlations = tuple(sorted(candidate.correlation_id for candidate in selected))
    scope = TraceScopeEvidence(
        marker=marker,
        marker_external_id=marker_external_id,
        cpu_process_id=cpu_pid,
        cpu_thread_id=cpu_tid,
        sdpa_external_id=sdpa_external_id,
        nested_cpu_external_ids=tuple(sorted(nested_ids)),
        runtime_correlations=correlations,
        gpu_stream=gpu_stream,
    )
    return ScopedCUDAActivities(
        scope=scope,
        device_events=_canonical_device_events(tuple(selected)),
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


@dataclass(frozen=True, slots=True)
class BackendControlEvidence:
    """Fail-closed Flash forcing, eligibility, and build identity evidence."""

    enabled_backends: tuple[str, ...]
    flash_eligible: bool
    fused_backend_name: str | None
    rejected_control_failed: bool
    rejected_control_error: str | None
    rejected_control_warnings: tuple[str, ...]
    rejected_control_synchronized: bool
    source_build_fingerprint: str
    source_build_verified: bool
    eligibility_diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _SHA256_RE.fullmatch(self.source_build_fingerprint):
            raise ValueError("backend source/build fingerprint is invalid")
        if any(not item.strip() for item in self.eligibility_diagnostics):
            raise ValueError("backend diagnostics must be non-empty strings")
        if self.rejected_control_error is not None and not (
            isinstance(self.rejected_control_error, str)
            and self.rejected_control_error.strip()
        ):
            raise ValueError("rejected-control error must be non-empty")
        if any(not item.strip() for item in self.rejected_control_warnings):
            raise ValueError("rejected-control warnings must be non-empty")

    @property
    def passed(self) -> bool:
        return bool(
            self.enabled_backends == ("FLASH_ATTENTION",)
            and self.flash_eligible
            and self.fused_backend_name == "FLASH_ATTENTION"
            and self.rejected_control_failed
            and self.rejected_control_synchronized
            and self.source_build_verified
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled_backends": list(self.enabled_backends),
            "flash_eligible": self.flash_eligible,
            "fused_backend_name": self.fused_backend_name,
            "rejected_control_failed": self.rejected_control_failed,
            "rejected_control_error": self.rejected_control_error,
            "rejected_control_warnings": list(self.rejected_control_warnings),
            "rejected_control_synchronized": self.rejected_control_synchronized,
            "source_build_fingerprint": self.source_build_fingerprint,
            "source_build_verified": self.source_build_verified,
            "eligibility_diagnostics": list(self.eligibility_diagnostics),
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class TensorShapeEvidence:
    """Tensor metadata read without copying tensor contents to the host."""

    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    device: str
    element_size: int
    storage_bytes: int
    storage_offset: int
    is_contiguous: bool

    def __post_init__(self) -> None:
        if not self.shape or len(self.shape) != len(self.stride):
            raise ValueError("tensor shape and stride ranks must match")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in self.shape
        ):
            raise ValueError("tensor dimensions must be positive integers")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in self.stride
        ):
            raise ValueError("tensor strides must be nonnegative integers")
        _positive_integer(self.element_size, "element_size")
        _positive_integer(self.storage_bytes, "storage_bytes")
        if (
            not isinstance(self.storage_offset, int)
            or isinstance(self.storage_offset, bool)
            or self.storage_offset < 0
        ):
            raise ValueError("tensor storage offset must be nonnegative")
        if not isinstance(self.is_contiguous, bool):
            raise ValueError("tensor contiguity must be boolean")
        if not self.dtype.strip() or not self.device.strip():
            raise ValueError("tensor dtype and device must be recorded")

    @property
    def logical_bytes(self) -> int:
        elements = 1
        for dimension in self.shape:
            elements *= dimension
        return elements * self.element_size

    def to_dict(self) -> dict[str, object]:
        return {
            "shape": list(self.shape),
            "stride": list(self.stride),
            "dtype": self.dtype,
            "device": self.device,
            "element_size": self.element_size,
            "logical_bytes": self.logical_bytes,
            "storage_bytes": self.storage_bytes,
            "storage_offset": self.storage_offset,
            "is_contiguous": self.is_contiguous,
        }


@dataclass(frozen=True, slots=True)
class SourceFileEvidence:
    """Identity and forbidden-path findings for one selected SUT source."""

    relative_path: str
    sha256: str
    findings: tuple[str, ...]

    def __post_init__(self) -> None:
        path = PurePosixPath(self.relative_path)
        if (
            not self.relative_path
            or path.is_absolute()
            or ".." in path.parts
            or "." in path.parts
        ):
            raise ValueError("source audit path must be safe and relative")
        if not _SHA256_RE.fullmatch(self.sha256):
            raise ValueError("source audit SHA-256 is invalid")
        if any(not finding.strip() for finding in self.findings):
            raise ValueError("source audit findings must be non-empty strings")

    @property
    def passed(self) -> bool:
        return not self.findings

    def to_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "findings": list(self.findings),
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class SourceShapeEvidence:
    """Source identity and exact Q/K/V/output metadata for one control."""

    sources: tuple[SourceFileEvidence, ...]
    query: TensorShapeEvidence
    key: TensorShapeEvidence
    value: TensorShapeEvidence
    output: TensorShapeEvidence
    native_kv_storage_verified: bool

    def __post_init__(self) -> None:
        if not self.sources:
            raise ValueError("source audit must retain at least one source")
        paths = tuple(source.relative_path for source in self.sources)
        if len(set(paths)) != len(paths) or paths != tuple(sorted(paths)):
            raise ValueError("source audit paths must be unique and sorted")

    @property
    def source_verified(self) -> bool:
        return all(source.passed for source in self.sources)

    def shape_verified_for(self, control: DispatchControlEvidence) -> bool:
        expected_query = (
            control.batch_size,
            control.num_query_heads,
            control.query_length,
            control.head_dim,
        )
        expected_kv = (
            control.batch_size,
            control.num_kv_heads,
            control.context_length,
            control.head_dim,
        )
        tensors = (self.query, self.key, self.value, self.output)
        return bool(
            self.native_kv_storage_verified
            and self.query.shape == expected_query
            and self.key.shape == expected_kv
            and self.value.shape == expected_kv
            and self.output.shape == expected_query
            and all(tensor.dtype == "torch.bfloat16" for tensor in tensors)
            and len({tensor.device for tensor in tensors}) == 1
            and all(tensor.element_size == 2 for tensor in tensors)
            and all(all(stride > 0 for stride in tensor.stride) for tensor in tensors)
            and self.key.storage_bytes == self.key.logical_bytes
            and self.value.storage_bytes == self.value.logical_bytes
            and self.key.storage_offset == 0
            and self.value.storage_offset == 0
            and self.key.is_contiguous
            and self.value.is_contiguous
        )

    def to_dict(self, *, control: DispatchControlEvidence) -> dict[str, object]:
        return {
            "sources": [source.to_dict() for source in self.sources],
            "query": self.query.to_dict(),
            "key": self.key.to_dict(),
            "value": self.value.to_dict(),
            "output": self.output.to_dict(),
            "native_kv_storage_verified": self.native_kv_storage_verified,
            "source_verified": self.source_verified,
            "shape_verified": self.shape_verified_for(control),
        }


@dataclass(frozen=True, slots=True)
class DispatchControlEvidence:
    """One held-constant GQA or MHA direct device control."""

    role: str
    batch_size: int
    context_length: int
    query_length: int
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    dtype: str
    dtype_bytes: int
    is_causal: bool
    warmup_count: int
    backend: BackendControlEvidence
    raw_trace: RawTraceArtifact | None
    trace_scope: TraceScopeEvidence | None
    device_events: tuple[CUDADeviceEvent, ...]

    def __post_init__(self) -> None:
        if self.role not in {"gqa", "mha_control"}:
            raise ValueError("dispatch control role is unsupported")
        for name in (
            "batch_size",
            "context_length",
            "query_length",
            "num_query_heads",
            "num_kv_heads",
            "head_dim",
            "dtype_bytes",
            "warmup_count",
        ):
            _positive_integer(getattr(self, name), name)
        if self.num_query_heads != 32 or self.head_dim != 128:
            raise ValueError("dispatch control differs from frozen HQ/D geometry")
        expected_kv_heads = 8 if self.role == "gqa" else 32
        if self.num_kv_heads != expected_kv_heads:
            raise ValueError("dispatch control differs from frozen KV geometry")
        if self.dtype != "torch.bfloat16" or self.dtype_bytes != 2:
            raise ValueError("dispatch control must use frozen BF16")
        if tuple(event.order for event in self.device_events) != tuple(
            range(len(self.device_events))
        ):
            raise ValueError("dispatch device event ordering is not canonical")
        if self.trace_scope is not None:
            correlations = tuple(
                sorted({event.correlation_id for event in self.device_events})
            )
            if correlations != self.trace_scope.runtime_correlations:
                raise ValueError("trace scope correlations differ from device events")
            if any(
                event.external_id not in self.trace_scope.nested_cpu_external_ids
                or event.stream != self.trace_scope.gpu_stream
                for event in self.device_events
            ):
                raise ValueError("device events differ from retained trace scope")

    @property
    def byte_evidence(self) -> KVByteEvidence:
        return calculate_kv_bytes(
            batch_size=self.batch_size,
            num_query_heads=self.num_query_heads,
            num_kv_heads=self.num_kv_heads,
            context_length=self.context_length,
            head_dim=self.head_dim,
            dtype_bytes=self.dtype_bytes,
        )

    @property
    def attention_families(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    event.kernel_family
                    for event in self.device_events
                    if event.kernel_family is not None
                }
            )
        )

    @property
    def events_before_attention(self) -> tuple[CUDADeviceEvent, ...]:
        attention_orders = [
            event.order
            for event in self.device_events
            if event.classification == "flash_attention"
        ]
        if not attention_orders:
            return self.device_events
        first_attention = min(attention_orders)
        return tuple(
            event for event in self.device_events if event.order < first_attention
        )

    def held_constants(self) -> tuple[object, ...]:
        return (
            self.batch_size,
            self.context_length,
            self.query_length,
            self.num_query_heads,
            self.head_dim,
            self.dtype,
            self.dtype_bytes,
            self.is_causal,
            self.warmup_count,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "geometry": {
                "batch_size": self.batch_size,
                "context_length": self.context_length,
                "query_length": self.query_length,
                "num_query_heads": self.num_query_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
                "dtype": self.dtype,
                "dtype_bytes": self.dtype_bytes,
                "is_causal": self.is_causal,
                "warmup_count": self.warmup_count,
            },
            "backend": self.backend.to_dict(),
            "raw_trace": None if self.raw_trace is None else self.raw_trace.to_dict(),
            "trace_scope": (
                None if self.trace_scope is None else self.trace_scope.to_dict()
            ),
            "byte_evidence": self.byte_evidence.to_dict(),
            "device_events": [event.to_dict() for event in self.device_events],
            "attention_families": list(self.attention_families),
            "events_before_attention": [
                event.to_dict() for event in self.events_before_attention
            ],
        }


@dataclass(frozen=True, slots=True)
class KernelFamilyComparison:
    """Relationship between the held-constant GQA and MHA controls."""

    relation: str
    gqa_families: tuple[str, ...]
    mha_families: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.relation not in {"same", "related", "unrelated", "unverified"}:
            raise ValueError("kernel family relation is unsupported")

    @property
    def passed(self) -> bool:
        return self.relation in {"same", "related"}

    def to_dict(self) -> dict[str, object]:
        return {
            "relation": self.relation,
            "gqa_families": list(self.gqa_families),
            "mha_families": list(self.mha_families),
            "passed": self.passed,
        }


def compare_kernel_families(
    gqa: DispatchControlEvidence | None,
    mha: DispatchControlEvidence | None,
    *,
    explicitly_related: Set[tuple[str, str]] = frozenset(),
) -> KernelFamilyComparison:
    """Compare normalized families with no implicit related-family waiver."""

    gqa_families = () if gqa is None else gqa.attention_families
    mha_families = () if mha is None else mha.attention_families
    if len(gqa_families) != 1 or len(mha_families) != 1:
        relation = "unverified"
    elif gqa_families == mha_families:
        relation = "same"
    elif (
        (gqa_families[0], mha_families[0]) in explicitly_related
        or (mha_families[0], gqa_families[0]) in explicitly_related
    ):
        relation = "related"
    else:
        relation = "unrelated"
    return KernelFamilyComparison(
        relation=relation,
        gqa_families=gqa_families,
        mha_families=mha_families,
    )


@dataclass(frozen=True, slots=True)
class GQAProofEvaluation:
    """Combined result for every preregistered GQA proof layer."""

    verdict: GQAVerdict
    dispatch_verified: bool
    no_replication_kernel_verified: bool
    allocation_verified: bool
    source_verified: bool
    shape_verified: bool
    family_comparison: KernelFamilyComparison
    positive_materialization_evidence: tuple[str, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "verdict": self.verdict.value,
            "dispatch_verified": self.dispatch_verified,
            "no_replication_kernel_verified": (
                self.no_replication_kernel_verified
            ),
            "allocation_verified": self.allocation_verified,
            "source_verified": self.source_verified,
            "shape_verified": self.shape_verified,
            "family_comparison": self.family_comparison.to_dict(),
            "positive_materialization_evidence": list(
                self.positive_materialization_evidence
            ),
            "reasons": list(self.reasons),
            "performance_timing_reported": False,
        }


def evaluate_gqa_device_dispatch(
    *,
    gqa: DispatchControlEvidence | None,
    mha: DispatchControlEvidence | None,
    allocation_verified: bool,
    source_verified: bool,
    shape_verified: bool,
    expanded_kv_allocation_detected: bool = False,
    expanded_kv_tensor_detected: bool = False,
    explicitly_related_families: Set[tuple[str, str]] = frozenset(),
) -> GQAProofEvaluation:
    """Evaluate device proof through the corrected four-state taxonomy."""

    family = compare_kernel_families(
        gqa,
        mha,
        explicitly_related=explicitly_related_families,
    )
    held_constants_match = bool(
        gqa is not None
        and mha is not None
        and gqa.held_constants() == mha.held_constants()
    )
    source_build_match = bool(
        gqa is not None
        and mha is not None
        and gqa.backend.source_build_fingerprint
        == mha.backend.source_build_fingerprint
    )
    dispatch_verified = bool(
        held_constants_match
        and source_build_match
        and gqa is not None
        and mha is not None
        and gqa.raw_trace is not None
        and mha.raw_trace is not None
        and gqa.trace_scope is not None
        and mha.trace_scope is not None
        and gqa.backend.passed
        and mha.backend.passed
        and family.passed
    )

    positive: list[str] = []
    if expanded_kv_allocation_detected:
        positive.append("expanded_kv_allocation")
    if expanded_kv_tensor_detected:
        positive.append("expanded_kv_tensor")
    if gqa is not None:
        expanded_copy_sizes = {
            gqa.byte_evidence.expanded_kv_bytes,
            gqa.byte_evidence.expanded_kv_bytes // 2,
        }
        positive.extend(
            f"device_event:{event.classification}:{event.name}"
            for event in gqa.events_before_attention
            if event.classification in MATERIALIZATION_CLASSIFICATIONS
        )
        positive.extend(
            f"expanded_kv_copy:{event.copy_bytes}:{event.name}"
            for event in gqa.events_before_attention
            if event.classification in COPY_CANDIDATE_CLASSIFICATIONS
            and event.copy_bytes in expanded_copy_sizes
        )

    no_replication_kernel_verified = bool(
        dispatch_verified
        and gqa is not None
        and mha is not None
        and not gqa.events_before_attention
        and not mha.events_before_attention
        and not any(
            event.classification
            in (MATERIALIZATION_CLASSIFICATIONS | COPY_CANDIDATE_CLASSIFICATIONS)
            for event in (*gqa.device_events, *mha.device_events)
        )
    )
    verdict = classify_gqa_evidence(
        materialization_evidence=bool(positive),
        dispatch_verified=dispatch_verified,
        no_replication_kernel_verified=no_replication_kernel_verified,
        allocation_verified=allocation_verified,
        source_verified=source_verified,
        shape_verified=shape_verified,
    )
    reasons: list[str] = []
    if positive:
        reasons.append("positive materialization evidence exists")
    if not held_constants_match:
        reasons.append("GQA and MHA controls do not hold constants fixed")
    if not source_build_match:
        reasons.append("GQA and MHA backend source/build identities differ")
    if not dispatch_verified:
        reasons.append("device dispatch is not fully verified")
    if dispatch_verified and not no_replication_kernel_verified:
        reasons.append("no-preceding-materialization proof is incomplete")
    if not allocation_verified:
        reasons.append("allocation-size proof is incomplete")
    if not source_verified:
        reasons.append("source proof is incomplete")
    if not shape_verified:
        reasons.append("shape and storage proof is incomplete")
    return GQAProofEvaluation(
        verdict=verdict,
        dispatch_verified=dispatch_verified,
        no_replication_kernel_verified=no_replication_kernel_verified,
        allocation_verified=allocation_verified,
        source_verified=source_verified,
        shape_verified=shape_verified,
        family_comparison=family,
        positive_materialization_evidence=tuple(positive),
        reasons=tuple(reasons),
    )


def collect_torch_profiler_trace(
    operation: Callable[[], Any],
    output_path: Path,
    *,
    artifact_relative_path: str,
    marker: str,
    warmup_count: int,
    device: Any,
) -> RawTraceArtifact:
    """Export one untouched CPU+CUDA Chrome trace outside benchmark timing."""

    if not callable(operation):
        raise TypeError("dispatch operation must be callable")
    _positive_integer(warmup_count, "warmup_count")
    if not marker.strip():
        raise ValueError("dispatch trace marker must be non-empty")
    path = Path(output_path)
    if os.path.lexists(path):
        raise GQADeviceDispatchError("raw trace output already exists")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise GQADeviceDispatchError("raw trace parent must be a real directory")
    staging_name = tempfile.mkdtemp(
        prefix=f".{path.name}.",
        suffix=".staging-dir",
        dir=path.parent,
    )
    staging_directory = Path(staging_name)
    staging_path = staging_directory / "trace.chrome.json"
    reserved_stat = os.lstat(staging_directory)
    if (
        not stat.S_ISDIR(reserved_stat.st_mode)
        or stat.S_IMODE(reserved_stat.st_mode) & 0o077
    ):
        raise GQADeviceDispatchError("raw trace staging directory is not private")

    torch = importlib.import_module("torch")
    profiler = importlib.import_module("torch.profiler")
    for _ in range(warmup_count):
        operation()
    torch.cuda.synchronize(device=device)
    retained_output: Any = None
    with profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as trace:
        with torch.autograd.profiler.record_function(marker):
            retained_output = operation()
    torch.cuda.synchronize(device=device)
    trace.export_chrome_trace(str(staging_path))
    staged_directory_stat = os.lstat(staging_directory)
    staged_stat = os.lstat(staging_path)
    if (
        not stat.S_ISDIR(staged_directory_stat.st_mode)
        or staged_directory_stat.st_dev != reserved_stat.st_dev
        or staged_directory_stat.st_ino != reserved_stat.st_ino
        or not stat.S_ISREG(staged_stat.st_mode)
        or staged_stat.st_size <= 0
    ):
        raise GQADeviceDispatchError(
            "profiler replaced or invalidated the private trace staging area"
        )
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    trace_fd = os.open(staging_path, os.O_RDONLY | nofollow)
    try:
        opened_stat = os.fstat(trace_fd)
        if (
            opened_stat.st_dev != staged_stat.st_dev
            or opened_stat.st_ino != staged_stat.st_ino
        ):
            raise GQADeviceDispatchError("raw trace file changed during validation")
        os.fsync(trace_fd)
        try:
            os.link(staging_path, path, follow_symlinks=False)
        except FileExistsError as error:
            raise GQADeviceDispatchError(
                "raw trace destination appeared before no-replace promotion"
            ) from error
        published_stat = os.lstat(path)
        if (
            not stat.S_ISREG(published_stat.st_mode)
            or published_stat.st_dev != opened_stat.st_dev
            or published_stat.st_ino != opened_stat.st_ino
        ):
            raise GQADeviceDispatchError("published raw trace inode differs")
        artifact = RawTraceArtifact.from_path(
            path,
            relative_path=artifact_relative_path,
        )
    finally:
        os.close(trace_fd)
    os.unlink(staging_path)
    os.rmdir(staging_directory)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    del retained_output
    return artifact


@dataclass(frozen=True, slots=True)
class BackendIdentityEvidence:
    """Canonical, checksum-bound copy of the frozen backend manifest."""

    canonical_json: str
    sha256: str

    def __post_init__(self) -> None:
        try:
            payload = json.loads(self.canonical_json)
        except (json.JSONDecodeError, RecursionError) as error:
            raise ValueError("backend identity JSON is invalid") from error
        if not isinstance(payload, dict):
            raise ValueError("backend identity must be a JSON object")
        canonical = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        if canonical != self.canonical_json:
            raise ValueError("backend identity JSON is not canonical")
        if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != self.sha256:
            raise ValueError("backend identity SHA-256 differs")

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> BackendIdentityEvidence:
        canonical = json.dumps(
            dict(payload),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        return cls(
            canonical_json=canonical,
            sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        )

    def to_dict(self) -> dict[str, object]:
        payload = json.loads(self.canonical_json)
        if not isinstance(payload, dict):  # pragma: no cover - guarded at creation
            raise AssertionError("validated backend identity changed type")
        return {"sha256": self.sha256, "manifest": payload}


def audit_gqa_source_files(
    source_root: Path,
    source_paths: tuple[Path, ...],
) -> tuple[SourceFileEvidence, ...]:
    """Hash and scan the selected frozen SUT sources, failing on ambiguity."""

    root = Path(source_root)
    if not root.is_dir() or root.is_symlink():
        raise GQADeviceDispatchError("source root must be a real directory")
    root = root.resolve(strict=True)
    if not source_paths:
        raise GQADeviceDispatchError("source audit paths are absent")
    evidence: list[SourceFileEvidence] = []
    for raw_path in source_paths:
        supplied = Path(raw_path)
        candidate = supplied if supplied.is_absolute() else root / supplied
        if not candidate.is_file() or candidate.is_symlink():
            raise GQADeviceDispatchError("source audit target is not a real file")
        resolved = candidate.resolve(strict=True)
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError as error:
            raise GQADeviceDispatchError(
                "source audit target escapes the source root"
            ) from error
        raw = resolved.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise GQADeviceDispatchError(
                f"source audit target is not UTF-8: {relative}"
            ) from error
        findings = tuple(
            label
            for label, pattern in _FORBIDDEN_SOURCE_PATTERNS
            if pattern.search(text) is not None
        )
        evidence.append(
            SourceFileEvidence(
                relative_path=relative,
                sha256=hashlib.sha256(raw).hexdigest(),
                findings=findings,
            )
        )
    evidence.sort(key=lambda item: item.relative_path)
    paths = tuple(item.relative_path for item in evidence)
    if len(set(paths)) != len(paths):
        raise GQADeviceDispatchError("source audit paths contain duplicates")
    return tuple(evidence)


def tensor_shape_evidence(tensor: Any) -> TensorShapeEvidence:
    """Read tensor metadata without copying tensor contents to the host."""

    required = (
        "shape",
        "stride",
        "dtype",
        "device",
        "element_size",
        "untyped_storage",
        "storage_offset",
        "is_contiguous",
    )
    if any(not hasattr(tensor, attribute) for attribute in required):
        raise GQADeviceDispatchError("shape evidence requires a tensor object")
    storage = tensor.untyped_storage()
    return TensorShapeEvidence(
        shape=tuple(int(item) for item in tensor.shape),
        stride=tuple(int(item) for item in tensor.stride()),
        dtype=str(tensor.dtype),
        device=str(tensor.device),
        element_size=int(tensor.element_size()),
        storage_bytes=int(storage.nbytes()),
        storage_offset=int(tensor.storage_offset()),
        is_contiguous=bool(tensor.is_contiguous()),
    )


def _backend_enum_name(backend: object) -> str:
    name = getattr(backend, "name", None)
    if isinstance(name, str) and name:
        return name
    rendered = str(backend)
    return rendered.rsplit(".", maxsplit=1)[-1]


@dataclass(frozen=True, slots=True)
class _RejectedBackendControl:
    passed: bool
    error: str | None
    warning_messages: tuple[str, ...]
    synchronized: bool


def _collect_rejected_backend_control(
    torch: Any,
    *,
    device: Any,
    num_kv_heads: int,
    is_causal: bool,
    scale: float,
    forced_flash_execution: Callable[[], Any],
) -> _RejectedBackendControl:
    query = torch.empty((1, 32, 1, 128), dtype=torch.float64, device=device)
    key = torch.empty((1, num_kv_heads, 1, 128), dtype=torch.float64, device=device)
    value = torch.empty_like(key)
    torch.cuda.synchronize(device=device)
    error_message: str | None = None
    output: Any = None
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            with forced_flash_execution():
                output = torch.nn.functional.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=is_causal,
                    scale=float(scale),
                    enable_gqa=True,
                )
        except RuntimeError as error:
            error_message = str(error).strip()
    warning_messages = tuple(str(item.message) for item in captured)
    try:
        torch.cuda.synchronize(device=device)
    except RuntimeError as error:
        raise GQADeviceDispatchError(
            "rejected-backend control left an asynchronous CUDA error"
        ) from error
    del output, query, key, value
    expected_error = "No available kernel. Aborting execution."
    has_flash_diagnostic = any(
        "Flash attention kernel not used because:" in item
        for item in warning_messages
    )
    has_dtype_diagnostic = any(
        "Expected query, key and value to all be of dtype: {Half, BFloat16}."
        in item
        for item in warning_messages
    )
    return _RejectedBackendControl(
        passed=bool(
            error_message == expected_error
            and has_flash_diagnostic
            and has_dtype_diagnostic
        ),
        error=error_message,
        warning_messages=warning_messages,
        synchronized=True,
    )


def _collect_backend_control(
    torch: Any,
    *,
    query: Any,
    key: Any,
    value: Any,
    is_causal: bool,
    scale: float,
    identity: BackendIdentityEvidence,
    forced_flash_execution: Callable[[], Any],
) -> BackendControlEvidence:
    current_backends = getattr(
        torch.nn.attention,
        "_cur_sdpa_kernel_backends",
        None,
    )
    if not callable(current_backends):
        raise GQADeviceDispatchError(
            "enabled-SDPA-backend diagnostics are unavailable"
        )
    diagnostics: list[str] = []
    with forced_flash_execution():
        enabled = tuple(_backend_enum_name(item) for item in current_backends())
        params = torch.backends.cuda.SDPAParams(
            query,
            key,
            value,
            None,
            0.0,
            is_causal,
            True,
        )
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            flash_eligible = bool(
                torch.backends.cuda.can_use_flash_attention(params, debug=True)
            )
        diagnostics.extend(str(item.message) for item in captured)
        try:
            choice = int(
                torch._fused_sdp_choice(
                    query,
                    key,
                    value,
                    None,
                    0.0,
                    is_causal,
                    scale=float(scale),
                    enable_gqa=True,
                )
            )
        except RuntimeError:
            choice = -1
    expected = int(torch.nn.attention.SDPBackend.FLASH_ATTENTION.value)
    fused_backend_name = "FLASH_ATTENTION" if choice == expected else None
    diagnostics.extend(
        (
            "enabled_backends=" + ",".join(enabled),
            f"can_use_flash_attention={flash_eligible}",
            f"fused_sdp_choice={choice}",
        )
    )
    rejected = _collect_rejected_backend_control(
        torch,
        device=query.device,
        num_kv_heads=int(key.shape[1]),
        is_causal=is_causal,
        scale=scale,
        forced_flash_execution=forced_flash_execution,
    )
    return BackendControlEvidence(
        enabled_backends=enabled,
        flash_eligible=flash_eligible,
        fused_backend_name=fused_backend_name,
        rejected_control_failed=rejected.passed,
        rejected_control_error=rejected.error,
        rejected_control_warnings=rejected.warning_messages,
        rejected_control_synchronized=rejected.synchronized,
        source_build_fingerprint=identity.sha256,
        source_build_verified=True,
        eligibility_diagnostics=tuple(diagnostics),
    )


def _validate_control_tensors(
    torch: Any,
    *,
    role: str,
    query: Any,
    key: Any,
    value: Any,
) -> None:
    if role not in {"gqa", "mha_control"}:
        raise ValueError("dispatch control role is unsupported")
    if not all(isinstance(item, torch.Tensor) for item in (query, key, value)):
        raise GQADeviceDispatchError("dispatch controls require tensors")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise GQADeviceDispatchError("dispatch control tensors must be rank four")
    expected_kv_heads = 8 if role == "gqa" else 32
    if int(query.shape[1]) != 32 or int(key.shape[1]) != expected_kv_heads:
        raise GQADeviceDispatchError("dispatch control head geometry differs")
    if int(query.shape[-1]) != 128 or int(key.shape[-1]) != 128:
        raise GQADeviceDispatchError("dispatch control head dimension differs")
    if int(query.shape[-2]) != FROZEN_CONTROL_QUERY_LENGTH:
        raise GQADeviceDispatchError("dispatch control must isolate one decode query")
    if tuple(key.shape) != tuple(value.shape):
        raise GQADeviceDispatchError("dispatch control K/V shapes differ")
    if int(query.shape[0]) != int(key.shape[0]):
        raise GQADeviceDispatchError("dispatch control batch dimensions differ")
    if int(query.shape[0]) != FROZEN_CONTROL_BATCH_SIZE:
        raise GQADeviceDispatchError("dispatch control batch differs from frozen one")
    if int(key.shape[-2]) != FROZEN_CONTROL_CONTEXT_LENGTH:
        raise GQADeviceDispatchError("dispatch control context differs from frozen one")
    if query.dtype != torch.bfloat16 or key.dtype != query.dtype:
        raise GQADeviceDispatchError("dispatch control must use BF16 Q/K/V")
    if value.dtype != query.dtype:
        raise GQADeviceDispatchError("dispatch control V dtype differs")
    if query.device.type != "cuda":
        raise GQADeviceDispatchError("dispatch control must use CUDA tensors")
    if key.device != query.device or value.device != query.device:
        raise GQADeviceDispatchError("dispatch control devices differ")


def _read_verified_trace(
    path: Path,
    artifact: RawTraceArtifact,
    *,
    marker: str,
) -> ScopedCUDAActivities:
    if not path.is_file() or path.is_symlink():
        raise GQADeviceDispatchError("raw trace is not a real file")
    raw = path.read_bytes()
    if len(raw) != artifact.size_bytes:
        raise GQADeviceDispatchError("raw trace size changed after collection")
    if hashlib.sha256(raw).hexdigest() != artifact.sha256:
        raise GQADeviceDispatchError("raw trace digest changed after collection")
    return parse_scoped_chrome_cuda_events(raw, marker=marker)


@dataclass(frozen=True, slots=True)
class GQADeviceDispatchAudit:
    """Small frozen operator controls, not campaign/cache integration proof."""

    backend_identity: BackendIdentityEvidence
    gqa: DispatchControlEvidence
    mha: DispatchControlEvidence
    gqa_source_shape: SourceShapeEvidence
    mha_source_shape: SourceShapeEvidence
    raw_trace_bytes_verified: bool
    allocation_proof_sha256: str | None
    expanded_kv_allocation_detected: bool
    expanded_kv_tensor_detected: bool
    explicitly_related_families: tuple[tuple[str, str], ...]
    evaluation: GQAProofEvaluation

    def __post_init__(self) -> None:
        if self.gqa.role != "gqa" or self.mha.role != "mha_control":
            raise ValueError("dispatch audit controls have incorrect roles")
        if (
            self.gqa.batch_size != FROZEN_CONTROL_BATCH_SIZE
            or self.mha.batch_size != FROZEN_CONTROL_BATCH_SIZE
            or self.gqa.context_length != FROZEN_CONTROL_CONTEXT_LENGTH
            or self.mha.context_length != FROZEN_CONTROL_CONTEXT_LENGTH
            or self.gqa.query_length != FROZEN_CONTROL_QUERY_LENGTH
            or self.mha.query_length != FROZEN_CONTROL_QUERY_LENGTH
            or self.gqa.is_causal
            or self.mha.is_causal
        ):
            raise ValueError("dispatch audit differs from frozen small controls")
        source_paths = tuple(
            item.relative_path for item in self.gqa_source_shape.sources
        )
        if source_paths != REQUIRED_SUT_SOURCES:
            raise ValueError("dispatch audit source set differs from the SUT")
        if self.mha_source_shape.sources != self.gqa_source_shape.sources:
            raise ValueError("GQA/MHA source audit evidence differs")
        if not self.raw_trace_bytes_verified:
            raise ValueError("dispatch audit requires reverified raw traces")
        if self.gqa.raw_trace is None or self.mha.raw_trace is None:
            raise ValueError("dispatch audit raw traces are absent")
        fingerprints = {
            self.gqa.backend.source_build_fingerprint,
            self.mha.backend.source_build_fingerprint,
        }
        if fingerprints != {self.backend_identity.sha256}:
            raise ValueError("backend controls differ from retained identity")
        if self.allocation_proof_sha256 is not None and not _SHA256_RE.fullmatch(
            self.allocation_proof_sha256
        ):
            raise ValueError("allocation proof SHA-256 is invalid")
        if (
            self.evaluation.allocation_verified
            or self.expanded_kv_allocation_detected
        ) and self.allocation_proof_sha256 is None:
            raise ValueError("allocation conclusions require a checksum-bound proof")
        if self.explicitly_related_families != tuple(
            sorted(set(self.explicitly_related_families))
        ):
            raise ValueError("related kernel families are not canonical")
        expected = evaluate_gqa_device_dispatch(
            gqa=self.gqa,
            mha=self.mha,
            allocation_verified=self.evaluation.allocation_verified,
            source_verified=(
                self.gqa_source_shape.source_verified
                and self.mha_source_shape.source_verified
            ),
            shape_verified=(
                self.gqa_source_shape.shape_verified_for(self.gqa)
                and self.mha_source_shape.shape_verified_for(self.mha)
            ),
            expanded_kv_allocation_detected=(
                self.expanded_kv_allocation_detected
            ),
            expanded_kv_tensor_detected=self.expanded_kv_tensor_detected,
            explicitly_related_families=set(self.explicitly_related_families),
        )
        if expected != self.evaluation:
            raise ValueError("dispatch audit evaluation is not reproducible")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "kvbench-phase3-gqa-device-dispatch-audit-1.0.0",
            "run_kind": "dispatch_audit",
            "measurement_scope": "small_frozen_isolated_operator_controls",
            "control_geometry_scope": {
                "batch_size": FROZEN_CONTROL_BATCH_SIZE,
                "context_length": FROZEN_CONTROL_CONTEXT_LENGTH,
                "query_length": FROZEN_CONTROL_QUERY_LENGTH,
                "campaign_geometries_covered": False,
                "production_endpoint_binding": "integration_required",
                "actual_cache_geometry_binding": "integration_required",
            },
            "profiler_instrumented": True,
            "performance_timing_reported": False,
            "benchmark_timing_eligible": False,
            "backend_identity": self.backend_identity.to_dict(),
            "raw_trace_bytes_verified": self.raw_trace_bytes_verified,
            "gqa": self.gqa.to_dict(),
            "mha_control": self.mha.to_dict(),
            "source_shape": {
                "gqa": self.gqa_source_shape.to_dict(control=self.gqa),
                "mha_control": self.mha_source_shape.to_dict(control=self.mha),
            },
            "allocation_size_proof": {
                "verified": self.evaluation.allocation_verified,
                "sha256": self.allocation_proof_sha256,
                "expanded_kv_allocation_detected": (
                    self.expanded_kv_allocation_detected
                ),
                "expanded_kv_tensor_detected": self.expanded_kv_tensor_detected,
            },
            "explicitly_related_families": [
                list(pair) for pair in self.explicitly_related_families
            ],
            "evaluation": self.evaluation.to_dict(),
        }


def collect_gqa_mha_device_dispatch(
    *,
    gqa_query: Any,
    gqa_key: Any,
    gqa_value: Any,
    mha_query: Any,
    mha_key: Any,
    mha_value: Any,
    output_directory: Path,
    artifact_relative_root: str,
    source_root: Path,
    source_paths: tuple[Path, ...],
    is_causal: bool,
    scale: float,
    warmup_count: int,
    allocation_verified: bool,
    allocation_proof_sha256: str | None = None,
    expanded_kv_allocation_detected: bool = False,
    expanded_kv_tensor_detected: bool = False,
    explicitly_related_families: Set[tuple[str, str]] = frozenset(),
) -> GQADeviceDispatchAudit:
    """Collect small frozen public-SDPA controls for B-011 only.

    This does not bind the proof to every campaign geometry, the production
    endpoint, or an actual model cache. Those remain explicit integration work.
    """

    _positive_integer(warmup_count, "warmup_count")
    if not isinstance(is_causal, bool):
        raise TypeError("is_causal must be boolean")
    if is_causal:
        raise ValueError("frozen small dispatch controls use decode causality")
    if not isinstance(scale, (int, float)) or isinstance(scale, bool):
        raise TypeError("scale must be numeric")
    if not math.isfinite(float(scale)) or float(scale) <= 0.0:
        raise ValueError("scale must be finite and positive")
    if float(scale) != 128**-0.5:
        raise ValueError("dispatch control scale differs from frozen head scale")
    relative_root = PurePosixPath(artifact_relative_root)
    if (
        not artifact_relative_root
        or relative_root.is_absolute()
        or "." in relative_root.parts
        or ".." in relative_root.parts
    ):
        raise ValueError("artifact trace root must be safe and relative")
    output_root = Path(output_directory)
    if not output_root.is_dir() or output_root.is_symlink():
        raise GQADeviceDispatchError("trace output directory must be real")
    torch = importlib.import_module("torch")
    backend_module = importlib.import_module("kvbench.runtime.backend")
    forced_flash_execution = backend_module.forced_flash_execution
    identity_payload = backend_module.backend_identity()
    identity = BackendIdentityEvidence.from_payload(identity_payload)
    controls = (
        ("gqa", gqa_query, gqa_key, gqa_value),
        ("mha_control", mha_query, mha_key, mha_value),
    )
    for role, query, key, value in controls:
        _validate_control_tensors(
            torch,
            role=role,
            query=query,
            key=key,
            value=value,
        )
    if (
        tuple(int(item) for item in gqa_query.shape[:1] + gqa_query.shape[2:])
        != tuple(int(item) for item in mha_query.shape[:1] + mha_query.shape[2:])
        or int(gqa_key.shape[-2]) != int(mha_key.shape[-2])
        or gqa_query.dtype != mha_query.dtype
        or gqa_query.device != mha_query.device
    ):
        raise GQADeviceDispatchError("GQA/MHA held-constant controls differ")
    sources = audit_gqa_source_files(source_root, source_paths)
    if tuple(item.relative_path for item in sources) != REQUIRED_SUT_SOURCES:
        raise GQADeviceDispatchError(
            "composite dispatch audit requires the exact frozen SUT source set"
        )
    backend_controls = {
        role: _collect_backend_control(
            torch,
            query=query,
            key=key,
            value=value,
            is_causal=is_causal,
            scale=float(scale),
            identity=identity,
            forced_flash_execution=forced_flash_execution,
        )
        for role, query, key, value in controls
    }

    def operation(query: Any, key: Any, value: Any) -> Callable[[], Any]:
        def invoke() -> Any:
            with forced_flash_execution():
                return torch.nn.functional.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=is_causal,
                    scale=float(scale),
                    enable_gqa=True,
                )

        return invoke

    retained: dict[str, tuple[DispatchControlEvidence, SourceShapeEvidence]] = {}
    for role, query, key, value in controls:
        filename = "gqa.chrome.json" if role == "gqa" else "mha.chrome.json"
        trace_path = output_root / filename
        relative_path = (relative_root / filename).as_posix()
        marker = f"kvbench.phase3.{role}.dispatch_audit"
        invoke = operation(query, key, value)
        artifact = collect_torch_profiler_trace(
            invoke,
            trace_path,
            artifact_relative_path=relative_path,
            marker=marker,
            warmup_count=warmup_count,
            device=query.device,
        )
        scoped = _read_verified_trace(trace_path, artifact, marker=marker)
        output = invoke()
        output_shape = tensor_shape_evidence(output)
        del output
        torch.cuda.synchronize(device=query.device)
        control = DispatchControlEvidence(
            role=role,
            batch_size=int(query.shape[0]),
            context_length=int(key.shape[-2]),
            query_length=int(query.shape[-2]),
            num_query_heads=int(query.shape[1]),
            num_kv_heads=int(key.shape[1]),
            head_dim=int(query.shape[-1]),
            dtype=str(query.dtype),
            dtype_bytes=int(query.element_size()),
            is_causal=is_causal,
            warmup_count=warmup_count,
            backend=backend_controls[role],
            raw_trace=artifact,
            trace_scope=scoped.scope,
            device_events=scoped.device_events,
        )
        query_shape = tensor_shape_evidence(query)
        key_shape = tensor_shape_evidence(key)
        value_shape = tensor_shape_evidence(value)
        source_shape = SourceShapeEvidence(
            sources=sources,
            query=query_shape,
            key=key_shape,
            value=value_shape,
            output=output_shape,
            native_kv_storage_verified=bool(
                key_shape.storage_bytes == key_shape.logical_bytes
                and value_shape.storage_bytes == value_shape.logical_bytes
                and key_shape.storage_offset == 0
                and value_shape.storage_offset == 0
                and key_shape.is_contiguous
                and value_shape.is_contiguous
            ),
        )
        retained[role] = (control, source_shape)
    gqa, gqa_source_shape = retained["gqa"]
    mha, mha_source_shape = retained["mha_control"]
    related = tuple(sorted(set(explicitly_related_families)))
    source_verified = bool(
        gqa_source_shape.source_verified and mha_source_shape.source_verified
    )
    shape_verified = bool(
        gqa_source_shape.shape_verified_for(gqa)
        and mha_source_shape.shape_verified_for(mha)
    )
    evaluation = evaluate_gqa_device_dispatch(
        gqa=gqa,
        mha=mha,
        allocation_verified=allocation_verified,
        source_verified=source_verified,
        shape_verified=shape_verified,
        expanded_kv_allocation_detected=expanded_kv_allocation_detected,
        expanded_kv_tensor_detected=expanded_kv_tensor_detected,
        explicitly_related_families=set(related),
    )
    return GQADeviceDispatchAudit(
        backend_identity=identity,
        gqa=gqa,
        mha=mha,
        gqa_source_shape=gqa_source_shape,
        mha_source_shape=mha_source_shape,
        raw_trace_bytes_verified=True,
        allocation_proof_sha256=allocation_proof_sha256,
        expanded_kv_allocation_detected=expanded_kv_allocation_detected,
        expanded_kv_tensor_detected=expanded_kv_tensor_detected,
        explicitly_related_families=related,
        evaluation=evaluation,
    )
