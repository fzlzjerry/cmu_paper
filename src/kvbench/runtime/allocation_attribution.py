"""Fail-closed CUDA allocator attribution for Phase 3 remediation.

Collection remains separate from this module: these routines validate and
attribute an already captured chronological allocator trace.  Profiler or
audit execution therefore cannot be mistaken for normal benchmark timing.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
import hashlib
import json
from typing import Any


CUDA_ALLOCATOR_MIN_BLOCK_BYTES = 512
FLASH_SPLIT_K_CPP_MARKERS = (
    "pytorch_flash::set_params_splitkv",
    "pytorch_flash::mha_fwd",
    "_flash_attention_forward_no_dropout_inplace",
)


class AllocationAttributionError(ValueError):
    """Attribution evidence or configuration is structurally invalid."""


class AllocationClass(StrEnum):
    """Preregistered Phase 3 allocation classes."""

    CACHE_GROWTH = "cache_growth"
    GQA_EXPANSION = "gqa_expansion"
    CONTEXT_SCALED_WORKSPACE = "context_scaled_workspace"
    FIXED_OUTPUT = "fixed_output"
    FIXED_SHARED_ACTIVATION = "fixed_shared_activation"
    FRAMEWORK_BOOKKEEPING = "framework_bookkeeping"
    AUDIT_INSTRUMENTATION = "audit_instrumentation"
    UNKNOWN = "unknown"


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AllocationAttributionError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class AllocationGeometry:
    """Frozen geometry and exact byte formulas for one operation."""

    batch: int
    query_heads: int
    kv_heads: int
    context: int
    head_dim: int
    dtype_bytes: int
    query_length: int = 1

    def __post_init__(self) -> None:
        for name in (
            "batch",
            "query_heads",
            "kv_heads",
            "context",
            "head_dim",
            "dtype_bytes",
            "query_length",
        ):
            _positive_int(name, getattr(self, name))
        if self.kv_heads > self.query_heads:
            raise AllocationAttributionError(
                "kv_heads cannot exceed query_heads"
            )

    @property
    def output_bytes(self) -> int:
        return (
            self.batch
            * self.query_heads
            * self.query_length
            * self.head_dim
            * self.dtype_bytes
        )

    @property
    def expanded_kv_single_bytes(self) -> int:
        return (
            self.batch
            * self.query_heads
            * self.context
            * self.head_dim
            * self.dtype_bytes
        )

    @property
    def expanded_kv_combined_bytes(self) -> int:
        return 2 * self.expanded_kv_single_bytes

    @property
    def native_kv_single_bytes(self) -> int:
        return (
            self.batch
            * self.kv_heads
            * self.context
            * self.head_dim
            * self.dtype_bytes
        )

    @property
    def native_kv_combined_bytes(self) -> int:
        return 2 * self.native_kv_single_bytes

    def flash_split_k_output_accumulator_bytes(self, splits: int) -> int:
        """FP32 partial-output storage for frozen Flash split-K."""

        _positive_int("splits", splits)
        return (
            splits
            * self.batch
            * self.query_heads
            * self.query_length
            * self.head_dim
            * 4
        )

    def flash_split_k_lse_bytes(self, splits: int) -> int:
        """FP32 log-sum-exp storage for frozen Flash split-K."""

        _positive_int("splits", splits)
        return (
            splits
            * self.batch
            * self.query_heads
            * self.query_length
            * 4
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "batch": self.batch,
            "query_heads": self.query_heads,
            "kv_heads": self.kv_heads,
            "context": self.context,
            "head_dim": self.head_dim,
            "dtype_bytes": self.dtype_bytes,
            "query_length": self.query_length,
            "output_bytes": self.output_bytes,
            "expanded_kv_single_bytes": self.expanded_kv_single_bytes,
            "expanded_kv_combined_bytes": self.expanded_kv_combined_bytes,
            "native_kv_single_bytes": self.native_kv_single_bytes,
            "native_kv_combined_bytes": self.native_kv_combined_bytes,
        }


@dataclass(frozen=True, slots=True)
class DependencyFlags:
    """Whether requested bytes depend on each frozen dimension.

    None means the dependency is unproven and is fail-closed by the gate.
    """

    batch: bool | None
    context: bool | None
    query_heads: bool | None
    kv_heads: bool | None

    def to_dict(self) -> dict[str, bool | None]:
        return {
            "batch": self.batch,
            "context": self.context,
            "query_heads": self.query_heads,
            "kv_heads": self.kv_heads,
        }


UNKNOWN_DEPENDENCIES = DependencyFlags(None, None, None, None)


@dataclass(frozen=True, slots=True)
class StackFrame:
    """One normalized Python or C++ stack frame."""

    name: str
    filename: str | None = None
    line: int | None = None

    def render(self) -> str:
        location = self.filename or ""
        if self.line is not None:
            location = f"{location}:{self.line}"
        return f"{self.name} {location}".strip()

    def to_dict(self) -> dict[str, str | int | None]:
        return {
            "name": self.name,
            "filename": self.filename,
            "line": self.line,
        }


@dataclass(frozen=True, slots=True)
class TraceEventEvidence:
    """One allocator trace event retained in original chronological order."""

    index: int
    action: str
    address: int | None
    size_bytes: int | None
    stream: int | None
    allocated_block_bytes: int | None
    python_stack: tuple[StackFrame, ...]
    cpp_stack: tuple[StackFrame, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "action": self.action,
            "address": self.address,
            "size_bytes": self.size_bytes,
            "stream": self.stream,
            "allocated_block_bytes": self.allocated_block_bytes,
            "python_stack": [frame.to_dict() for frame in self.python_stack],
            "cpp_stack": [frame.to_dict() for frame in self.cpp_stack],
        }


@dataclass(frozen=True, slots=True)
class AllocationLifetime:
    """One address generation from alloc through free completion."""

    allocation_id: int
    address: int
    requested_bytes: int
    rounded_minimum_bytes: int
    allocated_block_bytes: int | None
    allocated_block_size_proven: bool
    stream: int | None
    alloc_event_index: int
    free_requested_event_index: int | None
    free_completed_event_index: int | None
    fully_freed: bool
    reused_from_cache: bool
    triggered_segment_alloc: bool
    python_stack: tuple[StackFrame, ...]
    cpp_stack: tuple[StackFrame, ...]
    event_class: AllocationClass
    size_formula: str | None
    formula_parameters: tuple[tuple[str, int], ...]
    dependencies: DependencyFlags

    def to_dict(self) -> dict[str, Any]:
        return {
            "allocation_id": self.allocation_id,
            "address": self.address,
            "requested_bytes": self.requested_bytes,
            "rounded_minimum_bytes": self.rounded_minimum_bytes,
            "allocated_block_bytes": self.allocated_block_bytes,
            "allocated_block_size_proven": self.allocated_block_size_proven,
            "stream": self.stream,
            "alloc_event_index": self.alloc_event_index,
            "free_requested_event_index": self.free_requested_event_index,
            "free_completed_event_index": self.free_completed_event_index,
            "fully_freed": self.fully_freed,
            "reused_from_cache": self.reused_from_cache,
            "triggered_segment_alloc": self.triggered_segment_alloc,
            "python_stack": [frame.to_dict() for frame in self.python_stack],
            "cpp_stack": [frame.to_dict() for frame in self.cpp_stack],
            "event_class": self.event_class.value,
            "size_formula": self.size_formula,
            "formula_parameters": dict(self.formula_parameters),
            "dependencies": self.dependencies.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class AllocatorCounterEvidence:
    """Cumulative counters spanning exactly the traced operation.

    Nullable fields preserve unavailable evidence.  Gate evaluation treats
    every unavailable counter as a failure.
    """

    allocation_count: int | None
    requested_bytes: int | None
    allocated_block_bytes: int | None
    device_allocation_count: int | None
    device_free_count: int | None

    def __post_init__(self) -> None:
        for name in (
            "allocation_count",
            "requested_bytes",
            "allocated_block_bytes",
            "device_allocation_count",
            "device_free_count",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise AllocationAttributionError(
                    f"{name} must be a nonnegative integer or None"
                )

    @property
    def complete(self) -> bool:
        return all(
            value is not None
            for value in (
                self.allocation_count,
                self.requested_bytes,
                self.allocated_block_bytes,
                self.device_allocation_count,
                self.device_free_count,
            )
        )

    def to_dict(self) -> dict[str, int | bool | None]:
        return {
            "allocation_count": self.allocation_count,
            "requested_bytes": self.requested_bytes,
            "allocated_block_bytes": self.allocated_block_bytes,
            "device_allocation_count": self.device_allocation_count,
            "device_free_count": self.device_free_count,
            "complete": self.complete,
        }


@dataclass(frozen=True, slots=True)
class AttributionRules:
    """Preregistered facts not inferable from one allocator trace."""

    expected_split_k_splits: int | None = None
    frozen_backend_identity: str | None = None
    fixed_shared_activation_sizes: frozenset[int] = field(
        default_factory=frozenset
    )
    framework_bookkeeping_sizes: frozenset[int] = field(
        default_factory=frozenset
    )
    fixed_shared_stack_markers: tuple[str, ...] = ()
    framework_stack_markers: tuple[str, ...] = ()
    cache_stack_markers: tuple[str, ...] = (
        "static_cache",
        "cache.update",
        "cache_growth",
    )
    audit_stack_markers: tuple[str, ...] = (
        "allocation_audit",
        "_record_memory_history",
        "torch.profiler",
    )
    flash_split_k_cpp_markers: tuple[str, ...] = FLASH_SPLIT_K_CPP_MARKERS

    def __post_init__(self) -> None:
        if self.expected_split_k_splits is not None:
            _positive_int(
                "expected_split_k_splits", self.expected_split_k_splits
            )
            if self.expected_split_k_splits <= 1:
                raise AllocationAttributionError(
                    "expected_split_k_splits must exceed one"
                )
        for name in (
            "fixed_shared_activation_sizes",
            "framework_bookkeeping_sizes",
        ):
            for size in getattr(self, name):
                _positive_int(f"{name} entry", size)
        if self.frozen_backend_identity == "":
            raise AllocationAttributionError(
                "frozen_backend_identity must be nonempty when supplied"
            )


@dataclass(frozen=True, slots=True)
class AllocatorTraceAttribution:
    """Validated trace, address lifetimes, and formula classifications."""

    trace_sha256: str
    expected_trace_sha256: str | None
    backend_identity: str | None
    geometry: AllocationGeometry
    counters: AllocatorCounterEvidence
    events: tuple[TraceEventEvidence, ...]
    allocations: tuple[AllocationLifetime, ...]
    action_counts: dict[str, int]
    segment_alloc_count: int
    segment_free_count: int
    integrity_errors: tuple[str, ...]

    @property
    def all_block_sizes_proven(self) -> bool:
        return all(
            item.allocated_block_size_proven for item in self.allocations
        )

    @property
    def all_lifetimes_fully_freed(self) -> bool:
        return all(item.fully_freed for item in self.allocations)

    @property
    def all_allocations_cache_reused(self) -> bool:
        return all(item.reused_from_cache for item in self.allocations)

    def class_counts(self) -> dict[str, int]:
        counts = Counter(item.event_class.value for item in self.allocations)
        return dict(sorted(counts.items()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_sha256": self.trace_sha256,
            "expected_trace_sha256": self.expected_trace_sha256,
            "backend_identity": self.backend_identity,
            "geometry": self.geometry.to_dict(),
            "counters": self.counters.to_dict(),
            "events": [event.to_dict() for event in self.events],
            "allocations": [item.to_dict() for item in self.allocations],
            "action_counts": dict(sorted(self.action_counts.items())),
            "class_counts": self.class_counts(),
            "segment_alloc_count": self.segment_alloc_count,
            "segment_free_count": self.segment_free_count,
            "all_block_sizes_proven": self.all_block_sizes_proven,
            "all_lifetimes_fully_freed": self.all_lifetimes_fully_freed,
            "all_allocations_cache_reused": (
                self.all_allocations_cache_reused
            ),
            "integrity_errors": list(self.integrity_errors),
        }


@dataclass(frozen=True, slots=True)
class MemoryDeltaEvidence:
    """Persistent PyTorch and device memory around the isolated operation."""

    allocated_before: int
    allocated_after: int
    reserved_before: int
    reserved_after: int
    device_used_before: int | None
    device_used_after: int | None

    def __post_init__(self) -> None:
        for name in (
            "allocated_before",
            "allocated_after",
            "reserved_before",
            "reserved_after",
            "device_used_before",
            "device_used_after",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise AllocationAttributionError(
                    f"{name} must be a nonnegative integer or None"
                )
        if (self.device_used_before is None) != (
            self.device_used_after is None
        ):
            raise AllocationAttributionError(
                "device-used samples must both be present or both absent"
            )

    @property
    def allocated_delta(self) -> int:
        return self.allocated_after - self.allocated_before

    @property
    def reserved_delta(self) -> int:
        return self.reserved_after - self.reserved_before

    @property
    def device_used_delta(self) -> int | None:
        if (
            self.device_used_before is None
            or self.device_used_after is None
        ):
            return None
        return self.device_used_after - self.device_used_before

    @property
    def non_pytorch_delta(self) -> int | None:
        """Device growth unexplained by PyTorch reserved-memory growth."""

        device_delta = self.device_used_delta
        if device_delta is None:
            return None
        return device_delta - self.reserved_delta

    def to_dict(self) -> dict[str, int | None]:
        return {
            "allocated_before": self.allocated_before,
            "allocated_after": self.allocated_after,
            "allocated_delta": self.allocated_delta,
            "reserved_before": self.reserved_before,
            "reserved_after": self.reserved_after,
            "reserved_delta": self.reserved_delta,
            "device_used_before": self.device_used_before,
            "device_used_after": self.device_used_after,
            "device_used_delta": self.device_used_delta,
            "non_pytorch_delta": self.non_pytorch_delta,
        }


@dataclass(frozen=True, slots=True)
class SplitKControlEvidence:
    """Independent GQA/MHA controls required to permit split-K buffers."""

    frozen_backend_identity: str
    gqa_control_verified: bool
    mha_control_verified: bool
    common_workspace_formula_verified: bool
    same_or_related_kernel_family_verified: bool
    source_build_identity_verified: bool

    def __post_init__(self) -> None:
        if not self.frozen_backend_identity:
            raise AllocationAttributionError(
                "frozen_backend_identity must be nonempty"
            )

    def passes_for(self, backend_identity: str | None) -> bool:
        return (
            backend_identity == self.frozen_backend_identity
            and self.gqa_control_verified
            and self.mha_control_verified
            and self.common_workspace_formula_verified
            and self.same_or_related_kernel_family_verified
            and self.source_build_identity_verified
        )


@dataclass(frozen=True, slots=True)
class AllocationCriterionResult:
    """Machine-readable gate result.

    An eager pass is described only as attributed bounded ephemeral allocation,
    never as zero-allocation.
    """

    criterion_id: str
    passed: bool
    failure_reasons: tuple[str, ...]
    allocation_event_count: int
    class_counts: dict[str, int]
    no_context_dependent_allocation: bool
    fully_attributed_bounded_ephemeral: bool
    strict_graph_zero_events: bool | None

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "criterion_id": self.criterion_id,
            "passed": self.passed,
            "failure_reasons": list(self.failure_reasons),
            "allocation_event_count": self.allocation_event_count,
            "class_counts": dict(sorted(self.class_counts.items())),
            "no_context_dependent_allocation": (
                self.no_context_dependent_allocation
            ),
            "fully_attributed_bounded_ephemeral": (
                self.fully_attributed_bounded_ephemeral
            ),
        }
        if self.strict_graph_zero_events is not None:
            value["strict_graph_zero_events"] = self.strict_graph_zero_events
        return value


@dataclass(slots=True)
class _OpenAllocation:
    allocation_id: int
    event: TraceEventEvidence
    rounded_minimum_bytes: int
    explicit_block_bytes: int | None
    reused_from_cache: bool
    triggered_segment_alloc: bool
    free_requested_event_index: int | None = None


def cuda_allocator_rounded_minimum(requested_bytes: int) -> int:
    """Frozen 512-byte lower-bound rounding used for aggregate proofs."""

    _positive_int("requested_bytes", requested_bytes)
    return max(
        CUDA_ALLOCATOR_MIN_BLOCK_BYTES,
        (
            (requested_bytes + CUDA_ALLOCATOR_MIN_BLOCK_BYTES - 1)
            // CUDA_ALLOCATOR_MIN_BLOCK_BYTES
        )
        * CUDA_ALLOCATOR_MIN_BLOCK_BYTES,
    )


def allocator_trace_sha256(
    trace: Sequence[Mapping[str, Any]],
) -> str:
    """Hash a raw trace canonically so later mutation can be rejected."""

    try:
        payload = json.dumps(
            list(trace),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise AllocationAttributionError(
            "allocator trace is not canonical JSON"
        ) from error
    return hashlib.sha256(payload).hexdigest()


def verify_allocator_trace_sha256(
    trace: Sequence[Mapping[str, Any]],
    expected_sha256: str,
) -> bool:
    """Return whether a raw trace still matches its recorded digest."""

    return allocator_trace_sha256(trace) == expected_sha256


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _frame_from_value(value: object) -> StackFrame | None:
    if isinstance(value, str):
        return StackFrame(name=value)
    if not isinstance(value, Mapping):
        return None
    raw_name = value.get("name", value.get("function", ""))
    raw_filename = value.get("filename", value.get("file"))
    raw_line = value.get("line", value.get("line_number"))
    name = raw_name if isinstance(raw_name, str) else ""
    filename = raw_filename if isinstance(raw_filename, str) else None
    line = _optional_int(raw_line)
    if not name and filename is None:
        return None
    return StackFrame(name=name, filename=filename, line=line)


def _frames(value: object) -> tuple[StackFrame, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    result: list[StackFrame] = []
    for item in value:
        frame = _frame_from_value(item)
        if frame is not None:
            result.append(frame)
    return tuple(result)


def _looks_cpp(frame: StackFrame) -> bool:
    filename = (frame.filename or "").lower()
    return (
        "::" in frame.name
        or filename.endswith(
            (".cc", ".cpp", ".cxx", ".cu", ".cuh", ".h", ".hpp")
        )
        or "/aten/" in filename
        or "/c10/" in filename
    )


def _event_stacks(
    raw: Mapping[str, Any],
) -> tuple[tuple[StackFrame, ...], tuple[StackFrame, ...]]:
    python_stack = _frames(raw.get("python_stack"))
    cpp_stack = _frames(raw.get("cpp_stack"))
    if python_stack or cpp_stack:
        return python_stack, cpp_stack
    generic = _frames(raw.get("frames"))
    return (
        tuple(frame for frame in generic if not _looks_cpp(frame)),
        tuple(frame for frame in generic if _looks_cpp(frame)),
    )


def _stack_text(lifetime: AllocationLifetime) -> str:
    return "\n".join(
        frame.render()
        for frame in (*lifetime.python_stack, *lifetime.cpp_stack)
    )


def _has_any_marker(text: str, markers: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in markers)


def _has_all_markers(text: str, markers: Sequence[str]) -> bool:
    lowered = text.lower()
    return bool(markers) and all(
        marker.lower() in lowered for marker in markers
    )


def _classify(
    lifetime: AllocationLifetime,
    geometry: AllocationGeometry,
    rules: AttributionRules,
) -> AllocationLifetime:
    size = lifetime.requested_bytes
    stack = _stack_text(lifetime)
    if _has_any_marker(stack, rules.audit_stack_markers):
        return replace(
            lifetime,
            event_class=AllocationClass.AUDIT_INSTRUMENTATION,
        )

    if geometry.query_heads > geometry.kv_heads:
        if size == geometry.expanded_kv_combined_bytes:
            return replace(
                lifetime,
                event_class=AllocationClass.GQA_EXPANSION,
                size_formula="expanded_kv_combined",
                dependencies=DependencyFlags(True, True, True, False),
            )
        if size == geometry.expanded_kv_single_bytes:
            return replace(
                lifetime,
                event_class=AllocationClass.GQA_EXPANSION,
                size_formula="expanded_kv_single",
                dependencies=DependencyFlags(True, True, True, False),
            )

    if size == geometry.native_kv_combined_bytes:
        return replace(
            lifetime,
            event_class=AllocationClass.CACHE_GROWTH,
            size_formula="native_kv_combined",
            dependencies=DependencyFlags(True, True, False, True),
        )
    if size == geometry.native_kv_single_bytes:
        return replace(
            lifetime,
            event_class=AllocationClass.CACHE_GROWTH,
            size_formula="native_kv_single",
            dependencies=DependencyFlags(True, True, False, True),
        )
    if _has_any_marker(stack, rules.cache_stack_markers):
        return replace(
            lifetime,
            event_class=AllocationClass.CACHE_GROWTH,
            dependencies=DependencyFlags(None, True, None, None),
        )

    splits = rules.expected_split_k_splits
    if splits is not None:
        formulas = {
            "flash_split_k_output_accumulator": (
                geometry.flash_split_k_output_accumulator_bytes(splits)
            ),
            "flash_split_k_lse": geometry.flash_split_k_lse_bytes(splits),
        }
        for name, expected in formulas.items():
            if size == expected:
                parameters = (("num_splits", splits),)
                if _has_all_markers(
                    stack, rules.flash_split_k_cpp_markers
                ):
                    return replace(
                        lifetime,
                        event_class=(
                            AllocationClass.CONTEXT_SCALED_WORKSPACE
                        ),
                        size_formula=name,
                        formula_parameters=parameters,
                        dependencies=DependencyFlags(
                            True, True, True, True
                        ),
                    )
                return replace(
                    lifetime,
                    size_formula=f"{name}_stack_unverified",
                    formula_parameters=parameters,
                )

    if size == geometry.output_bytes:
        return replace(
            lifetime,
            event_class=AllocationClass.FIXED_OUTPUT,
            size_formula="fixed_decode_output",
            dependencies=DependencyFlags(True, False, True, False),
        )
    if size in rules.fixed_shared_activation_sizes and (
        not rules.fixed_shared_stack_markers
        or _has_any_marker(stack, rules.fixed_shared_stack_markers)
    ):
        return replace(
            lifetime,
            event_class=AllocationClass.FIXED_SHARED_ACTIVATION,
            size_formula="preregistered_fixed_shared_activation",
            dependencies=DependencyFlags(True, False, True, False),
        )
    if size in rules.framework_bookkeeping_sizes and (
        not rules.framework_stack_markers
        or _has_any_marker(stack, rules.framework_stack_markers)
    ):
        return replace(
            lifetime,
            event_class=AllocationClass.FRAMEWORK_BOOKKEEPING,
            size_formula="preregistered_framework_bookkeeping",
            dependencies=DependencyFlags(False, False, False, False),
        )
    return lifetime


def _make_lifetime(
    opened: _OpenAllocation,
    free_completed_index: int | None,
) -> AllocationLifetime:
    event = opened.event
    if event.address is None or event.size_bytes is None:
        raise AssertionError("invalid allocation cannot form a lifetime")
    return AllocationLifetime(
        allocation_id=opened.allocation_id,
        address=event.address,
        requested_bytes=event.size_bytes,
        rounded_minimum_bytes=opened.rounded_minimum_bytes,
        allocated_block_bytes=opened.explicit_block_bytes,
        allocated_block_size_proven=opened.explicit_block_bytes is not None,
        stream=event.stream,
        alloc_event_index=event.index,
        free_requested_event_index=opened.free_requested_event_index,
        free_completed_event_index=free_completed_index,
        fully_freed=free_completed_index is not None,
        reused_from_cache=opened.reused_from_cache,
        triggered_segment_alloc=opened.triggered_segment_alloc,
        python_stack=event.python_stack,
        cpp_stack=event.cpp_stack,
        event_class=AllocationClass.UNKNOWN,
        size_formula=None,
        formula_parameters=(),
        dependencies=UNKNOWN_DEPENDENCIES,
    )


def _reconcile_block_sizes(
    allocations: list[AllocationLifetime],
    counters: AllocatorCounterEvidence,
    errors: list[str],
) -> list[AllocationLifetime]:
    observed_count = len(allocations)
    observed_requested = sum(item.requested_bytes for item in allocations)
    if counters.allocation_count is None:
        errors.append("allocator_counter_allocation_count_missing")
    elif counters.allocation_count != observed_count:
        errors.append("allocator_counter_allocation_count_mismatch")
    if counters.requested_bytes is None:
        errors.append("allocator_counter_requested_bytes_missing")
    elif counters.requested_bytes != observed_requested:
        errors.append("allocator_counter_requested_bytes_mismatch")

    explicit_total = 0
    unresolved_minimum = 0
    for item in allocations:
        block = item.allocated_block_bytes
        if block is None:
            unresolved_minimum += item.rounded_minimum_bytes
        else:
            if (
                block < item.rounded_minimum_bytes
                or block % CUDA_ALLOCATOR_MIN_BLOCK_BYTES != 0
            ):
                errors.append(
                    "invalid_allocated_block_size:"
                    f"allocation_id={item.allocation_id}"
                )
            explicit_total += block

    aggregate = counters.allocated_block_bytes
    if aggregate is None:
        errors.append("allocator_counter_allocated_block_bytes_missing")
        if unresolved_minimum:
            errors.append("allocated_block_sizes_unresolved")
        return allocations
    minimum_total = explicit_total + unresolved_minimum
    if aggregate < minimum_total:
        errors.append(
            "allocator_counter_allocated_block_bytes_below_minimum"
        )
    elif unresolved_minimum and aggregate == minimum_total:
        allocations = [
            replace(
                item,
                allocated_block_bytes=item.rounded_minimum_bytes,
                allocated_block_size_proven=True,
            )
            if item.allocated_block_bytes is None
            else item
            for item in allocations
        ]
    elif unresolved_minimum:
        errors.append("allocated_block_sizes_unresolved")
    elif aggregate != explicit_total:
        errors.append("allocator_counter_allocated_block_bytes_mismatch")
    return allocations


def attribute_allocator_trace(
    trace: Sequence[Mapping[str, Any]],
    *,
    geometry: AllocationGeometry,
    counters: AllocatorCounterEvidence,
    rules: AttributionRules | None = None,
    backend_identity: str | None = None,
    expected_trace_sha256: str | None = None,
) -> AllocatorTraceAttribution:
    """Parse a trace with a reuse-safe chronological address state machine."""

    selected_rules = rules or AttributionRules()
    digest = allocator_trace_sha256(trace)
    errors: list[str] = []
    if expected_trace_sha256 is not None and digest != expected_trace_sha256:
        errors.append("allocator_trace_sha256_mismatch")
    if backend_identity == "":
        errors.append("backend_identity_empty")
    if (
        selected_rules.frozen_backend_identity is not None
        and backend_identity != selected_rules.frozen_backend_identity
    ):
        errors.append("backend_identity_mismatch")

    events: list[TraceEventEvidence] = []
    counts: Counter[str] = Counter()
    active: dict[int, _OpenAllocation] = {}
    completed: list[AllocationLifetime] = []
    active_trace_segments: dict[int, tuple[int, int]] = {}
    next_allocation_id = 0

    for index, raw in enumerate(trace):
        if not isinstance(raw, Mapping):
            errors.append(f"malformed_trace_event:index={index}")
            event = TraceEventEvidence(
                index, "malformed", None, None, None, None, (), ()
            )
            events.append(event)
            counts[event.action] += 1
            continue
        raw_action = raw.get("action")
        action = raw_action if isinstance(raw_action, str) else "malformed"
        address = _optional_int(raw.get("addr", raw.get("address")))
        size = _optional_int(raw.get("size"))
        stream = _optional_int(raw.get("stream"))
        block = _optional_int(raw.get("allocated_block_size"))
        python_stack, cpp_stack = _event_stacks(raw)
        event = TraceEventEvidence(
            index,
            action,
            address,
            size,
            stream,
            block,
            python_stack,
            cpp_stack,
        )
        events.append(event)
        counts[action] += 1

        if action == "alloc":
            if (
                address is None
                or address < 0
                or size is None
                or size <= 0
            ):
                errors.append(f"invalid_alloc_event:index={index}")
                continue
            if address in active:
                errors.append(
                    f"alloc_before_prior_free_completed:index={index}"
                )
                continue
            if block is not None and block <= 0:
                errors.append(
                    f"invalid_allocated_block_size:index={index}"
                )
                block = None
            segment_index = next(
                (
                    segment_index
                    for start, (
                        segment_size,
                        segment_index,
                    ) in active_trace_segments.items()
                    if start <= address < start + segment_size
                ),
                None,
            )
            active[address] = _OpenAllocation(
                allocation_id=next_allocation_id,
                event=event,
                rounded_minimum_bytes=(
                    cuda_allocator_rounded_minimum(size)
                ),
                explicit_block_bytes=block,
                reused_from_cache=segment_index is None,
                triggered_segment_alloc=segment_index is not None,
            )
            next_allocation_id += 1
            continue

        if action in ("free_requested", "free_completed"):
            if address is None or address < 0:
                errors.append(f"invalid_free_event:index={index}")
                continue
            opened = active.get(address)
            if opened is None:
                errors.append(
                    f"free_without_matching_alloc:index={index}"
                )
                continue
            if size is None:
                errors.append(f"free_event_size_missing:index={index}")
            elif size != opened.event.size_bytes:
                errors.append(f"free_event_size_mismatch:index={index}")
            if action == "free_requested":
                if opened.free_requested_event_index is not None:
                    errors.append(
                        f"duplicate_free_requested:index={index}"
                    )
                else:
                    opened.free_requested_event_index = index
                continue
            if opened.free_requested_event_index is None:
                errors.append(
                    "free_completed_before_free_requested:"
                    f"index={index}"
                )
                continue
            completed.append(_make_lifetime(opened, index))
            del active[address]
            continue

        if action == "segment_alloc":
            if (
                address is None
                or address < 0
                or size is None
                or size <= 0
            ):
                errors.append(f"invalid_segment_alloc:index={index}")
            elif address in active_trace_segments:
                errors.append(f"duplicate_segment_alloc:index={index}")
            else:
                active_trace_segments[address] = (size, index)
            continue
        if action == "segment_free":
            if address is None or address < 0:
                errors.append(f"invalid_segment_free:index={index}")
            elif address not in active_trace_segments:
                errors.append(
                    f"segment_free_without_matching_alloc:index={index}"
                )
            else:
                segment_size, _ = active_trace_segments[address]
                if size is not None and size != segment_size:
                    errors.append(
                        f"segment_free_size_mismatch:index={index}"
                    )
                del active_trace_segments[address]
            continue
        if action in ("oom", "alloc_retry"):
            errors.append(
                f"allocator_failure_event:{action}:index={index}"
            )
        elif action not in ("snapshot", "malformed"):
            errors.append(
                f"unknown_allocator_action:{action}:index={index}"
            )
        elif action == "malformed":
            errors.append(f"missing_allocator_action:index={index}")

    for opened in active.values():
        errors.append(
            "allocation_not_fully_freed:"
            f"allocation_id={opened.allocation_id}"
        )
        completed.append(_make_lifetime(opened, None))
    completed.sort(key=lambda item: item.allocation_id)
    completed = _reconcile_block_sizes(completed, counters, errors)
    completed = [
        _classify(item, geometry, selected_rules) for item in completed
    ]
    return AllocatorTraceAttribution(
        trace_sha256=digest,
        expected_trace_sha256=expected_trace_sha256,
        backend_identity=backend_identity,
        geometry=geometry,
        counters=counters,
        events=tuple(events),
        allocations=tuple(completed),
        action_counts=dict(counts),
        segment_alloc_count=int(counts.get("segment_alloc", 0)),
        segment_free_count=int(counts.get("segment_free", 0)),
        integrity_errors=tuple(dict.fromkeys(errors)),
    )


def _append_memory_failures(
    reasons: list[str],
    memory: MemoryDeltaEvidence,
) -> None:
    if memory.allocated_delta != 0:
        reasons.append("persistent_allocated_delta_nonzero")
    if memory.reserved_delta != 0:
        reasons.append("persistent_reserved_delta_nonzero")
    if memory.device_used_delta is None:
        reasons.append("device_used_delta_unavailable")
    elif memory.device_used_delta != 0:
        reasons.append("persistent_device_used_delta_nonzero")
    if memory.non_pytorch_delta is None:
        reasons.append("non_pytorch_delta_unavailable")
    elif memory.non_pytorch_delta != 0:
        reasons.append("persistent_non_pytorch_delta_nonzero")


def _append_common_failures(
    reasons: list[str],
    attribution: AllocatorTraceAttribution,
) -> None:
    if attribution.integrity_errors:
        reasons.append("allocator_trace_integrity_failure")
    if not attribution.counters.complete:
        reasons.append("allocator_counter_evidence_incomplete")
    if not attribution.all_block_sizes_proven:
        reasons.append("allocated_block_size_evidence_incomplete")
    if not attribution.all_lifetimes_fully_freed:
        reasons.append("allocation_lifetime_not_fully_freed")
    if attribution.segment_alloc_count:
        reasons.append("segment_alloc_detected")
    if attribution.segment_free_count:
        reasons.append("segment_free_detected")
    if attribution.counters.device_allocation_count != 0:
        reasons.append("device_allocation_detected_or_unavailable")
    if attribution.counters.device_free_count != 0:
        reasons.append("device_free_detected_or_unavailable")


def evaluate_refined_eager_criterion(
    attribution: AllocatorTraceAttribution,
    memory: MemoryDeltaEvidence,
    *,
    split_k_controls: SplitKControlEvidence | None = None,
) -> AllocationCriterionResult:
    """Evaluate the attributed-bounded-ephemeral eager criterion.

    Known context-dependent Flash split-K buffers can pass only with the
    independent control contract.  Their presence is always recorded by
    no_context_dependent_allocation=false.
    """

    reasons: list[str] = []
    _append_common_failures(reasons, attribution)
    _append_memory_failures(reasons, memory)
    if not attribution.all_allocations_cache_reused:
        reasons.append("allocation_not_reused_from_cache")

    forbidden = {
        AllocationClass.CACHE_GROWTH,
        AllocationClass.GQA_EXPANSION,
        AllocationClass.AUDIT_INSTRUMENTATION,
        AllocationClass.UNKNOWN,
    }
    for event_class in forbidden:
        if any(
            item.event_class is event_class
            for item in attribution.allocations
        ):
            reasons.append(
                f"forbidden_allocation_class:{event_class.value}"
            )

    workspaces = [
        item
        for item in attribution.allocations
        if item.event_class is AllocationClass.CONTEXT_SCALED_WORKSPACE
    ]
    if workspaces:
        if split_k_controls is None or not split_k_controls.passes_for(
            attribution.backend_identity
        ):
            reasons.append("flash_split_k_control_evidence_failed")
        formula_counts = Counter(item.size_formula for item in workspaces)
        if formula_counts[
            "flash_split_k_output_accumulator"
        ] != formula_counts["flash_split_k_lse"]:
            reasons.append("flash_split_k_workspace_pair_mismatch")
        if any(
            not _has_all_markers(
                _stack_text(item), FLASH_SPLIT_K_CPP_MARKERS
            )
            for item in workspaces
        ):
            reasons.append("flash_split_k_stack_attribution_failed")

    no_context_dependent = all(
        item.dependencies.context is False
        for item in attribution.allocations
    )
    fully_attributed = not any(
        item.event_class in forbidden for item in attribution.allocations
    )
    unique = tuple(dict.fromkeys(reasons))
    return AllocationCriterionResult(
        criterion_id=(
            "phase3_refined_eager_attributed_bounded_ephemeral_v1"
        ),
        passed=not unique,
        failure_reasons=unique,
        allocation_event_count=len(attribution.allocations),
        class_counts=attribution.class_counts(),
        no_context_dependent_allocation=no_context_dependent,
        fully_attributed_bounded_ephemeral=(
            not unique and fully_attributed
        ),
        strict_graph_zero_events=None,
    )


def evaluate_strict_graph_criterion(
    attribution: AllocatorTraceAttribution,
    memory: MemoryDeltaEvidence,
) -> AllocationCriterionResult:
    """Require exactly zero graph-replay allocation and device events."""

    reasons: list[str] = []
    _append_common_failures(reasons, attribution)
    _append_memory_failures(reasons, memory)
    if attribution.action_counts.get("alloc", 0) != 0 or (
        attribution.allocations
    ):
        reasons.append("graph_allocation_event_detected")
    counters = attribution.counters
    if any(
        value != 0
        for value in (
            counters.allocation_count,
            counters.requested_bytes,
            counters.allocated_block_bytes,
        )
    ):
        reasons.append(
            "graph_allocator_counters_nonzero_or_unavailable"
        )
    unique = tuple(dict.fromkeys(reasons))
    strict_zero = not unique
    return AllocationCriterionResult(
        criterion_id="phase3_graph_strict_zero_allocation_v1",
        passed=strict_zero,
        failure_reasons=unique,
        allocation_event_count=len(attribution.allocations),
        class_counts=attribution.class_counts(),
        no_context_dependent_allocation=strict_zero,
        fully_attributed_bounded_ephemeral=False,
        strict_graph_zero_events=strict_zero,
    )
