"""Strict cumulative CUDA allocator-event audit for Phase 3 admission."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import importlib
import time
from typing import Any


_TORCH: Any | None = None


class AllocationAuditError(RuntimeError):
    """The allocator event stream could not be collected faithfully."""


@dataclass(frozen=True, slots=True)
class AllocationAudit:
    """Cumulative events and snapshots; net-zero never erases an event."""

    audit_available: bool
    passed: bool
    allocation_event_count: int
    allocation_event_bytes: int
    event_counts: dict[str, int]
    allocated_before: int
    allocated_after: int
    reserved_before: int
    reserved_after: int
    peak_allocated: int
    peak_reserved: int
    failure_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_available": self.audit_available,
            "passed": self.passed,
            "allocation_event_count": self.allocation_event_count,
            "allocation_event_bytes": self.allocation_event_bytes,
            "event_counts": dict(sorted(self.event_counts.items())),
            "allocated_before": self.allocated_before,
            "allocated_after": self.allocated_after,
            "allocated_delta": self.allocated_after - self.allocated_before,
            "reserved_before": self.reserved_before,
            "reserved_after": self.reserved_after,
            "reserved_delta": self.reserved_after - self.reserved_before,
            "peak_allocated": self.peak_allocated,
            "peak_reserved": self.peak_reserved,
            "failure_reason": self.failure_reason,
            "instrumented_duration_reported_as_timing": False,
        }


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    """One outside-boundary CUDA allocator snapshot."""

    label: str
    host_timestamp_ns: int
    allocated_bytes: int
    reserved_bytes: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "label": self.label,
            "host_timestamp_ns": self.host_timestamp_ns,
            "allocated_bytes": self.allocated_bytes,
            "reserved_bytes": self.reserved_bytes,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "peak_reserved_bytes": self.peak_reserved_bytes,
        }


@dataclass(frozen=True, slots=True)
class NormalTimingMemoryEvidence:
    """Non-instrumented timing-lane memory snapshots, separate from audit."""

    model_baseline: MemorySnapshot
    post_cache_allocation: MemorySnapshot
    post_setup: MemorySnapshot
    timing_before: MemorySnapshot
    timing_after: MemorySnapshot
    timing_executed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_baseline": self.model_baseline.to_dict(),
            "post_cache_allocation": self.post_cache_allocation.to_dict(),
            "post_setup": self.post_setup.to_dict(),
            "timing_before": self.timing_before.to_dict(),
            "timing_after": self.timing_after.to_dict(),
            "timing_allocated_delta_bytes": (
                self.timing_after.allocated_bytes
                - self.timing_before.allocated_bytes
            ),
            "timing_reserved_delta_bytes": (
                self.timing_after.reserved_bytes
                - self.timing_before.reserved_bytes
            ),
            "timing_peak_allocated_bytes": (
                self.timing_after.peak_allocated_bytes
            ),
            "timing_peak_reserved_bytes": self.timing_after.peak_reserved_bytes,
            "timing_executed": self.timing_executed,
            "peak_reset_before_timing": self.timing_executed,
            "peak_reset_inside_measured_boundary": False,
            "instrumented_audit_separate": True,
            "profiler_duration_reported": False,
        }


def _torch() -> Any:
    global _TORCH
    if _TORCH is None:
        try:
            _TORCH = importlib.import_module("torch")
        except ModuleNotFoundError as error:
            raise AllocationAuditError("PyTorch is unavailable") from error
    return _TORCH


def _trace_for_device(snapshot: Mapping[str, Any], index: int) -> list[Any]:
    traces = snapshot.get("device_traces")
    if not isinstance(traces, list) or index >= len(traces):
        raise AllocationAuditError("allocator snapshot lacks the selected device trace")
    trace = traces[index]
    if not isinstance(trace, list):
        raise AllocationAuditError("allocator device trace is malformed")
    return trace


def capture_cuda_memory_snapshot(
    label: str,
    *,
    device: Any | None = None,
) -> MemorySnapshot:
    """Read allocator counters outside a measured operation."""

    torch = _torch()
    selected = torch.device(
        f"cuda:{torch.cuda.current_device()}" if device is None else device
    )
    if selected.type != "cuda":
        raise AllocationAuditError("memory snapshot requires a CUDA device")
    if not label:
        raise ValueError("memory snapshot label must be nonempty")
    return MemorySnapshot(
        label=label,
        host_timestamp_ns=time.time_ns(),
        allocated_bytes=int(torch.cuda.memory_allocated(device=selected)),
        reserved_bytes=int(torch.cuda.memory_reserved(device=selected)),
        peak_allocated_bytes=int(
            torch.cuda.max_memory_allocated(device=selected)
        ),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device=selected)),
    )


def audit_cuda_allocations(
    operation: Callable[[], Any],
    *,
    device: Any | None = None,
    max_entries: int = 100000,
) -> AllocationAudit:
    """Run one exact instrumented operation and fail on every alloc event."""

    torch = _torch()
    selected = torch.device(
        f"cuda:{torch.cuda.current_device()}" if device is None else device
    )
    if selected.type != "cuda":
        raise AllocationAuditError("allocation audit requires a CUDA device")
    if max_entries <= 0:
        raise ValueError("max_entries must be positive")
    torch.cuda.synchronize(device=selected)
    torch.cuda.reset_peak_memory_stats(device=selected)
    allocated_before = int(torch.cuda.memory_allocated(device=selected))
    reserved_before = int(torch.cuda.memory_reserved(device=selected))
    recorder = getattr(torch.cuda.memory, "_record_memory_history", None)
    snapshot_function = getattr(torch.cuda.memory, "_snapshot", None)
    if not callable(recorder) or not callable(snapshot_function):
        return AllocationAudit(
            audit_available=False,
            passed=False,
            allocation_event_count=0,
            allocation_event_bytes=0,
            event_counts={},
            allocated_before=allocated_before,
            allocated_after=allocated_before,
            reserved_before=reserved_before,
            reserved_after=reserved_before,
            peak_allocated=allocated_before,
            peak_reserved=reserved_before,
            failure_reason="allocator_history_unavailable",
        )
    snapshot: Mapping[str, Any]
    try:
        recorder(
            enabled="all",
            context="all",
            stacks="python",
            max_entries=max_entries,
            device=selected,
            clear_history=True,
        )
        operation()
        torch.cuda.synchronize(device=selected)
        snapshot = snapshot_function(device=selected)
    except BaseException as error:
        raise AllocationAuditError(
            "instrumented allocation operation failed"
        ) from error
    finally:
        recorder(enabled=None, device=selected)
    allocated_after = int(torch.cuda.memory_allocated(device=selected))
    reserved_after = int(torch.cuda.memory_reserved(device=selected))
    trace = _trace_for_device(snapshot, int(selected.index or 0))
    counts: Counter[str] = Counter()
    allocation_bytes = 0
    for entry in trace:
        if not isinstance(entry, Mapping):
            continue
        action = entry.get("action")
        if not isinstance(action, str):
            continue
        counts[action] += 1
        if action == "alloc":
            size = entry.get("size", 0)
            if isinstance(size, int) and not isinstance(size, bool):
                allocation_bytes += size
    allocation_count = int(counts.get("alloc", 0))
    passed = (
        allocation_count == 0
        and allocated_after <= allocated_before
        and reserved_after <= reserved_before
    )
    reason = None if passed else "measured_region_allocation_detected"
    return AllocationAudit(
        audit_available=True,
        passed=passed,
        allocation_event_count=allocation_count,
        allocation_event_bytes=allocation_bytes,
        event_counts=dict(counts),
        allocated_before=allocated_before,
        allocated_after=allocated_after,
        reserved_before=reserved_before,
        reserved_after=reserved_after,
        peak_allocated=int(torch.cuda.max_memory_allocated(device=selected)),
        peak_reserved=int(torch.cuda.max_memory_reserved(device=selected)),
        failure_reason=reason,
    )
