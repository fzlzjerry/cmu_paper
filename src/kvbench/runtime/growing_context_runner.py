"""Preallocated eager growing-context Phase 3 runner."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Any

from kvbench.runtime.allocation import (
    AllocationAudit,
    NormalTimingMemoryEvidence,
    audit_cuda_allocations,
    capture_cuda_memory_snapshot,
)
from kvbench.runtime.backend import (
    BackendAudit,
    audit_backend_choice,
    forced_flash_execution,
)
from kvbench.runtime.bf16_endpoint import BF16DecodeEndpoint
from kvbench.runtime.gqa_audit import (
    OperatorAudit,
    SourceAudit,
    audit_cache_geometry,
    audit_gqa_operator,
    audit_mha_operator_control,
    audit_source_paths,
)
from kvbench.runtime.numerical import tensor_sha256_untimed
from kvbench.runtime.static_cache import BF16StaticCache
from kvbench.runtime.telemetry import (
    TelemetryError,
    TelemetrySnapshot,
    collect_telemetry,
    telemetry_sampling_interval_seconds,
)
from kvbench.runtime.timing import TimingResult, measure_growing_trajectory


_TORCH: Any | None = None


class GrowingContextRunnerError(RuntimeError):
    """The growing-context runner could not preserve its frozen semantics."""


@dataclass(frozen=True, slots=True)
class GrowingStepEvidence:
    """Untimed checksum evidence for one declared growing step."""

    step: int
    historical_active_length: int
    output_checksum: str
    output_finite: bool

    def to_dict(self) -> dict[str, int | str]:
        return {
            "step": self.step,
            "historical_active_length": self.historical_active_length,
            "output_checksum": self.output_checksum,
            "output_finite": self.output_finite,
        }


@dataclass(frozen=True, slots=True)
class GrowingContextRunResult:
    """Machine-readable eager trajectory evidence with raw step sequence."""

    starting_context: int
    output_steps: int
    batch_size: int
    output_checksum: str
    output_finite: bool
    active_lengths: tuple[int, ...]
    step_evidence: tuple[GrowingStepEvidence, ...]
    cache_accounting: dict[str, int]
    cache_layout_fingerprint: str
    cache_pointers_stable: bool
    timing: TimingResult | None
    timing_skipped_reason: str | None
    step_checksum_source: str
    allocation: AllocationAudit
    memory_evidence: NormalTimingMemoryEvidence
    backend: BackendAudit
    prefill_backend: BackendAudit
    gqa_source: SourceAudit
    gqa_cache_geometry: dict[str, Any]
    gqa_operators: tuple[OperatorAudit, ...]
    mha_control: OperatorAudit
    telemetry_before: dict[str, Any]
    telemetry_after: dict[str, Any]
    telemetry_sampling_interval_seconds: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "runner": "growing_context",
            "context_convention": "historical_active_length_before_append",
            "starting_context": self.starting_context,
            "output_steps": self.output_steps,
            "batch_size": self.batch_size,
            "graph_mode": "eager",
            "output_checksum": self.output_checksum,
            "output_finite": self.output_finite,
            "active_lengths": list(self.active_lengths),
            "step_evidence": [item.to_dict() for item in self.step_evidence],
            "cache_accounting": self.cache_accounting,
            "cache_layout_fingerprint": self.cache_layout_fingerprint,
            "cache_pointers_stable": self.cache_pointers_stable,
            "timing": None if self.timing is None else self.timing.to_dict(),
            "timing_skipped_reason": self.timing_skipped_reason,
            "step_checksum_source": self.step_checksum_source,
            "allocation": self.allocation.to_dict(),
            "memory_evidence": self.memory_evidence.to_dict(),
            "backend": self.backend.to_dict(),
            "prefill_backend": self.prefill_backend.to_dict(),
            "gqa_source": self.gqa_source.to_dict(),
            "gqa_cache_geometry": self.gqa_cache_geometry,
            "gqa_operators": [item.to_dict() for item in self.gqa_operators],
            "mha_control": self.mha_control.to_dict(),
            "telemetry_before": self.telemetry_before,
            "telemetry_after": self.telemetry_after,
            "telemetry_sampling_interval_seconds": (
                self.telemetry_sampling_interval_seconds
            ),
            "telemetry_stability_inference": False,
            "quality_status": "unvalidated",
            "claim_eligibility": "performance_only",
            "performance_claim_eligible": False,
            "measurement_scope": "native_host_admission",
        }


def _torch() -> Any:
    global _TORCH
    if _TORCH is None:
        try:
            _TORCH = importlib.import_module("torch")
        except ModuleNotFoundError as error:
            raise GrowingContextRunnerError("PyTorch is unavailable") from error
    return _TORCH


def _telemetry_or_error() -> tuple[dict[str, Any], TelemetrySnapshot | None]:
    try:
        snapshot = collect_telemetry()
        return snapshot.to_dict(), snapshot
    except TelemetryError as error:
        return (
            {
                "error": type(error).__name__,
                "available": False,
                "raw_snapshot": False,
                "stability_inference": False,
            },
            None,
        )


def run_growing_context(
    model: Any,
    prefix_input_ids: Any,
    decode_input_ids: Any,
    *,
    starting_context: int,
    warmup_trajectories: int = 1,
) -> GrowingContextRunResult:
    """Run one measured eager trajectory after a fully separate warmup state."""

    torch = _torch()
    if prefix_input_ids.ndim != 2 or decode_input_ids.ndim != 2:
        raise GrowingContextRunnerError("growing input tensors must be rank two")
    batch = int(prefix_input_ids.shape[0])
    if int(prefix_input_ids.shape[1]) != starting_context:
        raise GrowingContextRunnerError("prefix tensor differs from starting context")
    if int(decode_input_ids.shape[0]) != batch:
        raise GrowingContextRunnerError("decode batch differs from prefix batch")
    if prefix_input_ids.device != decode_input_ids.device:
        raise GrowingContextRunnerError("growing input devices differ")
    output_steps = int(decode_input_ids.shape[1])
    if output_steps <= 0:
        raise GrowingContextRunnerError("growing trajectory requires output tokens")
    if isinstance(warmup_trajectories, bool) or warmup_trajectories <= 0:
        raise GrowingContextRunnerError("warmup_trajectories must be positive")
    device = prefix_input_ids.device
    workspace_bytes = 32 * batch * (32 + 8) * 1 * 64 * 2
    model_memory = capture_cuda_memory_snapshot(
        "model_baseline",
        device=device,
    )
    torch.cuda.reset_peak_memory_stats(device=device)
    cache = BF16StaticCache(
        num_layers=32,
        batch_size=batch,
        num_kv_heads=8,
        capacity=starting_context + output_steps,
        head_dim=128,
        device=device,
        workspace_bytes=workspace_bytes,
    )
    cache_memory = capture_cuda_memory_snapshot(
        "post_cache_allocation",
        device=device,
    )
    endpoint = BF16DecodeEndpoint(model, cache)
    if endpoint.workspace_bytes != workspace_bytes:
        raise GrowingContextRunnerError("endpoint workspace accounting differs")
    cache_positions = tuple(
        torch.tensor(
            [starting_context + step],
            dtype=torch.long,
            device=device,
        )
        for step in range(output_steps)
    )
    position_embeddings = tuple(
        endpoint.prepare_position_embeddings(position.unsqueeze(0))
        for position in cache_positions
    )
    token_views = tuple(
        decode_input_ids[:, step : step + 1] for step in range(output_steps)
    )

    def fresh_prefix() -> None:
        cache.reset_active_length(0)
        endpoint.prefill(prefix_input_ids)
        cache.prepare_growing(starting_context, output_steps)

    def trajectory_step(step: int) -> Any:
        cache.select_growing_step(step)
        output = endpoint.decode(
            token_views[step],
            cache_positions[step],
            position_embeddings[step],
        )
        cache.finish_growing_step()
        return output

    def full_trajectory() -> Any:
        output: Any = None
        for step in range(output_steps):
            output = trajectory_step(step)
        return output

    with torch.inference_mode(), forced_flash_execution():
        for _ in range(warmup_trajectories):
            fresh_prefix()
            full_trajectory()
        torch.cuda.synchronize(device=device)
        fresh_prefix()
        runtime_root = Path(__file__).resolve().parent
        gqa_source = audit_source_paths(
            (
                runtime_root / "backend.py",
                runtime_root / "static_cache.py",
                runtime_root / "bf16_endpoint.py",
            )
        )
        gqa_cache_geometry = audit_cache_geometry(
            cache,
            num_query_heads=32,
        )
        prefill_query_probe = torch.empty(
            (batch, 32, starting_context, 128),
            dtype=torch.bfloat16,
            device=device,
        )
        prefill_backend = audit_backend_choice(
            prefill_query_probe,
            cache.keys[0, :, :, :starting_context, :],
            cache.values[0, :, :, :starting_context, :],
            is_causal=True,
            scale=128**-0.5,
        )
        del prefill_query_probe
        query_probe = torch.empty(
            (batch, 32, 1, 128),
            dtype=torch.bfloat16,
            device=device,
        )
        backend = audit_backend_choice(
            query_probe,
            cache.keys[0, :, :, : starting_context + 1, :],
            cache.values[0, :, :, : starting_context + 1, :],
            is_causal=False,
            scale=128**-0.5,
        )
        gqa_operators = tuple(
            audit_gqa_operator(
                query_probe,
                cache.keys[0, :, :, : starting_context + step + 1, :],
                cache.values[0, :, :, : starting_context + step + 1, :],
                is_causal=False,
                scale=128**-0.5,
            )
            for step in range(output_steps)
        )
        mha_query = torch.empty(
            (batch, 32, 1, 128),
            dtype=torch.bfloat16,
            device=device,
        )
        mha_key = torch.empty(
            (batch, 32, starting_context + 1, 128),
            dtype=torch.bfloat16,
            device=device,
        )
        mha_value = torch.empty_like(mha_key)
        mha_control = audit_mha_operator_control(
            mha_query,
            mha_key,
            mha_value,
            is_causal=False,
            scale=128**-0.5,
        )
        del mha_query, mha_key, mha_value, query_probe
        pointers_before = cache.pointers()
        setup_memory = capture_cuda_memory_snapshot(
            "post_setup",
            device=device,
        )
        allocation = audit_cuda_allocations(full_trajectory, device=device)
        fresh_prefix()
        telemetry_before, telemetry_before_snapshot = _telemetry_or_error()
        timing: TimingResult | None = None
        timing_skipped_reason: str | None = None
        if allocation.passed:
            torch.cuda.synchronize(device=device)
            torch.cuda.reset_peak_memory_stats(device=device)
            timing_memory_before = capture_cuda_memory_snapshot(
                "normal_timing_before",
                device=device,
            )
            timing = measure_growing_trajectory(
                trajectory_step,
                output_steps=output_steps,
                device=device,
            )
            timing_memory_after = capture_cuda_memory_snapshot(
                "normal_timing_after",
                device=device,
            )
        else:
            timing_skipped_reason = allocation.failure_reason or "allocation_failed"
            timing_memory_before = capture_cuda_memory_snapshot(
                "normal_timing_not_run_before",
                device=device,
            )
            timing_memory_after = capture_cuda_memory_snapshot(
                "normal_timing_not_run_after",
                device=device,
            )
        telemetry_after, telemetry_after_snapshot = _telemetry_or_error()
        telemetry_interval = (
            None
            if telemetry_before_snapshot is None
            or telemetry_after_snapshot is None
            else telemetry_sampling_interval_seconds(
                telemetry_before_snapshot,
                telemetry_after_snapshot,
            )
        )
        fresh_prefix()
        evidence: list[GrowingStepEvidence] = []
        for step in range(output_steps):
            output = trajectory_step(step)
            evidence.append(
                GrowingStepEvidence(
                    step=step,
                    historical_active_length=starting_context + step,
                    output_checksum=tensor_sha256_untimed(output),
                    output_finite=bool(torch.isfinite(output).all().item()),
                )
            )
        pointers_after = cache.pointers()
        output_checksum = evidence[-1].output_checksum
        output_finite = all(item.output_finite for item in evidence)
    accounting = cache.accounting().to_dict()
    accounting["model_baseline_allocated_bytes"] = model_memory.allocated_bytes
    memory_evidence = NormalTimingMemoryEvidence(
        model_baseline=model_memory,
        post_cache_allocation=cache_memory,
        post_setup=setup_memory,
        timing_before=timing_memory_before,
        timing_after=timing_memory_after,
        timing_executed=timing is not None,
    )
    return GrowingContextRunResult(
        starting_context=starting_context,
        output_steps=output_steps,
        batch_size=batch,
        output_checksum=output_checksum,
        output_finite=output_finite,
        active_lengths=tuple(
            starting_context + step for step in range(output_steps)
        ),
        step_evidence=tuple(evidence),
        cache_accounting=accounting,
        cache_layout_fingerprint=cache.layout_fingerprint(),
        cache_pointers_stable=pointers_before == pointers_after,
        timing=timing,
        timing_skipped_reason=timing_skipped_reason,
        step_checksum_source=(
            "untimed_deterministic_replay_after_timing"
            if timing is not None
            else "untimed_deterministic_replay_after_allocation_rejection"
        ),
        allocation=allocation,
        memory_evidence=memory_evidence,
        backend=backend,
        prefill_backend=prefill_backend,
        gqa_source=gqa_source,
        gqa_cache_geometry=gqa_cache_geometry,
        gqa_operators=gqa_operators,
        mha_control=mha_control,
        telemetry_before=telemetry_before,
        telemetry_after=telemetry_after,
        telemetry_sampling_interval_seconds=telemetry_interval,
    )
