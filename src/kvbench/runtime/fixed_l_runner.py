"""Fixed-historical-context eager and CUDA Graph Phase 3 runner."""

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
from kvbench.runtime.cuda_graph import CapturedFixedGraph, capture_fixed_graph
from kvbench.runtime.gqa_audit import (
    OperatorAudit,
    SourceAudit,
    audit_cache_geometry,
    audit_gqa_operator,
    audit_mha_operator_control,
    audit_source_paths,
)
from kvbench.runtime.numerical import (
    NumericalComparison,
    cache_history_sha256_untimed,
    compare_tensors_untimed,
    tensor_sha256_untimed,
)
from kvbench.runtime.static_cache import BF16StaticCache
from kvbench.runtime.telemetry import (
    TelemetryError,
    TelemetrySnapshot,
    collect_telemetry,
    telemetry_sampling_interval_seconds,
)
from kvbench.runtime.timing import (
    TimingResult,
    measure_fixed_batches,
    warmup_operations,
)


_TORCH: Any | None = None


class FixedLRunnerError(RuntimeError):
    """The fixed-L runner could not preserve its frozen semantics."""


@dataclass(frozen=True, slots=True)
class FixedLRunResult:
    """Machine-readable fixed-L runtime evidence with no scientific claim."""

    context_length: int
    total_attended_length: int
    batch_size: int
    graph_mode: str
    output_checksum: str
    output_finite: bool
    historical_checksum_before: str
    historical_checksum_after: str
    historical_cache_unchanged: bool
    cache_pointers_stable: bool
    cache_accounting: dict[str, int]
    cache_layout_fingerprint: str
    timing: TimingResult | None
    timing_skipped_reason: str | None
    allocation: AllocationAudit
    memory_evidence: NormalTimingMemoryEvidence
    backend: BackendAudit
    prefill_backend: BackendAudit
    gqa_source: SourceAudit
    gqa_cache_geometry: dict[str, Any]
    gqa_operator: OperatorAudit
    mha_control: OperatorAudit
    graph: dict[str, Any] | None
    eager_graph_comparison: NumericalComparison | None
    telemetry_before: dict[str, Any]
    telemetry_after: dict[str, Any]
    telemetry_sampling_interval_seconds: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "runner": "fixed_l",
            "context_convention": "historical_prefix_length",
            "context_length": self.context_length,
            "total_attended_length": self.total_attended_length,
            "batch_size": self.batch_size,
            "graph_mode": self.graph_mode,
            "output_checksum": self.output_checksum,
            "output_finite": self.output_finite,
            "historical_checksum_before": self.historical_checksum_before,
            "historical_checksum_after": self.historical_checksum_after,
            "historical_cache_unchanged": self.historical_cache_unchanged,
            "cache_pointers_stable": self.cache_pointers_stable,
            "cache_accounting": self.cache_accounting,
            "cache_layout_fingerprint": self.cache_layout_fingerprint,
            "timing": None if self.timing is None else self.timing.to_dict(),
            "timing_skipped_reason": self.timing_skipped_reason,
            "allocation": self.allocation.to_dict(),
            "memory_evidence": self.memory_evidence.to_dict(),
            "backend": self.backend.to_dict(),
            "prefill_backend": self.prefill_backend.to_dict(),
            "gqa_source": self.gqa_source.to_dict(),
            "gqa_cache_geometry": self.gqa_cache_geometry,
            "gqa_operator": self.gqa_operator.to_dict(),
            "mha_control": self.mha_control.to_dict(),
            "graph": self.graph,
            "eager_graph_comparison": (
                None
                if self.eager_graph_comparison is None
                else self.eager_graph_comparison.to_dict()
            ),
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
            raise FixedLRunnerError("PyTorch is unavailable") from error
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


def _validate_inputs(
    prefix_input_ids: Any,
    current_input_ids: Any,
    context_length: int,
) -> int:
    if prefix_input_ids.ndim != 2 or current_input_ids.ndim != 2:
        raise FixedLRunnerError("fixed-L input tensors must be rank two")
    batch = int(prefix_input_ids.shape[0])
    if int(prefix_input_ids.shape[1]) != context_length:
        raise FixedLRunnerError("prefix tensor length differs from fixed L")
    if tuple(current_input_ids.shape) != (batch, 1):
        raise FixedLRunnerError("current token fixture must have shape [B,1]")
    if prefix_input_ids.device != current_input_ids.device:
        raise FixedLRunnerError("fixed-L input devices differ")
    return batch


def run_fixed_l(
    model: Any,
    prefix_input_ids: Any,
    current_input_ids: Any,
    *,
    context_length: int,
    graph_mode: str,
    warmup_steps: int = 16,
    measured_steps: int = 32,
    measured_batches: int = 5,
) -> FixedLRunResult:
    """Run the exact frozen fixed-L endpoint and retain every batch sample."""

    torch = _torch()
    if graph_mode not in {"eager", "cuda_graph"}:
        raise FixedLRunnerError("graph_mode must be eager or cuda_graph")
    batch = _validate_inputs(
        prefix_input_ids,
        current_input_ids,
        context_length,
    )
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
        capacity=context_length + 1,
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
        raise FixedLRunnerError("endpoint workspace accounting differs")
    graph_evidence: dict[str, Any] | None = None
    graph_comparison: NumericalComparison | None = None
    with torch.inference_mode(), forced_flash_execution():
        endpoint.prefill(prefix_input_ids)
        cache.prepare_fixed(context_length)
        cache_position = torch.tensor(
            [context_length],
            dtype=torch.long,
            device=device,
        )
        position_ids = cache_position.unsqueeze(0)
        positions = endpoint.prepare_position_embeddings(position_ids)

        def eager_operation() -> Any:
            return endpoint.decode(
                current_input_ids,
                cache_position,
                positions,
            )

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
            (batch, 32, context_length, 128),
            dtype=torch.bfloat16,
            device=device,
        )
        prefill_backend = audit_backend_choice(
            prefill_query_probe,
            cache.keys[0, :, :, :context_length, :],
            cache.values[0, :, :, :context_length, :],
            is_causal=True,
            scale=128**-0.5,
        )
        del prefill_query_probe
        key_view = cache.keys[0, :, :, : context_length + 1, :]
        value_view = cache.values[0, :, :, : context_length + 1, :]
        query_probe = torch.empty(
            (batch, 32, 1, 128),
            dtype=torch.bfloat16,
            device=device,
        )
        backend = audit_backend_choice(
            query_probe,
            key_view,
            value_view,
            is_causal=False,
            scale=128**-0.5,
        )
        gqa_operator = audit_gqa_operator(
            query_probe,
            key_view,
            value_view,
            is_causal=False,
            scale=128**-0.5,
        )
        mha_query = torch.empty(
            (batch, 32, 1, 128),
            dtype=torch.bfloat16,
            device=device,
        )
        mha_key = torch.empty(
            (batch, 32, context_length + 1, 128),
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
        history_before = cache_history_sha256_untimed(
            cache.keys,
            cache.values,
            historical_length=context_length,
        )
        last_setup_output = warmup_operations(
            eager_operation,
            count=warmup_steps,
            device=device,
        )
        operation = eager_operation
        if graph_mode == "cuda_graph":
            eager_reference = eager_operation().detach().cpu()
            torch.cuda.synchronize(device=device)
            captured: CapturedFixedGraph = capture_fixed_graph(
                eager_operation,
                warmup_steps=3,
                device=device,
            )
            graph_output = captured.replay()
            torch.cuda.synchronize(device=device)
            first_replay = graph_output.detach().cpu().clone()
            captured.replay()
            torch.cuda.synchronize(device=device)
            second_replay = captured.output.detach().cpu().clone()
            graph_comparison = compare_tensors_untimed(
                first_replay,
                eager_reference,
                atol=0.02,
                rtol=0.02,
            )
            graph_evidence = captured.to_dict()
            graph_evidence.update(
                {
                    "consecutive_replay_outputs_exact": bool(
                        torch.equal(first_replay, second_replay)
                    ),
                    "first_replay_checksum": tensor_sha256_untimed(
                        first_replay
                    ),
                    "second_replay_checksum": tensor_sha256_untimed(
                        second_replay
                    ),
                }
            )
            last_setup_output = captured.output
            operation = captured.replay
        setup_memory = capture_cuda_memory_snapshot(
            "post_setup",
            device=device,
        )
        allocation = audit_cuda_allocations(operation, device=device)
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
            timing = measure_fixed_batches(
                operation,
                operations_per_batch=measured_steps,
                batches=measured_batches,
                device=device,
            )
            timing_memory_after = capture_cuda_memory_snapshot(
                "normal_timing_after",
                device=device,
            )
            last_output = timing.last_output
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
            last_output = last_setup_output
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
        output_checksum = tensor_sha256_untimed(last_output)
        output_finite = bool(torch.isfinite(last_output).all().item())
        history_after = cache_history_sha256_untimed(
            cache.keys,
            cache.values,
            historical_length=context_length,
        )
        pointers_after = cache.pointers()
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
    return FixedLRunResult(
        context_length=context_length,
        total_attended_length=context_length + 1,
        batch_size=batch,
        graph_mode=graph_mode,
        output_checksum=output_checksum,
        output_finite=output_finite,
        historical_checksum_before=history_before,
        historical_checksum_after=history_after,
        historical_cache_unchanged=history_before == history_after,
        cache_pointers_stable=pointers_before == pointers_after,
        cache_accounting=accounting,
        cache_layout_fingerprint=cache.layout_fingerprint(),
        timing=timing,
        timing_skipped_reason=timing_skipped_reason,
        allocation=allocation,
        memory_evidence=memory_evidence,
        backend=backend,
        prefill_backend=prefill_backend,
        gqa_source=gqa_source,
        gqa_cache_geometry=gqa_cache_geometry,
        gqa_operator=gqa_operator,
        mha_control=mha_control,
        graph=graph_evidence,
        eager_graph_comparison=graph_comparison,
        telemetry_before=telemetry_before,
        telemetry_after=telemetry_after,
        telemetry_sampling_interval_seconds=telemetry_interval,
    )
