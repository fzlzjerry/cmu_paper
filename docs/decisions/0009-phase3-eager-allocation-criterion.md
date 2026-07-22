# Decision 0009: Phase 3 eager allocation attribution criterion

- Status: Accepted
- Date: 2026-07-22
- Authority: AGENTS.md, the Phase 3 remediation instruction, Decision 0007,
  and the immutable Phase 3 G1 report
- Supersedes: the undifferentiated eager zero-allocator-event interpretation
  for the frozen BF16 baseline only
- Superseded by: none

## Context

The immutable Phase 3 runs recorded eager allocator events but retained only
counts and requested-byte totals. A bounded remediation diagnostic using the
exact frozen model endpoint showed zero `segment_alloc`, zero persistent
allocated or reserved growth, and zero device-used-memory growth, while also
showing that the requested-byte total changes with attended context. The old
evidence therefore cannot be described as fixed-output-only allocation.

The public frozen PyTorch Flash SDPA API has neither an output-buffer nor a
workspace-buffer argument. The same frozen build exposes a private
`aten::_flash_attention_forward_no_dropout_inplace` operator, but the requested
fixed `num_splits=1` control is rejected because the selected backend is FA2
and fixed split counts require FA3. A separately labelled FA2 auto-split
control accepted native GQA and a caller-provided output, but at context 4096
it allocated split-K accumulation and LSE workspaces. Those workspaces were
identified by C++ stacks under `pytorch_flash::set_params_splitkv`, matched the
observed split count and geometry exactly, occurred in both GQA and MHA
controls, were fully freed, reused cached allocator blocks, caused no segment
or device allocation, and caused no persistent allocator or device-memory
growth. The private operator therefore removes only the final output
allocation and does not solve workspace allocation.

Changing to that private operator would add an unstable ABI without satisfying
a stricter no-context-allocation rule. No backend change is accepted.
Decision 0007's public, forced-Flash SDPA backend remains selected.

## Decision

### Eager lane

The frozen BF16 baseline uses criterion
`phase3_eager_attributed_ephemeral_v1`. An eager allocation audit passes only
when all of the following independently rederived conditions hold:

1. Allocator history is complete, uses Python and C++ stacks, has not reached
   its ring-buffer limit, and every chronological address lifecycle is valid.
2. Every allocation is attributed to exactly one of
   `fixed_output`, `fixed_shared_activation`,
   `framework_bookkeeping`, or `context_scaled_workspace`.
3. A `context_scaled_workspace` event is permitted only when its C++ stack,
   requested bytes, frozen geometry, and observed Flash split-K kernel identify
   the exact accumulation or LSE formula; the same mechanism must be present
   in the frozen MHA control. Merely changing with context is insufficient.
4. No event is `cache_growth`, `gqa_expansion`, `unknown`, or
   `audit_instrumentation`.
5. No `segment_alloc`, `cudaMalloc`, or positive device-allocation counter
   delta occurs after warmup, and every allocation is proven reused from the
   PyTorch cache.
6. Allocated, reserved, and device-used memory have no positive persistent
   delta; any non-PyTorch residual is zero.
7. Requested and allocator-rounded block accounting is complete. A missing or
   inconsistent transient block-size derivation fails closed.
8. Exact single-tensor and combined expanded-K/V sizes are checked. An
   attention, repeat, expand, copy, or unknown stack at either expanded size is
   `gqa_expansion`; a size collision is explainable only by an independently
   proven non-attention stack and formula.

This is not a claim that eager execution has zero allocation events or zero
context-dependent workspace. Evidence must explicitly record
`no_context_dependent_allocation: false` whenever the attributed split-K
workspace occurs. The scientific guarantee is narrower: no new device
allocation, no persistent growth, no cache growth, no GQA expansion, and no
unexplained event, with all remaining ephemeral storage bounded by recorded
formulas and lifetimes.

### Graph lane

CUDA Graph replay retains criterion `phase3_graph_zero_allocation_v1`:

- zero allocation, free, segment, and device-allocation events during replay;
- zero allocated, reserved, and device-used deltas;
- no eager exception for output, activation, bookkeeping, or workspace.

### Gate and reporting semantics

- Any incomplete stack, lifetime, block-size, dependency, device-memory, or
  raw-snapshot proof fails G1.
- Any context-dependent allocation not meeting the exact frozen Flash
  workspace proof is `unknown` and fails G1.
- Eager and graph criteria are reported separately and never compared as if
  their execution modes were interchangeable.
- Profiler-instrumented traces are separate dispatch evidence and never supply
  benchmark timing or allocator-gate timing.
- Report derivation must replay this criterion from raw events; it must not
  trust a serialized `passed` field.

## Consequences

- B-012 can close only after production allocator evidence reproduces the
  event-level attribution across all preregistered Phase 3 geometries.
- The direct private FA2 candidate is rejected; there is no silent backend
  switch.
- A future build, kernel family, split policy, or workspace formula change
  invalidates the context-workspace attribution and fails closed pending a new
  decision.
- No physical HBM-traffic conclusion follows from allocator sizes or latency.
