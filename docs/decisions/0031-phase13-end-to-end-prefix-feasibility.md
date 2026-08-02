# Decision 0031 — Phase 13 end-to-end prefix feasibility

- Status: Accepted
- Date: 2026-08-02
- Scope: Phase 13 feasibility only

## Context

The stopped Phase 13R campaign classified `tq_3bit_nc`, B=8, L=98304 as
feasible from model weights, adapter-owned cache bytes, and a scaled Graph
reserve.  Prefix construction then failed before capture, warmup, or timing:
the attention output projection requested 6.00 GiB with 5.27 GiB free.  The
omitted allocation is exactly `B * L * 4096 * sizeof(BF16)`.

## Decision

Feasibility is the end-to-end allocator high-water contract for the unchanged
fixed-L setup sequence.  For every frozen point define:

- `H = B * L * 4096 * 2`, one BF16 hidden tensor;
- `I = B * L * 14336 * 2`, one BF16 MLP-intermediate tensor;
- `K = B * L * 8 * 128 * 2`, one native-KV projection tensor;
- attention/output-projection peak =
  `2H + H + 2K + H/2 + K/2 + 2H`;
- MLP peak = `2H + 3I`;
- prefix compute peak = the larger of those two peaks.

The required bytes are the sum of exact model-weight bytes, adapter-owned
cache bytes (including its persistent workspace), endpoint RoPE workspace,
prefix token/position/RoPE control tensors, prefix compute peak, and the
existing conservative net Graph reserve scaled from the immutable Phase 12
reference.  This additive lifecycle rule is required because the runner does
not empty the CUDA caching allocator between prefill and private Graph-pool
capture.  The limit remains exactly:

`floor(0.88 * 101970345984)` bytes.

Every one of the frozen 810 records is recomputed.  A point above the limit is
`capacity_infeasible` and is not launched.  Replicate identity must not change
classification.

## Preservation

This changes only the pre-launch feasibility contract.  It does not change the
grid, execution order, adapters, CUDA, cache layouts, calibration, fixtures,
admission evidence, timing boundaries, or numerical tolerances.  Both stopped
Phase 13 campaigns and their classifications remain immutable historical
evidence.  Phase 13F performs no Pilot timing and makes no performance,
capacity-benefit, HBM, or quality claim.

