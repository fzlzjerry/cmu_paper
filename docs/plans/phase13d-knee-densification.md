# Phase 13D — Preregistered Knee Densification

Status: frozen before CUDA execution. Phase 14 is explicitly deferred.

## Immutable source

- Source campaign: `phase13-20260822t150835736582z-4ddd7b17-3a8fb3`
- Source root: `feb2e5a8ebba8b729c182fc8170107c9acf8128edd3e5618c8f1b90530557531`
- Source URI: `r2://kvbench-artifacts/kvbench/sha256/feb2e5a8ebba8b729c182fc8170107c9acf8128edd3e5618c8f1b90530557531/`
- Candidate authority: `docs/plans/phase13d-candidate-table.json`
- Execution authority: `docs/plans/phase13d-execution-order.json`
- The source campaign is referenced, never copied or rewritten.

## Exact targets

All rows have original fit status `knee_observed`; each listed interval is the
stored session-bootstrap 95% interval.

| Configuration | B | Original L* | 95% interval | Original below/near/above |
|---|---:|---:|---|---|
| bf16 | 1 | 4096 | [4096, 4096] | 0/1/8 |
| bf16 | 4 | 4096 | [4096, 4096] | 0/1/6 |
| bf16 | 8 | 4096 | [4096, 4096] | 0/1/4 |
| tq_4bit_nc | 1 | 4096 | [4096, 4096] | 0/1/8 |
| tq_4bit_nc | 4 | 4096 | [4096, 4096] | 0/1/7 |
| tq_4bit_nc | 8 | 4096 | [4096, 4096] | 0/1/5 |
| tq_k3v4_nc | 1 | 4096 | [4096, 4096] | 0/1/8 |
| tq_k3v4_nc | 4 | 4096 | [4096, 4096] | 0/1/7 |
| tq_k3v4_nc | 8 | 4096 | [4096, 4096] | 0/1/5 |
| tq_3bit_nc | 1 | 4096 | [4096, 4096] | 0/1/8 |
| tq_3bit_nc | 4 | 4096 | [4096, 4096] | 0/1/7 |
| tq_3bit_nc | 8 | 4096 | [4096, 4096] | 0/1/5 |
| k4v4 | 1 | 6144 | [6144, 6144] | 1/0/8 |
| k4v4 | 4 | 4096 | [4096, 4096] | 0/1/7 |
| k4v4 | 8 | 4096 | [4096, 4096] | 0/1/5 |
| k2v4 | 1 | 6144 | [6144, 6144] | 1/0/8 |
| k2v4 | 4 | 4096 | [4096, 4096] | 0/1/7 |
| k2v4 | 8 | 4096 | [4096, 4096] | 0/1/5 |
| k2v2 | 1 | 6144 | [6144, 6144] | 1/0/8 |
| k2v2 | 4 | 4096 | [4096, 4096] | 0/1/7 |
| k2v2 | 8 | 4096 | [4096, 4096] | 0/1/5 |
| kvq4 | 8 | 4096 | [4096, 4096] | 0/1/5 |
| kvq3 | 8 | 4096 | [4096, 4096] | 0/1/5 |
| kvq2 | 4 | 4096 | [4096, 4096] | 0/1/7 |
| kvq2 | 8 | 4096 | [4096, 4096] | 0/1/5 |

The five source rows already marked sufficient are excluded.

## Candidate and feasibility contract

For every target, generate `0.60, 0.75, 0.90, 1.00, 1.10, 1.25, 1.50`
times its stored L*. Round to the nearest 128 tokens with deterministic
half-up rounding, clamp historical context to `[4096, 131071]`, then remove
within-target duplicates and contexts already present for that config/batch.
Every proposal and exclusion remains in the committed candidate table.

Recompute feasibility for every included candidate with the current admitted
fingerprint and the frozen Phase 13 end-to-end formula: model weights, owned
cache and method metadata/residual/sink/outlier bytes, workspace, prefix peak,
graph reserve, and the 0.88 GPU-memory limit. Known-infeasible candidates are
recorded and not launched; no replacement point is permitted.

## Execution and QC

- Runner/graph: `fixed_l` / `cuda_graph`
- Warmup/measured: 64 / 128 operations, using the existing five measured batches
- Independent processes: 3
- Seeds: `20260823`, `20260824`, `20260825`
- Randomization: config/batch target blocks, randomized contexts within block,
  and rotated/randomized block order across replicates
- Retry policy: no selective rerun; a runtime implementation failure preserves
  the failure and aborts the remaining schedule
- Stability: process-median CV `<= 0.03`, plus exact output/path/allocation and
  finite/no-fallback/no-foreign-process checks
- Timing semantics, telemetry, container, model, tokenizer, adapters, and graph
  strategy are identical to the successful source Pilot.

## Combined fit and density

Reference stable source observations plus stable Phase 13D observations through
a checksum-bound combined index; do not copy source raw samples. Reuse the
existing constant, linear, and `max(tau, a+sL)` implementation and session-level
bootstrap. Reuse the existing density criterion exactly: at least one stable
context below `0.75 L*`, one within `[0.75 L*, 1.25 L*]`, and one above
`1.25 L*`. Allowed resolved statuses are `density_sufficient`,
`knee_below_range`, `knee_above_range`, `no_positive_slope`, and
`insufficient_feasible_span` after all preregistered feasible candidates are
exhausted. `densification_required`, `unstable_data`, and `fit_failed` remain
unresolved.

## Outputs and custody

Use one new append-only `artifacts/phase13d/<campaign_id>/` bundle containing
the target/candidate/order/feasibility/run/QC/combined-fit indices, diagnostics,
inventory, checksums, and `COMPLETE` written last. Publish content-addressed to
R2, then perform one clean empty-directory retrieval. Full Scan remains CLOSED,
quality remains LOCKED, `PERFORMANCE_DATA_FROZEN` remains absent, and all ratios
remain Pilot-only, quality-unvalidated, and claim-ineligible.
