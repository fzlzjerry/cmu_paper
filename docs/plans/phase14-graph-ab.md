# Phase 14 CUDA Graph OFF/ON Mechanism Experiment

Status: preregistered and frozen before CUDA timing.

## Scope and authority

Phase 14 uses the ten admitted main configurations and their successful
Phase 13 fingerprints: `bf16`; `tq_4bit_nc`, `tq_k3v4_nc`, `tq_3bit_nc`;
`k4v4`, `k2v4`, `k2v2`; and `kvq4`, `kvq3`, `kvq2`.  Execution uses only the
authorized Measurement Container
`sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e`.
Adapters, CUDA, cache layouts, numerical tolerances, prefix inputs, runner
timing, and telemetry sampling are unchanged.  Phase 15, Full Scan, profiling,
and quality evaluation are deferred.

## Frozen paired grid

- Runner: fixed-L; `L` is historical prefix length, with label `131072`
  mapped to historical prefix `131071` and total attended length `131072`.
- Batch sizes: `1, 4`.
- Context labels: `4096, 16384, 24576, 32768, 65536, 131072`.
- Modes: `eager`, `cuda_graph`.
- Warmup operations: `64`; measured operations per batch: `128`; existing
  measured-batch count: `5`.
- Independent process replicates: `3`.
- Seeds: `20260826`, `20260827`, `20260828`.
- Design: 360 A/B pairs and 720 mode-specific process records before
  feasibility.  Within each configuration block, B/L pairs are randomized;
  block order rotates by replicate; pair members are adjacent and their first
  mode is randomized.  The complete order is checksum-bound in
  `docs/plans/phase14-graph-ab-execution-order.json`.

Pair feasibility uses the current Phase 13 allocation formulas and the Graph
reserve, because both modes must fit.  A pair is either `pair_feasible` or
`pair_capacity_infeasible`; one-sided execution is forbidden.  Every feasible
mode runs in a fresh process.  The same exact-batch, exact-context,
checksum-bound Phase 13 prefix state is restored read-only for both members,
outside timing.

## Failure and QC

CUDA/runtime failure, numerical mismatch, fallback, identity drift, actual
foreign GPU activity, or timing-semantic drift aborts the campaign.  Snapshot,
telemetry, report, or publication-query failures preserve their raw payload and
do not by themselves abort other timing records.  No slow or non-monotonic run
is selectively repeated.

For each mode, the three process wall-time medians must have `CV <= 0.03`.
Output, backend, cache, path, and allocation identities must agree.  Paired
metrics are eager/Graph wall-time ratio and host-wall minus CUDA-event proxies;
the latter are not direct launch-gap measurements.  The existing knee model is
fit independently by configuration, batch, and mode using process observations.
The preregistered descriptive interpretation flags are: material floor
reduction when `tau_eager / tau_graph >= 1.05`, and similar slope when
`0.80 <= s_graph / s_eager <= 1.20`; neither flag is a performance or quality
claim.

## Evidence and publication

The append-only campaign is created under `artifacts/phase14/<campaign_id>/`
with raw run records, paired summaries, mode fits, effects, QC, plots,
inventory, checksums, and `COMPLETE` written last.  It is published through the
existing content-addressed R2 client and verified by one clean retrieval.
Quality remains unvalidated, performance claims remain ineligible, Full Scan
remains closed, and a PASS makes only Phase 15 ready.

Machine-readable claim boundaries remain `quality_status = unvalidated` and
`performance_claim_eligible = false`; Full Scan remains CLOSED and Phase 15 is
deferred until a separate task.

Phase 15 is deferred.
