# Decision 0032 — Phase 13 bounded stage-aware supervision

- Status: Accepted
- Date: 2026-08-02
- Scope: Phase 13 process supervision only

## Context

The stopped Phase 13R2 campaign applied one 7,200-second wall-clock deadline
to every feasible point.  The exact-container diagnostic for
`kvq2/B=1/L=131072` showed normal forward progress during the unchanged
32-layer prefix construction: model loading completed quickly, while each
long-prefix layer continued consuming GPU work.  The direct child was neither
stalled nor deadlocked; the global deadline expired during source-faithful
prefill before Graph capture, warmup, measurement, or finalization.

## Decision

Phase 13 workers use exclusive ordered transitions for `model_load`,
`prefix_construction`, `graph_capture`, `warmup_and_audit`,
`measurement`, and `finalization`.  The supervisor validates each event
and its SHA-256, retains the direct-child identity, and applies one immutable
deadline per active stage.  Repeated observation or any heartbeat cannot
extend a deadline.  Missing, reordered, altered, duplicated, or unknown stage
evidence fails closed.

The frozen deadlines in seconds are:

- startup and each inter-stage transition: 600;
- model load: 900;
- prefix construction:
  `max(3600, 1800 + ceil(B * historical_context / 4))`;
- Graph capture and post-capture correctness: 7,200;
- Phase 13 warmup and audit: 7,200;
- measurement: 7,200;
- finalization: 1,800.

The prefix formula is determined only by preregistered point geometry.  For
`kvq2/B=1/L=131072` it is 34,568 seconds; for the largest frozen feasible
`B * historical_context` it is 100,104 seconds.  A stage deadline remains
finite and a timeout preserves the failed run with its exact stage.

## Preservation

This changes only Pilot process supervision and stage evidence.  It does not
change adapters, CUDA, cache layouts, grid, feasibility classification,
timing or synchronization boundaries, calibration, fixtures, admission
evidence, or analysis rules.  All stopped campaigns remain immutable.  A
successful remediation permits only a wholly new append-only Pilot campaign.
