# Decision 0034 — Phase 13 batch-exact prefix states

- Status: Accepted
- Date: 2026-08-04
- Scope: Phase 13 untimed prefix setup only

## Context

Phase 13P-D proved that restoration of a leading B=8 row is byte-exact, but
the restored row is not numerically identical to a prefix constructed directly
at B=1.  The batch-independence assumption in Decision 0033 is therefore false
for the admitted model execution path.

## Decision

Decision 0033's cross-batch slicing rule is superseded.  Each reusable state is
keyed by the exact tuple:

    configuration × batch × context

Separate untimed states are constructed at B=1, B=4, and B=8 wherever the
frozen feasibility contract marks that geometry feasible.  Restoration
requires source batch and target batch to be identical.  Configuration,
method fingerprint, context, tensor inventory/shape/dtype, capacity, cache
layout fingerprint, and lifecycle mismatches fail closed.  Cross-batch
restoration is prohibited.

Every consumer still allocates fresh caller-owned cache and workspace buffers.
State copying occurs before Graph capture, warmup, or timing.  No live pointer,
cache, workspace, CUDA Graph, or runtime prefix cache is shared.  Decision
0033's remaining setup-only and preservation rules continue unchanged.

## Validation and scope

One focused exact-container matrix must pass for all ten configurations at
B=1, B=4, and B=8 with L=17.  Direct and restored cache bytes, lifecycle,
layout, allocation, output checksum, kernel path, pointer invariants, and CUDA
Graph results must agree exactly.  Cross-batch controls must be rejected before
cache mutation.

This decision changes no adapter, CUDA source, cache layout, feasibility
classification, grid, timing boundary, calibration, fixture, or admission
evidence.  It authorizes only remediation evidence and does not authorize a
Pilot, Phase 14, Full Scan, profiling, or quality evaluation.
