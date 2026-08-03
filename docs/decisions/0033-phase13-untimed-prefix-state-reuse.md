# Decision 0033 — Phase 13 untimed deterministic prefix-state reuse

- Status: Accepted
- Date: 2026-08-04
- Scope: Phase 13 Pilot setup only

## Context

The stopped Phase 13T-R staging campaign proved that source-faithful direct
prefix construction progresses, but repeating the same deterministic prefill
in every independent process dominates wall time.  The campaign
`phase13-20260803t005021130867z-3e022662-6b36c5` is preserved locally as
`stopped_non_claim_bearing`; it is never resumed, deleted, or analyzed.

## Decision

Before formal timing, Phase 13 constructs one checksum-bound safe-format
prefix state for each configuration and context at the largest feasible batch.
Because model execution and cache writes are batch-independent, target B=1 or
B=4 restores only the corresponding leading rows.  A focused exact-container
control must first prove direct construction and restoration agree for cache
contents, layout, allocation, output checksum, kernel path, and CUDA Graph for
all ten configurations.

Every formal timing point remains a new independent process.  It loads the
model, constructs the admitted adapter, allocates fresh caller-owned cache and
workspace buffers, and copies the immutable prefix state into those buffers
before Graph capture, warmup, or timing.  No CUDA pointer, Graph object, live
cache, or runtime prefix cache is shared between timing processes.  Temporary
prefix payloads are local setup material and are not published; the final
bundle retains their hashes, catalog, restoration receipts, and equivalence
evidence.

Checksum-bound prefix witnesses replace repeated full-history buffer scans in
formal Pilot workers.  Per-point fixture replay, sanitizer execution, and full
G1-G4 re-admission audits are not repeated.  GPU exclusivity, source/container
identity, Graph capture, 64 warmups, 128 measured operations, raw samples,
telemetry, output checksum, kernel path, allocation consistency, and immutable
finalization remain mandatory.

## Preservation

This changes no adapter, CUDA source, cache layout, quantization, calibration,
fixture, admitted fingerprint, grid, execution order, synchronization point,
or timing boundary.  Decision 0031's 684 feasible and 126
`capacity_infeasible` records remain frozen.  A failed equivalence control
blocks execution; a pass authorizes only a wholly new append-only 810-record
Pilot campaign.  Phase 14, Full Scan, and quality evaluation remain deferred.
