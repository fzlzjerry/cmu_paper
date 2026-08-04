# PHASE 13P-B REPORT

Status: PASS

## Change

Decision 0034 supersedes cross-batch prefix slicing. Prefix snapshots are now
keyed by exact `configuration × batch × context`, with separate B=1, B=4, and
B=8 construction and same-batch restoration only. Every restored session owns
fresh caller-allocated buffers; restoration remains before Graph capture,
warmup, or timing. Batch, context, configuration, fingerprint, tensor shape,
layout, capacity, and lifecycle mismatches fail closed.

The focused pointer harness also distinguishes actual allocations from exact
nonallocating placeholders. Configuration-bound zero-byte tensors and the
q3/q2 zero-element workspace view are validated against their live tensor and
storage contracts; every positive direct/restored allocation remains stable
and address-disjoint.

## 30-case result

- Evidence ID: `phase13pb-20260804t065806037759z-6fa70df3-869de9`
- Execution HEAD: `6fa70df3064d4953309f4edb9ae20c4f8cc227ef`
- Authorized container:
  `sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e`
- Matrix: 10 configurations × B={1,4,8} × L=17 = 30/30 PASS
- Cross-batch negative controls: 20/20 rejected before target mutation
- Direct/restored cache bytes, lifecycle, layout, allocation, output checksum,
  kernel path, pointer invariants, and CUDA Graph result: exact
- Runtime prefix/cache/pointer/Graph sharing: none
- Timing, Pilot, QC, fitting, profiling, Full Scan, Phase 14, and quality: not
  run

## Tests

- Focused unit, tamper, mismatch, and exact-scope tests in the authorized
  container: 35/35 PASS
- Exact source/container reconstruction and full 30-case CUDA Graph matrix:
  PASS
- Local append-only artifact inventory, checksum ledger, and COMPLETE-last:
  PASS
- Repository package-lock, full unit suite, and exact-path checks: PASS

## Durable publication

- Local root:
  `8fa941c7b35245955373f5e25e6895119a9f9c4e9c7f7737342fbc33017dc50b`
- Objects: 7
- R2 URI:
  `r2://kvbench-artifacts/kvbench/sha256/8fa941c7b35245955373f5e25e6895119a9f9c4e9c7f7737342fbc33017dc50b/`
- Conditional content-addressed publication / COMPLETE last: PASS / yes
- One clean retrieval; inventory, ledger, marker, root, and object set: PASS
- Private bucket and indefinite Bucket Lock: PASS
- Credential leakage: none; `.env` was not read, hashed, mounted, committed,
  or uploaded

## Preservation, blocker, and next action

All stopped Phase 13 campaigns remain unchanged and were not mounted or
analyzed. Adapters, CUDA, cache layouts, feasibility, grid, timing boundaries,
calibration, fixtures, and admission evidence are unchanged. Phase 13P-B has
no remaining technical blocker. A wholly new Phase 13 Pilot campaign may be
proposed separately; no stopped campaign may be resumed, and Pilot is not
started by this phase.
