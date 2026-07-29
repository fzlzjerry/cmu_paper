# Decision 0025: Deterministic KVQuant kvq3 Value parallel pack

- Status: Proposed
- Date: 2026-07-29
- Parent source proposal: Decision 0024 commit
  `7fa389ecf5a5e198c76096d52fc2949dde844532`, tree
  `9ab495025a7f5c7fb86e929e7144a86fb6cf7024`
- Frozen calibration root:
  `8148306d08205af376994b022f189a0d6837915cd279ca8af6b104e1f4b46ccf`
- Immutable Phase 10 fixture root:
  `32cdf465a361dd6695b66ccbea0a462bddc075fd9778d0aa8cdaa3f94e6f63ab`
- Corrected fixture ID:
  `kvqref-2e0a0e9022c50cbc6fb497d88cae973e`

## Context

The legacy kvq3 parallel Value-store kernel initializes the per-token shared
LUT at `deq2[value][off]` but searches candidates through
`deq2[value][k]`. For the frozen 12-token store this reads uninitialized shared
columns 12 through 127, so the old kvq3 packed bytes depend on execution
history.

## Decision

1. Correct only that candidate lookup to `deq2[value][off]`, matching the
   existing kvq4, kvq2, and single-token kvq3 per-channel semantics.
2. Add no barrier: after the correction each thread reads only the shared lane
   it wrote, so no cross-thread shared state remains.
3. Quantizers, calibration, sparse selection, caps, sinks, GQA geometry,
   kvq4, kvq2, and the authorized Measurement Container remain unchanged.
4. Decision 0024's caller-owned APIs remain part of the derived source.
5. The immutable Phase 10 bundle and reports remain unchanged. A new
   nine-fixture bundle byte-reuses every kvq4 and kvq2 case member and
   regenerates only the three kvq3 cases.
6. Reused case manifests retain their original Phase 10 fixture ID and source
   hashes. The new bundle root records mixed per-family provenance explicitly;
   copied bytes must match the old bundle exactly.
7. Corrected kvq3 store and append payloads must equal an independent scalar
   nearest-code and bit-pack control exactly. Execution-history dependence
   fails closed.
8. This decision does not implement the KVQuant Measurement Adapter or
   authorize performance, profiling, Pilot, Full Scan, or quality work.

## Acceptance

This decision becomes Accepted only after the corrected source identity,
all-nine bundle validation, non-default-stream execution, CUDA Graph replay
with zero allocation, memcheck and initcheck, MHA/GQA regression, immutable R2
publication, and clean retrieval all pass inside the required authority
boundaries.
