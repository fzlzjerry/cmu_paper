# PHASE 13F REPORT

Status: PASS

## Root cause

The former feasibility formula counted model weights, adapter-owned cache, and
a scaled Graph reserve, but omitted the end-to-end prefix-construction peak.
Consequently, `tq_3bit_nc` B=8/L=98304 was called feasible even though prefix
attention/output projection required an additional 6.00 GiB and failed before
Graph capture, warmup, or timing.

## Formula correction

Decision 0031 freezes an additive high-water calculation containing exact model
weights, cache and persistent workspace, endpoint workspace, prefix control
tensors, the larger of the source-derived attention/output-projection and MLP
peaks, and the net Graph pool/capture reserve. The GPU limit remains
`floor(0.88 * 101970345984) = 89733904465` bytes. No grid, method, CUDA,
layout, timing, calibration, fixture, or admission semantics changed.

## 810-record classification

- Evidence ID: `phase13f-20260802t035116023475z-43e0065e-32d26f`
- Source HEAD: `43e00659340b38f77708c8d2d37c82a2cb7035a0`
- Authorized container:
  `sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e`
- Planned records: 810/810
- Feasible: 684 records / 228 unique points
- Capacity-infeasible: 126 records / 42 unique points
- Replicate classifications: deterministic
- Former point: `tq_3bit_nc` B=8/L=98304 is source-faithfully
  `capacity_infeasible`; corrected required bytes 150255517718 versus limit
  89733904465. Its three replicate records agree.

## Tests

- Focused boundary, underestimation, deterministic-recomputation,
  fingerprint/order tamper, source/evidence tamper, custody, and exact
  810-record tests: PASS
- Exact authorized-container identity and focused diagnostic tests: PASS
- Local immutable artifact, inventory, ledger, and COMPLETE-last validation:
  PASS
- Pilot, Phase 14, Full Scan, profiling, performance, and quality execution:
  not run

## Durable publication

- Local root:
  `5090b193c046637cb3836f7d5a3ee5ebbad95d9a459672ab4aa3ff0ddb756589`
- Objects: 9
- R2 URI:
  `r2://kvbench-artifacts/kvbench/sha256/5090b193c046637cb3836f7d5a3ee5ebbad95d9a459672ab4aa3ff0ddb756589/`
- Conditional publication / COMPLETE last: PASS / yes
- One new-empty-directory clean retrieval: PASS
- Inventory, checksum ledger, root, object set, private bucket, and indefinite
  Bucket Lock: PASS
- Credential leakage: none; `.env` was not read, hashed, mounted, committed,
  or uploaded

## Preservation and next action

Both stopped Phase 13 campaigns remain byte- and checksum-unchanged at roots
`be8680d3d94dba35d58a98ac13aa5ae3aa2ba47e767c301418b129060466babc`
and
`94104865452017fbfd3c87fffd34e82b12248e379ef2680b22153dd5b71d90b8`.
Phase 13F has no remaining technical blocker. Phase 13 itself remains
incomplete: propose a wholly new preregistered Pilot campaign with a new ID;
never resume either stopped campaign. Phase 14 remains not ready, Full Scan
remains closed, and quality remains locked.
