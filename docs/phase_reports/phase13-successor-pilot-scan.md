# PHASE 13 SUCCESSOR PILOT REPORT

Status: PASS

## Execution and preservation

- Starting and execution HEAD: `4ddd7b1784de0327a8ed03508b1769f35e2cb8da`
- Campaign: `phase13-20260822t150835736582z-4ddd7b17-3a8fb3`
- Authorized container: `sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e`
- Frozen seeds: `20260805`, `20260806`, `20260807`
- Planned / feasible / capacity-infeasible records: 810 / 684 / 126
- Completed / failed / aborted feasible runs: 684 / 0 / 0
- Selective reruns: 0
- The immutable blocked campaign
  `phase13-20260804t111810342595z-a127b0d1-8649c3` remains unchanged at
  69 completed, one failed, 614 aborted, 126 capacity-infeasible, and 3,246
  objects. No timing sample from it was reused.

## QC and coverage

- Stable / unstable points: 228 / 0
- Maximum three-process CV: `0.002218507701406616` (0.2218507701%)
- Output mismatches, kernel-path drift, allocation drift, NaN/Inf, backend
  fallback, GPU-exclusivity failures: 0
- BF16 feasible / capacity-infeasible records: 63 / 18
- Each of the other nine configurations: 69 / 12
- Pilot-only same-work ratio records / calculated: 243 / 189
- Ratio labels remain `pilot_only=true`, `quality_status=unvalidated`, and
  `performance_claim_eligible=false`; `r_hbm` remains null.

## Provisional fits

All 30 configuration-by-batch records fit the preregistered constant, linear,
and knee-aware Pilot models with status `knee_observed`; all 30 session
bootstrap intervals were estimable. The discrete provisional knees are:

- `L*=40960`, interval `[40960,40960]`: `kvq4` B=1/4, `kvq3` B=1/4, and
  `kvq2` B=1. These five have sufficient below/near/above density.
- `L*=6144`, interval `[6144,6144]`: `k4v4`, `k2v4`, and `k2v2` at B=1.
- `L*=4096`, interval `[4096,4096]`: the remaining 22 records.

Knee density is insufficient for 25/30 records, so a separately
preregistered densification campaign is required before Phase 14. These are
Pilot-only provisional fits, not final knee or performance claims.

## Durable publication and decision

- Local root: `feb2e5a8ebba8b729c182fc8170107c9acf8128edd3e5618c8f1b90530557531`
- R2 URI: `r2://kvbench-artifacts/kvbench/sha256/feb2e5a8ebba8b729c182fc8170107c9acf8128edd3e5618c8f1b90530557531/`
- Objects / bytes: 17,384 / 12,326,995,814
- Conditional content-addressed publication: PASS
- COMPLETE uploaded last: yes
- One clean empty-directory retrieval: PASS
- Inventory, checksum ledger, COMPLETE marker, root, and object set: PASS
- Bucket Lock: exact private prefix, indefinite retention, PASS
- Credential values recorded, mounted, or uploaded: no

Phase 13 successor Pilot passes its execution, QC, custody, and publication
contract. Phase 14 is not started and remains gated on the separately
preregistered knee-densification work. Full Scan remains CLOSED, quality
execution remains LOCKED, and `PERFORMANCE_DATA_FROZEN` remains absent. No
final speedup, HBM-traffic, capacity-benefit, quality, or knee claim is made.
