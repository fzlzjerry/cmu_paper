# PHASE 13R REPORT

Status: BLOCKED

## Entry and design

- Starting HEAD: `3dcd075d987db2452793408f8dd2b8f97f87530b`
- Pilot execution HEAD: `e886592fd83de9e037cf154d1512c53192bfb0aa`
- Working tree at launch: clean
- Phase 12 unified admission and all Phase 13B successor admissions: PASS
- Authorized container: `sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e`
- Configurations: `bf16`, `tq_4bit_nc`, `tq_k3v4_nc`, `tq_3bit_nc`, `k4v4`, `k2v4`, `k2v2`, `kvq4`, `kvq3`, `kvq2`
- Batches: 1, 4, 8
- Context labels: 4096, 8192, 16384, 24576, 32768, 49152, 65536, 98304, 131072
- Top label mapping: historical prefix 131071, total attended length 131072
- Runner/Graph: `fixed_l` / `cuda_graph`
- Warmup/measured steps/process replicates: 64 / 128 / 3
- Seeds: 20260801, 20260802, 20260803
- Planned point records: 810
- Precomputed feasible GPU run records: 792
- Capacity-infeasible records: 18 across 6 points
- Selective reruns: 0

## Execution and QC

Fresh campaign `phase13-20260801t184243094922z-e886592f-99b6ca` completed 14
timing runs before the next preregistered point, `tq_3bit_nc`, B=8, L=98304,
replicate 0, failed while constructing the prefix.  The model endpoint attempted
an additional 6.00 GiB allocation with 5.27 GiB free and raised
`torch.OutOfMemoryError` in the attention output projection.  This occurred
before Graph capture, warmup, or timing.  The point had been precomputed as
feasible, so it is a concrete end-to-end peak-memory feasibility blocker, not a
capacity-infeasible result and not a timing observation.

- Completed: 14
- Runtime failed: 1
- Aborted after fail-closed stop: 777
- Capacity infeasible: 18
- Unstable points: 0; no point obtained three independent processes
- Maximum CV: not estimable
- Output mismatches, fallback, allocation drift, kernel-path drift, NaN/Inf,
  and GPU-exclusivity failures: 0 in the available QC records
- All 810 planned records are preserved; no point was selectively rerun or
  reclassified

Only `tq_3bit_nc` reached execution: 14 records completed, one failed, and 66
were aborted.  The remaining method records were aborted after the fail-closed
stop or retained as preregistered capacity-infeasible records.

## Provisional analysis

- Constant, linear, and knee-aware records: 30
- Fit status: 30 `insufficient_feasible_span`
- Valid provisional knees: 0
- Knee-density assessments: 0
- Pilot-only ratio records: 243; calculated: 0
- Quality status: unvalidated
- Performance claim eligibility: false
- `r_hbm`: null

No performance, speedup, HBM, capacity-benefit, quality, or final-knee claim is
made from the 14 incomplete Pilot runs.

## Durable custody

- Local campaign: `artifacts/phase13/phase13-20260801t184243094922z-e886592f-99b6ca/`
- Local root: `94104865452017fbfd3c87fffd34e82b12248e379ef2680b22153dd5b71d90b8`
- Objects: 1807
- Local size: 64,415,750 bytes
- Pilot QC SHA-256: `09a1fdd135bd15a17c5897761971f4260152674b1354b38993369329a7ec4c8c`
- Provisional-knee Parquet SHA-256: `fe14b9f128bff3fd13d175e2174fa5336faa91e825c61d715a2e4a235e5705b3`
- Checksum-ledger SHA-256: `b7e533e367612298f50bf2ba24ea0017bff25dbcdb838c1bb2901d21283fa4ae`
- R2 URI: `r2://kvbench-artifacts/kvbench/sha256/94104865452017fbfd3c87fffd34e82b12248e379ef2680b22153dd5b71d90b8/`
- Publication: PASS, 1807/1807 objects uploaded conditionally
- COMPLETE uploaded last: yes
- Clean retrieval: PASS; inventory, ledger, marker, root, and object set valid
- Bucket Lock: exact private prefix, indefinite retention, PASS
- Credential leakage: none; `.env` was not read, hashed, committed, mounted, or uploaded

The old stopped Phase 13 campaign, Phase 13B admissions, adapters, CUDA,
calibration, fixtures, timing boundaries, and historical evidence remain
unchanged.

## Phase decision

- G0-G5: remain PASS from unified admission
- Phase 13R: BLOCKED
- Phase 14 readiness: NOT READY
- Full Scan: CLOSED
- Quality execution: LOCKED
- `PERFORMANCE_DATA_FROZEN`: absent

The minimum remaining blocker is a separately authorized correction to the
Pilot feasibility contract that accounts for end-to-end prefix-construction
peak memory at the failed TurboQuant B=8/L=98304 point.  This campaign must
remain immutable and must never be resumed or selectively repaired.  Phase 14
was not started.
