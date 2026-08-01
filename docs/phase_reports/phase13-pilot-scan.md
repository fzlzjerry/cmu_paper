# Phase 13 — Pilot Scan

Status: BLOCKED

## Entry and design

- Starting Phase 12 PASS HEAD: `7379e808ff687b10bf18c56364ae1c545cd00fe4`
- Pilot execution HEAD: `009dfd7128a029bdbb941d7d55a4fcc017eff588`
- Phase 12 report and G0-G5: PASS
- Pilot at entry: READY
- Authorized container: `sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e`
- Configurations: `bf16`, `tq_4bit_nc`, `tq_k3v4_nc`, `tq_3bit_nc`, `k4v4`, `k2v4`, `k2v2`, `kvq4`, `kvq3`, `kvq2`
- Batches: 1, 4, 8
- Context labels: 4096, 8192, 16384, 24576, 32768, 49152, 65536, 98304, 131072
- Top label mapping: historical prefix 131071, total attended length 131072
- Runner/Graph: `fixed_l` / `cuda_graph`
- Warmup/measured steps/process replicates: 64 / 128 / 3
- Seeds: 20260801, 20260802, 20260803
- Planned point records: 810
- Precomputed memory-feasible records: 792
- Capacity-infeasible records: 18
- Selective reruns: 0

## Execution result

Campaign `phase13-20260801t080641686374z-009dfd71-14b7e3` launched the first preregistered feasible point: `tq_3bit_nc`, B=8, L=24576, replicate 0. The existing `TurboQuantStaticCache` rejected construction with `ValueError: TurboQuant cache requires frozen B=1 Llama GQA geometry`. The failure occurred before graph capture, warmup, or timing. It is a method-geometry boundary, not a memory-capacity result.

- Completed timing runs: 0
- Runtime failed: 1
- Aborted after the method bug: 791
- Capacity infeasible: 18
- Unstable points: 0 (no complete three-process point existed)
- Foreign-process failures: 0 observed
- Backend fallbacks: 0 observed
- Maximum CV: not estimable
- Output/kernel/allocation agreement: not evaluated for Pilot points
- NaN/Inf: not evaluated for Pilot points

The failed run is `phase13-20260801t080641686374z-009dfd71-14b7e3-r0-o000-tq_3bit_nc-b8-l24576`. Every planned point has an immutable machine-readable record. No individual point was retried, and no preregistered point was removed.

## Provisional analysis

- Constant, linear, and knee-aware records: emitted for all 30 configuration/batch groups
- Fit status: 30 `insufficient_feasible_span`
- Valid/below-range/above-range/no-positive-slope/unstable/failed knees: 0
- Knee-density sufficient: 0
- Densification: not proposed while the B>1 method boundary remains unresolved
- Pilot-only same-work ratio records: 243
- Ratios calculated: 0
- Quality status: unvalidated
- Performance claim eligibility: false
- `r_hbm`: null

No timing, speedup, HBM, capacity-benefit, quality, or final-knee claim is made.

## Outputs and custody

- Local campaign: `artifacts/phase13/phase13-20260801t080641686374z-009dfd71-14b7e3/`
- Local root: `be8680d3d94dba35d58a98ac13aa5ae3aa2ba47e767c301418b129060466babc`
- Objects: 1653
- Local size: 3,215,501 bytes
- QC JSON SHA-256: `1c30c0b98296caa2a4ba352cea0c98fc8f4f894ad2b12f20a362ee5c2c4a94d7`
- Provisional knee Parquet SHA-256: `7af36115e957b92d9215475130fc798424bcf31bbcd8dc02e31826cce9cf1058`
- Checksum ledger SHA-256: `baec9e845ff8310e6ef27bf3bfc983c816a08353ca61e8efcb263602c14b541c`
- R2 URI: `r2://kvbench-artifacts/kvbench/sha256/be8680d3d94dba35d58a98ac13aa5ae3aa2ba47e767c301418b129060466babc/`
- Publication: PASS, 1653/1653 objects uploaded conditionally
- COMPLETE uploaded last: yes
- Clean retrieval: PASS, exactly once
- Bucket Lock: exact prefix, private bucket, indefinite retention, PASS
- Credential leakage: none; `.env` was not read, hashed, committed, mounted, or uploaded

Adapters, CUDA, cache layouts, calibration, fixtures, admission evidence, timing boundaries, and historical evidence were not changed. The two earlier launch-only reservations stopped before any GPU worker because the temporary clone lacked a read-only nested mountpoint; they remain local staging evidence and were not used in this campaign.

## Phase decision

- Phase 13: BLOCKED
- Phase 14 readiness: NOT READY
- Full Scan: CLOSED
- Quality execution: LOCKED
- `PERFORMANCE_DATA_FROZEN`: absent

The minimum remaining blocker is a separately authorized method phase that either adds admitted static B=4/B=8 cache geometry to every compressed method in the fixed Pilot grid, or revises the preregistered Pilot batch contract before a completely new campaign. Phase 14 was not started.
