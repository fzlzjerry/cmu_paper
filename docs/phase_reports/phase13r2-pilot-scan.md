# PHASE 13R2 REPORT

Status: BLOCKED

## Entry and campaign

- Starting HEAD: bdfcb5ba41f150b1f56ddc8ac3515cd5a0b7bde3
- Pilot execution HEAD: 3127f1d1b358e0d884b146f6fab2a397f6ca8c31
- Working tree at launch: clean
- Phase 12 unified admission, Phase 13B successor admissions, Decision 0031,
  and Phase 13F evidence: PASS
- Authorized container:
  sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e
- Campaign ID: phase13-20260802t045837322693z-3127f1d1-486dcb
- Planned / feasible / capacity-infeasible records: 810 / 684 / 126
- Decision 0031 target tq_3bit_nc B=8/L=98304:
  capacity_infeasible, not launched
- Grid, seeds, order, timing boundaries, and selective-rerun count: unchanged /
  unchanged / unchanged / unchanged / 0

## Execution, QC, and provisional analysis

- Completed: 23
- Runtime failed: 1
- Aborted after fail-closed stop: 660
- Capacity infeasible: 126
- Unstable points: 0
- Maximum CV: not estimable; no point obtained three process medians
- Fits: 30 records, all insufficient_feasible_span
- Valid knees / knee-density assessments: 0 / 0
- Pilot-only ratio records / calculated: 243 / 0
- Output mismatch, backend fallback, allocation drift, kernel-path drift,
  NaN/Inf, GPU-exclusivity failure: 0 in available QC records
- Quality status: unvalidated
- Performance claim eligibility: false
- r_hbm: null

The next preregistered point was kvq2, B=1, label L=131072 (historical prefix
131071), replicate 0. Decision 0031 correctly classified it feasible:
predicted required bytes were 34,800,441,952 versus the frozen 89,733,904,465
byte limit. The worker did not finalize within the frozen 7,200-second
supervision deadline. The supervisor requested SIGTERM, recorded return code
-15 and supervised_worker_failed, and the campaign stopped fail-closed. This is
not an OOM reclassification and produces no timing observation.

## Durable custody and preservation

- Local campaign:
  artifacts/phase13/phase13-20260802t045837322693z-3127f1d1-486dcb/
- Local root:
  581b02a6ca1a09c976a899b2b5d7eeb7897c0ad8f7ed8ad9fb11be5f6475f327
- Objects / local size: 1906 / 104,192,511 bytes
- Pilot QC SHA-256:
  453fdcfecf11622e1d1d49b342d9ef2433eb4e1dd92e47c52e7ab4172b73b88a
- Provisional-knee Parquet SHA-256:
  fe14b9f128bff3fd13d175e2174fa5336faa91e825c61d715a2e4a235e5705b3
- Checksum-ledger SHA-256:
  9a25a98cc9083ba91ea5f14fc9bbcb0d7359aae3ac0d2843e9cd1b48f30c5dc9
- R2 URI:
  r2://kvbench-artifacts/kvbench/sha256/581b02a6ca1a09c976a899b2b5d7eeb7897c0ad8f7ed8ad9fb11be5f6475f327/
- Publication / COMPLETE-last / clean retrieval: PASS / yes / PASS
- Clean retrieval: 1906 objects; inventory, ledger, marker, root, and object set
  valid; no unexpected object
- Bucket Lock: private exact prefix, indefinite retention, PASS
- Credential leakage: none; .env was not read, hashed, committed, mounted, or
  uploaded

Both earlier stopped Phase 13 campaigns, Phase 13F evidence, adapters, CUDA,
runners, cache layouts, calibration, fixtures, admission evidence, timing
boundaries, and historical evidence remain unchanged.

## Phase decision and next action

- G0-G5: remain PASS from unified admission
- Phase 13R2: BLOCKED
- Phase 14 readiness: NOT READY
- Full Scan: CLOSED
- Quality execution: LOCKED
- PERFORMANCE_DATA_FROZEN: absent

The minimum remaining blocker is the frozen 7,200-second worker-supervision
deadline for a Decision 0031-feasible top-context point. Correct and
preregister that timeout contract in a separate remediation task, preserve
this campaign unchanged, then start another wholly new Pilot campaign ID.
Do not resume this campaign or begin Phase 14.
