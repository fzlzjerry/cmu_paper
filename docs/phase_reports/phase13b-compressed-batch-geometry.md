# PHASE 13B REPORT

Status: PASS

## Scope and authority

- Decision 0030 authorizes only static compressed-cache geometry for
  `B in {1,4,8}` in the exact Decision 0016 Measurement Container.
- TurboQuant `tq_4bit_nc`, `tq_k3v4_nc`, `tq_3bit_nc`; KIVI `k4v4`,
  `k2v4`, `k2v2`; and KVQuant `kvq4`, `kvq3`, `kvq2` are admitted at all
  three batch sizes.
- CUDA source, quantization, packing, sparse semantics, caps, sinks,
  calibration, fixtures, native 32Q/8KV GQA, runners, tolerances, and timing
  boundaries are unchanged.  No timing was collected.

## Implementation and preservation

- TurboQuant uses fixed per-batch packed banks and a batch-aware block table.
- KIVI folds the fixed batch dimension into the unchanged head-major GEMV ABI.
- KVQuant invokes the unchanged caller-owned APIs on fixed per-batch cache
  banks.  Dense, metadata, sparse, sink, staging, output, and workspace
  storage is preallocated.
- Frozen B=1 fixtures, address/order contracts, and numerical outputs remain
  unchanged.  The stopped Phase 13 campaign and every historical admission
  bundle/report remain immutable.

## Validation

- The authorized-container matrix passed 27/27 points at fixed-L 128:
  nine configurations crossed with B=1, B=4, and B=8.
- Numerical controls and finite-output checks passed.  Identical batched rows
  produced equal outputs and isolated cache banks.
- Native eight-KV-head GQA, direct execution paths, no fallback, no complete-
  prefix materialization, stable pointers, and byte/allocation accounting
  passed.  Eager execution retained the exact admitted common outer-control
  event topology with zero persistent allocated/reserved delta; graph replay
  recorded zero allocation events and no cache growth.
- Fixed-L eager/Graph agreement, repeated Graph replay, non-default-stream
  ordering, and historical-prefix preservation passed for every point.
- Formal container suites passed: `test-cuda` 26/26 and `test-graph` 7/7.
- Compute Sanitizer passed nine B=8 memcheck cases plus three distinct
  initcheck cases with zero errors and zero leaked bytes.
- Current TurboQuant, KIVI, and checksum-bound Decision 0029 KVQuant admission
  validations pass.  The historical Q23 wrapper is validated against its
  exact execution Git blob, not current source bytes.

## Successor admission reports

- TurboQuant: `49799ef89646ec008a530c5180fdcef6cd4af9ca0d5772fe2b01d6e775e3b1c0`
- KIVI: `1e91730ac56af37e03d80edce7979a509d52049428faad89f61e61dc6bd48c51`
- KVQuant: `e1cee8e1c514f9cf6323b5e710480c1fefab2804e5f4eafe6c473b29f4768481`

## Durable publication

- Local bundle:
  `artifacts/phase13b/phase13b-20260801t143138050263z-b862af64-batch-admission`
- Local checksum-ledger SHA-256:
  `456cbc1d23a6cc94934b960c2ed30554aeb84faa5fde267defc899a1d09c38a2`
- R2 root:
  `f1c96eaacbbace1c23b249d1afe8d892aa26c3f6b8d04e07f373a2becafba1fe`
- URI:
  `r2://kvbench-artifacts/kvbench/sha256/f1c96eaacbbace1c23b249d1afe8d892aa26c3f6b8d04e07f373a2becafba1fe/`
- 52 objects were published through conditional content-addressed writes with
  `COMPLETE` last.  One retrieval into a new empty directory passed inventory,
  ledger, root, unexpected-object, and indefinite Bucket Lock checks.  No
  credential value entered Git, an artifact, or the Measurement Container.

## Changed files

- Adapter/cache geometry: TurboQuant, KIVI, and KVQuant adapter/cache/session
  files only; no CUDA source.
- Admission support: Phase 13B schema/coordinator, exact historical validators,
  artifact lifecycle support, and exact-path scope validation.
- Evidence and governance: Decision 0030, plan, focused unit/CUDA/Graph/
  sanitizer tests, 27-point evidence, three successor reports, publication
  receipt, and minimal status/task/risk/blocker updates.

## Scientific interpretation

The nine compressed configurations support preallocated native-HKV static
cache geometry at B=1, B=4, and B=8 while preserving their frozen B=1
numerical behavior and passing the bounded correctness, path, allocation,
Graph, stream, and sanitizer controls.  This phase makes no performance,
speedup, HBM, capacity, knee, or quality claim.

## Remaining blocker and next action

No Phase 13B technical blocker remains.  A completely new, preregistered
Phase 13 Pilot campaign may be proposed separately; the stopped campaign must
never be resumed.  Phase 14, Full Scan, profiling, and quality remain closed.
