# Phase 13B — compressed static-cache batch geometry

## Scope

Phase 13B extends only the admitted TurboQuant, KIVI, and KVQuant adapter/cache
geometry from `B=1` to the frozen set `B in {1, 4, 8}`.  Quantization,
packing, sparse selection, sinks, native `H_KV=8` GQA mapping, numerical
tolerances, runner timing boundaries, calibration, fixtures, CUDA source, and
the authorized Measurement Container remain unchanged.

The stopped Phase 13 campaign is immutable.  No Pilot point is resumed or
executed in this phase.

## Static layout

- TurboQuant uses batch-indexed packed block banks and a batch-aware block
  table while retaining native eight-KV-head slots.
- KIVI keeps its existing batch-leading native-HKV tensors and folds only
  `B*H_Q` / `B*H_KV` into the unchanged direct GEMV ABI.
- KVQuant adds batch banks to dense, per-token metadata, fixed-cap sparse, and
  count storage.  Existing caller-owned CUDA APIs are invoked on fixed
  per-batch views; the batch loop is fixed at adapter construction geometry.
- All dense, metadata, sparse, sink, staging, output, and workspace buffers are
  allocated before measured execution.  No cache grows.

## Validation and admission

For each of the nine compressed main configurations, validate `B=1,4,8` at a
bounded fixed-L correctness point in eager and CUDA Graph modes.  Require
finite numerical controls, exact B=1 oracle preservation, byte/accounting
agreement, native GQA/path checks, stable pointers, zero graph-replay
allocation, non-default-stream ordering, and focused Compute Sanitizer
coverage inside the Decision 0016 container.

Exact B=1 preservation is proved by the frozen method fixtures and unchanged
B=1 address/order contracts.  For `B=4,8`, identical input rows must produce
equal output rows and byte-identical persistent cache banks.  The full-model
B=1-to-batched logits delta is retained as a diagnostic only: changing the
cuBLAS GEMM row count can change BF16 rounding, so the frozen single-layer
cache tolerance is not misapplied to that cross-shape comparison.  No
numerical tolerance is changed.

Create append-only successor TurboQuant, KIVI, and KVQuant
MethodAdmissionReports bound to the new adapter/cache fingerprints.  Finalize
one Phase 13B evidence bundle, publish it to R2 with `COMPLETE` last, and verify
one clean retrieval.

## Deferred work

Phase 13 Pilot execution, Phase 14, Full Scan, profiling, performance
comparison, and quality evaluation are explicitly deferred.  A successful
Phase 13B requires a completely new Pilot campaign; the stopped campaign is
never resumed.
