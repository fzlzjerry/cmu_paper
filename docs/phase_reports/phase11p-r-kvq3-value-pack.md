# PHASE 11P-R REPORT

Status: PASS

## New source identities

- Corrected commit/tree:
  `0d9df350bd1788284e1ce76a8bf6e886beca5efa` /
  `a85cf7bf093982a4bf89c33d4e6794d9a85f846d`
- Aggregate patch SHA-256:
  `23a15db86790299392412c3ce2da7d971f4f073cfaf6839d82d3746c8b56b551`
- Extension SHA-256:
  `46c41aad8f56d58608d4c1273bd3a72fd36c8f69f9ca2c5a046f0c811631bf51`
- Changed source file: `deployment/kvquant/quant_cuda_kernel.cu`;
  `deq2[val][k]` became `deq2[val][off]`, with no barrier required because
  each thread now reads only its own initialized per-channel lane.
- Quantization mathematics, calibration, quantizers, sparse semantics, caps,
  sinks, GQA geometry, kvq4, kvq2, and Measurement Container: unchanged.

## Fixture root

- Fixture ID: `kvqref-2e0a0e9022c50cbc6fb497d88cae973e`
- Root:
  `c28682d58706b58812dc1db69ba5eb4982339ba13f39bf67f751794cdaabfdec`
- kvq4 and kvq2: byte-identical reuse of the immutable Phase 10 bundle.
- kvq3: exactly three regenerated fixtures; scalar store and append controls
  match exactly. No execution-history-dependent output remains.

## Tests

- All nine fixtures: PASS.
- Non-default CUDA stream: PASS.
- CUDA Graph selector to append to decode: capture/replay PASS; zero replay
  allocation; pointer and output stability PASS.
- Existing MHA and native 32Q/8KV GQA numerical tests: PASS.
- `make package-lock-check`, `make test`, and `make checks`: PASS.
- Phase 10 bundle and reports: unchanged.

## Sanitizer

- Compute Sanitizer 2025.3.1.0 memcheck: PASS, zero errors, zero leaked
  allocations, zero leaked bytes.
- Compute Sanitizer initcheck: PASS, zero uninitialized-memory errors.

## R2 result

- URI:
  `r2://kvbench-artifacts/kvbench/sha256/c28682d58706b58812dc1db69ba5eb4982339ba13f39bf67f751794cdaabfdec/`
- Conditional publication: PASS, 118 objects, `COMPLETE` last.
- Clean retrieval: PASS; inventory, checksum ledger, root, all objects, and
  unexpected-object check verified.
- Bucket Lock: PASS, exact indefinite rule
  `kvbench-evidence-indefinite`.

## Remaining blocker

None for Phase 11P-R. The KVQuant Measurement Adapter remains intentionally
unimplemented and G2-KVQ remains NOT EVALUATED.

## Next action

Phase 11 adapter work may be proposed as a separate task. It was not started.
