# PHASE 11D REPORT

Status: PASS

## Source identities

- Source identifier: `kvquant_gqa_longctx_deterministic_v3`
- New commit/tree:
  `4b8533b29b04f8c4bf55f688a41fefe20487637b` /
  `46f2149a0369d5c97d9a6bc77d57b5f3a5a5fb3b`
- Aggregate patch SHA-256:
  `bae63bced549479709b10d7f6a8ee35a8f21ec18cc040a7424591cee47c1b0a6`
- Parent-relative delta SHA-256:
  `d12db702ec6d625330e791f4556af1dd368bfff7f00a45d5817eabcff4298ce9`
- Deterministically stripped extension SHA-256:
  `a79644923ba131e56abe95029e669346dbbb11fd210d2b9f8b2086819ffeaad1`

## Changed source files

- `deployment/kvquant/quant_cuda.cpp`
- `deployment/kvquant/quant_cuda_kernel.cu`

The new Measurement-only q4 Value-decode API uses caller-owned float32
workspace `[1, 32, 32, 128]` and a fixed token/slot/tile reduction order.
Potentially read q4/q3/q2 tail shared-memory lanes are initialized. No
quantization, calibration, packing, metadata, sparse, cap, sink, or GQA
semantic changed.

## Determinism results

- Width 4092 dense-only: 100/100 executions produced exact SHA-256
  `3e90e30349cf53de0b711b3a345e6169befbd10166c134cfde497b7295082d2e`.
- Width 4092 dense+sparse: 100/100 executions produced exact SHA-256
  `bc2c31cdb5dc3293110311c928b09dc61aa0ae46c9cc3be25ea0ac75340e5fb5`.
- Independent deterministic controls: PASS; maximum absolute difference
  `0.000244140625`, maximum relative difference
  `0.000003451496240813867`, frozen `atol=rtol=0.01`.
- Execution-history-dependent output: absent.

## Fixture preservation

- Current oracle root:
  `c28682d58706b58812dc1db69ba5eb4982339ba13f39bf67f751794cdaabfdec`
- All nine Phase 11P-R fixtures: PASS and unchanged.
- Historical Phase 10 root, calibration, Adapter, existing methods, and
  Measurement Container: unchanged.

## Stream, Graph, allocation, and sanitizer

- Non-default PyTorch CUDA stream: PASS.
- CUDA Graph capture/replay and eager/graph output agreement: PASS.
- Caller-owned pointers: stable.
- Eager and graph replay allocation events/bytes and persistent
  allocated/reserved deltas: zero.
- SM120 cubin and compute_120 PTX: PASS.
- Compute Sanitizer 2025.3.1.0 memcheck: PASS, zero errors and leaks.
- Compute Sanitizer initcheck: PASS, zero errors and uninitialized reads.
- Append-only evidence root:
  `7dd03673e74d12fc5218416b162994d007bcaf31aeb53774094381fa351e1007`

## Remaining blocker

None for Phase 11D. The Phase 11 Adapter remains deliberately unchanged and
is not yet bound to this new API/workspace; G2-KVQ remains NOT EVALUATED.

## Next action

Propose a separate full Phase 11 task to bind the Adapter to Decision 0027 and
rerun the nine-point admission grid. Phase 12 was not started.
