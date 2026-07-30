# PHASE 11D-Q23 REPORT

Status: PASS

## Source identities

- Source identifier: `kvquant_gqa_longctx_deterministic_q23_v4`
- Parent Decision 0027 commit/tree:
  `4b8533b29b04f8c4bf55f688a41fefe20487637b` /
  `46f2149a0369d5c97d9a6bc77d57b5f3a5a5fb3b`
- New commit/tree:
  `34b0bdfa83082e1f30387d9ac5cca369006e089c` /
  `1f85af65fe03061583ffe8bd91e47d7ecffdd312`
- Aggregate patch SHA-256:
  `7b9d3cc6773e8ef37697601c885f2c5ec581dffd57cf59424d03e68f147bd55a`
- Parent-relative delta SHA-256:
  `a3dc04b35371662603aaa68e00ba8bfa04f264dd63177c7c6c123e66ea08e736`
- Changed source files:
  `deployment/kvquant/quant_cuda.cpp`,
  `deployment/kvquant/quant_cuda_kernel.cu`
- Deterministically stripped extension SHA-256:
  `b3c33badb8e55b19d6b2ce535182e964ce51e5102d8413b29701dd3d817ad73d`

Decision 0029 adds Measurement-only q3/q2 caller-owned deterministic Value
decode APIs. The Adapter binds both APIs to a preallocated strided view of its
existing FP32 storage, adding zero owned bytes. Quantization, packing,
metadata, sparse behavior, calibration, sinks, caps, q4, and native 32Q/8KV
GQA semantics are unchanged.

## Determinism results

- `kvq3`, width 4092: 100/100 executions produced exact SHA-256
  `5d8e11ec50de71b04a072fdec5b0863ab9e607122edd10006d6b5a098bdcff02`.
  Independent control PASS; maximum absolute difference
  `0.000396728515625`.
- `kvq2`, width 4092: 100/100 executions produced exact SHA-256
  `4a62b1e203b1861ba276855d6ed5fa5e4b35a365f266584d92b7bc3bfa69752a`.
  Independent control PASS; maximum absolute difference `0.00048828125`.
- Frozen tolerance: `atol=rtol=0.01`.
- Execution-history-dependent output: absent.

## Fixture preservation

- Numerical-oracle root:
  `c28682d58706b58812dc1db69ba5eb4982339ba13f39bf67f751794cdaabfdec`
- All nine Phase 11P-R fixtures: PASS.
- Payload, metadata, sparse state, sink state, store state, and append state:
  exact.
- Decode agreement: PASS under the frozen tolerance.
- Calibration and fixture files changed: no.
- q4 deterministic/stream/Graph regressions: PASS.
- Existing CUDA MHA/GQA tests: 4/4 PASS.
- Checksum-bound frozen-source MHA/GQA helper control: PASS.

## Stream, Graph, allocation, and sanitizer

- Current/non-default PyTorch CUDA stream: PASS.
- CUDA Graph capture/replay and eager/Graph agreement: PASS.
- Caller-owned pointers: stable.
- Eager and replay allocation events/bytes: zero.
- Eager and replay allocated/reserved deltas: zero.
- SM120 cubin and compute_120 PTX: PASS.
- Compute Sanitizer 2025.3.1.0 memcheck: PASS, zero errors and leaks.
- Compute Sanitizer initcheck: PASS, zero errors and uninitialized reads.
- Authorized Measurement Container changed: no.
- Phase 12 campaign/admission grid run: no.
- Performance or quality work: no.
- Append-only evidence root:
  `8b65112ea2d49b58ee07c1533b429fac1a8af7466e09adad073d9a22ae2ec790`

## Remaining blocker

None for Phase 11D-Q23.

## Next action

Propose the complete Phase 12 30-run unified-admission campaign as a separate
task. Phase 12 was not begun automatically.
