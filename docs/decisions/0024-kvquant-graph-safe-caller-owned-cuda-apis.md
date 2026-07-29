# Decision 0024: KVQuant graph-safe caller-owned CUDA APIs

- Status: Proposed
- Date: 2026-07-29
- Parent authority: Decision 0021 patched commit
  `4ad80bc8c942d0a05516d2be8f8d443a77a05900`, tree
  `c4f1490c9c0c4ec46099f1e95c092516df2adb4e`
- Reference oracle: Phase 10 fixture root
  `32cdf465a361dd6695b66ccbea0a462bddc075fd9778d0aa8cdaa3f94e6f63ab`
- Supersedes: none

## Context

The Phase 11 entry audit found three source-level blockers in the otherwise
frozen Decision 0021 deployment path: sparse selection allocated dynamically
and synchronized through Python, single-token Value packing transported
extrema through host floats, and Measurement-path append/pack launches used the
legacy default CUDA stream.

## Decision

1. Add only caller-owned, fixed-shape CUDA/pybind APIs needed to remove those
   blockers.
2. Quantization mathematics, calibration tensors, quantizers, bit packing,
   sparse capacity 12, Key threshold semantics, Value six-lowest-plus-six-
   highest semantics, stable value/native-index ordering, sink behavior, and
   numerical tolerances do not change.
3. The new selector writes float32 sparse values, int32 sparse indices, int32
   active counts, and required dense bounds into caller-owned tensors. It
   performs no dynamically sized output construction or tensor-to-host
   conversion.
4. The Measurement Value append path consumes extrema and zero-point as CUDA
   tensors and writes dense, metadata, sparse, and count state into
   caller-owned buffers.
5. Only the new Measurement append, parallel-pack, selector, and sparse-output
   launches use the current PyTorch CUDA stream with launch-error checks.
   Existing Reference Lane APIs remain unchanged.
6. The nine immutable Phase 10 fixtures remain the numerical oracle. No
   calibration artifact or Phase 10 fixture is regenerated or modified.
7. This decision authorizes a source API compatibility patch only. It does not
   implement the KVQuant Measurement Adapter, admit G2-KVQ, or start Phase 12.
8. The derived commit/tree, delta and aggregate patch digests, changed-file
   hashes, and extension digest become accepted authority only after exact
   fixture conformance, allocation, non-default-stream, CUDA Graph, and Compute
   Sanitizer validation inside the authorized Measurement Container.

## Consequences

Decision 0021 and Decision 0023 remain unchanged. A later, separately
authorized Phase 11 task may consume these APIs only after this decision
becomes Accepted.

## Phase 11P validation outcome

The proposed local source is commit
`7fa389ecf5a5e198c76096d52fc2949dde844532`, tree
`9ab495025a7f5c7fb86e929e7144a86fb6cf7024`. Its Decision-0021-relative delta
SHA-256 is
`36a78cee6a14654352f36aa83efdbd69bc82666bc60540378d4165be03777536`;
the aggregate upstream-base-relative patch SHA-256 is
`b4b9d172437fe76edf6450afd6befda5400bdc92e64308ecdbbffed400104b52`.
The SM120 extension SHA-256 is
`a9b1eacdb7fc584fb04bdfa201763e6fc2947273b14638140f4202cb8f7605b2`.

Decision 0024 remains Proposed. The frozen 3-bit parallel Value store kernel
initializes only the 12 active shared-memory columns but reads all 128 columns.
The Phase 10 kvq3 packed bytes therefore depend on uninitialized shared memory.
Repairing that defect changes the immutable kvq3 oracle; reproducing it violates
the deterministic graph-safe and sanitizer-zero requirements. No proposed
source identity is accepted while those requirements conflict.
