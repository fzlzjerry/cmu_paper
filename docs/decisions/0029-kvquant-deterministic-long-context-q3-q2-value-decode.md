# Decision 0029: KVQuant deterministic q3/q2 long-context Value decode

- Status: Accepted
- Date: 2026-07-30
- Scope: Phase 11D-Q23 only
- Parent source authority: Decision 0027

## Context

Phase 12 stopped before its G5 campaign because the admitted `kvq3` and
`kvq2` Value-decode paths still reduce 128-token tile contributions into one
output with inter-block floating-point `atomicAdd`.  At Value width 4092 this
creates 32 independently scheduled contributors, so the existing evidence does
not establish exact process-reproducible output.

## Decision

Add one Measurement-only caller-owned out API for each of `kvq3` and `kvq2`.
Each API writes per-tile FP32 partials into preallocated caller-owned storage
and then reduces tiles in ascending order with one writer per output element.
Within each tile, tokens and sparse slots are accumulated in ascending order.

The new APIs:

- preserve the canonical q3/q2 packed bitstream and frozen LUT metadata;
- preserve fixed sparse capacity 12, sparse ordering, sinks, caps, and native
  32-query-head/8-KV-head GQA mapping;
- use the current PyTorch CUDA stream and checked launches;
- perform no floating-point output atomics, allocation, host synchronization,
  tensor-to-host conversion, CPU reduction, GQA expansion, or fallback.

The Measurement Adapter must bind `kvq3` and `kvq2` directly to these APIs.
It reuses caller-owned FP32 storage already present in the static cache, so the
physical cache allocation and byte-accounting formulas do not change.

Quantization mathematics, calibration, quantizers, packing, metadata, sparse
selection, Phase 11P-R fixtures, q4 behavior, Reference APIs, BF16,
TurboQuant, KIVI, and the authorized Measurement Container are unchanged.
The nine Phase 11P-R fixtures remain the numerical oracle.

## Checksum-bound source authority

- Parent Decision 0027 commit:
  `4b8533b29b04f8c4bf55f688a41fefe20487637b`
- Parent Decision 0027 tree:
  `46f2149a0369d5c97d9a6bc77d57b5f3a5a5fb3b`
- Corrected commit: `34b0bdfa83082e1f30387d9ac5cca369006e089c`
- Corrected tree: `1f85af65fe03061583ffe8bd91e47d7ecffdd312`
- Aggregate patch SHA-256:
  `7b9d3cc6773e8ef37697601c885f2c5ec581dffd57cf59424d03e68f147bd55a`
- Decision-0027-parent-relative delta SHA-256:
  `a3dc04b35371662603aaa68e00ba8bfa04f264dd63177c7c6c123e66ea08e736`

## Evidence required for acceptance

This decision may become Accepted only after the checksum-bound patch
reconstructs exactly and the authorized Measurement Container proves, for both
q3 and q2 at Value width 4092:

- 100 identical executions produce one exact output SHA-256;
- agreement with the frozen independent deterministic numerical control;
- non-default-stream ordering;
- CUDA Graph capture/replay, stable pointers, and zero replay allocation;
- Compute Sanitizer memcheck and initcheck with zero errors or leaks;
- unchanged nine-fixture conformance and passing q4/MHA/GQA regressions.

Phase 12 execution is explicitly deferred.

## Accepted identities and evidence

- Corrected source commit:
  `34b0bdfa83082e1f30387d9ac5cca369006e089c`
- Corrected source tree:
  `1f85af65fe03061583ffe8bd91e47d7ecffdd312`
- Aggregate patch SHA-256:
  `7b9d3cc6773e8ef37697601c885f2c5ec581dffd57cf59424d03e68f147bd55a`
- Decision-0027-parent-relative delta SHA-256:
  `a3dc04b35371662603aaa68e00ba8bfa04f264dd63177c7c6c123e66ea08e736`
- Deterministically stripped extension SHA-256:
  `b3c33badb8e55b19d6b2ce535182e964ce51e5102d8413b29701dd3d817ad73d`
- Validation evidence SHA-256:
  `04759580cf6ddbd6d5108f5069058ce71994a12c0ce6b951b36093ab222b934c`
- Append-only validation root:
  `8b65112ea2d49b58ee07c1533b429fac1a8af7466e09adad073d9a22ae2ec790`

The authorized Measurement Container passed q3 and q2 width-4092
100-execution exact-SHA checks, independent numerical controls, all nine
fixture checks, q4 and MHA/GQA regressions, non-default-stream ordering, CUDA
Graph capture/replay with stable pointers and zero replay allocation, and
Compute Sanitizer memcheck/initcheck with zero errors, uninitialized reads, or
leaked allocations.
