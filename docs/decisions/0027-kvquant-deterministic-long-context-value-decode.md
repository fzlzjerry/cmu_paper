# Decision 0027: KVQuant deterministic long-context Value decode

- Status: Proposed
- Date: 2026-07-30
- Parent authority: Decision 0025 corrected commit
  `0d9df350bd1788284e1ce76a8bf6e886beca5efa`, tree
  `a85cf7bf093982a4bf89c33d4e6794d9a85f846d`
- Frozen calibration root:
  `8148306d08205af376994b022f189a0d6837915cd279ca8af6b104e1f4b46ccf`
- Numerical oracle:
  `c28682d58706b58812dc1db69ba5eb4982339ba13f39bf67f751794cdaabfdec`

## Context

The blocked Phase 11 `kvq4` diagnostic showed byte-stable Value decode at
width 124 but execution-history-dependent output at width 4092. The legacy
path performs floating `atomicAdd` reductions from 32 independently scheduled
width tiles and from the sparse correction kernel. Its tail tile also leaves
inactive shared-memory lanes uninitialized before an unconditional read.

## Decision

1. Add one explicit Measurement-only `kvq4` Value-decode out-API. Keep the
   legacy Reference API unchanged.
2. Write dense-plus-sparse tile partials into caller-owned float32 workspace
   `[1, 32, max_tiles, 128]`, then reduce tiles in ascending order into the
   caller-owned output. Each workspace and output address has one writer; the
   new path uses no floating output atomic.
3. Process tokens and sparse slots in ascending native order. Preserve
   `kv_head = query_head // 4` and native eight-KV-head packed storage.
4. Initialize every potentially read shared LUT and score lane in the legacy
   q4, q3, and q2 Value kernels before a tail-lane guard. This is a
   deterministic initialization correction, not a quantization change.
5. Launch the new tile and reduction kernels on the current PyTorch CUDA
   stream with launch-error checks.
6. Quantization, calibration, quantizers, packing, metadata, sparse selection,
   caps, sinks, dtypes, GQA geometry, and all nine fixtures remain unchanged.
7. The new API performs no dynamic allocation, host synchronization,
   tensor-to-host scalar conversion, CPU reduction, `torch.cat`, GQA
   expansion, or fallback.
8. This decision authorizes source/API remediation only. It does not modify or
   admit the Phase 11 Adapter, rerun the admission grid, publish a
   MethodAdmissionReport, start Phase 12, or authorize performance or quality
   work.

## Acceptance

This decision becomes Accepted only after the exact source reconstructs,
width 4092 produces one exact output SHA across 100 executions, an independent
numerical control agrees within the frozen tolerance, all nine fixtures remain
conformant and byte-unchanged, non-default-stream and CUDA Graph execution
pass with zero replay allocation, and Compute Sanitizer memcheck and initcheck
report zero errors, uninitialized reads, and leaked allocations inside the
authorized Measurement Container.

## Proposed identities

- Derived source commit:
  `4b8533b29b04f8c4bf55f688a41fefe20487637b`
- Derived source tree:
  `46f2149a0369d5c97d9a6bc77d57b5f3a5a5fb3b`
- Aggregate patch SHA-256:
  `bae63bced549479709b10d7f6a8ee35a8f21ec18cc040a7424591cee47c1b0a6`
- Parent-relative delta SHA-256:
  `d12db702ec6d625330e791f4556af1dd368bfff7f00a45d5817eabcff4298ce9`

The extension and validation-evidence identities are recorded only after the
required exact-container checks finish.
