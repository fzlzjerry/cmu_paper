# Decision 0026: KVQuant pre-RoPE adapter boundary

- Status: Accepted
- Date: 2026-07-29
- Phase: 11

## Context

The existing endpoint forms K immediately after `k_proj`, then applies RoPE
in place before calling `KVCacheMethod`. KVQuant must quantize the former
representation while its sink and attention path require the latter. Passing
only the attention-ready tensor would silently change the frozen method.

## Decision

1. The existing positional Key argument remains the attention-ready post-RoPE
   tensor.
2. The method protocol gains one optional keyword-only pre-RoPE Key argument
   and one static capability flag. Only KVQuant declares the capability.
3. After query RoPE has completed and before Key RoPE begins, the endpoint
   copies native-eight-head pre-RoPE K into a correctly shaped view of the
   already allocated query-RoPE scratch. That scratch is no longer needed by
   the query operation and has sufficient storage for the frozen geometry.
4. The endpoint passes both references to KVQuant. Existing adapters receive
   the same arguments as before and remain behaviorally and allocation
   equivalent.
5. This is a narrow backward-compatible tensor-semantic correction. It adds no
   method-specific runner branch, model-hook framework, persistent
   query-head-sized K/V storage, or measured-region CUDA allocation.
6. Quantization mathematics, corrected CUDA source, calibration, quantizers,
   sparse semantics, sinks, fixtures, and existing methods remain unchanged.

Any geometry for which the existing scratch cannot safely hold the native
pre-RoPE Key fails closed.
