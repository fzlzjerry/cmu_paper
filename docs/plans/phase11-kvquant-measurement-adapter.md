# Phase 11 KVQuant Measurement Adapter

## Authority and boundary

Phase 11 starts from clean commit
`72f1897af78b738cc8c74fd335a8957a8e8f5d6c`. Algorithm authority remains
`kvquant_gqa_upstream_patch_v1`: upstream commit
`57a238357f0ffe50084670fcd5781c9848f80ea2`, Decision 0021 patch
`db3b6fb7ec0a72e25001e1c83a5158d86512248db5c3a06c61895598d1d482d6`,
and calibration root
`8148306d08205af376994b022f189a0d6837915cd279ca8af6b104e1f4b46ccf`.
Measurement execution binds Decision 0025 source
`kvquant_gqa_graphsafe_kvq3_v2`, aggregate patch
`23a15db86790299392412c3ce2da7d971f4f073cfaf6839d82d3746c8b56b551`,
commit `0d9df350bd1788284e1ce76a8bf6e886beca5efa`, and tree
`a85cf7bf093982a4bf89c33d4e6794d9a85f846d`.

The numerical oracle is corrected fixture ID
`kvqref-2e0a0e9022c50cbc6fb497d88cae973e`, root
`c28682d58706b58812dc1db69ba5eb4982339ba13f39bf67f751794cdaabfdec`.
The immutable Phase 10 root
`32cdf465a361dd6695b66ccbea0a462bddc075fd9778d0aa8cdaa3f94e6f63ab`
is preservation evidence only. CUDA execution is confined to Measurement
Container
`sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e`.

The common endpoint retains attention-ready `key_states` and supplies one
optional pre-RoPE Key view only to methods declaring that requirement.
KVQuant quantizes `key_pre_rope_states`; sink K uses attention-ready
`key_states`; V is the native projection output. Existing adapters retain
their current boundary and measured allocation behavior.

## Static implementation

Add one adapter and one cache state for `kvq4`, `kvq3`, and `kvq2` at
`B=1,H_Q=32,H_KV=8,D=128`, BF16 interface, five sink tokens, and fixed
Key/Value sparse capacities of 12. Before measurement, allocate packed K/V,
frozen metadata, float32 sparse values, int32 indices, count/mask state,
full-precision sink K/V, pre-/post-RoPE staging, query/output staging, and
softmax/merge/correction workspace. No persistent K/V region uses 32 heads.

Store and append use Decision 0024 caller-owned APIs incorporated by Decision
0025. Key prefill uses the deterministic single-token current-stream pack path
outside timing, avoiding the legacy parallel Key-prefill shared-memory path;
each destination is cleared before source `+=` packing. Value extrema and
zero-point remain device-resident. Fixed-L overwrites only its scratch slot;
growing context writes the next bounded slot without growth.

Decode consumes the packed cache, metadata, fixed-cap sparse corrections, and
sink buffers directly with `kv_head=query_head//4`. It preallocates all
outputs and workspaces and permits no complete-prefix materialization, GQA
expansion, CPU sparse work, host scalar transport, host synchronization,
dynamic allocation, cache growth, or fallback.

## Conformance and admission

Replay all nine corrected fixtures. Require exact dense payload, metadata,
sparse value/index/count, unused-slot, sink, store, append, and byte results;
freeze decode comparison at `atol=0.01, rtol=0.01` before admission. Preserve
mixed-family provenance: kvq4/kvq2 reuse Phase 10 bytes, while kvq3 binds
Decision 0025.

Report dense K/V, metadata, sparse values/indices, count state, sink,
staging, padding, workspace, and temporary peak separately at
`L=5,17,18,128,4096`. Enforce predicted-versus-owned error below 1%,
`abs(rho_alloc*r_alloc-1)<=1e-9`, and `r_hbm=null`.

Reuse the existing path, GQA, allocation, graph, sanitizer, process,
artifact, MethodAdmissionReport, and R2 machinery. Fixed-L graph capture must
show pointer stability, eager/graph agreement, and zero replay allocation.
The minimal sanitizer set covers kvq4 cap, corrected kvq3, kvq2, sink,
native 32Q/8KV decode, fixed-L overwrite, and graph replay.

Only after those checks pass, run the preregistered nine admission points:
L=128 eager/graph for all three bit widths, kvq4 L=4096 eager/graph, and kvq4
growing L=17 with four eager outputs. Compute no speedup. Finalize one
append-only admission bundle, publish it content-addressed with `COMPLETE`
last, and perform one clean retrieval. Phase 12 and all performance, profiling,
campaign, and quality work remain deferred.
