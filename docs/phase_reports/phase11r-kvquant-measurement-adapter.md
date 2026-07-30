# PHASE 11 REPORT

Status: PASS

## Authority and scope

- Starting HEAD: `f0f02364a556da70e67b3107a0c0afad5f75eae9`
- Admission execution HEAD: `3b52b42674c23e2be8e7a5b2355c77629b904b2a`
- Working tree: CLEAN
- Algorithm identifier: kvquant_gqa_upstream_patch_v1
- Execution-source identifier: kvquant_gqa_longctx_deterministic_v3
- Decisions: 0021, 0023, 0024, 0025, 0026, 0027
- Aggregate patch SHA: bae63bced549479709b10d7f6a8ee35a8f21ec18cc040a7424591cee47c1b0a6
- Corrected commit: 4b8533b29b04f8c4bf55f688a41fefe20487637b
- Corrected tree: 46f2149a0369d5c97d9a6bc77d57b5f3a5a5fb3b
- Extension SHA: a79644923ba131e56abe95029e669346dbbb11fd210d2b9f8b2086819ffeaad1
- Calibration ID: kvqcal-cdb724c806d64d095c040d2673a987a3
- Calibration root: 8148306d08205af376994b022f189a0d6837915cd279ca8af6b104e1f4b46ccf
- Historical Phase 10 root: 32cdf465a361dd6695b66ccbea0a462bddc075fd9778d0aa8cdaa3f94e6f63ab
- Corrected fixture ID: kvqref-2e0a0e9022c50cbc6fb497d88cae973e
- Corrected fixture root: c28682d58706b58812dc1db69ba5eb4982339ba13f39bf67f751794cdaabfdec

Phase 11R changed only the existing KVQuant Adapter binding and its evidence
plumbing. CUDA source, calibration, fixtures, existing methods, decisions, and
the authorized Measurement Container were not changed.

## Adapter and cache

- Adapter location: src/kvbench/adapters/kvquant.py
- Supported configurations: kvq4, kvq3, kvq2
- Boundary semantics: PRE-ROPE KEY QUANTIZATION; ATTENTION-READY SINK KEY
- Static cache: PASS

The q4 decode path now calls Decision 0027's deterministic API with one
cache-owned, preallocated FP32 workspace of shape `[1, 32, 32, 128]`
(524,288 bytes). q3 and q2 retain their existing corrected paths. Packed K/V,
metadata, fixed-cap sparse values/indices/counts, native-eight-head sink K/V,
staging, outputs, and workspaces remain caller-owned and pointer-stable.

Cache layout fingerprints are
`kvq4=f070f5851c825baa5c8c723b65fe65b134a83a0eedb33dcd2b6825f8ceb00cd5`,
`kvq3=c5ca6905b5369965c884e1f1608e14c376b8b391513de4801ea17c5c5cccf722`,
and
`kvq2=3d6819bd9096196c9a7d4807c77c1fb102d606facb9ff531ab62e47fe48302d5`.

## Correctness and execution controls

- Fixture conformance: 9/9 PASS
- Execution-path and GQA audit: PASS
- Eager allocation: PASS
- CUDA Graph: PASS
- Sanitizer: PASS

All nine corrected fixtures match exactly for inputs, pre-RoPE K, dense
payloads, metadata, sparse values/indices/counts, unused slots, sink state,
store state, append state, and byte records. Decode agrees at the frozen
`atol=rtol=0.01`, and all outputs are finite.

Native 32Q/8KV mapping is preserved with `kv_head = query_head // 4`. The
measured path has no CPU top-k, dynamic sparse allocation, tensor-to-host
conversion, host synchronization, `torch.cat`, complete-prefix
materialization, GQA expansion, cache growth, or backend fallback.

Byte accounting passes for L=5, 17, 18, 128, and 4096 for all three
configurations. Predicted-versus-actual relative error is 0, the maximum
reciprocal residual is `2.220446049250313e-16`, and `r_hbm` remains null.
Eager persistent allocation deltas and unknown allocations are zero. All four
required fixed-L graph points capture and replay with exact repeated outputs,
stable pointers, eager/graph agreement, and zero replay allocation.

Compute Sanitizer 2025.3.1.0 memcheck/initcheck covers q4 cap, corrected q3,
q2, sink/native GQA, fixed-L overwrite, and graph replay. It reports zero
memory errors and zero leaked allocations.

## Bounded admission and publication

- Bounded admission: 9/9 PASS
- Admission run IDs: phase11-20260730t030934524z-3b52b426-424aec-kvquant-p00-kvq4-fixed-l128-eager, phase11-20260730t030934524z-3b52b426-424aec-kvquant-p01-kvq4-fixed-l128-graph, phase11-20260730t030934524z-3b52b426-424aec-kvquant-p02-kvq3-fixed-l128-eager, phase11-20260730t030934524z-3b52b426-424aec-kvquant-p03-kvq3-fixed-l128-graph, phase11-20260730t030934524z-3b52b426-424aec-kvquant-p04-kvq2-fixed-l128-eager, phase11-20260730t030934524z-3b52b426-424aec-kvquant-p05-kvq2-fixed-l128-graph, phase11-20260730t030934524z-3b52b426-424aec-kvquant-p06-kvq4-fixed-l4096-eager, phase11-20260730t030934524z-3b52b426-424aec-kvquant-p07-kvq4-fixed-l4096-graph, phase11-20260730t030934524z-3b52b426-424aec-kvquant-p08-kvq4-growing-l17-eager
- MethodAdmissionReport SHA-256: 59ef5bfc581a68cdc4d21c4c0a840f046e698633f7475f79906063c6e333ae6a
- Inner R2 URI: r2://kvbench-artifacts/kvbench/sha256/0834410509ea7324a41715e0e84e09617bf9b188b10394a234f9a57e804dd1f2/

The 165-object inner admission bundle validates locally, was conditionally
published with `COMPLETE` last under the indefinite Bucket Lock, and passed
one clean retrieval. One transient transport failure and one rejected
container-view path attempt remain preserved as append-only local evidence.

## Gates and preservation

- G2-KVQ: PASS
- Global G2: NOT EVALUATED
- G3: NOT EVALUATED
- G4: NOT EVALUATED
- G5: NOT EVALUATED
- Full Scan: CLOSED
- Quality execution: LOCKED
- PERFORMANCE_DATA_FROZEN: ABSENT
- Performance claim eligible: FALSE
- Speedup calculated: NO
- r_hbm: NULL
- Historical evidence changed: NO
- Existing methods changed: NO
- Measurement Container changed: NO
- Phase 12 started: NO

Commits through admission and receipt correction are
`3b52b42674c23e2be8e7a5b2355c77629b904b2a` and
`03ae147b2b6dac745afcc70e395c2cf77eb85ff5`. No concrete Phase 11R blocker
remains.

Scientific interpretation is limited to this result: the static KVQuant
Measurement Adapter conforms to the frozen corrected numerical oracle and
satisfies G2-KVQ under the bounded admission contract. This is not a speedup,
physical-HBM, knee, capacity, performance, or quality claim.

Next action: Phase 12 unified admission may be proposed as a separate task; it
was not started here.
