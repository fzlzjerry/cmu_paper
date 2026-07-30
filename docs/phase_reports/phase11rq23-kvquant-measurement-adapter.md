# PHASE 11R-Q23 REPORT

Status: PASS

## Authority and scope

- Starting HEAD: d99920e5dd7ea94bce7c98b4301bd035c073dfea
- Admission execution HEAD: 8a708407825d5f3f3eaa0af476a55631ad546059
- Working tree: CLEAN
- Algorithm identifier: kvquant_gqa_upstream_patch_v1
- Execution-source identifier: kvquant_gqa_longctx_deterministic_q23_v4
- Decisions: 0021, 0023, 0024, 0025, 0026, 0027, 0029
- Aggregate patch SHA: 7b9d3cc6773e8ef37697601c885f2c5ec581dffd57cf59424d03e68f147bd55a
- Corrected commit: 34b0bdfa83082e1f30387d9ac5cca369006e089c
- Corrected tree: 1f85af65fe03061583ffe8bd91e47d7ecffdd312
- Extension SHA: b3c33badb8e55b19d6b2ce535182e964ce51e5102d8413b29701dd3d817ad73d
- Calibration ID: kvqcal-cdb724c806d64d095c040d2673a987a3
- Calibration root: 8148306d08205af376994b022f189a0d6837915cd279ca8af6b104e1f4b46ccf
- Historical Phase 10 root: 32cdf465a361dd6695b66ccbea0a462bddc075fd9778d0aa8cdaa3f94e6f63ab
- Corrected fixture ID: kvqref-2e0a0e9022c50cbc6fb497d88cae973e
- Corrected fixture root: c28682d58706b58812dc1db69ba5eb4982339ba13f39bf67f751794cdaabfdec

Phase 11R-Q23 changes admission authority and evidence only. CUDA, Adapter
behavior, cache layout, runners, calibration, fixtures, existing methods,
numerical tolerances, Decisions 0021 through 0029, and the authorized
Measurement Container remain unchanged.

## Adapter and controls

- Adapter location: src/kvbench/adapters/kvquant.py
- Supported configurations: kvq4, kvq3, kvq2
- Boundary semantics: PRE-ROPE KEY QUANTIZATION; ATTENTION-READY SINK KEY
- Static cache: PASS
- Fixture conformance: 9/9 PASS
- Execution-path and GQA audit: PASS
- Eager allocation: PASS
- CUDA Graph: PASS
- Sanitizer: PASS

All nine corrected fixtures match for inputs, pre-RoPE Key, dense payloads,
metadata, sparse values, indices and counts, unused slots, sink state, store
state, append state, and byte records. Decode agrees at the frozen
`atol=rtol=0.01`, and all outputs are finite.

The measured path retains native 32Q/8KV mapping and direct compressed-cache
decode. It has no CPU top-k, dynamic sparse allocation, tensor-to-host
conversion, host synchronization, measured `torch.cat`, complete-prefix
materialization, GQA expansion, query-head-sized cache, or backend fallback.
All cache, sparse, sink, staging, output, and Decision 0029 Value-decode
workspace pointers remain stable.

Byte accounting passes at L=5, 17, 18, 128, and 4096 for all three
configurations. Predicted-versus-actual allocation error is zero,
`rho_alloc` and `r_alloc` are reciprocal within the frozen tolerance, and
`r_hbm` remains null. Fixed-L CUDA Graph capture/replay passes at all four
required points with eager/graph agreement and zero replay allocation.
Compute Sanitizer 2025.3.1.0 memcheck/initcheck reports zero memory errors and
zero leaked allocations across q4, q3, q2, sink/native-GQA, overwrite, and
graph paths.

## Bounded admission and publication

- Bounded admission: 9/9 PASS
- Admission run IDs: phase11rq23-20260730t152550624z-8a708407-685bb7-kvquant-p00-kvq4-fixed-l128-eager, phase11rq23-20260730t152550624z-8a708407-685bb7-kvquant-p01-kvq4-fixed-l128-graph, phase11rq23-20260730t152550624z-8a708407-685bb7-kvquant-p02-kvq3-fixed-l128-eager, phase11rq23-20260730t152550624z-8a708407-685bb7-kvquant-p03-kvq3-fixed-l128-graph, phase11rq23-20260730t152550624z-8a708407-685bb7-kvquant-p04-kvq2-fixed-l128-eager, phase11rq23-20260730t152550624z-8a708407-685bb7-kvquant-p05-kvq2-fixed-l128-graph, phase11rq23-20260730t152550624z-8a708407-685bb7-kvquant-p06-kvq4-fixed-l4096-eager, phase11rq23-20260730t152550624z-8a708407-685bb7-kvquant-p07-kvq4-fixed-l4096-graph, phase11rq23-20260730t152550624z-8a708407-685bb7-kvquant-p08-kvq4-growing-l17-eager
- MethodAdmissionReport SHA-256: 9cfed618cee9514a1071392d0a2dca327dcf6acd33d81ac72cc477c7880c09e2
- Inner R2 URI: r2://kvbench-artifacts/kvbench/sha256/8ea533b9544e99140aec04b4cb9b1ad26f271273206d170e7abefa195c0581aa/

The finalized 212-object inner bundle validates locally, was conditionally
published with `COMPLETE` last under Bucket Lock rule
`kvbench-evidence-indefinite`, and passed one clean retrieval. One local
validator invocation error and three transient transport failures remain
preserved as read-only local evidence; no failed attempt overwrote an object
or changed the content-addressed root.

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

No concrete Phase 11R-Q23 blocker remains. Scientific interpretation is
limited to this result: the existing static KVQuant Measurement Adapter
conforms to the corrected fixture oracle and satisfies current-source G2-KVQ
under Decision 0029 and the bounded admission contract. This is not a
speedup, physical-HBM, knee, capacity, general performance, or quality claim.

Next action: rerun Phase 12 unified admission in a separate task; it was not
started here.
