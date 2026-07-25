# Phase 6 TurboQuant Measurement Adapter

- Status: BLOCKED
- Entry HEAD: `e06f638f4b913f9bd1be2975a478657f5bf2338e`
- Final execution HEAD: `ace9261a0cd4f5bf1d7a0b55549ffbe85811998b`
- Authorized container: `sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e`
- G0/G1: PASS
- G2-TQ: BLOCKED
- Global G2-G5: NOT EVALUATED
- Full Scan: CLOSED
- Quality execution: LOCKED
- `PERFORMANCE_DATA_FROZEN`: absent

## Minimal integration

The existing fixed-L and growing-context runners delegate through one
`TurboQuantMethodAdapter` and one static cache-state class to the exact
vLLM v0.25.1 source at commit
`752a3a504485790a2e8491cacbb35c137339ad34`, tree
`3ec7a4eb00f9bc8fec399bea6cf7de27a7936372`. Only narrow package-coupling
compatibility is carried; upstream algorithmic source is unchanged. No new
framework, runner, server, scheduler, profiler, or R2 client was introduced.
KIVI, KVQuant, Phase 7, Pilot, Full Scan, Nsight, fitting, figures, and quality
work remain unopened.

The cache uses block size 16, compressed layers 2-29, BF16 layers 0,1,30,31,
and exact slots 134/118/102 bytes. At fixture capacity 18 (rounded to 32),
predicted and allocated bytes match exactly: 1,945,732 / 1,830,980 / 1,716,292
bytes for 4bit/k3v4/3bit. Relative error is 0.0 for all. `r_hbm` is null.

## Passing evidence before the blocker

All three frozen fixtures pass exact store, append, appended-slot, slot-layout,
source identity, and byte accounting. Decode passes the frozen
`atol=0.02, rtol=0.02` tolerance with finite outputs. Execution-path audits
identify `_tq_fused_store_mse`, `_tq_decode_stage1`, and
`_fwd_kernel_stage2`, with no full-prefix dequantization, GQA materialization,
query-head-sized K/V temporary, host synchronization, cache growth, or backend
fallback. Fixture eager operations and graph replays have zero allocation;
capture/replay, eager/graph agreement, pointer stability, and replay stability
pass for all mandatory configurations.

The exact authorized container passed `make test-cuda` (16/16) and
`make test-graph` (5/5) at the final execution SHA.

## Blocking evidence

Final run `phase6-20260725t065153714z-ace9261a-083f14-4bit_nc-fixed-l128-eager` is a valid immutable `runtime_failed` artifact. Its
4bit sanitizer probe completes store, append, and decode correctly, but
Compute Sanitizer 2025.3.1.0 memcheck exits 99 with
`LEAK SUMMARY: 2093260 bytes leaked in 28 allocations` and
`ERROR SUMMARY: 28 errors`. Per the frozen stop condition, k3v4/3bit
sanitizers and all nine bounded grid points were not attempted. No speedup or
normal timing was computed.

Three earlier finalized failures remain preserved and show the same gate while
probe-only teardown was narrowed. No prior run was overwritten. The final
failure artifact independently validates with 24 files, COMPLETE, inventory,
and checksum ledger.

## Durable evidence

The finalized BLOCKED artifact was conditionally published with COMPLETE last
to `r2://kvbench-artifacts/kvbench/sha256/f319c4b05054ee2f31bdcbfe15fa67850ea784c17718192b82e527567c5cf343/`. Root SHA-256 is `f319c4b05054ee2f31bdcbfe15fa67850ea784c17718192b82e527567c5cf343`. A clean retrieval verified all 27
objects, inventory, COMPLETE, and checksums with no unexpected object. Bucket
`kvbench-artifacts` remains private under exact enabled indefinite rule
`kvbench-evidence-indefinite`. This is durable BLOCKED evidence, not a
completed admission bundle.

The strict MethodAdmissionReport is
`docs/evidence/phase6/turboquant-method-admission.json`, SHA-256
`2954feffeb8a8d33c69fd4f051a2ff8647a9bdcd02348f0e250b804c0dd70f44`. It records B-018, no admitted configuration, G2-TQ BLOCKED,
global G2-G5 NOT EVALUATED, Full Scan CLOSED, quality LOCKED, no speedup, and
no `r_hbm`.

## Minimum remediation

Under the unchanged authorized image and pinned algorithm, establish a
sanitizer probe lifecycle that releases every adapter-owned CUDA tensor before
context teardown and produces one zero ERROR SUMMARY plus one zero LEAK
SUMMARY for each mandatory configuration. Do not weaken the sanitizer parser
or leak criterion. Then use new run IDs to rerun the three sanitizer probes;
only if all pass may the frozen nine-point admission grid and complete durable
admission bundle be attempted.
