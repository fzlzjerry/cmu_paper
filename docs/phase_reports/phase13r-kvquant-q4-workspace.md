# PHASE 13R REPORT

Status: PASS

## Identity and scope

- Starting HEAD: `a127b0d12f1815f9e51f21767686d4536b9b35e3`
- CUDA execution HEAD: `ab4e0b84dca127be0cde5450162cf2235de02ce8`
- Evidence-finalization HEAD: `2eafbb63f883cf27308018146e20e25e1d4b39eb`
- Final HEAD: the clean report-bearing descendant reported at handoff
- Decision: 0036, Accepted
- Authorized container: `sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e`
- CUDA source changed: no
- q3/q2 behavior changed: no
- Pilot executed: no

## Workspace correction

The q4 Value-decode workspace now derives once from declared total-attended
capacity:

`quantized_value_capacity = max(0, total_attended_capacity - 5)`

`q4_tile_capacity = ceil_div(quantized_value_capacity, 128)`

The preallocated FP32 shape is
`[B, 32 query heads, q4_tile_capacity, 128 head dimension]`. It is allocated
before prefill and never resized during decode. Formula version is
`kvbench-kvquant-q4-value-workspace-capacity-v1`; the workspace is accounted
separately from cache payload and `r_hbm` remains null.

Boundary tests passed: L=4096/8192/16384/32768/65536 use respectively
32/64/128/256/512 tiles. The maximum attended length 131072 uses 1024 tiles
with historical prefix 131071. Tile-boundary and undersized/exact-capacity
controls, batch-scaling, shape, byte, and pointer-stability tests pass.

Observed q4 workspace geometry:

- B=1/L=4096: `[1,32,32,128]`, 524,288 bytes.
- B=4/L=16384: `[4,32,128,128]`, 8,388,608 bytes.
- B=8/L=16384: `[8,32,128,128]`, 16,777,216 bytes.
- B=8/L=65536: `[8,32,512,128]`, 67,108,864 bytes; the unchanged feasibility
  contract classified the point `capacity_infeasible` before launch
  (100,247,423,739 required versus 89,733,904,465 limit).

Predicted and actual workspace bytes agree exactly. Cache-owned byte totals,
workspace bytes, feasibility, layout fingerprints, method fingerprints, and
manifests include the corrected capacity.

## Numerical, path, allocation, Graph, and sanitizer results

All nine corrected KVQuant fixtures pass. Dense payload, metadata, sparse
values/indices, sink state, store, and append remain exact; decode remains
within the pre-existing frozen tolerance with finite output. q3 and q2
fixtures, fingerprints, and behavior are unchanged.

Targeted immutable run results:

- `phase13rq4-b4-l16384-eager`: PASS
- `phase13rq4-b4-l16384-cuda_graph`: PASS
- `phase13rq4-b8-l16384-eager`: PASS
- `phase13rq4-b8-l16384-cuda_graph`: PASS
- `phase13rq4-b8-l65536-cuda_graph`: `capacity_infeasible`, not launched

Native 32Q/8KV GQA, non-default-stream ordering, execution path, numerical
agreement, historical-prefix preservation, and fallback rejection pass.
Workspace and cache pointers are stable. The workspace performs no measured
allocation or resize. Eager persistent allocated/reserved deltas are zero and
the existing batch-linear outer allocation control matches exactly. CUDA Graph
capture/replay passes with exact repeated output, zero replay allocation events,
zero persistent deltas, and no fallback.

Compute Sanitizer 2025.3.1 memcheck and initcheck pass the q4 B=8/L=16384
store/append/decode plus Graph path with zero errors, zero uninitialized reads,
and zero leaked bytes.

## Successor admission and unified refresh

Successor q4 MethodAdmissionReport: PASS; SHA-256
`75605637f460a309081e1e0a4065e90e8e14194d365250cb513092616ef89ec7`.
The standardized q4 method fingerprint is
`27b3af27e153491112ef974ea3a6d813987bfcb5083a699ce131fa3332c9703b`;
the standardized layout fingerprint is
`41c2c6b92b6a81c5e6991e398de72034f71b9989e5d32b2cc2c6037f25d61988`.

The refreshed standardized q4 G5 process medians are 151.18264770507812,
150.7449188232422, and 151.15745544433594 ms. CV is
0.0016273336621252625 (0.162733%), below the frozen 3% threshold. Output
checksum, kernel path, and allocation agree across all three processes; no
fallback occurred. These are non-claim Measurement Container admission data;
no speedup or comparative latency was calculated.

The refreshed unified report reuses nine unchanged configuration records and
the successor q4 record. G0-G5 are PASS, Pilot is READY, Full Scan is CLOSED,
quality execution is LOCKED, and PERFORMANCE_DATA_FROZEN remains absent.
Unified report SHA-256:
`13553823a68f34a0a538674caeb5abf2f9e638d5086b90d21fbdb94aa66be05a`.

## Durable evidence and preservation

Append-only artifact:
`artifacts/phase13rq4/phase13rq4-20260820t094629495794z-ab4e0b84-b8c7bd`.
Its 14-object root is
`9f027d64424844d0d62311daad5740e2b76960d1d5e7b8be26aa1ccba100a8db`.
It was conditionally published COMPLETE-last to
`r2://kvbench-artifacts/kvbench/sha256/9f027d64424844d0d62311daad5740e2b76960d1d5e7b8be26aa1ccba100a8db/`;
one clean empty-directory retrieval passed every object, checksum, inventory,
COMPLETE, root, and indefinite Bucket Lock check. No credentials were recorded
or passed to the Measurement Container.

The blocked campaign
`phase13-20260804t111810342595z-a127b0d1-8649c3` remains immutable at root
`c5523a894bc38f41b45b6bdf38ad437fb0b28fc29db4a959b4a592fdd825277b`
with 3,246 objects. Its 69 completed runs, one failure, 614 aborted records,
and COMPLETE marker were not modified, resumed, or reused. Phase 11R evidence,
original Phase 12 evidence, calibration, fixtures, q3/q2, and all other methods
remain unchanged.

## Scientific interpretation and next action

The evidence supports only that the capacity-derived, preallocated q4
Value-decode workspace preserves the frozen KVQuant numerical behavior and
restores q4 long-context admission and unified G0-G5. It makes no speedup,
HBM-traffic, capacity-benefit, final-knee, or quality claim.

Next action: propose a wholly new Phase 13 Pilot campaign in a separate task;
never resume or reuse the blocked Pilot timing data.
