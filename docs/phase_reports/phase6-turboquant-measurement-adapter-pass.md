# Phase 6 TurboQuant Measurement Adapter — PASS

- Status: PASS
- Execution HEAD: `0df5bb4d445d48e6cba17e30723733f8de35cb14`
- Authorized container:
  `sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e`
- Admission bundle:
  `phase6-20260726t035257468z-0df5bb4d-4139a6-4bit_nc-fixed-l128-eager`
- G0/G1: PASS
- G2-TQ: PASS
- Global G2-G5: NOT EVALUATED
- Full Scan: CLOSED
- Quality execution: LOCKED
- `PERFORMANCE_DATA_FROZEN`: absent

## Reference and admission evidence

The final bundle uses the pinned vLLM TurboQuant source at commit
`752a3a504485790a2e8491cacbb35c137339ad34`, source tree
`3ec7a4eb00f9bc8fec399bea6cf7de27a7936372`, and fixture set
`774ec946a8839d4de012bc6fba0ee5a933ab1488ecc43354d8573b4481b12f76`.
All mandatory configurations passed fixture store, append, appended-slot,
slot-layout, byte-accounting, decode, execution-path, allocation, and CUDA
Graph checks:

- `turboquant_4bit_nc`
- `turboquant_k3v4_nc`
- `turboquant_3bit_nc`

The three mandatory Compute Sanitizer probes passed with exit code 0, probe
PASS, `ERROR SUMMARY: 0 errors`, and `LEAK SUMMARY: 0 bytes leaked`. The
authorized container also passed `make test-cuda` (18/18) and
`make test-graph` (5/5).

The frozen bounded grid completed 9/9 points with no failure or
capacity-infeasible point:

- each mandatory configuration at fixed-L, B=1, L=128, eager and CUDA Graph;
- `turboquant_4bit_nc` at fixed-L, B=1, L=4096, eager and CUDA Graph;
- `turboquant_4bit_nc` at growing-context, B=1, L=128, O=4, eager.

Every point records the exact authorized container and execution SHA. The
bundle records native GQA, finite output, stable cache pointers, zero unknown
allocation, no backend fallback, no cache growth, and exact predicted versus
allocated byte accounting. CUDA Graph points passed capture and replay with
zero replay allocation. The admission records explicitly set
`performance_claim_eligible` to false, `speedup_calculated` to false, and
`r_hbm` to null.

## Durable publication

The finalized bundle contains 167 objects and has content-addressed tree root:

```text
f003bc3dc5de6b67a6d8f1b8bed7fa49b7f90f9d7edc4d1383e2d97c8aa19d6d
```

Its durable locator is:

```text
r2://kvbench-artifacts/kvbench/sha256/f003bc3dc5de6b67a6d8f1b8bed7fa49b7f90f9d7edc4d1383e2d97c8aa19d6d/
```

The first publication attempt ended with a `TransportError`. Conditional
no-replace continuation then reported 89 uploaded and 78 already-existing
identical objects, totaling all 167 expected objects. `COMPLETE` was
published last. Clean retrieval into a new empty directory passed inventory,
object-set, lifecycle, and SHA-256 verification under the existing locked R2
evidence prefix.

## Historical evidence

All earlier B-018, forced-Flash, and bounded-grid failures remain preserved as
immutable completed failure artifacts. In particular, the earlier
`01fb93cd` grid evidence remains a failed attempt and was neither overwritten
nor promoted. The pre-existing BLOCKED Phase 6 report remains an accurate
record of its earlier evidence state; this PASS report records the subsequent
successful execution and publication at `0df5bb4d`.

## Gate disposition

G2-TQ is PASS because all three mandatory configurations satisfy the frozen
fixture, sanitizer, execution-path, allocation, Graph, bounded-grid, and
durable-publication requirements. This is a method-specific admission only.
It does not mark global G2, G3, G4, or G5 as evaluated, open Full Scan, unlock
quality execution, or create `PERFORMANCE_DATA_FROZEN`.

## Scientific interpretation

The TurboQuant Measurement Adapter conforms to the pinned reference for the
mandatory configurations and satisfies G2-TQ. This report makes no speedup,
physical-HBM-traffic, knee, capacity, performance, or quality claim.

Phase 7 KIVI Reference Lane may be proposed only as a separate new task.
