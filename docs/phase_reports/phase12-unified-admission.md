# Phase 12 — Unified Admission Gates

Status: PASS

- Campaign: `phase12-20260731t062914664948z-6165f78d-c78b9a`
- Execution Git SHA: `6165f78ddfbc96f7e65568b6a960d4b47f85e813`
- Container: `sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e`
- Common point: `fixed_l`, `cuda_graph`, B=1, L=4096, warmup=64, measured_steps=128
- Independent processes: 3 per configuration (30 total)
- Selective reruns: 0

## Configuration results

| Configuration | CV | G1 | G2 | G3 | G4 | G5 |
| --- | ---: | --- | --- | --- | --- | --- |
| `bf16` | 0.00005592 | PASS | PASS | PASS | PASS | PASS |
| `tq_4bit_nc` | 0.00023520 | PASS | PASS | PASS | PASS | PASS |
| `tq_k3v4_nc` | 0.00057858 | PASS | PASS | PASS | PASS | PASS |
| `tq_3bit_nc` | 0.00080931 | PASS | PASS | PASS | PASS | PASS |
| `k4v4` | 0.00015331 | PASS | PASS | PASS | PASS | PASS |
| `k2v4` | 0.00014399 | PASS | PASS | PASS | PASS | PASS |
| `k2v2` | 0.00006022 | PASS | PASS | PASS | PASS | PASS |
| `kvq4` | 0.00012556 | PASS | PASS | PASS | PASS | PASS |
| `kvq3` | 0.00002661 | PASS | PASS | PASS | PASS | PASS |
| `kvq2` | 0.00006289 | PASS | PASS | PASS | PASS | PASS |

Held-out controls `turboquant_k8v4` and `k4v2` did not participate in the main gate.

## Global decision and custody

- G0: PASS
- G1: PASS
- G2: PASS
- G3: PASS
- G4: PASS
- G5: PASS
- Pilot: READY
- Full Scan: CLOSED
- Quality execution: LOCKED
- PERFORMANCE_DATA_FROZEN: absent
- Published campaign root: `42ab15b6617d072f9b0825b701d1df4519caa110166b8edd48b8359fe8e588e5`
- Published campaign R2 URI: `r2://kvbench-artifacts/kvbench/sha256/42ab15b6617d072f9b0825b701d1df4519caa110166b8edd48b8359fe8e588e5/`
- Published campaign objects: 391
- COMPLETE uploaded last: yes
- Clean retrieval: PASS

Adapters, CUDA, cache layouts, calibration, fixtures, and method-specific evidence were not changed. This report establishes only unified numerical, memory, execution-path, graph, and three-process reproducibility admission; it makes no speedup, HBM, knee, capacity, performance, or quality claim.

Phase 13 is deferred to a separate task.
