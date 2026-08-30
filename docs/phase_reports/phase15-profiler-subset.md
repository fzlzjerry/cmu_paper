# Phase 15 Profiler Subset

Status: **PASS**

## Scope and identity

- Campaign: `phase15-20260828t144810363697z-446b334e-90460f`
- Starting HEAD: `68cd50b912d5d91ffe57710b7e36714f4fe43f22`
- Profiler execution HEADs: `446b334e7bb3175bb5c28217a473c976990d60ed`, then NCU continuation `e00cce073be6946db2d8fd301f593f74f020b4c2`
- Publication closeout code HEAD: `cd61b52a92748f148a02c8f8a575272dc5ee6453`
- Measurement Container: `sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e`
- Tools: Nsight Systems 2026.1.3; Nsight Compute 2026.2.1; NVIDIA RTX PRO 6000 Blackwell, SM120
- Profiler records use only `run_kind=nsys` or `run_kind=ncu`; zero profiler duration entered Phase 13/14 timing, fits, or performance claims.

## Coverage

- Nsight Systems: 32/32 selected profiles and 16 eager/Graph pairs complete.
- Nsight Compute: 22/22 selected profiles complete; all used the live SM120 metric map and recorded 10 replay passes.
- Formal selected failures: 0. Five selected NCU points used preserved retry attempts; eight failed tool-configuration attempts and one incomplete diagnostic attempt remain append-only.
- Anchors: `bf16`, `tq_4bit_nc`, `k4v4`, `kvq4`. Regime contexts were deterministically selected from stable Phase 13/13D evidence.
- Common same-work point: B=1, context label 131072, historical prefix 131071, total attended 131072, CUDA Graph, shared by all ten configurations.

## CUDA Graph mechanism

Across all 16 anchor eager/Graph pairs, Graph reduced traced CPU CUDA API call count to eight graph launches: BF16 eager had 10,832 calls, TurboQuant 12,400, KIVI 54,608 or 63,312, and KVQuant 53,328. GPU inter-kernel idle was lower in 16/16 pairs. Kernel count and ordering were unchanged in 16/16, and no overlap change was observed.

CPU submission *interval* and the API-to-node-start correlation metric were lower in 0/16 because eager calls and one-to-many Graph replay nodes have different timestamp granularity; the latter is not a direct per-kernel launch-latency measure. Both modes contain one required profiler-boundary synchronization. Graph's boundary wait was longer in 16/16 because it drains rapidly queued replays; it is not a measured-path synchronization or normal timing result.

Classification: `method_specific_mixed`. Direct profiling confirms fewer CPU submissions and lower GPU idle, but Phase 14's heterogeneous slopes are not explained by a pure launch-floor-only model. Graph-mode-only NCU does not supply an eager/Graph traffic delta.

## Same-work physical traffic

Bytes below are decimal GB. Canonical `r_hbm` is BF16 cache-path DRAM bytes divided by method cache-path DRAM bytes; `A_traffic = rho_hbm / rho_alloc`.

| Configuration | Cache DRAM GB | Total decode DRAM GB | r_hbm | A_traffic |
|---|---:|---:|---:|---:|
| bf16 | 17.376 | 32.498 | 1.000 | 1.000 |
| tq_4bit_nc | 6.253 | 21.384 | 2.779 | 0.803 |
| tq_k3v4_nc | 5.752 | 20.885 | 3.021 | 0.786 |
| tq_3bit_nc | 5.185 | 20.319 | 3.351 | 0.758 |
| k4v4 | 6.177 | 23.081 | 2.813 | 1.115 |
| k2v4 | 5.103 | 22.008 | 3.405 | 1.146 |
| k2v2 | 4.029 | 20.936 | 4.313 | 1.197 |
| kvq4 | 4.544 | 25.208 | 3.824 | 0.822 |
| kvq3 | 3.775 | 23.606 | 4.603 | 0.880 |
| kvq2 | 3.158 | 22.448 | 5.502 | 1.008 |

At the common point, BF16 L2 hit rate was 3.16%; compressed configurations ranged from 40.43% to 59.60%. SM activity was 15.84% for BF16, 3.27-5.40% for TurboQuant, 14.49-17.74% for KIVI, and 26.60-27.36% for KVQuant. Achieved occupancy was 11.09%, 2.42-2.58%, 17.93-20.99%, and 28.58-30.40%, respectively. These are profiler mechanism observations, not normal benchmark timing.

All ten same-work rows have complete classification and zero unclassified DRAM bytes. No complete-prefix materialization or query-head-expanded K/V signature was observed. Split-KV intermediates, KVQuant sparse value/index/metadata kernels, and deterministic workspace tile/reduce kernels were retained explicitly. Available exact KIVI symbols do not isolate residual-copy traffic, and sink/metadata bytes are not separately identifiable; neither is inferred from allocation.

## Validation and custody

- Phase 15 focused tests: 25/25 PASS.
- R2 transfer tests after the large-object transport remediation: 36/36 PASS.
- Profiler smoke: BF16 B=1/L=4096 Graph Nsys PASS and NCU PASS.
- Local bundle: 893 objects, root `641fc02d8fa598097885b74a336b1b1f454d9844b90025cf0c4b427bee02d5e8`.
- R2: `r2://kvbench-artifacts/kvbench/sha256/641fc02d8fa598097885b74a336b1b1f454d9844b90025cf0c4b427bee02d5e8/`; conditional content-addressed writes and COMPLETE-last PASS; one clean empty-directory retrieval verified all 893 objects and the root.
- Interrupted R2 transfer attempts are preserved in `docs/evidence/phase15/r2-publication-attempt0.json` through `attempt3.json`; no profiler run was repeated.

Phase 16 is **READY**, not started. Full Scan remains **CLOSED**, Quality remains **LOCKED**, and `PERFORMANCE_DATA_FROZEN` remains absent. No final latency, speedup, capacity, knee, HBM-critical-path, or quality claim is made.
