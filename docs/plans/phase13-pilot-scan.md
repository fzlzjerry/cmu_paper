# Phase 13R — Fresh Pilot Scan after Phase 13B

## Frozen scope and authority

Phase 13 reuses the unchanged common fixed-L runner, timing implementation,
CUDA Graph harness, telemetry, process supervision, admitted adapters, artifact
lifecycle, and R2 publisher. All CUDA work runs in Measurement Container
`sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e`.
Adapters, CUDA, cache layouts, calibration, fixtures, tolerances, and timing
boundaries remain unchanged. Phase 14, Full Scan, profiling, and quality work
are deferred. The stopped campaign
`phase13-20260801t080641686374z-009dfd71-14b7e3` remains immutable and is never
resumed, amended, or used as timing input; Phase 13R always creates a fresh
append-only campaign ID.

Decision 0030 and the checksum-bound Phase 13B successor reports are the static
B=1/4/8 execution authority: TurboQuant
`49799ef89646ec008a530c5180fdcef6cd4af9ca0d5772fe2b01d6e775e3b1c0`, KIVI
`1e91730ac56af37e03d80edce7979a509d52049428faad89f61e61dc6bd48c51`, and
KVQuant `e1cee8e1c514f9cf6323b5e710480c1fefab2804e5f4eafe6c473b29f4768481`.
Their durable root is
`f1c96eaacbbace1c23b249d1afe8d892aa26c3f6b8d04e07f373a2becafba1fe`.
Each compressed worker binds the applicable successor report, admitted batch,
adapter version, current source hashes, and live capacity-specific layout
fingerprint. B=1 preservation remains part of that authority.

The main IDs and Phase 12 configuration fingerprints are exactly:

| ID | Fingerprint |
| --- | --- |
| `bf16` | `81ca6a0d74727a9c8a54f14d6d222dee502ee5e9a9ab6e8c9a03b4fed98f371b` |
| `tq_4bit_nc` | `5b56167c81ef2042be5fa45ed4e7f8ddc60670d5611283e19e848307076c27eb` |
| `tq_k3v4_nc` | `ff92cd334059888584564bb84999353a698baf3b991a602a827a53aab726908e` |
| `tq_3bit_nc` | `0d17950c2502fe7399cf5a896efa429861eb53194060ab95a7369287889dc49a` |
| `k4v4` | `97289ed9c875e27013ddcf7659fc6e849b3d438c58d0d86bd3dcac5d82eefb09` |
| `k2v4` | `568493e09cad122088716533c954beb6b25a01209fa28016f761b2ede4930a3f` |
| `k2v2` | `667395cefa882efc7c54f9088e3706dcdc3ba33c8734bdf8de9e0dd8ae1124b8` |
| `kvq4` | `8f3ea4f49056a5c4ada715a853ec506de4b6bcab262cfd88dc5796bacc032fa0` |
| `kvq3` | `2f0d1a99db2e6884745b6cd54c50eedfa17744b89a3e8b2ffd840986126bd802` |
| `kvq2` | `eb75d6cbf8ff27365cd2799c4e0232649c94d6f094cb4d041bbe8c3ac1cda5ee` |

TurboQuant `k8v4` and KIVI `k4v2` remain excluded held-out controls.

## Pilot design

The grid is `fixed_l`, `cuda_graph`, batches `1,4,8`, context labels
`4096,8192,16384,24576,32768,49152,65536,98304,131072`, 64 warmups,
128 operations per measured batch, the unchanged five raw measured batches,
and three independent processes. This creates exactly 810 planned records.
The fixed-L runner defines context as historical prefix length, so only label
131072 maps to historical prefix 131071 and total attended length 131072.

Before CUDA, every record receives a source-formula allocation prediction:
model weights + adapter-owned cache/workspace + a conservative graph reserve
scaled from the checksum-bound Phase 12 point. The limit is the frozen
`0.88 * 101970345984` bytes. The source formulas replay the exact Phase 13B
B=1/4/8 owned allocations at L=128 before extrapolation. Known
capacity-infeasible records are preserved and not launched; all ten
configurations are admitted at B=1/4/8, and geometry rejection is never
relabeled as a memory failure.

The complete immutable order is
`docs/plans/phase13-pilot-execution-order.json`, SHA-256
`e82c2ba5502a373989e6e82fa03fb8a30e4f104aa70320c9f4ff9cf87f0c0342`;
its ordered-record digest is
`eed1165b6cff332d9bce5264807d4eb47a72e74b8c4475ed1fb52cb8d7b1dc37`.
Configuration block orders are:

- seed 20260801: `tq_3bit_nc, kvq2, k2v2, kvq4, tq_4bit_nc, k2v4, bf16, tq_k3v4_nc, k4v4, kvq3`;
- seed 20260802: `kvq4, k4v4, k2v4, kvq2, tq_k3v4_nc, k2v2, tq_4bit_nc, bf16, kvq3, tq_3bit_nc`;
- seed 20260803: `k2v2, tq_4bit_nc, kvq3, k4v4, kvq4, kvq2, bf16, k2v4, tq_3bit_nc, tq_k3v4_nc`.

Within each block all 27 `(B,L)` records are deterministically randomized in
the committed JSON. The order cannot change after execution begins. A normal
point failure is retained. An implementation-boundary failure stops the Pilot,
marks all remaining planned records `aborted`, and requires a separate method
phase; it is never selectively retried. A machine-wide invalidation requires a
fresh campaign ID and complete affected-replicate rerun.

## QC, provisional analysis, and closure

For each complete point, QC uses the three process medians, sample standard
deviation, `CV=sd/mean`, and the frozen `CV <= 0.03` stable criterion. It also
requires output, kernel path, allocation, finite-value, exclusivity, and
fallback agreement. Latency non-monotonicity is warning-only. `r_hbm` remains
null.

Stable session observations feed only the preregistered constant, linear, and
`max(tau,a+sL)` provisional fits. A knee is not forced; fit failures and
insufficient span remain explicit. A deterministic 1,000-draw bootstrap
resamples independent process medians within each observed context; it never
resamples individual decode operations. Near-knee diagnostics use
`[0.75 L_star,1.25 L_star]`. Same-work ratios require matching BF16 work and
are labeled Pilot-only, quality-unvalidated, and claim-ineligible.

One append-only bundle under `artifacts/phase13/<campaign_id>/` retains all 810
records, raw samples, feasibility, summaries, provisional fits, exclusions,
and simple plots. It is finalized with inventory, checksum ledger, and
`COMPLETE` written last, then conditionally published to R2 and cleanly
retrieved once. Phase 13 PASS makes only Phase 14 ready; Full Scan remains
closed, quality remains locked, and no final speedup, HBM, capacity, quality,
or knee claim is authorized.
