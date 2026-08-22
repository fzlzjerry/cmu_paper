# Phase 13 — Fresh Successor Pilot Scan

## Authority and boundary

This campaign reuses the admitted common fixed-L runner, timing, CUDA Graph,
telemetry, supervision, artifact, analysis, and R2 paths without changing an
adapter, CUDA kernel, cache layout, numerical tolerance, or timing boundary.
All CUDA work runs in Measurement Container
`sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e`.
Phase 12 and the Phase 13B successor admissions prove G1-G4; Decision 0036 and
MethodAdmissionReport SHA-256
`75605637f460a309081e1e0a4065e90e8e14194d365250cb513092616ef89ec7`
are the current `kvq4` workspace authority. The refreshed global evidence keeps
G0-G5 PASS and Pilot READY.

Decisions 0037 and 0038 govern setup-only prefix construction.  The stopped
`phase13-20260821t151036274823z-b27442c4-8a39d6` coordinator was terminated
only after `prefix-kvq4-b8-l16384` finalized `COMPLETE`.  Its 168 fully
checksum-validated states are frozen as local seed
`phase13-prefix-seed-20260822t185619z-b27442c4-168`.  A fresh campaign
hardlinks those immutable states into a new persistent catalog and constructs
only the 60 absent KVQuant states; no historical timing sample is reused.
KVQuant setup packing uses one fixed 128-token tile and never materializes a
complete-prefix FP32 Key or Value copy.  The tile scratch is included in the
same conservative 0.88 memory calculation and changes no adapter fingerprint,
cache layout, formal decode, grid, or timing boundary.  Only the disposable
prefix-builder children bind `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
before CUDA initialization; formal timing workers retain their frozen allocator
environment.  The child records both allocated and reserved peaks and fails if
either exceeds the unchanged 0.88 limit.

The stopped successor attempt
`phase13-20260822t130430172378z-356ee0ab-86ac68` remains local, immutable, and
non-claim-bearing.  It completed no formal timing run and is not resumed or
used as timing input.

The immutable blocked campaign
`phase13-20260804t111810342595z-a127b0d1-8649c3` is historical failure evidence
only. Its root
`c5523a894bc38f41b45b6bdf38ad437fb0b28fc29db4a959b4a592fdd825277b`,
3,246 objects, 69 completed, one runtime failure, and 614 aborted records are
not modified, republished, resumed, or used as timing input.

The ten main configurations and fingerprints are frozen:

| Configuration | Fingerprint |
| --- | --- |
| `bf16` | `81ca6a0d74727a9c8a54f14d6d222dee502ee5e9a9ab6e8c9a03b4fed98f371b` |
| `tq_4bit_nc` | `5b56167c81ef2042be5fa45ed4e7f8ddc60670d5611283e19e848307076c27eb` |
| `tq_k3v4_nc` | `ff92cd334059888584564bb84999353a698baf3b991a602a827a53aab726908e` |
| `tq_3bit_nc` | `0d17950c2502fe7399cf5a896efa429861eb53194060ab95a7369287889dc49a` |
| `k4v4` | `97289ed9c875e27013ddcf7659fc6e849b3d438c58d0d86bd3dcac5d82eefb09` |
| `k2v4` | `568493e09cad122088716533c954beb6b25a01209fa28016f761b2ede4930a3f` |
| `k2v2` | `667395cefa882efc7c54f9088e3706dcdc3ba33c8734bdf8de9e0dd8ae1124b8` |
| `kvq4` | `27b3af27e153491112ef974ea3a6d813987bfcb5083a699ce131fa3332c9703b` |
| `kvq3` | `2f0d1a99db2e6884745b6cd54c50eedfa17744b89a3e8b2ffd840986126bd802` |
| `kvq2` | `eb75d6cbf8ff27365cd2799c4e0232649c94d6f094cb4d041bbe8c3ac1cda5ee` |

TurboQuant `k8v4` and KIVI `k4v2` remain excluded held-out controls.

## Frozen campaign

The grid is `fixed_l`, `cuda_graph`, B=`1,4,8`, context labels
`4096,8192,16384,24576,32768,49152,65536,98304,131072`, 64 warmups,
128 measured operations, and three independent processes: exactly 810 planned
records. Context is historical prefix length; only label 131072 maps to
historical 131071 so total attended length is 131072.

Feasibility is recomputed for every new record from current fingerprints and
actual owned-byte formulas. It includes model weights, cache payload and
metadata/residual/sink/sparse regions, persistent workspace, Decision 0036 q4
capacity workspace, Decision 0037's fixed KVQuant prefix tile, end-to-end
prefix-construction peak, Graph reserve, and the frozen 0.88 memory limit. The
result is 684 feasible launches and 126 explicit
`capacity_infeasible` records. Known-infeasible records are never launched or
removed.

The complete order is frozen in
`docs/plans/phase13-pilot-execution-order.json`: file SHA-256
`d64fc06cda5d6f594ea82247eca5b3a400b39dc32605b93f308461591641d35a`
and ordered-record SHA-256
`3fcf59663ce7bc9379e4a6b16c9b50839aca251fdbe7d35a1b1cf87c15ab3205`.
Block orders are:

- seed 20260805: `k2v4, bf16, k2v2, tq_4bit_nc, kvq4, tq_3bit_nc, k4v4, kvq2, tq_k3v4_nc, kvq3`;
- seed 20260806: `tq_3bit_nc, tq_4bit_nc, kvq4, kvq2, kvq3, bf16, tq_k3v4_nc, k2v2, k4v4, k2v4`;
- seed 20260807: `kvq4, k4v4, tq_4bit_nc, k2v2, tq_k3v4_nc, kvq2, tq_3bit_nc, kvq3, k2v4, bf16`.

Within each block the feasible B/L points are deterministically randomized.
The order, seeds, code, and execution HEAD cannot change after the first run.
Exact configuration x batch x context prefix snapshots restore only into the
same geometry outside timing. Each process still allocates fresh caller-owned
cache/workspace buffers; no cache, pointer, or Graph object is shared.

Every result is retained. A method/runtime failure stops the campaign and
marks the scheduled remainder `aborted`; there is no selective retry. A genuine
machine-wide invalidation preserves this campaign and requires a wholly new ID.

## QC, analysis, and closure

For each feasible point, QC uses the three process medians, sample standard
deviation, and `CV = sd / mean`; stable means `CV <= 0.03`. It also requires
finite and agreeing outputs, kernel path and allocation agreement, GPU
exclusivity, and no fallback. Non-monotonic latency is warning-only.

Stable process observations feed only constant, linear, and
`max(tau, a+sL)` provisional fits with process-bootstrap intervals when
estimable. A knee is never forced; fit failure and insufficient density are
explicit. Same-work ratios are Pilot-only, quality-unvalidated, and
claim-ineligible. `r_hbm` remains null.

One new append-only tree under `artifacts/phase13/<campaign_id>/` retains all
810 records, raw samples, feasibility, QC, provisional fits, exclusions, and
simple Pilot plots. It is finalized with `COMPLETE` last, conditionally
published to R2, and cleanly retrieved once into an empty directory. Phase 13
PASS makes Phase 14 READY only. Phase 14, Full Scan, profiling, final figures,
and quality execution remain deferred and closed.
