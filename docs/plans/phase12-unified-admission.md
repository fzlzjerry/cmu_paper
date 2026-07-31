# Phase 12 — Unified Admission Gates

## Scope and authority

Phase 12 aggregates the existing BF16, TurboQuant, KIVI, and KVQuant
method-admission evidence and runs only the common G5 reproducibility matrix.
Adapters, CUDA sources, cache layouts, tolerances, calibration, fixtures,
method-specific admission evidence, and the common runners remain unchanged.
Phase 13, Pilot, Full Scan, profiling, fitting, figures, and quality evaluation
are deferred.

All CUDA execution uses the authorized Measurement Container:

`sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e`

The ten main configuration IDs and frozen method fingerprints are:

| Phase 12 ID | Factory variant | Method fingerprint |
| --- | --- | --- |
| `bf16` | `bf16` | `81ca6a0d74727a9c8a54f14d6d222dee502ee5e9a9ab6e8c9a03b4fed98f371b` |
| `tq_4bit_nc` | `turboquant_4bit_nc` | `5b56167c81ef2042be5fa45ed4e7f8ddc60670d5611283e19e848307076c27eb` |
| `tq_k3v4_nc` | `turboquant_k3v4_nc` | `ff92cd334059888584564bb84999353a698baf3b991a602a827a53aab726908e` |
| `tq_3bit_nc` | `turboquant_3bit_nc` | `0d17950c2502fe7399cf5a896efa429861eb53194060ab95a7369287889dc49a` |
| `k4v4` | `k4v4` | `97289ed9c875e27013ddcf7659fc6e849b3d438c58d0d86bd3dcac5d82eefb09` |
| `k2v4` | `k2v4` | `568493e09cad122088716533c954beb6b25a01209fa28016f761b2ede4930a3f` |
| `k2v2` | `k2v2` | `667395cefa882efc7c54f9088e3706dcdc3ba33c8734bdf8de9e0dd8ae1124b8` |
| `kvq4` | `kvq4` | `8f3ea4f49056a5c4ada715a853ec506de4b6bcab262cfd88dc5796bacc032fa0` |
| `kvq3` | `kvq3` | `2f0d1a99db2e6884745b6cd54c50eedfa17744b89a3e8b2ffd840986126bd802` |
| `kvq2` | `kvq2` | `eb75d6cbf8ff27365cd2799c4e0232649c94d6f094cb4d041bbe8c3ac1cda5ee` |

TurboQuant `k8v4` and KIVI `k4v2` remain held-out controls and are excluded
from G5 and the main Pilot set.

## Existing G1–G4 evidence

The aggregator verifies the exact report bytes, every selected evidence
reference and checksum, and the recognized method-specific checks. It does not
copy or regenerate raw evidence.

| Method | MethodAdmissionReport | SHA-256 | G1–G4 authority |
| --- | --- | --- | --- |
| BF16 | `docs/evidence/phase4/method-admission.json` | `1362fd1817b8bb5706baaa09ed6e5115789fbc4d35d394f184d0b132a0e58d22` | Phase 3 G1 and Phase 4 checks, plus exact Phase 6A eager/graph authorized-container artifacts and the Decision 0026 endpoint transition |
| TurboQuant | `docs/evidence/phase6/turboquant-method-admission.json` | `388e8107b649a9093491699357c8b1ad1d8e12c8c75378bce658f8a09bf9ab2a` | frozen fixture/decode, byte/static-allocation, direct-path/GQA, and CUDA Graph checks, plus the Decision 0026 endpoint transition |
| KIVI | `docs/evidence/phase8/kivi-method-admission.json` | `3a4b63b9da0eab12db9a916ebdc1cffd788ea6f93678d87964a8332ae7cec83a` | Decision 0028 historical replay plus fixture/token, byte/static-allocation, direct-path/GQA, and CUDA Graph checks |
| KVQuant | `docs/evidence/phase11rq23/kvquant-method-admission.json` | `9cfed618cee9514a1071392d0a2dca327dcf6acd33d81ac72cc477c7880c09e2` | Decision 0029 current-source fixture/sparse/sink/store/append, byte/static-allocation, direct-path/native-GQA, and CUDA Graph checks |

The durable TurboQuant, KIVI, and KVQuant outer-bundle receipts remain
checksum-bound input evidence. The KVQuant numerical oracle is
`kvqref-2e0a0e9022c50cbc6fb497d88cae973e`, root
`c28682d58706b58812dc1db69ba5eb4982339ba13f39bf67f751794cdaabfdec`;
superseded Phase 10 kvq3 packed bytes are not used.

For BF16, the aggregator semantically replays the exact Phase 6A eager and
graph artifacts at their recorded execution commit and verifies their
authorized-container digest, adapter identity, byte accounting, allocation,
and graph results. For BF16 and TurboQuant, it resolves the historical model
endpoint, adapter, and session sources as exact Git blobs at their recorded
execution commits, verifies Decision 0026 as the recognized backward-compatible
endpoint transition, and separately proves the method adapter/cache authority
remains unchanged. Historical endpoint bytes are not compared directly with
current HEAD.

KVQuant execution is additionally bound to Decision 0029, source commit/tree
`34b0bdfa83082e1f30387d9ac5cca369006e089c` /
`1f85af65fe03061583ffe8bd91e47d7ecffdd312`, aggregate patch
`7b9d3cc6773e8ef37697601c885f2c5ec581dffd57cf59424d03e68f147bd55a`,
extension
`b3c33badb8e55b19d6b2ce535182e964ce51e5102d8413b29701dd3d817ad73d`,
and Q23 evidence root
`8b65112ea2d49b58ee07c1533b429fac1a8af7466e09adad073d9a22ae2ec790`.

## G5 campaign

One new append-only campaign is reserved at
`artifacts/phase12/<campaign_id>/`. Each run has a unique child directory under
`runs/`; per-configuration normalized references are written under
`admission/`; aggregate material is written under `unified/`. Existing campaign
or run IDs are refused. Any interrupted campaign is terminally sealed under its
reserved ID, retained, and never resumed or selectively retried.

Every run uses the unchanged common fixed-L runner and:

- `runner_kind=fixed_l`
- `graph_mode=cuda_graph`
- `batch_size=1`
- `context_length=4096`
- `warmup_steps=64`
- `measured_steps=128`
- the common runner's frozen `measured_batches=5`
- three independent child processes per configuration

The Phase 12 coordinator has one private fixed-point session binding for this
exact matrix. It constructs only the frozen factory adapters and static caches,
then delegates graph capture, allocation audit, and measurement to the existing
utilities. It does not alter or bypass the historical Phase 3/6/8/11 grids.
Every configuration and replicate uses the same token recipe:
`prefix[i] = (i + 12000) % 120000 + 1000` for 4096 positions and
`decode[0] = (0 + 12000 + 4096 + 257) % 120000 + 1000`.

The blocked-randomized process orders are frozen before execution:

| Replicate | Seed | Order |
| --- | ---: | --- |
| 0 | 20260730 | `k2v4`, `k2v2`, `kvq3`, `tq_3bit_nc`, `bf16`, `tq_4bit_nc`, `kvq4`, `k4v4`, `tq_k3v4_nc`, `kvq2` |
| 1 | 20260731 | `tq_k3v4_nc`, `k2v4`, `k2v2`, `kvq4`, `kvq2`, `tq_4bit_nc`, `kvq3`, `tq_3bit_nc`, `k4v4`, `bf16` |
| 2 | 20260732 | `tq_k3v4_nc`, `tq_4bit_nc`, `k4v4`, `bf16`, `k2v4`, `k2v2`, `tq_3bit_nc`, `kvq2`, `kvq3`, `kvq4` |

The randomization seed changes order only; all processes use the same frozen
deterministic model inputs. Before each child process, the coordinator verifies
GPU exclusivity, container and source authority, and the unused run ID. Compile,
autotune, cache setup, graph capture, allocation audit, and 64 warmups occur
outside timing. R2 credentials are never passed into the container.

For each configuration, G5 uses the three independent process medians and
reports median, minimum, maximum, mean, sample standard deviation, coefficient
of variation, temperature/clock/power ranges, and output/kernel/allocation
agreement. The frozen criterion is `CV = standard_deviation / mean <= 0.03`.
Kernel-path agreement binds each method's immutable G3 audit and report hash to
the current runtime backend identity, adapter/layout identity, native-GQA
geometry, observed graph fallback verdict, replay-allocation audit, and the
actual retained CUDA Graph topology. The Phase 12 worker temporarily requests
`keep_graph=True` only while calling the existing common capture harness, then
restores the constructor. Before and after measurement it writes the graph's
debug DOT, strips process-local graph IDs and pointer handles, and requires the
normalized node/kernel/edge topology checksum and graph-exec pointer to remain
stable. The normalized topology checksum must also agree across the three
independent processes. This is a non-timing CUDA Graph topology witness, not
profiler evidence.
Exactly three completed processes, stable output checksums, stable kernel paths,
stable allocation, no fallback, GPU exclusivity failure, or NaN/Inf are also
required.

No selective rerun or majority vote is permitted. A point failure preserves the
campaign and blocks G5. A machine-level interruption may only be retried as an
entire new 30-run campaign with a new campaign ID.

## Reports and closure

The coordinator creates checksum-bound per-config reports in the raw campaign.
The finalized campaign is the sole Phase 12 publication bundle. It contains all
30 raw run records, per-configuration reports, the local unified admission
candidate, inventory, checksum ledger, and `COMPLETE`. The candidate keeps
publication `PENDING`, global G5 `NOT_EVALUATED`, and Pilot `NOT_READY`; it does
not claim closure before its exact root is durable.

The campaign is published once with `COMPLETE` last and then retrieved exactly
once into a new empty directory. After that clean retrieval, the external
receipt and final JSON/Markdown derived from the candidate plus that receipt are
recorded at:

- `docs/evidence/phase12/unified-admission.json`
- `docs/phase_reports/phase12-unified-admission.md`
- `docs/evidence/phase12/r2-publication.json`

Global G1, G2, G3, or G4 passes only when all ten recognized configurations
pass its existing evidence contract. Global G5 passes only when all ten pass
the common campaign and the single campaign publication and clean retrieval
validate. The receipt remains external to avoid a self-referential root digest
and authorizes deterministic derivation of final G5 `PASS` and Pilot `READY`
reports.

Only a full G0–G5 PASS marks Pilot `READY`. Full Scan remains `CLOSED`, quality
execution remains `LOCKED`, `PERFORMANCE_DATA_FROZEN` remains absent, and no
speedup, HBM, knee, capacity, performance, or quality claim is made. Phase 13
is explicitly deferred to a separate task.
