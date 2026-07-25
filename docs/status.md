# Project status

Last updated: 2026-07-25
Authoritative contracts: CODEX_WORKFLOW.md for active performance engineering;
CODEX_POST_PERFORMANCE_QUALITY_VALIDATION.md for post-performance quality
scheduling; CODEX_QUALITY_EVALUATION_ADDENDUM.md for non-conflicting quality
requirements; and AGENTS.md. Decision 0005 records precedence.

## Current state

- Latest completed prerequisite phase: Phase 6A PASS. The exact authorized
  Measurement Container Docker image ID / OCI image-index digest is
  `sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e`;
  container G0 and both BF16 parity smokes pass, and Decision 0016 binds
  Measurement Lane CUDA execution to that digest only. Phase 5 remains the
  latest completed method reference lane. Official vLLM
  `v0.25.1` commit `752a3a504485790a2e8491cacbb35c137339ad34` is pinned.
- Phase 6 status: BLOCKED at the Compute Sanitizer gate. The minimal adapter,
  static cache, common-runner integration, fixture/path/allocation/Graph
  audits, and focused tests are present. Final run `phase6-20260725t065153714z-ace9261a-083f14-4bit_nc-fixed-l128-eager` passes all three
  fixture audit families but records a nonzero 4bit memcheck leak summary.
  The nine-point bounded grid was not attempted. G2-TQ is BLOCKED by B-018.
- Phase 6A prerequisite status: PASS. B-010 is RESOLVED for the exact Decision
  0016 image after full image identity/layer verification, container G0, and
  separate BF16 eager and CUDA Graph parity runs.
- The synthetic R2 artifact
  `phase6a-r2-synthetic-20260724t135642z`, root SHA-256
  `bbb80210dc729dedc9dd25a24d61cfbedbbe9d05661b1f95e6af278df3d0c11e`,
  remains checksum-valid after a new clean retrieval. Cloudflare REST confirms
  bucket `kvbench-artifacts` is private and covered by enabled indefinite rule
  `kvbench-evidence-indefinite` at exact prefix `kvbench/sha256/`. Container-G0
  root `85e1f49dea76d08b2cba4477d089a71759d529f03b2bc3538da3d15d8639455c`
  was published COMPLETE-last and cleanly retrieved. B-009 is RESOLVED.
- Phase 0 status: PASS
- Phase 1 remediation status: PASS
- Next authorized task: remediate only B-018's sanitizer resource lifecycle,
  then retry the mandatory sanitizer matrix with new run IDs. Phase 7, Pilot,
  Full Scan, profiling, fitting, figures, and quality remain unauthorized
- Active admission gate: native-host G0 PASS; authorized-container G0 PASS;
  native-host BF16 G1 PASS; G2-TQ BLOCKED
- Benchmark implementation changes: exact BF16 static cache, fixed-L and
  growing-context runners, eager and CUDA Graph lanes, timing, allocation,
  telemetry, campaign lifecycle, and source-backed G1 reporting are
  implemented. Phase 4 adds only a thin method adapter, explicit BF16-only
  factory, shared audit facades, and a strict admission report schema. Phase 5
  adds only an isolated upstream TurboQuant reference lane and compact
  fixtures. Phase 6 adds one TurboQuant adapter and static cache through the
  same runners; it is implemented but not admitted
- CUDA builds or executions: the new formal E00 run passed extension build,
  native execution, forced PTX/JIT, numerical golden, CUDA Graph, allocation,
  SASS/PTX inspection, and all required Compute Sanitizer lanes
- Benchmark, performance-profiler, or quality data produced: all earlier Phase 3
  runs, campaigns, and reports remain immutable. The B-015 execution preserved
  one 16-run fixed-L campaign with 13 completed and 3 graph aborts; B-016 added
  only an untimed diagnostic. Execution SHA
  `9def265ab613cde7a06b0e51850f066d0564d635` then preserved two complete new
  campaigns with 20/20 completed runs and immutable FAIL report
  `phase3-g1-20260723t123322160580z-9def265a-08dc69`. Reporting-only commit
  `7f72c95f9932c608f9bd68f1971d6e86378596a2` then published immutable PASS
  report `phase3-g1-20260723t132609515797z-7f72c95f-f31ccb` from those same
  source runs without executing timing. Phase 4 produced exactly three
  checksum-bound functional smoke records at B=1/L=128; they contain no
  latency, independent timing replicates, or formal performance data. No
  profiler campaign or quality evidence was produced. Phase 5 added only
  deterministic reference tensors and kernel-name traces; all profiler
  durations were discarded and no formal timing sample was created. Phase 6A
  added only untimed container certification and parity artifacts. Phase 6
  added only correctness/audit/sanitizer admission evidence; the sanitizer
  failure stopped the grid before any engineering timing. No formal
  performance sample, Nsight result, or quality result was created.
- Scientific performance claims: none
- Quality protocol: preregistered by Decision 0005 before any performance or
  quality result
- Quality execution: LOCKED; `PERFORMANCE_DATA_FROZEN` is absent
- Quality runs or quality-only dependency installations: none
- Full-scan admission: CLOSED
- Gate state: native-host and authorized-container G0 PASS; native-host BF16 G1
  PASS; method-specific G2-TQ BLOCKED; global G2-G5 NOT EVALUATED

## Phase 5 TurboQuant reference lane

The official vLLM repository is pinned at release `v0.25.1`, commit
`752a3a504485790a2e8491cacbb35c137339ad34`, tree
`3ec7a4eb00f9bc8fec399bea6cf7de27a7936372`, under Apache-2.0. Exact Git
blobs and SHA-256 values bind the preset, cache dtype, store, decode, backend,
centroid, and upstream-test sources. The installed wheel runtime files match
the pinned source bytes. No floating branch or local TurboQuant rewrite is
used.

The isolated reference environment records Python 3.12.3, PyTorch
2.11.0+cu130, CUDA 13.0, Triton 3.6.0, vLLM 0.25.1, driver 595.71.05, and the
SM120 GPU. The alternative official vLLM image is pinned by linux/amd64 digest
`sha256:f0b9a0dc75a9fca3b6811e3279367b2d6a448055a000bfd13859587d74cef268`.
This environment is not the Measurement Lane and did not itself close B-010.

With batch 1, 32 query heads, 8 KV heads, head dimension 128, 17 stored
tokens, one append, block size 16, seed 20260724, and BF16 inputs, the official
store/append/decode functions produced three mandatory fixtures and the
same-path optional held-out k8v4 fixture. Actual cache files agree with the
source-derived 134, 118, 102, and 196-byte slots. Kernel-name traces identify
the official MSE/FP8 store and split-KV decode kernels, with no observed
full-prefix dequantization, GQA materialization, or backend fallback in this
minimal path. Direct graph smoke is deferred to Phase 6; upstream declares
`AttentionCGSupport.UNIFORM_BATCH`.

`make reference-turboquant` first published the no-replace set, then a second
identical run returned `verified_existing` without replacing it.
`make validate-reference-turboquant` validates all manifests, 34 root checksum
entries, layouts, actual storage sizes, and claim boundaries. At Phase 5
completion, TurboQuant remained rejected by the Measurement Lane adapter
factory and G2-TQ was `NOT EVALUATED / READY`. The current Phase 6 state below
supersedes that historical entry state.

## Phase 6 retrospective entry-blocked record

Phase 6 was attempted from
`7bccb3217e257d2dbc72deefe8653e9f3556d4f2` and stopped BLOCKED at entry.
Method-specific G2-TQ was BLOCKED because B-009 and B-010 were unresolved;
global G2-G5 were NOT EVALUATED. Provisional plan commit
`1f8e29a8da97e3ad56567c319ec817bec91593be` was completely reverted by
`a9cb4833bfba15a01426bf314c31add7e1c1c698`. No TurboQuant Measurement
Adapter implementation, formal run, performance data, profiler data, or
quality data was retained or created.

The retrospective governance record is
`docs/phase_reports/phase6-turboquant-measurement-blocked.md`. It was created
after the complete revert and is not CUDA, correctness, performance, profiler,
quality, or method-admission evidence. Native-host G0 and native-host BF16 G1
remain PASS, Full Scan remains
CLOSED, quality execution remains LOCKED, and `PERFORMANCE_DATA_FROZEN`
remains absent.

## Phase 6A initial blocked prerequisite attempt

Phase 6A added one digest-pinned Measurement Container definition, an explicit
container mode for the existing E00 implementation, and one Cloudflare
R2-specific publisher/verifier. Unit and repository validation passed, but the
execution host had no Docker/OCI runtime or NVIDIA Container Toolkit. No
Docker image ID or OCI image-index digest, container G0 run, BF16 eager/graph
parity run, or execution authority existed in that attempt.

The bounded synthetic R2 object-path acceptance test passed conditional
creation, exact-existing verification, conflicting-byte rejection, and clean
retrieval at the content-addressed root
`bbb80210dc729dedc9dd25a24d61cfbedbbe9d05661b1f95e6af278df3d0c11e`.
The read-only Cloudflare REST certification returned HTTP 403, so the bucket's
public state and active Bucket Lock rule remain NOT VERIFIED. The required
container-G0 bundle was unavailable for the second publication test. The
historical report is
`docs/phase_reports/phase6a-measurement-container-and-r2-blocked.md`.
At that attempt, B-009 and B-010 remained OPEN.

## Phase 6A remediation

The existing implementation was reused. Docker 29.6.1 and NVIDIA Container
Toolkit 1.19.1 built and verified the exact linux/amd64 Docker image ID / OCI
image-index digest
`sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e`.
Its saved-layer scan found no model weights, operator environment files, or
configured credential bytes. Exact dpkg, Python, CUDA, compiler, profiler-tool,
and executable-hash identities validate against the reviewed container lock.

Container run `e00-20260724T195014.679255Z-a6025ae023e1-23dbe853` passed all
17 G0 checks, including native SASS, forced PTX/JIT, all four Compute Sanitizer
lanes, `sm_120`/`compute_120` inspection, Graph capture/replay, allocation, and
GPU exclusivity. Separate eager and CUDA Graph BF16 B=1/L=128 parity runs pass
with the frozen model, adapter, Flash backend, 32/8 GQA geometry, numerical,
non-materialization, eager-allocation, and zero graph-replay-allocation
controls. They are untimed, non-claim parity artifacts.

Read-only Cloudflare management verification confirms bucket
`kvbench-artifacts` is private and exact enabled rule
`kvbench-evidence-indefinite` covers `kvbench/sha256/` indefinitely. The
existing synthetic root cleanly reverified. Container-G0 root
`85e1f49dea76d08b2cba4477d089a71759d529f03b2bc3538da3d15d8639455c`
was conditionally published with COMPLETE last and cleanly retrieved with all
222 objects. Decision 0016 authorizes Measurement Lane CUDA only inside the
exact image digest. B-009 and B-010 are RESOLVED. The complete report is
`docs/phase_reports/phase6a-measurement-container-and-r2.md`.

At Phase 6A completion, Phase 6 had not yet been restarted and G2-TQ was
`NOT EVALUATED / READY`. The current Phase 6 state below supersedes that
historical entry state.

## Phase 4 common adapter

Clean implementation SHA `0cf160caa532c7cac23275c8a14fd8694789a86f`
places the existing BF16 static-cache and forced-Flash path behind the small
`KVCacheMethod` protocol. Fixed-L and growing-context runners now consume the
common session facade without changing timing boundaries or scientific
semantics. Quantized methods remain `phase_not_implemented`.

`make test`, `make test-cuda` (15/15), and `make test-graph` (4/4) passed.
The three bounded functional smokes passed with new run IDs and valid checksum
ledgers. The strict report at `docs/evidence/phase4/method-admission.json`
retains the historical native-host G0/G1 PASS, G2-G5 NOT EVALUATED, Full Scan
CLOSED, quality LOCKED, B-009/B-010 OPEN state at its publication, and
native-host non-claim status. No decision record was
needed because delegation was mechanical and changed no experiment semantics.

## Phase 3 remediation attempt

Clean execution SHA `7bd6dd48c1d88ac2b61684b02cc636f66b121054`
passed `make checks`, `make test`, `make test-cuda` (12/12), and
`make test-graph` (3/3). The prior 600-file Phase 3 evidence set remained
byte-identical before execution.

Fresh fixed-L campaign
`phase3-20260723t042422417332z-7bd6dd48-8a9cb6` preregistered and attempted
all 16 frozen points once. All 16 finalized `aborted`; no timing was retained.
Six operations completed raw B-011/B-012 audit, directly verified the
`pytorch_flash::flash_fwd_splitkv` GQA/MHA kernel family, found no
materialization/expanded-KV evidence, and passed the frozen eager or graph
allocation criterion. Those six then failed the retained-callable output
equivalence check. Seven runs recorded `owned_worker_failure`, and three
reproduced the registered-PID `[No data]`/missing-`pmon` race.

The campaign and all 16 runs independently validate, are read-only, and have
COMPLETE-last finalization. The stop condition prevented a growing-context
campaign and a new G1 report. No selective rerun occurred. See
`docs/phase_reports/phase3-remediation.md`.

## Phase 3 remediation execution 2

Clean execution SHA `eb908f6e372d6b232e6079e9344c2103bc90cdea` passed
`make checks`, `make test`, `make test-cuda` (13/13), and `make test-graph`
(3/3). Both prior Phase 3 evidence baselines remained byte-identical.

Fresh fixed-L campaign
`phase3-20260723t051939423712z-eb908f6e-b1039a` attempted all 16 frozen points
once and preserved 5 completed plus 11 aborted runs. Fresh growing campaign
`phase3-20260723t052647190745z-eb908f6e-5caf7f` attempted all 4 frozen points
once and preserved 4 aborts. No point was selectively rerun.

The five completed operations directly verified the
`pytorch_flash::flash_fwd_splitkv` GQA/MHA family with no materialization or
expanded-KV evidence. Two eager operations passed the source-backed 1,066-event
criterion and three graph operations retained strict zero allocation. All five
passed frozen numerical controls and exact audit/measured checksum equality.
The registered `compute_apps`/`pmon` race recurred eight times and correctly
joined to `owned_only`; foreign and PID-reuse controls remain fail-closed.
Thus B-013 and B-014 are resolved.

Fifteen workers aborted before measurement. Two preserve an explicit raw-audit
run hard-limit failure; thirteen preserve only the producer wrapper and omit
the lower-level cause. B-015 therefore keeps B-011/B-012 and G1 open.

Reporting-only SHA `3f2c365a5fd495cb3666b421e279b196b58dfb88` published
immutable report `phase3-g1-20260723t060636246041z-3f2c365a-26bf3c`, status
FAIL, SHA-256
`2bc0b4be6c1cc4a723b5b031e56b42520709de2d98cb35917bea857de70412c0`.
Independent validation passes with no errors and `COMPLETE` written last.
Quality remains locked, Full Scan remains closed, and Phase 4 did not begin.

## Phase 3 remediation execution 3 (B-015)

Untimed diagnostics at starting SHA
`8d64c673696ab3c8147310fa09b25217cac5104c` preserved the lower producer
exception, proved that the frozen Flash split heuristic selected 11 partitions
for GQA and 5 for the held-constant MHA geometry, and measured the exact
16-step worst-case raw bundle. Decision 0014 corrected only that disproven
cross-geometry equality and the source-backed raw transport envelope.

Clean execution SHA `52f41ce9d9be4edc07a833e00fe3404fbfa80b89`
passed `make checks`, `make test`, `make test-cuda` (13/13), and
`make test-graph` (3/3). The complete 1,978-file entry Phase 3 artifact baseline
and the original report SHA-256
`060a88283f083e281692a2c471d279da9bfc635e0f513e2dca588ed729d85c7d`
remained unchanged.

Fresh fixed-L campaign
`phase3-20260723t072710859854z-52f41ce9-88e3fd` preregistered and attempted all
16 frozen points once. It preserved 13 completed and 3 aborted runs, with no
unattempted point or selective rerun. All 13 completed operations independently
rederived `gqa_nonmaterialization_verified` for
`pytorch_flash::flash_fwd_splitkv`, found no replication/copy kernel or
expanded-KV allocation, passed frozen numerical/state/checksum controls, and
retained stable cache geometry. All 8 eager operations attributed 1,066 events
each with no forbidden/unknown event; all 5 graph operations had zero allocation
events and deltas.

The graph controls at `B1/L16384`, `B4/L4096`, and `B4/L16384` aborted before
measurement. Each immutable worker log preserves the same exact lower cause:
`ChromeTraceValidationError: graph GPU marker is not contained by its host
marker`. Each registered worker was correctly classified as an owned failure,
reaped with PID start-time protection, and never treated as foreign. This is
B-016; B-013 remains resolved.

Independent validation passed for the campaign and all 16 runs. The 17 new
directories contain 778 checksum-bound files, are read-only, and have
COMPLETE-last finalization; their aggregate manifest SHA-256 is
`7a584e456a253c4d583649a6c19ed538e6a8a1fb10e182ece3b5766467132dee`.
The stop condition prevented a growing campaign and new G1 report. G1 remains
FAIL; quality, Full Scan, pilot, and Phase 4 remained closed.

## Phase 3 B-016 remediation admission

At clean starting HEAD `7c4057c797230e21755812281bcfffe8e7319d5f`, an
untimed forced-Flash `B1/L16384` graph diagnostic reproduced the prior MHA
failure and preserved both raw traces plus the lower parser exception under
`/tmp/phase3-b016-7c4057c-b1-l16384-xbohqv1i`. The MHA GPU annotation extended
150.023 microseconds beyond host return, while the host still contained the
unique `cudaGraphLaunch` and the launch preceded the GPU range. Extending only
an in-memory copy of the host duration recovered the exact two correlated Flash
split-K graph nodes, proving host containment was the sole failed predicate.

Decision 0015 and commit `e7219e0dd714149e3eea783ce7a8602c4bf9bc54`
correct only that asynchronous parser boundary. The parser still requires the
host/GPU marker identity, unique launch, ordering, correlation, stream,
External-ID agreement, one graph ID, unique graph-node IDs, recognized Flash
kernel sequence, and no materialization activity. It now additionally rejects
launch-correlated device events outside the GPU marker and unknown device-like
categories across the union of host and GPU ranges.

The deterministic parser suite, pure replay of the preserved failing raw trace,
and actual long `B1/L16385` MHA graph control pass. `make checks`, `make test`,
`make test-cuda` (14/14), and `make test-graph` (3/3) pass, including strict
graph zero allocation, process ownership, and immutable-evidence checks. B-016
is resolved. B-011/B-012 and G1 remain open pending two entirely new complete
campaigns. No campaign, quality, Full Scan, pilot, or Phase 4 work ran during
this remediation admission.

## Phase 3 remediation execution 4 (complete post-B-016 campaigns)

Execution SHA `9def265ab613cde7a06b0e51850f066d0564d635`
passed `make checks`, `make test`, `make test-cuda` (14/14), and
`make test-graph` (3/3) before campaign admission. The complete 1,978-file
entry baseline, the B-015 778-file aggregate digest, and the original G1
report SHA-256 `060a88283f083e281692a2c471d279da9bfc635e0f513e2dca588ed729d85c7d`
remained unchanged.

Fresh fixed-L campaign
`phase3-20260723t112051327159z-9def265a-aa9c5e` preregistered and completed
all 16 frozen points. Fresh growing-context campaign
`phase3-20260723t121325332843z-9def265a-8fbf6a` preregistered and completed
all 4 frozen points. Each attempted exactly its expected process count, with
no failure, abort, capacity exclusion, unattempted point, or selective rerun.

All 20 source runs and both campaign records independently validate. The 22
campaign/run directories contain 1,368 checksum-ledger entries; every digest
matches, every directory is read-only, no unsafe link exists, and no file is
newer than its `COMPLETE` marker. Report generation revalidated the exact
source-run commitments without mutating them.

Campaign-side independent replay consumed all 80 checksum-bound raw audit
operation bundles rather than trusting worker verdict booleans. All 80 derive
`gqa_nonmaterialization_verified`; both held-constant controls identify the
`pytorch_flash::flash_fwd_splitkv` family and have no GQA failure reason. The
72 eager operations pass `phase3_eager_attributed_ephemeral_v1` with exactly
1,066 attributed events each and no allocation failure reason. All 8 graph
operations pass `phase3_graph_zero_allocation_v1` with zero events.

Append-only report `phase3-g1-20260723t123322160580z-9def265a-08dc69`
is valid, COMPLETE-last, immutable, source-checksum-valid, and SHA-256
`db044273f681bb66f5578c4c19327497302c903f1b4409a08b7b582a2d47ba07`.
It marks G1 FAIL on `no_torch_cat_growth`,
`no_unexplained_measured_region_allocation`, `gqa_not_materialized`,
`graph_replay_no_allocation`, and `no_backend_fallback`. The report derivation
does not consume the consolidated raw audit bundle and instead checks legacy
runtime summaries that are null in all 20 new runs.

This is B-017, a reporting-only raw-evidence join blocker. No campaign point
may be rerun to repair it. Quality execution remains LOCKED,
`PERFORMANCE_DATA_FROZEN` remains absent, Full Scan remains CLOSED, and pilot
and Phase 4 remain unstarted.

## Phase 3 B-017 reporting-only closure

Starting from clean HEAD `9c517ceeec1f9d0587be709166e62cdeca4d6831`, the
original report retained SHA-256
`060a88283f083e281692a2c471d279da9bfc635e0f513e2dca588ed729d85c7d`,
the B-017 FAIL report retained SHA-256
`db044273f681bb66f5578c4c19327497302c903f1b4409a08b7b582a2d47ba07`,
and every entry checksum remained unchanged.

Reporting-only commit `7f72c95f9932c608f9bd68f1971d6e86378596a2`
reuses the existing coordinator raw replay. It reconstructs the execution-SHA
source pin, binds each sidecar to its canonical index, binds every operation
key and declared raw file digest to the selected run, and derives report facts
from local raw replay rather than serialized worker `passed` booleans. The
legacy derivation remains generator-SHA-bound so every older immutable report
still validates under its original semantics. Targeted tests cover local
pass/worker-fail, local-fail/worker-pass, missing files, tampered files, and a
mismatched sidecar/index.

Before publication, `make checks`, `make test` (38 schema, 31 Phase 2, 226
Phase 3, and 167 remediation-control tests), `make test-cuda` (14/14), and
`make test-graph` (3/3) passed. The tree was clean at the report-generator SHA.
No campaign, performance timing, pilot, quality evaluation, Full Scan, or
Phase 4 work ran.

The append-only publisher reused exactly the complete fixed-L campaign
`phase3-20260723t112051327159z-9def265a-aa9c5e` and growing-context campaign
`phase3-20260723t121325332843z-9def265a-8fbf6a`. It created no new run and did
not modify any source run. New no-replace report
`phase3-g1-20260723t132609515797z-7f72c95f-f31ccb` has SHA-256
`c29aef1d9f22b328201599b3e6cdf9efe7c069e78abaf6b37bc3cb12931414c9`.
Independent validation returns `valid=true` with no errors; its ledger passes,
all 20 criteria are PASS, and no payload is newer than `COMPLETE`.

The resulting gate state at that report's publication was G0 PASS,
native-host BF16 G1 PASS, G2-G5 NOT EVALUATED, and Full Scan CLOSED. B-011,
B-012, and B-017 were resolved. At that point B-009 and B-010 still blocked
formal E02 closure, later method execution, ordinary timing, and every
performance claim. Quality execution remains LOCKED and
`PERFORMANCE_DATA_FROZEN` remains absent.

## Repository

The initial non-Git workspace contained three operator-provided inputs but no
implementation. The reviewed Phase 0 records are committed on branch main at
9569d938d9023a3e71d98f12234efa1897004533. The E00 collector and certification
tests are committed at 980eff7b6f5904c4828aa79d684c01a8dc45320d. Formal run
`e00-20260722T041628.190813Z-980eff7b6f59-0dd71f2d` remains immutable FAIL
evidence after `cuobjdump --dump-sass` could not find `nvdisasm`. The quality
protocol was preregistered at 6535a6f6a4e5caa53213e917e9fcf8fc9c0f0190,
and the exact `cuda-nvdisasm-13-0=13.0.85-1` package/tool identity was locked at
6442ba1f7554ea0ebf0b3bb1a920c94567cab689. New formal run
`e00-20260722T050632.375718Z-6442ba1f7554-02d5bd32` finalized as immutable PASS
evidence. At that point B-001, B-002, and B-004 were resolved while B-009 and
B-010 remained open. At that point, no remote was configured.

Phase 2 adds a dependency-free strict schema package, 11 versioned contract
templates, a fail-closed CLI, deterministic command reconstruction, and a
local append-only staging/finalization implementation. Its tests use temporary
roots only. The initial Phase 6A attempt later selected Cloudflare R2 and
exercised one synthetic content-addressed object path, but at that attempt
control-plane lock/public-state certification and the container-G0 publication
test remained incomplete, so B-009 remained open. No built and certified
digest-pinned Measurement Container or container-parity G0 existed then, so
B-010 remained open. The Phase 6A remediation recorded above subsequently
resolved both blockers.

Phase 3 execution used clean SHA
`457123b12220aa4a724968c1b4dd04340cf34a54`. The fixed-L campaign
`phase3-20260722t112917207390z-457123b1-36731e` attempted all 16 frozen
processes; the growing-context campaign
`phase3-20260722t113532869819z-457123b1-694228` attempted all four. Nineteen
runs finalized as `gqa_materialization_detected` because the exact operator
audit could not prove the required fused native-GQA kernel path; one fixed-L
eager run finalized `aborted` after a terminal process-query ambiguity. This
taxonomy is fail-closed: all 79 operator audits recorded no query-head-sized
KV temporary, so Phase 3 does not make a positive physical-materialization
claim. Eleven eager allocation audits also recorded allocator events. All
eight graph replay audits recorded zero allocation events and passed
eager/graph numerical agreement, but those facts cannot override G1.

Reporting-only descendant SHA
`ade0e86d2243ff193f684e008f99f35403dca293` produced immutable report
`phase3-g1-20260722t115413439499z-457123b1-e225cd`, status FAIL, SHA-256
`060a88283f083e281692a2c471d279da9bfc635e0f513e2dca588ed729d85c7d`.
Independent rederivation and repository governance validation pass. At that
immutable report publication, B-009 through B-013 remained open.

At Phase 0 start, the only top-level inputs were AGENTS.md,
CODEX_WORKFLOW.md, and Archive.zip. No implementation, model config, CUDA
extension, Dockerfile, build system, tests, artifact directory, or prior result
was present.

## Inputs

- Archive: /home/rockrock/cmu_paper/Archive.zip
- SHA-256: 20e5b6be5c3060012c48446d1b51067996cd4f13df1d6a73ee8eeb8f855e3ab1
- Contents: 23 PDFs plus 23 AppleDouble metadata files
- Extracted bytes: 123,745,542
- Extraction destination: literature/raw/
- Raw-tree writable entries: zero
- Source archive writable: no
- Checksum records: 47, covering the archive and all 46 extracted files
- Manifest records: 47 data rows
- Archive code executed: none

Static PDF checks found no encryption, declared JavaScript, or embedded files.
qpdf is unavailable; this residual defense-in-depth gap is non-gating and is
recorded in B-008/R-014.

## Source pins and commit plans for later validation

| Source | Exact revision | Phase 0 role |
|---|---|---|
| vLLM v0.25.1 | 752a3a504485790a2e8491cacbb35c137339ad34 | TurboQuant source/reference candidate |
| KIVI | 876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6 | post-paper official-repository candidate; equivalence unresolved |
| KVQuant | 57a238357f0ffe50084670fcd5781c9848f80ea2 | official-paper calibration/reference candidate |
| lm-evaluation-harness | c9bbec6e7de418b9082379da82797522eb173054 | direct KIVI Reference Lane dependency |

These are exact source pins, not admission decisions. KVQuant's three embedded
Transformers-derived trees are fixed by outer-commit tree hashes, while exact
upstream lineage remains unresolved; LOCK.json also assigns commit-resolution
plans to the GPTQ, GPTQ-for-LLaMA, and SqueezeLLM attributions. During the
Phase 0 audit, no upstream setup, binary, macro, kernel, or benchmark was
executed. Its temporary source snapshots were used only for read-only
inspection and are outside the repository. Phase 5's later bounded vLLM
reference execution is recorded separately above.

## Phase and gate ledger

| Phase/gate | Status | Evidence |
|---|---|---|
| Phase 0 repository/input audit | PASS | literature manifests; method notes; source lock; decision, risks, blockers, tasks |
| G0 native-host hardware certification | PASS | `docs/evidence/e00/e00-20260722T050632.375718Z-6442ba1f7554-02d5bd32/`; prior immutable FAIL retained |
| Phase 2 repository/contracts/tooling | PASS | strict schemas and examples; fail-closed CLI; append-only local writer; 54 Phase 2 tests; repository checks |
| G1 BF16 baseline | PASS — native_host_admission only | Native-host report `phase3-g1-20260723t132609515797z-7f72c95f-f31ccb` independently replays the unchanged 20 runs and 80 operations and passes all 20 criteria. Phase 6A eager/graph artifacts establish container parity only; no new unified or claim-bearing G1 result was created. |
| Phase 4 common method adapter | PASS | BF16 delegates through `KVCacheMethod`; fixed-L/growing, allocation, graph, and path checks pass; `docs/evidence/phase4/method-admission.json`; no quantized method implemented |
| Phase 5 TurboQuant reference lane | PASS | Exact vLLM v0.25.1 source/environment lock; 3 mandatory and 1 held-out deterministic fixtures; official store/append/decode paths; no measurement adapter or timing |
| Phase 6 TurboQuant measurement adapter | BLOCKED | Minimal adapter and static cache are implemented; all mandatory fixture audits pass, but final Compute Sanitizer evidence fails before the bounded grid. |
| Phase 6A Measurement Container and R2 prerequisites | PASS | Exact image built and scanned; container G0 and both BF16 parity smokes PASS; private R2 state and indefinite lock verified; synthetic and 222-object G0 roots cleanly retrieved; Decision 0016 accepted. B-009/B-010 RESOLVED. |
| G2-TQ | BLOCKED | B-018: final 4bit memcheck reports 2,093,260 leaked bytes in 28 allocations; no mandatory configuration is admitted. |
| G2-KIVI | NOT EVALUATED | requires E07-E08 |
| G2-KVQ | NOT EVALUATED | requires E09-E11 |
| G1-G5 unified admission | NOT EVALUATED | requires E12 |
| Pilot/full-scan gates | CLOSED / NOT EVALUATED | Phase 6 stopped before its bounded grid; no Pilot or Full Scan is authorized |
| Post-performance quality validation | LOCKED | Decision 0005; `PERFORMANCE_DATA_FROZEN` absent |

## Phase 0 acceptance

- Unknown archive code was not executed: pass.
- Every supplied archive/extracted input has a SHA-256 record: pass.
- Every directly fetched source/dependency has an exact commit; every currently
  identified embedded or attributed repository has an explicit commit-resolution
  plan: pass.
- Risk coverage includes CUDA compatibility, Graph support, GQA replication,
  full-prefix dequantization, legacy dependencies, and OOM: pass.
- Required status, risk, blocker, decision, method-note, provenance, and E00-E18
  planning records exist: pass.
- Graph A/B ownership and ignored-artifact audit policy are explicit: pass.

## Next action

Phase 6A is PASS and Decision 0016 authorizes Measurement Lane CUDA only in
the exact recorded image digest. The Phase 3 campaigns, all 20 runs, every
failed report, the PASS report, Phase 4 evidence, Phase 5 fixtures, the Phase 6
retrospective, and the initial blocked Phase 6A report remain unchanged.

The next action is the minimum B-018 remediation: under the unchanged
authorized image and pinned source, release every sanitizer-probe CUDA
allocation before teardown and obtain unique zero ERROR and LEAK summaries for
all three mandatory configurations using new run IDs. Only then may the frozen
bounded grid be attempted. Native-host and authorized-container G0 plus
native-host BF16 G1 remain PASS; G2-TQ is BLOCKED; global G2-G5 remain NOT
EVALUATED; Full Scan remains CLOSED; quality execution remains LOCKED; and
`PERFORMANCE_DATA_FROZEN` remains absent. Phase 7, Pilot, profiling, fitting,
figures, and quality execution remain unauthorized.
