# Project status

Last updated: 2026-07-23
Authoritative contracts: CODEX_WORKFLOW.md for active performance engineering;
CODEX_POST_PERFORMANCE_QUALITY_VALIDATION.md for post-performance quality
scheduling; CODEX_QUALITY_EVALUATION_ADDENDUM.md for non-conflicting quality
requirements; and AGENTS.md. Decision 0005 records precedence.

## Current state

- Current phase: Phase 3 remediation FAIL after both complete fresh campaigns
  were attempted. Phase 2 remains PASS, and Phase 4 is closed.
- Phase 0 status: PASS
- Phase 1 remediation status: PASS
- Next action: remediate B-015 without weakening B-011/B-012, preserve the
  lower-level raw-audit producer cause, establish a source-backed worst-case
  bundle bound, rerun admission controls, then use a new clean SHA and entirely
  new complete campaigns; do not begin Phase 4
- Active admission gate: native-host G0 PASS; container-parity G0 remains a
  later E01 requirement before E02
- Benchmark implementation changes: exact BF16 static cache, fixed-L and
  growing-context runners, eager and CUDA Graph lanes, timing, allocation,
  telemetry, campaign lifecycle, and source-backed G1 reporting are
  implemented; no custom CUDA/C++ extension was introduced
- CUDA builds or executions: the new formal E00 run passed extension build,
  native execution, forced PTX/JIT, numerical golden, CUDA Graph, allocation,
  SASS/PTX inspection, and all required Compute Sanitizer lanes
- Benchmark, profiler, or quality data produced: the original 20 checksum-valid
  Phase 3 runs/report and first-remediation 16-run failed campaign remain
  immutable. The latest execution preserved 20 new runs, 2 campaigns, and a
  new immutable FAIL report; 5 runs completed and 15 aborted before measurement.
  No profiler or quality evidence was produced.
- Scientific performance claims: none
- Quality protocol: preregistered by Decision 0005 before any performance or
  quality result
- Quality execution: LOCKED; `PERFORMANCE_DATA_FROZEN` is absent
- Quality runs or quality-only dependency installations: none
- Full-scan admission: CLOSED
- Gate state: G0 PASS; BF16 G1 FAIL; G2-G5 NOT EVALUATED

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
evidence. B-001, B-002, and B-004 are resolved; B-009 and B-010 remain open. No remote
is configured.

Phase 2 adds a dependency-free strict schema package, 11 versioned contract
templates, a fail-closed CLI, deterministic command reconstruction, and a
local append-only staging/finalization implementation. Its tests use temporary
roots only. The local controls do not select a durable backing store or an
immutable publication locator, so B-009 remains open. No digest-pinned
measurement container or container-parity G0 exists, so B-010 remains open.

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
plans to the GPTQ, GPTQ-for-LLaMA, and SqueezeLLM attributions. No upstream
setup, binary, macro, kernel, or benchmark was executed. Temporary source
snapshots were used only for read-only inspection and are outside the
repository.

## Phase and gate ledger

| Phase/gate | Status | Evidence |
|---|---|---|
| Phase 0 repository/input audit | PASS | literature manifests; method notes; source lock; decision, risks, blockers, tasks |
| G0 native-host hardware certification | PASS | `docs/evidence/e00/e00-20260722T050632.375718Z-6442ba1f7554-02d5bd32/`; prior immutable FAIL retained |
| Phase 2 repository/contracts/tooling | PASS | strict schemas and examples; fail-closed CLI; append-only local writer; 54 Phase 2 tests; repository checks |
| G1 BF16 baseline | FAIL | Original and first-remediation evidence remain immutable; latest complete campaigns attempted 20/20 points, preserved 5 completed plus 15 aborted runs, and produced independently valid FAIL report `phase3-g1-20260723t060636246041z-3f2c365a-26bf3c` |
| G2-TQ | NOT EVALUATED | requires E05-E06 |
| G2-KIVI | NOT EVALUATED | requires E07-E08 |
| G2-KVQ | NOT EVALUATED | requires E09-E11 |
| G1-G5 unified admission | NOT EVALUATED | requires E12 |
| Pilot/full-scan gates | CLOSED / NOT EVALUATED | BF16 G1 failed; no method admitted; graph timing is non-claim admission evidence only |
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

Phase 3 remains G1 FAIL. B-013 and B-014 are resolved without changing the
frozen process-exclusivity or numerical contracts. The minimum remaining
remediation is B-015: preserve each raw-audit producer exception and establish
a source-backed worst-case evidence bound so all B-011/B-012 operations can
complete. Any new execution requires passing all admission controls, a clean
new Git SHA, and both entirely new complete campaigns; every existing result
remains immutable and no point may be selectively rerun.
B-010 still requires a digest-pinned measurement container and container-parity
G0 before formal E02 closure, ordinary timing, later method admission, or a
performance claim. B-009 still requires durable append-only storage and an
immutable locator/publication mechanism. G2-G5 remain NOT EVALUATED, Full Scan
is CLOSED, quality execution is LOCKED, and `PERFORMANCE_DATA_FROZEN` is absent.
