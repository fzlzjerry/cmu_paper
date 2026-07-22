# Project status

Last updated: 2026-07-22
Authoritative contracts: CODEX_WORKFLOW.md for active performance engineering;
CODEX_POST_PERFORMANCE_QUALITY_VALIDATION.md for post-performance quality
scheduling; CODEX_QUALITY_EVALUATION_ADDENDUM.md for non-conflicting quality
requirements; and AGENTS.md. Decision 0005 records precedence.

## Current state

- Current phase: Phase 3 implementation and bounded admission complete with
  status FAIL. Phase 2 remains PASS. Phase 4 is closed.
- Phase 0 status: PASS
- Phase 1 remediation status: PASS
- Next action: remediate B-011 through B-013 in a new task and new Git SHA,
  then repeat the complete bounded Phase 3 campaigns with new run IDs; do not
  selectively rerun points and do not begin Phase 4
- Active admission gate: native-host G0 PASS; container-parity G0 remains a
  later E01 requirement before E02
- Benchmark implementation changes: exact BF16 static cache, fixed-L and
  growing-context runners, eager and CUDA Graph lanes, timing, allocation,
  telemetry, campaign lifecycle, and source-backed G1 reporting are
  implemented; no custom CUDA/C++ extension was introduced
- CUDA builds or executions: the new formal E00 run passed extension build,
  native execution, forced PTX/JIT, numerical golden, CUDA Graph, allocation,
  SASS/PTX inspection, and all required Compute Sanitizer lanes
- Benchmark, profiler, or quality data produced: 20 checksum-valid Phase 3
  native-host admission runs and one immutable G1 report exist; eight graph
  lanes contain claim-ineligible engineering timing, while profiler and
  quality data remain absent
- Scientific performance claims: none
- Quality protocol: preregistered by Decision 0005 before any performance or
  quality result
- Quality execution: LOCKED; `PERFORMANCE_DATA_FROZEN` is absent
- Quality runs or quality-only dependency installations: none
- Full-scan admission: CLOSED
- Gate state: G0 PASS; BF16 G1 FAIL; G2-G5 NOT EVALUATED

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
Independent rederivation and repository governance validation pass. B-009,
B-010, B-011, B-012, and B-013 remain open.

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
| G1 BF16 baseline | FAIL | 20/20 processes preserved; 19 `gqa_materialization_detected`, one `aborted`; immutable report `phase3-g1-20260722t115413439499z-457123b1-e225cd` |
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

Phase 3 is complete with G1 FAIL. The minimum remediation is to make the
geometry-specific fused native-GQA path directly provable, remove every eager
decode allocation event, and correct the terminal process-monitor race. Any
remediation requires a new Git SHA and complete new bounded campaigns; the
failed evidence remains immutable and no point may be selectively rerun.
B-010 still requires a digest-pinned measurement container and container-parity
G0 before formal E02 closure, ordinary timing, later method admission, or a
performance claim. B-009 still requires durable append-only storage and an
immutable locator/publication mechanism. G2-G5 remain NOT EVALUATED, Full Scan
is CLOSED, quality execution is LOCKED, and `PERFORMANCE_DATA_FROZEN` is absent.
