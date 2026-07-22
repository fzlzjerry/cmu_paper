# Project status

Last updated: 2026-07-22
Authoritative contract: CODEX_WORKFLOW.md plus AGENTS.md

## Current state

- Current phase: Phase 0 complete
- Phase 0 status: PASS
- Next action: review and commit Phase 0 records, then Phase 1 / E00 preflight
- Active admission gate: G0 not started
- Benchmark implementation changes: none
- CUDA builds or executions: none
- Benchmark or profiler data produced: none
- Scientific performance claims: none

## Repository

The initial non-Git workspace contained three operator-provided inputs but no
implementation. It was initialized as a Git repository on branch main and
remains unborn: there is no initial commit, HEAD SHA, remote, or tracked file.
This is acceptable for the repository/input audit. B-001 requires a reviewed
initial commit before E00 produces durable gate evidence or any later run
begins.

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
| G0 hardware certification | NOT STARTED | E00 is next |
| G1 BF16 baseline | NOT EVALUATED | requires G0 and E01-E04 |
| G2-TQ | NOT EVALUATED | requires E05-E06 |
| G2-KIVI | NOT EVALUATED | requires E07-E08 |
| G2-KVQ | NOT EVALUATED | requires E09-E11 |
| G1-G5 unified admission | NOT EVALUATED | requires E12 |
| Pilot/full-scan gates | NOT EVALUATED | no method admitted and no timing collected |

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

Review and commit the Phase 0 records to close B-001, then run Phase 1 exactly
as E00: collect the hardware/toolchain/container manifest, compile and execute
only the minimal CUDA certification extension, verify forced PTX/JIT and
Compute Sanitizer, and stop if G0 fails. E01 must implement the append-only
artifact policy and close B-009. Do not begin BF16 or method implementation
before G0 passes.
