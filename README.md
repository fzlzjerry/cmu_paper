# KV compression measurement project

This repository is building a reviewable measurement system for BF16,
TurboQuant, KIVI, and KVQuant KV-cache implementations on one NVIDIA RTX PRO
6000 Blackwell GPU. The scientific objective is to measure same-work decode
latency, allocated cache bytes, and directly observed mechanism data without
mixing incompatible execution modes or overstating the evidence.

## Current state

- Phase 0 repository and input audit: PASS.
- Native-host E00 / G0 certification: PASS for the recorded environment.
- Phase 2: repository contracts, strict schemas, CLI skeleton, and local
  append-only artifact infrastructure.
- G1 through G5: NOT EVALUATED.
- Full scan: CLOSED.
- Quality execution: LOCKED.
- PERFORMANCE_DATA_FROZEN: absent.
- Performance claims: none.
- Quality claims: none.

Phase 2 does not load a model, implement a KV cache, execute a decode runner,
collect latency, invoke a profiler, or run a quality benchmark. Phase 3 may be
proposed only as a separate task after Phase 2 is reviewed.

## Authority

The governing order is:

1. CODEX_WORKFLOW.md controls active performance engineering, experiment
   structure, admission gates, and measurement semantics.
2. CODEX_POST_PERFORMANCE_QUALITY_VALIDATION.md controls quality scheduling,
   performance-data freeze, execution unlock, and exact-fingerprint joins.
3. CODEX_QUALITY_EVALUATION_ADDENDUM.md supplies only non-conflicting quality
   metrics and scientific requirements.
4. AGENTS.md supplies repository-wide research invariants.

Decision 0005 records this precedence. Quality work must not execute until the
post-performance contract's unlock conditions are met.

## Phase 2 interfaces

The intended validation-only entry points are:

    make bootstrap
    make test
    kvbench validate-config configs/plans/smoke.yaml
    kvbench run --plan configs/plans/smoke.yaml --dry-run
    kvbench validate-run <run_dir>
    kvbench summarize <run_dir>

The run command fails closed before model or performance execution because
that execution belongs to Phase 3 or later. The preflight command delegates to
the independently certified E00 launcher; it is never an implicit prerequisite
of a run command.

Later-phase Make targets either validate a dry run or return an explicit
phase-not-implemented error. They do not install packages, download models,
collect timing or profiler output, or invoke quality evaluation.

## Repository map

- configs/ contains versioned contract examples with unresolved identities
  represented explicitly rather than invented.
- src/kvbench/ contains Phase 2 schemas, validation, CLI, command
  reconstruction, and local artifact lifecycle code.
- preflight/ and scripts/preflight.sh contain the certified E00 implementation.
- tests/ contains E00 certification tests and Phase 2 unit/schema/governance
  tests.
- docs/ contains contracts, decisions, status, blockers, risks, and phase
  reports.
- docs/evidence/e00/ contains immutable, Git-tracked E00 evidence.
- artifacts/ is the ignored local raw-run root; only its README is tracked.
- reference/ and calibration/ reserve later correctness-only inputs and frozen
  calibration artifacts.
- analysis/ reserves offline validation and analysis code for later phases.

## Evidence boundaries

Completed E00 evidence must never be changed. General raw artifacts are written
through the Phase 2 append-only API into a caller-selected root, first in a
unique staging directory and then by no-replace atomic promotion. Completed and
failed finalized runs remain immutable through supported APIs.

The local writer does not provide durable remote retention. B-009 therefore
remains open until an append-only durable store, immutable locator scheme, and
publication/retention process are selected and demonstrated. B-010 also remains
open because no digest-pinned measurement container or container-parity G0
exists.

No result is claim-bearing without the provenance required by AGENTS.md and
the experiment contract.
