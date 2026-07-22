# Decision 0006: Phase 2 schema tooling and local artifact boundary

- Status: Accepted
- Date: 2026-07-22
- Authority: CODEX_WORKFLOW.md Phase 2, AGENTS.md, and the operator's Phase 2
  execution instruction
- Supersedes: none
- Superseded by: none

## Context

The certified E00 virtual environment is an exact 35-distribution closure.
Its lock validation rejects added packages, and it contains neither Pydantic
nor a YAML parser. Installing Phase 2 dependencies into that environment would
invalidate the environment identity supporting native-host G0. No separate,
digest-pinned measurement environment exists because B-010 remains open.

Phase 2 also needs a general local append-only writer, while B-009 separately
requires a selected durable append-only store and immutable locator/publication
mechanism. A local writer alone cannot satisfy that operational requirement.

## Decision

1. Phase 2 uses frozen, typed standard-library models with strict explicit
   parsing and validation rather than Pydantic. Unknown fields are rejected,
   canonical JSON is deterministic, and every schema-bearing document carries
   a semantic schema version.
2. Versioned `.yaml` templates use the JSON-compatible YAML 1.2 subset. This is
   valid YAML while remaining parseable without adding a dependency to E00.
3. `pyproject.toml` declares no runtime dependency. Phase 2 tooling runs with a
   system Python and `PYTHONPATH=src`; `make bootstrap` verifies prerequisites
   but never installs packages. The E00 `.venv`, its requirements lock, and
   `make preflight` remain unchanged.
4. The general artifact writer is implemented independently under
   `src/kvbench/runtime/`. The certified E00 writer is not refactored or made to
   depend on new Phase 2 code.
5. The Phase 2 writer provides local staging, exclusive creation, checksums,
   inventory, completion marker, no-replace atomic promotion, and immutable
   supported APIs. It does not claim durable publication or remote retention.
6. B-009 stays open for durable-store selection, immutable locator publication,
   and demonstrated retention. B-010 stays open for a digest-pinned measurement
   container and identical container-parity G0.

## Consequences

- Phase 2 can be tested without mutating the certified E00 Python closure or
  installing quality-only or general project dependencies.
- Broader YAML syntax may be introduced only with a separately locked Phase 2
  environment; existing JSON-compatible YAML remains canonical and valid.
- Local append-only tests support the Phase 2 lifecycle acceptance criteria but
  do not admit G1-G5, close the full scan, or support a performance claim.
- Container construction and parity certification remain later, separately
  reviewed work.
