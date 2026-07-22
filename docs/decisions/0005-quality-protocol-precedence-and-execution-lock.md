# Decision 0005: Quality protocol precedence and execution lock

- Status: Accepted
- Date: 2026-07-22
- Authority: operator instruction dated 2026-07-22, CODEX_WORKFLOW.md,
  CODEX_POST_PERFORMANCE_QUALITY_VALIDATION.md, and
  CODEX_QUALITY_EVALUATION_ADDENDUM.md
- Supersedes: quality scheduling instructions in the older addendum where they
  conflict with the post-performance quality contract
- Superseded by: none

## Context

The quality addendum and post-performance quality contract were supplied after
the formal E00 failure and before any benchmark timing, profiler result,
performance grid, method ranking, or quality result existed. Their initial
SHA-256 identities are:

- `CODEX_QUALITY_EVALUATION_ADDENDUM.md`:
  `62a8978e04732caff101487275d8b22f14358254538a7b377db2153597a1f332`
- `CODEX_POST_PERFORMANCE_QUALITY_VALIDATION.md`:
  `b6178566f239ca6ae598b477754f2ebb9d34d0f44c4fd25593b7ea58aa844620`

The addendum already opens with a notice preserving its scientific quality
requirements while assigning execution order to the post-performance contract.
No `PERFORMANCE_DATA_FROZEN` milestone currently exists.

## Decision

1. `CODEX_WORKFLOW.md` remains authoritative for active performance phases,
   hardware gates, implementation rules, measurement semantics, and
   performance-data governance.
2. `CODEX_POST_PERFORMANCE_QUALITY_VALIDATION.md` is authoritative for quality
   scheduling, performance-data freeze requirements, quality unlock conditions,
   exact-fingerprint joins, and the post-performance quality workflow.
3. `CODEX_QUALITY_EVALUATION_ADDENDUM.md` supplies only non-conflicting quality
   metrics, scientific principles, statistical requirements, gates, and
   reporting requirements. Its older pre-performance or mid-performance
   scheduling instructions are superseded.
4. The two quality documents are preregistered now. This records protocol
   intent only; it does not approve a populated quality contract or create a
   quality result.
5. Quality execution remains `LOCKED` until an explicit, reviewed
   `PERFORMANCE_DATA_FROZEN` milestone proves the performance plan closed and
   checksummed and the later quality contract receives its required approval.
6. While locked, no PPL, NLL, LongBench, LongBench-E, LongBench v2, RULER,
   NIAH, lm-evaluation-harness, or other quality benchmark may run.
7. No quality-only dependency may be installed into the active performance
   environment before the performance freeze. Quality infrastructure must not
   alter the timing hot path or completed performance artifacts.

## Consequences

- Phase 1 remains blocked only on G0/B-002; preregistration does not advance the
  performance phase or authorize Phase 2.
- Quality status remains locked and no performance or quality claim is created.
- When quality execution is later unlocked, results must join frozen
  performance data only by exact method-configuration fingerprint.
