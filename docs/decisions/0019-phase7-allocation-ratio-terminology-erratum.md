# Decision 0019: Phase 7 allocation-ratio terminology erratum

- Status: Accepted
- Date: 2026-07-27
- Authority: Phase 8 KIVI Measurement Adapter authorization, AGENTS.md,
  CODEX_WORKFLOW.md, and the project measurement protocol
- Supersedes: Phase 7 use of the field name `r_alloc` only
- Superseded by: none

## Context

The immutable Phase 7 KIVI reference report and fixture records label
`C_method / C_BF16` as `r_alloc`. The stored byte counts, tensor identities,
checksums, and calculated values are correct; only the ratio name is inverted
relative to the project-wide canonical definition already enforced by the
sample schema.

## Decision

1. Interpret the old Phase 7 `r_alloc` field as `rho_alloc_legacy`. Do not
   rewrite the Phase 7 report, fixture manifests, traces, ledgers, or any other
   historical evidence.
2. Phase 8 and later records must emit both canonical ratios:

       rho_alloc = C_method_allocated / C_BF16_allocated
       r_alloc   = C_BF16_allocated / C_method_allocated

3. The frozen reciprocal absolute tolerance is `1e-9`:

       abs(r_alloc * rho_alloc - 1) <= 1e-9

4. The legacy field must be normalized to `rho_alloc_legacy` before use. It
   must never be consumed directly as `r_alloc`, including by model fitting.
5. The existing global `r_alloc` schema remains unchanged because it already
   uses the canonical definition. Phase 8 compatibility is narrow and
   KIVI-specific.

## Consequences

Phase 7 storage evidence remains valid and checksum-identical. Phase 8 tests
must reject swapped ratios, validate the legacy mapping, and enforce the
reciprocal check. This terminology correction does not authorize fitting,
performance, HBM, capacity, or quality claims.
