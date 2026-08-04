# Decision 0035 — Phase 13 prefix-equivalence CUDA-context isolation

- Status: Accepted
- Date: 2026-08-04
- Scope: Phase 13 Pilot setup supervision only

## Context

The stopped campaign
`phase13-20260804t091434161290z-91b98406-97ac82` completed the 30-case
batch-exact prefix-equivalence matrix in the campaign coordinator.  The
coordinator then retained a process-level CUDA context even after releasing
sessions and calling `torch.cuda.empty_cache()`.  The following prefix-builder
idle-GPU snapshot correctly failed closed before prefix construction or Pilot
timing.

## Decision

The complete 30-case equivalence gate executes in one dedicated child process
inside the exact authorized Measurement Container.  The parent coordinator
does not import CUDA for the gate, load the model, or own a CUDA context.

The parent requires an idle snapshot before launch.  After the child exits it
requires the exact 30/30 Decision 0034 evidence to validate, the child exit to
be successful, and a second idle snapshot with no remaining, foreign, or
unknown CUDA process.  Malformed or altered evidence, child failure, timeout,
residual context, or foreign GPU activity fails closed.  The coordinator is
not allowlisted, and `torch.cuda.empty_cache()` is not an isolation mechanism.

## Preservation

This is a harness-only process-boundary correction.  It changes no adapter,
CUDA source, cache layout, grid, feasibility classification, prefix-state
semantics, timing boundary, calibration, fixture, or admission evidence.  All
stopped Phase 13 campaigns remain immutable and non-claim-bearing.  A passing
remediation permits only a wholly new append-only Pilot campaign.
