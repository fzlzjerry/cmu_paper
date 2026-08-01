# Decision 0030 — Compressed static-cache batch geometry

- Status: Accepted
- Date: 2026-08-01

## Context

The first Phase 13 Pilot attempt stopped before timing because the admitted
TurboQuant, KIVI, and KVQuant cache states rejected `B=4` and `B=8`.  Their
CUDA paths already expose native batch indexing or can consume fixed
caller-owned per-batch views; the blocker is adapter/cache geometry rather
than quantization or kernel mathematics.

## Decision

The nine compressed main configurations admit only `B in {1,4,8}` with
preallocated native-eight-KV-head batch storage.  TurboQuant receives separate
packed block banks per batch, KIVI folds batch into its existing head-major
GEMV ABI, and KVQuant invokes its unchanged caller-owned APIs on deterministic
per-batch cache banks.

No CUDA source, quantization, packing, sparse selection, capacity, sink,
calibration, fixture, GQA, runner, tolerance, or timing-boundary change is
authorized.  `B=1` numerical results remain the oracle and must be preserved.
Any discovered need for a CUDA change blocks Phase 13B.

Successor admission reports must bind the new adapter/cache fingerprints and
the exact Decision 0016 container.  Historical admission evidence and the
stopped Phase 13 campaign remain immutable.  A later Pilot must use a fresh
append-only campaign ID.
