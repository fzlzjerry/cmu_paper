# Decision 0038 — Phase 13 prefix-child allocator binding

- Status: Accepted
- Date: 2026-08-22
- Scope: disposable, untimed KVQuant prefix-builder children only

## Context

The first Decision 0037 successor attempt preserved 168 checksum-bound prefix
states and proved the bounded q4 B=8/L=16384 builder byte-exact, but the later
q4 B=8/L=49152 setup reached about 91,987 MiB in the caching allocator.  That
exceeded the frozen 0.88 limit even though the fixed 128-token KVQuant scratch
was only 11,369,600 bytes.  The excess was allocator reservation fragmentation
during full-prefix model setup, not cache, workspace, quantization, or decode
geometry.

## Decision

Every disposable KVQuant prefix-builder child, including the q4 exact-state
oracle child, binds:

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

The binding is established before the child imports CUDA and is verified again
inside prefix construction.  It is not inherited by formal timing workers,
does not alter the Measurement Container, and does not change the frozen
feasibility classifications or 0.88 limit.  Missing, changed, or differently
scoped allocator authority fails closed.

The exact-container q4 B=8/L=49152 diagnostic recorded peak allocated bytes
76,017,167,360 and peak reserved bytes 89,347,063,808 against the unchanged
89,733,904,465-byte limit.  The completed state SHA-256 is
`e70762a2de64ebd5fac187e5ffa7bf1f3e898317a09a4dc49779e8871d0e473a`;
the append-only diagnostic is retained under
`artifacts/phase13_prefix_catalogs/diagnostics/phase13-prefix-memory-diag-20260822t140409z-356ee0ab-q4b8l49152-expandable/`.

## Preservation

This decision changes no adapter, CUDA source, cache layout, quantization,
input, grid, feasibility result, decode path, timing boundary, or formal timing
allocator.  The stopped campaign
`phase13-20260822t130430172378z-356ee0ab-86ac68` and all earlier stopped
campaigns remain immutable and excluded from timing analysis.  A successor
campaign must use a new ID and may reuse only the frozen 168-state seed.
