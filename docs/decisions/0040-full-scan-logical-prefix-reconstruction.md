# Decision 0040 — Full Scan logical-prefix reconstruction

- Status: Accepted
- Date: 2026-08-31
- Scope: Phase 16R Full Scan only

The canonical scientific prefix evidence is the deterministic logical input,
not a materialized method-specific KV cache. Its identity binds model and
tokenizer IDs/revisions, batch, configured context label, actual historical
context, token IDs, deterministic positions, input-generator version, input
seed, and token checksum.

Each independent worker allocates fresh caller-owned method cache/workspace
buffers and reconstructs the cache from the same checksum-bound logical input
outside timing. Prefill, quantization, packing, cache population, restoration,
validation decode, and Graph capture remain excluded from measured decode.
Materialized snapshots are optional setup accelerators only when method and
cache-layout fingerprints, model identity, exact batch/context, and input
checksum all match.

A legacy snapshot with parseable metadata but a different layout fingerprint
is `readable_historical_layout` and `restore_incompatible`; the worker rebuilds
from logical input without converting it. Corrupted snapshots and token,
position, shape, or identity mismatches fail closed. Existing historical
snapshots and the stopped Phase 16 attempt remain immutable; its timing is
non-claim-bearing and is not reused.

Completed Full Scan replicate segments become remote-authoritative only after
content-addressed R2 publication, COMPLETE-last verification, and one clean
retrieval. Compact local indexes and receipts are retained before local raw
staging eviction. This changes no adapter, CUDA, cache layout, quantization,
numerical tolerance, graph behavior, or timing boundary.
