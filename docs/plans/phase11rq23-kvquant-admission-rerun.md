# Phase 11R-Q23 KVQuant Admission Rerun

## Scope and authority

Re-admit the existing static KVQuant Measurement Adapter after Decision 0029.
The execution source is `kvquant_gqa_longctx_deterministic_q23_v4`, commit
`34b0bdfa83082e1f30387d9ac5cca369006e089c`, tree
`1f85af65fe03061583ffe8bd91e47d7ecffdd312`, aggregate patch SHA-256
`7b9d3cc6773e8ef37697601c885f2c5ec581dffd57cf59424d03e68f147bd55a`,
and extension SHA-256
`b3c33badb8e55b19d6b2ce535182e964ce51e5102d8413b29701dd3d817ad73d`.
Calibration root
`8148306d08205af376994b022f189a0d6837915cd279ca8af6b104e1f4b46ccf`
and corrected fixture root
`c28682d58706b58812dc1db69ba5eb4982339ba13f39bf67f751794cdaabfdec`
remain read-only. All CUDA execution uses authorized Measurement Container
digest
`sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e`.

This phase changes admission bindings and evidence only. It does not change
the Adapter, cache, CUDA source, fixtures, calibration, existing methods,
authorized Measurement Container, or Decisions 0021 through 0029.

## Admission

Replay all nine corrected fixtures and the existing byte, execution-path,
native-GQA, allocation, non-default-stream, CUDA Graph, pointer-stability, and
Compute Sanitizer controls. Run the unchanged nine-point bounded grid with
fresh run IDs:

- `kvq4`, `kvq3`, and `kvq2`: fixed-L 128 eager and CUDA Graph;
- `kvq4`: fixed-L 4096 eager and CUDA Graph;
- `kvq4`: growing context from 17 for four eager appends.

The current-source G2-KVQ gate remains not evaluated until every check passes.
No speedup, comparative latency, HBM, capacity, profiling, or quality claim is
permitted.

## Append-only closure

Write a new MethodAdmissionReport and receipts under
`docs/evidence/phase11rq23/`. Preserve the historical Phase 11R report,
MethodAdmissionReport, local roots, R2 receipts, the Phase 11D-Q23 evidence,
and the stopped Phase 12 staging tree byte-for-byte. Use fresh append-only
inner and outer run IDs, publish with conditional content-addressed writes and
`COMPLETE` last, then perform one clean retrieval and checksum verification.

Phase 12, Pilot, Full Scan, profiling, fitting, figures, and quality execution
remain deferred.
