# Decision 0014: Phase 3 B-015 raw-audit bounds and split controls

- Status: Accepted
- Date: 2026-07-23
- Authority: AGENTS.md, the Phase 3 remediation directive, the Phase 3
  scope-reduction directive, and Decisions 0007, 0009, 0012, and 0013
- Amends: the cross-geometry split-count equality in Decision 0013 and the
  raw-audit run-size transport bound
- Supersedes: no experiment semantic, scientific gate, or immutable evidence
- Superseded by: none

## Context

The immutable post-remediation campaigns retained five completed runs and
fifteen pre-measurement aborts. Two aborts exposed `ValueError: raw audit run
size exceeds the hard limit`; thirteen exposed only the wrapper
`Phase3RawAuditProducerError: raw-audit collection did not complete before
measurement`. B-015 therefore required an untimed worst-case diagnostic before
another campaign could be admitted.

At execution Git SHA `8d64c673696ab3c8147310fa09b25217cac5104c`, an untimed direct producer run
for the preregistered worst-case growing point (`B=4`, `L=4096`, 16 decode
steps, eager) preserved the underlying step-zero exception:

`WorkerProtocolError: paired allocator controls did not verify`

The exact `partial/error.json` is 118 bytes with SHA-256
`6f04a6cefb6d4453293d030508ec66752c9a410e1ae281019b2b4dc1e8faabbd`.
The diagnostic summary has SHA-256
`bd16fd4ac3947aab54f5b344549a5a4ccbde526842cb494f7cb35b151fd55ca5`.
No timing was collected.

An untimed interception preserved both raw allocator controls and independently
replayed them. Each control passed its own complete verifier. The GQA control
selected 11 split-K partitions, with one 720,896-byte output accumulator and
one 5,632-byte LSE allocation. The held-constant MHA control selected 5
partitions, with one 327,680-byte output accumulator and one 2,560-byte LSE
allocation. The old paired verifier failed only because it required the two
geometries to select identical split counts and sizes. The diagnostic summary
has SHA-256
`b8305a15062a14c78cff4899d17f91a4c911e6e612d9b959c0cbf8f1526f92f7`.

The frozen `libtorch_cuda.so` contains
`pytorch_flash::num_splits_heuristic(int, int, int, int)` and
`pytorch_flash::set_params_splitkv(..., cudaDeviceProp*)`. Its executed split
selection is geometry- and device-dependent. Equal batch, context, dtype,
query length, backend, execution mode, build, and GPU therefore do not imply
an equal split count when the K/V-head geometry changes from 8 to 32.

With only that disproven cross-geometry equality removed diagnostically, all
16 worst-case operations completed while both controls still passed
independently. The full worker-preamble diagnostic produced 97 files totaling
998,314,455 bytes; its largest file was 26,722,385 bytes. Its summary SHA-256
is `ecb88701a5d1a4611d76e78e90dd96f96dc8541c59f658e4a0a0e044781695fd`.
It was explicitly not admission eligible and collected no timing.

Across immutable completed Phase 3 source operations, the largest declared
single-operation bundle is 71,766,119 bytes. The frozen growing runner has at
most 16 operation records per process. Rounding the observed operation maximum
up to 72 MiB and multiplying by the frozen operation count yields a source- and
measurement-backed run envelope of 1,207,959,552 bytes (1,152 MiB).

## Decision

1. The GQA and MHA allocator controls must each independently pass the existing
   formula, stack, lifetime, cache-reuse, segment/device-allocation, allocator
   counter, and memory-delta checks.
2. Each geometry must contain exactly one internally matching split-K output
   accumulator/LSE pair. A missing, duplicate, malformed, or unmatched pair
   remains a failure.
3. The two different geometries are no longer required to choose the same
   split count or workspace sizes. Non-split known formulas, held constants,
   query identity, backend identity, runtime identity, and recorder identity
   must still match.
4. Because the production endpoint is GQA, only the independently verified GQA
   split count is bound into the exact full-endpoint production replay. Every
   production layer must still contain one exact output/LSE workspace pair for
   that count and the frozen Flash stack.
5. `MAX_RAW_AUDIT_RUN_SIZE_BYTES` is 1,207,959,552 bytes. The 256 MiB per-file,
   512-file, and 16 MiB index limits are unchanged.
6. A failed producer record must preserve a bounded machine-readable form of
   the underlying exception. Admission errors must identify its decode step
   and failure reason. The raw `partial/error.json` continues to preserve the
   human-readable exception type and message.

## Scientific gates retained

- Forced Flash with fallback rejection and raw CUDA device-kernel proof.
- No preceding repeat, expand-copy, H_KV-to-H_Q copy, or equivalent
  materialization kernel.
- No expanded-K/V allocation, cache growth, context-scaled unknown, unknown
  event, segment/device allocation, persistent growth, or incomplete evidence.
- Source, shape, stride, cache-view, and native-KV-storage evidence.
- Strict zero allocation for graph replay.
- Exact endpoint, measured region, model, tokenizer, weights, dtype, backend,
  grid, numerical tolerances, timing boundaries, and process supervision.

## Consequences

This decision changes an evidence transport bound and removes one disproven
cross-geometry equality. It does not make B-011, B-012, B-015, or G1 pass.
Those gates can close only after targeted tests and all CPU/CUDA/graph admission
controls pass, the tree is committed and clean, and entirely new complete
fixed-L and growing campaigns independently validate every raw operation. A
failure at any admission gate stops remediation without campaign execution.
