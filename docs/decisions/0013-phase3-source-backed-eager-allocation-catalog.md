# Decision 0013: Source-backed Phase 3 eager allocation catalog

- Status: Accepted
- Date: 2026-07-23
- Authority: AGENTS.md, the Phase 3 remediation directive, the Phase 3
  scope-reduction directive, and Decisions 0007, 0009, 0011, and 0012
- Amends: the empty production allocation catalog reserved by Decisions 0009
  and 0011
- Supersedes: none
- Superseded by: none

## Context

Decision 0009 froze the eager scientific criterion but deliberately left its
production event catalog empty until the exact full endpoint supplied complete
allocator history. A no-timing pre-campaign control now executes the retained
BF16 endpoint for the smallest frozen point (fixed-L, B=1, L=128) after the
required warmup. The captured operation has:

- 1,066 allocation lifetimes and 10,960,908 requested bytes;
- complete Python and C++ stacks and proven allocator block sizes;
- 1,066 matching free-requested and free-completed events;
- cache reuse for every allocation;
- zero segment allocations/frees and zero device allocations/frees;
- zero allocated, reserved, device-used, and non-PyTorch persistent deltas;
- no cache-sized or expanded-K/V-sized allocation; and
- no trace-integrity, lifetime, counter, or ring-saturation failure.

The empty catalog correctly classified all 1,066 events as unknown and kept
B-012 open. The raw stacks and the frozen endpoint/model source identify a
deterministic single-token allocation set:

| Source-backed role | Count | B=1 bytes per event |
|---|---:|---:|
| embedding hidden activation | 1 | 8,192 |
| RMSNorm BF16 hidden activations | 130 | 8,192 |
| RMSNorm FP32 hidden activations | 195 | 16,384 |
| RMSNorm FP32 batch scalars | 195 | 4 |
| K/V projection activations | 64 | 2,048 |
| hidden-width linear activations | 96 | 8,192 |
| residual additions | 64 | 8,192 |
| MLP intermediate activations | 128 | 28,672 |
| Flash fixed scalars | 64 | 8 or 16 |
| Flash fixed LSE | 32 | 128 |
| Flash attention output | 32 | 8,192 |
| Flash split-K LSE workspace | 32 | 256 |
| Flash split-K output workspace | 32 | 32,768 |
| vocabulary output | 1 | 256,512 |

The two split-K sizes are not inferred from size alone. Their C++ stacks contain
the frozen public Flash forward, `pytorch_flash::mha_fwd`, and
`pytorch_flash::set_params_splitkv`. The existing paired allocator-control
verifier independently requires the same exact output/LSE formula, split count,
and multiplicity in held-constant GQA and MHA controls.

## Decision

The production catalog is amended with exact policies for the non-workspace
roles above. Each policy freezes:

- one Python function and canonical source suffix;
- one C++ function and canonical source suffix;
- one geometry-derived or exact byte formula;
- one allocation class;
- the exact single-token multiplicity; and
- explicit B, L, H_Q, and H_KV dependency flags.

The geometry formulas are:

- hidden BF16: `B * query_length * H_Q * head_dim * 2`;
- hidden FP32: `B * query_length * H_Q * head_dim * 4`;
- single K or V projection: `B * query_length * H_KV * head_dim * 2`;
- intermediate BF16: `B * query_length * 14,336 * 2`;
- RMSNorm scalar: `B * query_length * 4`;
- Flash LSE: `B * query_length * H_Q * 4`;
- attention output: `B * query_length * H_Q * head_dim * 2`; and
- vocabulary output: `B * query_length * 128,256 * 2`.

The fixed Flash 8-byte and 16-byte entries are framework bookkeeping. All other
non-workspace internal tensors are fixed shared activations except the final
vocabulary tensor, which is fixed output.

An eager production binding must include checksum-bound raw GQA and MHA
allocator controls. The coordinator must independently replay those bytes. A
split-K workspace is admissible only when:

1. both controls pass;
2. their split count, formulas, sizes, and pair multiplicities are identical;
3. the full endpoint contains exactly one matching pair per each of 32 layers;
4. every workspace has the exact frozen Flash C++ stack; and
5. all ordinary eager lifetime, cache-reuse, segment, counter, and memory-delta
   requirements pass.

The catalog is resolved from the frozen operation geometry. It does not encode
one observed context as a universal byte constant. Every preregistered point
must independently match its resolved sizes, stack selectors, multiplicities,
and raw paired controls.

Graph replay is unchanged: `phase3_graph_zero_allocation_v1` permits no
allocation or free event and no allocator/device-memory delta. The eager
catalog cannot waive a graph event.

## Failure semantics

- An absent or failed paired control keeps B-012 open.
- An unrecognized stack, size, count, dependency, or formula is `unknown`.
- Cache growth, expanded K/V, context-scaled unknown storage, segment/device
  allocation, persistent growth, incomplete stacks, or incomplete lifetimes
  remains a strict G1 failure.
- A source, model, backend, build, or policy-catalog digest change invalidates
  the catalog.
- The catalog does not establish B-012 closure by itself. B-012 closes only if
  all pre-campaign eager and graph controls independently replay successfully.

## Consequences

This amendment fills the deliberately empty catalog using observed,
source-backed event evidence. It does not change the model, tokenizer, weights,
dtype, backend, cache semantics, grid, numerical tolerance, graph semantics, or
timing boundary. It authorizes no campaign until all original admission gates
pass and the working tree is clean.
