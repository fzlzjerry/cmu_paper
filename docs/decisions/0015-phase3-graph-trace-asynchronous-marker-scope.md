# Decision 0015: Phase 3 CUDA Graph asynchronous marker scope

- Status: Accepted
- Date: 2026-07-23
- Authority: AGENTS.md, the Phase 3 remediation directive, the Phase 3 scope
  reduction directive, and Decisions 0008, 0012, 0013, and 0014
- Amends: the CUDA Graph host/GPU marker relation in Decision 0008
- Supersedes: no device-kernel, allocation, graph, or experiment semantic gate
- Superseded by: none

## Context

The immutable B-015 fixed-L campaign attempted all 16 frozen points once.
Thirteen completed, while the graph controls at `B1/L16384`, `B4/L4096`, and
`B4/L16384` stopped before measurement with:

`ChromeTraceValidationError: graph GPU marker is not contained by its host marker`

The collector wraps `captured_graph.replay()` in a CPU
`torch.autograd.profiler.record_function`, exits that host range after the
asynchronous `cudaGraphLaunch` returns, and synchronizes only afterward. The
old parser nevertheless required the complete GPU annotation to end before
the host range ended.

An untimed diagnostic at clean HEAD
`7c4057c797230e21755812281bcfffe8e7319d5f` reproduced the frozen
`B1/L16384` control with forced Flash, BF16, one query token, three control
warmups, and CUDA Graph replay. No normal benchmark timing was collected. The
raw files and summary are preserved under:

`/tmp/phase3-b016-7c4057c-b1-l16384-xbohqv1i`

The summary SHA-256 is
`bfc5f8d2762b62fd052a76a365b68ccb38c490ced1ddaeb62e0630781abe15c6`.
The GQA trace SHA-256 is
`b8dedf12670346a0341ae55efbb904ef0dedda4bc995a217ee1f79b8bfde57eb`;
its GPU marker completed inside the host range and the old parser passed. The
MHA trace SHA-256 is
`94ce52163c07f86df64552358ba0e27d858b8d412d7f71c572e319af4f6641fe`.
Its host range still contained exactly one `cudaGraphLaunch`, and the launch
ended before the matching GPU marker began. The GPU marker then extended
150.023 microseconds beyond the host return, reproducing the campaign error.

Changing only an in-memory copy of that host duration so it contained the GPU
range made the old parser pass. It recovered exactly two device events:
`pytorch_flash::flash_fwd_splitkv_kernel` followed by
`pytorch_flash::flash_fwd_splitkv_combine_kernel`. Both used stream 7,
correlation 343, graph ID 5, and distinct graph-node IDs. No copy, repeat,
expand, or other device event was exposed. Thus host containment was the sole
failed predicate.

The immutable passing `B1/L4096` MHA trace shows why the predicate was
accidental: its GPU marker ended only 7.998 microseconds before the CPU marker.
Whether asynchronous graph execution fits inside unrelated CPU profiler
overhead depends on geometry and scheduling, not on dispatch correctness.

## Decision

1. A CUDA Graph trace must retain exactly one host/GPU annotation pair with the
   same marker and External ID.
2. The host annotation must contain exactly one `cudaGraphLaunch` on its CPU
   process/thread.
3. The launch must finish before the GPU annotation begins.
4. The GPU annotation may finish after the host annotation because
   `cudaGraphLaunch` is asynchronous.
5. Every device event inside the GPU annotation must match the launch
   correlation, GPU stream, External-ID presence/value, graph ID, and a unique
   graph-node ID.
6. Any device event with the launch correlation outside the GPU annotation is
   a hard failure, including after the host range ends.
7. Unknown device-like categories are rejected across the union of the host
   launch interval and asynchronous GPU interval.
8. All kernel-family, no-preceding-materialization, raw-byte, source, shape,
   allocation, and graph-zero-allocation requirements remain unchanged.

## Consequences

This is a parser-boundary correction for asynchronous CUDA semantics. It adds
no schema, evidence kind, execution path, timing result, or gate waiver.
B-016 can be resolved only after deterministic parser tests, the actual long
CUDA Graph control, and all repository/CUDA/graph admission suites pass.
B-011, B-012, and G1 remain open until two entirely new complete campaigns
independently validate every operation.
