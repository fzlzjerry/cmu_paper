# Decision 0003: E00 CUDA certification protocol

- Status: Accepted
- Date: 2026-07-22
- Authority: CODEX_WORKFLOW.md Phase 1, AGENTS.md invariant 6,
  and Decision 0002
- Supersedes: none
- Superseded by: none

## Context

Phase 1 requires a minimal PyTorch CUDA extension and native, forced-PTX, and
Compute Sanitizer evidence before any benchmark implementation. The extension
must exercise the Blackwell compile and launch path without introducing model,
cache, attention, timing, or method semantics.

## Decision

1. The certification operation is an out-parameter integer transform named
   `xor_out(input, output)` with the exact elementwise definition
   `output[i] = input[i] ^ 0x5A5A5A5A` over contiguous CUDA int32 tensors. Its
   result has exact CPU-golden semantics and is not a benchmark workload.
2. The binding validates tensor device, dtype, shape, contiguity, and common
   device, then launches on PyTorch's current CUDA stream into the caller's
   preallocated output. The binding and kernel perform no allocation, host
   synchronization, tensor-to-host conversion, or output construction.
3. The collector derives the actual compute capability independently from
   PyTorch and NVIDIA inventory, requires agreement, and compiles both exact
   `sm_<cc>` native code and `compute_<cc>` PTX. The capability is recorded in
   evidence and is not silently hardcoded in source.
4. Fresh subprocesses prove both code paths. The native process sets
   `CUDA_DISABLE_PTX_JIT=1`; the forced-PTX process sets
   `CUDA_FORCE_PTX_JIT=1` and uses a unique initially empty cache or disables
   caching. Conflicting JIT environment variables are removed.
5. Numerical tests include signed int32 extrema, negative and positive values,
   and lengths 1, 255, 256, 257, and 4097. Every result is compared exactly to
   a CPU-computed golden result without NumPy.
6. CUDA Graph certification captures only `xor_out` with fixed pointers,
   replays changed input contents at least three times, and verifies each
   output outside capture. Separate eager and Graph tests require stable
   allocator counters for at least 1,000 post-warmup calls or replays.
7. Compute Sanitizer runs the assertion-based minimal probe under memcheck,
   initcheck, racecheck, and synccheck. Any nonzero exit, reported error, failed
   test, missing run, kernel-image error, or allocation/Graph failure fails G0.
8. No result from this protocol is benchmark timing or supports a performance
   claim. Its only interpretation is certification of the recorded execution
   environment and extension path.

## Consequences

- Any future change to the operation or required checks updates or supersedes
  this decision before the implementation changes.
- Native and PTX success are distinct evidence and cannot substitute for one
  another.
