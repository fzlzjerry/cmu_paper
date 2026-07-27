# Phase 8 KIVI Measurement Adapter

## Authority and scope

Phase 8 starts at clean commit
`8d6d766a34a15bd40bd42cc47c5482b0dd052cc0`. CUDA execution is authorized
only in Measurement Container
`sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e`;
the image will not be rebuilt, mutated, or supplemented. Phase 9, KVQuant,
Pilot, Full Scan, profiling, fitting, figures, and quality evaluation remain
out of scope.

## Pinned source and kernel seam

Conformance authority is the checksum-bound patched official KIVI source:
official commit `876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6`, base tree
`c94c31b2cfd44eeb9a18cff9dcdf03adff4ac49b`, Decision 0018 patched tree
`b617493dea5aff1a754cd27ad6be12ac512b2aee`, patch SHA-256
`c9c2dd52d4c81b844d1d1d7218ad2cd60a5b31574a387f716d466cb01310423d`,
and extension SHA-256
`45d29ec1a3cecc4b253d1d1dd6139ef4f91cff88993db61a9d73685314851aa9`.
The frozen binary is extracted from the verified Phase 7 reference OCI image
`sha256:f27e4cdef6bd15f18ab76b1fe0e4413ede004b42538c74e3dd90d04172406f75`,
checksum-verified host-side, and mounted read-only into the authorized
Measurement Container. It imports there against the pinned Torch/CUDA
runtime. A clean build from the same source inside the Measurement Container
is not execution authority if its ELF bytes are not bit-identical; only the
frozen checksum-matching extension may execute admission operations.

The thin wrapper directly reuses `bgemv2_kernel_outer_dim` and
`bgemv4_kernel_outer_dim` from `quant/csrc/gemv_cuda.cu` (blob
`1294414332d90f5fcd7523db89a8ad5e763547c4`, SHA-256
`850b9a33b1e320b3d77dab61a423cea6c73473ab7f33b064dd2e34a682dfef63`)
and the official min/max and pack kernels from `quant/new_pack.py` (blob
`72380af9dcc931547367deb00117dc2cbd5d1ebf`, SHA-256
`3678af0e34a0ba18e5d80a4128acf11d4070667c800a15540a16d07253a4f75e`).
History is stored in extension-ready layout so the upstream
transpose/`contiguous` launcher is not used. Because the frozen pybind wrapper
allocates its return tensor, the thin Python wrapper launches the exact
registered kernels from that checksum-bound ELF directly into caller-owned
storage. The bound host-stub offsets are `0xa970` for
`bgemv4_kernel_outer_dim` and `0xab70` for
`bgemv2_kernel_outer_dim`; the extension digest and method fingerprint bind
them. Canonical logits, kernel output, merge, decode output, and all other
workspace are preallocated. This changes neither kernel binary nor
quantization math. Any unknown allocation, graph replay allocation, kernel
substitution, or required CUDA source/ABI change blocks the phase.

## Static cache and rollover

Add one KIVI cache state and one adapter for `k4v4`, `k2v4`, `k2v2`, plus the
held-out `k4v2`. Geometry is fixed at `B=1,H_Q=32,H_KV=8,D=128`, with
`group_size=32`, `residual_length=32`, and `kv_head=query_head//4`.

Preallocate quantized K/V history, K/V scales and additive minima, K residual,
a KIVI-specific indexed V residual, an ordered V residual staging view, FP16
query/K/V/metadata/output staging, canonical logits/softmax/merge/output
workspace, and all fixed index tensors for the declared maximum context.
K residual tokens are quantized as one 32-token group after the L=32 decode.
V uses a 32-slot KIVI-only circular index; after the L=33 decode the oldest
token is quantized and the logical residual becomes tokens 1..32. Precomputed
Python-side slice ranges copy the logical V order into fixed staging without
`torch.cat`, growth, host synchronization, or token duplication. Fixed-L
scratch never mutates historical state.

The semantic boundary is fixed and fingerprinted:

    BF16 common-runner input
      -> in-place copy/cast into FP16 staging
      -> official KIVI CUDA operations and FP16 residual math
      -> in-place copy/cast into the BF16 common output

No complete-prefix FP16/BF16 tensor or H_Q-sized K/V tensor is allowed.

## Accounting, conformance, and admission

Report actual allocated capacity and active source-faithful storage separately,
including packed K/V, scales/minima, other metadata, residual K/V, FP16
staging, padding/alignment, block/group rounding, persistent workspace, and
temporary peak. Emit canonical `rho_alloc=allocated/BF16_allocated` and
`r_alloc=BF16_allocated/allocated`, enforce their product within `1e-9`, and
leave `r_hbm` null. Block-rounded values are tested at L=31,32,33,64,128,4096.

Replay all four frozen Phase 7 configurations and require exact inputs, store,
append, packing, metadata, residual, rollover, byte, source, patch, and
extension identities. Freeze adapter decode and eager/graph comparison at
`atol=0.02, rtol=0.02` before bounded admission; it may not be loosened later.
Reuse the common fixed-L/growing runners, graph harness, execution/allocation
audits, artifact lifecycle, process supervision, and R2 publisher. Add only an
explicit factory mapping and narrow admission entrypoints.

After fixture, path, allocation, fixed-L graph, and minimal 2-bit/4-bit
sanitizer checks pass, run only the preregistered bounded grid: L=128
eager/graph for all three mandatory configurations; L=4096 eager/graph and
growing L=31,O=4 eager for k4v4; and L=128 eager for held-out k4v2. Compute no
speedup. G2-KIVI can pass only after a complete immutable bundle is published
content-addressed with COMPLETE last and cleanly retrieved from R2. Global
G2-G5 remain not evaluated, Full Scan remains closed, and quality remains
locked. Phase 9 is explicitly deferred.
