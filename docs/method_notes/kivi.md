# KIVI source note

Status: Phase 7 reference lane PASS and Phase 8 static Measurement Adapter
PASS at method-specific G2-KIVI under checksum-bound Decision 0018
patched-source authority.

## Paper and source

- Title: KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache
- Local input: literature/raw/2402.02750.pdf
- Version: arXiv:2402.02750v2, 2024-07-25
- Paper SHA-256: df31ef32d71bfb280c533c5db8220cadf5ef42076bf45d82ba4c8da8e50ea5f4
- Paper-provided repository: https://github.com/jy-yuan/KIVI
- Pinned commit: 876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6
- Pinned tree: c94c31b2cfd44eeb9a18cff9dcdf03adff4ac49b
- Commit authored: 2025-11-20, after arXiv v2 dated 2024-07-25
- Selection rationale: exact official-repository snapshot visible during the
  audit and the default branch that advertises Llama 3/GQA support. Decision
  0017 rejects the older `develop` and `lmeval` heads as substitutes. Decision
  0018 authorizes one checksum-bound project patch on that exact commit after
  a fresh remote-ref audit found no newer official revision. This selection is
  patched official source and is not presumed to be paper-era equivalent.
- License: MIT
- Paper-reported backend/hardware: Hugging Face Transformers with custom
  quantization/GEMV paths; efficiency results on one NVIDIA A100 80GB GPU.

## Algorithm

KIVI quantizes keys per channel, grouping along the token dimension, because
key outliers are channel-stable. It quantizes values per token, grouping along
the head-dimension direction. A recent unquantized window is retained at full
precision. When the residual window fills, a block is quantized and moved into
historical storage.

The workflow's initial canonical settings are group_size=32 and
residual_length=32. The paper's main experiments commonly use group size 32
and residual length 128; its appendix reports residual-length-32 results. The
project setting is therefore an explicit contract choice, not an inferred
upstream default.

The Phase 8 mandatory variants are K4/V4, K2/V4, and K2/V2, with K4/V2
retained as the held-out K/V-asymmetry control.

## Storage and metadata

Phase 7 identified, and Phase 8 accounts for, these source-owned components:

- packed historical key data;
- packed historical value data;
- per-group scale and minimum/zero metadata;
- recent full-precision key and value regions;
- padding and any kernel workspace.

KIVI has no sparse outlier indices and no attention-sink region in the
canonical method. Its effective allocation ratio is context dependent because
the FP16 residual fraction is large at short context and shrinks as context
grows.

Decision 0019 records a terminology-only correction: immutable Phase 7 fields
named `r_alloc` contain `C_method / C_BF16` and are interpreted only as
`rho_alloc_legacy`. Their bytes, values, and checksums remain correct. Phase 8
emits canonical `rho_alloc=C_method_allocated/C_BF16_allocated` and reciprocal
`r_alloc=C_BF16_allocated/C_method_allocated`, with product error bounded by
`1e-9`; `r_hbm` remains null.

## Reference implementation

Relevant pinned paths include:

- models/llama_kivi.py
- models/kivi_gqa.py (added by the Decision 0018 patch)
- models/mistral_kivi.py
- quant/new_pack.py
- quant/matmul.py
- quant/csrc/gemv_cuda.cu
- quant/csrc/pybind.cpp
- quant/setup.py
- third_party/patches/kivi/manifest.json

The pinned main commit includes GQA-aware configuration and kernels that accept
query-head and KV-head counts. Every relevant file is bound by Git blob and
SHA-256 in `third_party/LOCK.json`. Because this is a post-paper snapshot,
those paths are evidence of current repository behavior, not paper-era
equivalence.

## Phase 7 source-audit result

The default-branch README explicitly advertises GQA and Llama 3 support. The
primary Llama decode path imports Transformers `repeat_kv` and calls it for the
recent key and value regions. Pyproject pins Transformers 4.43.1; its exact
official helper is an `expand(...).reshape(...)` operation documented as
equivalent to `repeat_interleave`.

At the frozen geometry, a non-timing BF16 semantic audit maps shape
`[1, 8, 32, 128]` to `[1, 32, 32, 128]`, changes storage from 65,536 to
262,144 bytes, produces a contiguous output, and does not share storage with
the input. This is a physical H_Q-sized K/V temporary. The Mistral integration
also defines and calls the same expand/reshape pattern for quantized tensors.

The historical quantized-cache CUDA kernel itself accepts `nh=32` and
`nh_kv=8`, but native historical indexing does not cure the residual-window
expansion. Phase 7 therefore stops as BLOCKED under Decision 0017. No reference
environment, CUDA build, fixture, byte-layout result, trace, sanitizer result,
or R2 fixture root was produced.

## B-019 remediation result

A fresh author-maintained-ref audit found no newer official revision: `main`,
`develop`, and `lmeval` remain at the Decision 0017 commits. Decision 0018
therefore authorizes one project patch on the exact official commit. The patch
SHA-256 is `c9c2dd52d4c81b844d1d1d7218ad2cd60a5b31574a387f716d466cb01310423d`
and the resulting tree is `b617493dea5aff1a754cd27ad6be12ac512b2aee`.

The patch groups the 32 query heads under their eight owning KV heads and runs
both residual contractions as BMMs with leading dimension `batch * H_KV`.
Attention scores and outputs retain H_Q geometry, while K/V operands and cache
storage remain H_KV=8. CPU and SM120 BF16 checks at contexts 17 and 33 are
exactly equal to the original repeat formula and observe no H_Q-sized K/V
operand.

This resolves B-019 under the explicit patched-source authority. The code is
not an unmodified official implementation and must not be described that way.
The separately authorized continuation completed the remaining Phase 7
reference work without changing the quantization, packing, cache layout,
rollover, or CUDA extension source.

## Phase 7 reference runtime result

The isolated reference definition is
`docker/reference-kivi.Dockerfile`, SHA-256
`b319d0c15d43d70ce364123d447b820ad1e312d6c60aa737c9707c701da17912`,
with image manifest `sha256:f27e4cdef6bd15f18ab76b1fe0e4413ede004b42538c74e3dd90d04172406f75`
and OCI config `sha256:0915dc8488fd6c9a150a3b4f56bb4b97b5dbdb7c51d96cda2d431df20e856ce3`.
Python, PyTorch, CUDA, compiler, package, and extension identities are frozen in
`environment.json`, `python-freeze.txt`, and `build_manifest.json`.

The unchanged official extension builds through `quant/setup.py` with
`TORCH_CUDA_ARCH_LIST=12.0+PTX`, produces native `sm_120` plus
`compute_120` PTX, and imports with SHA-256
`45d29ec1a3cecc4b253d1d1dd6139ef4f91cff88993db61a9d73685314851aa9`.
A fresh PTX-only relink executes the same probes, and Compute Sanitizer reports
zero errors for the distinct two-bit and four-bit key/value kernel families.

The upstream CUDA ABI is half-only. Direct BF16 reaches the official binding
but is rejected with `expected scalar type Half but found BFloat16`. The
fixtures therefore start from the required BF16 inputs, require an exact
BF16-to-FP16-to-BF16 round trip on a fixed integer/64 grid, and execute the
unchanged official FP16 ABI without an algorithmic CUDA patch.

Four deterministic fixtures cover K4/V4, K2/V4, K2/V2, and held-out K4/V2 at
batch 1, H_Q=32, H_KV=8, head dimension 128, group size 32, residual length 32,
and seed 20260726. They bind store L=17, append/decode L=18, rollover
L=31/32/33, and post-rollover decode L=34. Quantized payloads, scale/minimum
metadata, residual tensors, outputs, manifests, and checksums are stored.

Rollover moves key tokens 0-31 at L=32 and the oldest value token at L=33;
one additional value token moves at L=34. There are no missing or duplicated
tokens. Actual source-owned tensor storage agrees exactly with the byte
categories at L=31/32/33, with a source-layout calculation at L=64; `r_hbm` is
never populated.

The non-performance trace records quantize/pack, append, BMM, and official
two-bit/four-bit GEMV operators while discarding all durations. Every observed
K/V operand and persistent cache remains at H_KV=8, with explicit
`query_head // 4` mapping and no repeat/expand materialization or backend
fallback. Packed history is consumed directly; no full-prefix temporary is
observed. At Phase 7 completion, dynamic Graph behavior remained deferred to
the separately authorized Phase 8 adapter.

The durable 30-object bundle is published COMPLETE-last at root
`abd164da0adf9e0c1404e8fba1f6a6e42e57944481cdf060b91e8cef175ed302`.
Clean retrieval validates every object, inventory, checksum ledger, and root.
The publication receipt is outside its own bundle to avoid self-reference.

## Phase 8 measurement adapter result

Phase 8 entered from clean commit
`8d6d766a34a15bd40bd42cc47c5482b0dd052cc0` and executed at clean SHA
`462325e9df809d3bcf24a06361bf004bc7383d73` only inside the unchanged
Decision 0016 Measurement Container. It adds one static KIVI cache state and
one adapter, reusing the existing fixed-L and growing-context runners, Graph
harness, audits, artifact lifecycle, process supervision, and R2 publisher.
There is no new algorithmic or CUDA source change.

The adapter preallocates packed K/V history, K/V scales and minimum offsets,
fixed token-index metadata, 32-token K and V residual regions, FP16
quantization and BF16-to-FP16 ABI staging, logits/softmax/merge/output
workspace, and all mapping/index tensors for declared capacity. K quantizes
tokens 0-31 at L=32; V moves token 0 at L=33 and token 1 at L=34 while a
KIVI-specific circular residual preserves order. No cache growth,
measured-region `torch.cat`, complete-prefix temporary, H_Q-sized K/V
temporary, host synchronization, or backend fallback is observed.

The exact half-only boundary is:

    BF16 common-runner input
      -> in-place copy/cast into preallocated FP16 staging
      -> exact official KIVI CUDA operations
      -> in-place copy/cast into the BF16 common output

The direct compressed path verifies
`bgemv2_kernel_outer_dim` and `bgemv4_kernel_outer_dim`, with persistent and
operand K/V at H_KV=8 and `kv_head=query_head//4`. All four Phase 7 fixture
configurations conform, L31/L32/L33/L34 rollover has no missing or duplicate
token, predicted and actual allocated bytes agree exactly, eager allocation is
fully attributed with zero unknown or persistent delta, and fixed-L Graph
replay has zero allocation.

The minimal sanitizer covers the distinct two-bit and four-bit families plus a
real rollover and reports zero errors and zero leaked bytes. The frozen
ten-point admission grid passes 10/10 without calculating speedup. Strict
MethodAdmissionReport
`docs/evidence/phase8/kivi-method-admission.json`, SHA-256
`3a4b63b9da0eab12db9a916ebdc1cffd788ea6f93678d87964a8332ae7cec83a`,
derives 17/17 PASS checks.

The 331-object inner root
`f0c72b5330d2f1f0ab4c6a1594d223fdf068a32cf58cdec63f4e254ef8aed515`
and 341-object report-bearing outer root
`de7d41f151af9fe1e716f27ae0f1fc24d2ef0a4b16e8e5c3ecf45d5f9983e132`
are content-addressed, COMPLETE-last, and cleanly retrieved under exact
indefinite Bucket Lock rule `kvbench-evidence-indefinite`. The external outer
receipt has SHA-256
`9e9d8a650c0c1ed35eb4ecad32a34ede75cf45d4953e0baa32d2c0d561476db4`
and is excluded from its own bundle.

G2-KIVI is PASS for mandatory K4/V4, K2/V4, and K2/V2; held-out K4/V2
conforms but is validation-only. Global G2-G5 remain NOT EVALUATED, Full Scan
remains CLOSED, quality execution remains LOCKED,
`PERFORMANCE_DATA_FROZEN` remains absent, and Phase 9 has not started. No
speedup, physical-HBM, knee, capacity, performance, or quality claim is made.

## Dependency and porting risks

1. The reference model path dynamically grows caches with torch.cat. It cannot
   be used as Measurement Lane timing code.
2. The unpatched official GQA path calls `repeat_kv`; its physical 8-to-32-head
   materialization remains proven historical evidence. The Decision 0018 patch
   removes that path under checksum-bound project authority, but it is not
   upstream. Any patch drift or upstream replacement requires a new audit.
3. The requirements file pins torch 2.1.2, transformers 4.36.2, Triton 2.1.0,
   FlashAttention 2.5.6, and CUDA 12.1 packages. It also contains conflicting
   duplicate packaging pins and directly pins lm-evaluation-harness at commit
   c9bbec6e7de418b9082379da82797522eb173054. Pyproject instead pins torch
   2.4.1 and transformers 4.43.1. This unresolved legacy environment is
   isolated to the Reference Lane; the Git dependency is recorded separately
   in third_party/LOCK.json.
4. The upstream CUDA setup enables use_fast_math and has no Blackwell default.
   Phase 7 therefore fixes explicit SM120 plus PTX flags and validates both;
   any source/toolchain change requires the same proof again.
5. The CUDA source contains group-size-specialized paths and comments centered
   on 64/128. Phase 7 directly demonstrates group size 32 for all four frozen
   bit combinations rather than inferring support from comments or the paper.
6. No CUDA Graph guarantee is documented for the dynamic reference path. The
   separate static Phase 8 adapter passes only its frozen fixed-L Graph
   admission shapes.
7. The reference still uses dynamic `torch.cat` and allocations, so no
   Measurement claim may use that path. Only the exact fingerprinted Phase 8
   static adapter has method-specific allocation and Graph admission.
