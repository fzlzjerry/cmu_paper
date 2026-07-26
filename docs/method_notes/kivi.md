# KIVI source note

Status: B-019 RESOLVED under checksum-bound patched-source authority; Phase 7
reference execution remains NOT STARTED.

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

Planned variants are K4/V4, K2/V4, and K2/V2, with K4/V2 and K2/V4 retained as
the K/V-asymmetry falsification pair.

## Storage and metadata

Source-owned components identified for later runtime verification:

- packed historical key data;
- packed historical value data;
- per-group scale and minimum/zero metadata;
- recent full-precision key and value regions;
- padding and any kernel workspace.

KIVI has no sparse outlier indices and no attention-sink region in the
canonical method. Its effective allocation ratio is context dependent because
the FP16 residual fraction is large at short context and shrinks as context
grows.

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

This resolves B-019 under the explicit patched-source authority. It does not
make the code an unmodified official implementation and does not complete
Phase 7. Reference environment, official extension build, fixtures, sanitizer,
trace, byte accounting, graph information, and durable publication remain
unstarted.

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
4. The CUDA setup does not explicitly name Blackwell architecture flags and
   enables use_fast_math. A future restarted Phase 7 must rebuild for the
   detected capability and validate native plus PTX/JIT execution.
5. The CUDA source contains group-size-specialized paths and comments centered
   on 64/128 even though the project requires group size 32. Actual support
   must be demonstrated from the pinned code, not assumed from the paper.
6. No CUDA Graph guarantee is documented for the dynamic reference path.
7. Rollover, metadata byte accounting, allocation freedom, full-prefix
   dequantization behavior, SM120 compatibility, and sanitizer status remain
   NOT EXECUTED because the earlier GQA source gate failed.
