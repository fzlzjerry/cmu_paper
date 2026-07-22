# KVQuant source note

Status: Phase 0 source audit; calibration and Reference Lane candidate only.

## Paper and source

- Title: KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization
- Local input: literature/raw/2401.18079.pdf
- Version: arXiv:2401.18079v6, 2025-05-28
- Paper SHA-256: 7ca1001fee6be5013d40ebf00e81df3bb677c90f1feb5c78b413cba64589ecfc
- Paper-provided repository: https://github.com/SqueezeAILab/KVQuant
- Pinned commit: 57a238357f0ffe50084670fcd5781c9848f80ea2
- License evidence: deployment metadata declares Apache Software License.
- Verification status: unresolved; the pinned repository has no root license
  file, so the package classifier is not treated as repository-wide license
  authority.
- Paper-reported hardware: A6000 for kernel latency, A100 80GB for
  single-GPU capacity, and eight A100 GPUs for the largest capacity examples.

## Algorithm

KVQuant combines:

- pre-RoPE, per-channel key quantization with offline-calibrated statistics;
- per-token value quantization;
- a per-layer sensitivity-weighted non-uniform datatype;
- dense-and-sparse decomposition for numerical outliers;
- fused RoPE application for key operations;
- a full-precision attention-sink prefix.

The planned main bitwidths are 4, 3, and 2. The workflow fixes sink_tokens=5
and requires a calibration-derived fixed outlier cap. The paper discusses
retaining the first token, while the repository describes configurable initial
tokens such as five. The exact five-token policy must be reproduced by the
pinned calibration/reference path before admission.

## Required storage accounting

The Measurement Lane must separately count:

- dense packed data;
- non-uniform lookup/scale/zero metadata;
- sparse outlier values;
- sparse outlier indices and structural metadata;
- full-precision sink keys and values;
- padding;
- fixed-cap workspace.

No outlier percentage, index dtype, metadata precision, or cap value is frozen
by Phase 0. Those are calibration outputs or configuration choices and must be
recorded before Phase 9 completes. Nominal bitwidth is not a substitute for
allocated bytes.

## Reference implementation

Relevant pinned paths include:

- deployment/kvquant/quant_cuda_kernel.cu
- deployment/kvquant/quant_cuda.cpp
- deployment/kvquant/simquant_module_quantizer.py
- deployment/llama.py
- deployment/kvquant/setup_cuda.py
- benchmarking/kvquant/quant_cuda_kernel.cu
- quant/ for Fisher-weighted quantizer construction and evaluation

The extension exposes 4/3/2-bit append and matvec kernels, sparse variants,
parallel packing variants, and fused pre-RoPE key operations. The deployment
environment requests Python 3.9, FlashAttention 2.5.5, and a vendored
Transformers 4.38.0.dev0 snapshot.

## Calibration state

Not frozen in Phase 0:

- calibration dataset and revision;
- preprocessing and sample selection;
- random seed;
- per-layer quantizer artifact;
- outlier cap;
- exact sparse value/index dtypes;
- lookup and scale precision.

Until these are fixed and checksummed, KVQuant cannot enter reference fixture
generation or the full scan.

## Porting and admission risks

1. The inspected Python path performs torch.topk and torch.cat while forming
   capped outliers. These operations are forbidden in measured decode.
2. Calibration uses NumPy/CPU conversions and data-dependent selection. All of
   it must occur before measurement and produce immutable artifacts.
3. The reference harness contains explicit CUDA synchronization for its own
   timing. Those numbers cannot be mixed with project timing.
4. The repository vendors a large, old Transformers snapshot with no recorded
   upstream commit, increasing compatibility and license risk. Three embedded
   trees and licenses are enumerated in third_party/LOCK.json: deployment/
   transformers and gradients at 4.38.0.dev0, and quant/dbrx at 4.41.0.dev0.
   Their exact upstream revisions and patch deltas remain unresolved.
5. The CUDA setup has no explicit Blackwell architecture selection. Native,
   PTX/JIT, and Compute Sanitizer evidence are absent.
6. GQA support is not established by the inspected reference interface; kernel
   names emphasize MHA. Unsupported geometry must be reported rather than
   expanded or silently routed elsewhere.
7. Sparse buffers in the reference path are data dependent. The Measurement
   Lane requires fixed-cap preallocation and graph-stable pointers.
8. CUDA Graph support, allocation freedom, cap-reached correctness, and
   component-level byte accounting are all unproven.
