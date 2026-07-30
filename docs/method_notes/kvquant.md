# KVQuant source note

Status: Phase 9P patched-upstream compatibility, Phase 9 calibration, Phase 10
reference, and Phase 11R method-specific G2-KVQ admission PASS. Global G2-G5
remain NOT EVALUATED.

## Paper and source

- Title: KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization
- Local input: literature/raw/2401.18079.pdf
- Version: arXiv:2401.18079v6, 2025-05-28
- Paper SHA-256: 7ca1001fee6be5013d40ebf00e81df3bb677c90f1feb5c78b413cba64589ecfc
- Paper-provided repository: https://github.com/SqueezeAILab/KVQuant
- Pinned base commit: 57a238357f0ffe50084670fcd5781c9848f80ea2
- Pinned base tree: 094e0f736f77ee327e5350cbd1eefb1c936aa77b
- Phase 9P decision: docs/decisions/0020-kvquant-upstream-gqa-patch.md
- Patched method identifier: `kvquant_gqa_upstream_patch_v1`
- Patched human name: **KVQuant-GQA patched upstream**
- Local/private patched commit: 4ad80bc8c942d0a05516d2be8f8d443a77a05900
- Local/private patched tree: c4f1490c9c0c4ec46099f1e95c092516df2adb4e
- License evidence: deployment metadata declares Apache Software License.
- Verification status: unresolved; the pinned repository has no root license
  file, so the package classifier is not treated as repository-wide license
  authority.
- Phase 9P-close publication status: neither modified source, patch contents,
  source archive, Docker source layer, nor extension had been committed to this
  project or uploaded to R2. Decision 0021 later authorizes exact-patch custody
  in the main repository; the current structured publication state is recorded
  in `third_party/LOCK.json`. The patch is not claimed to be an official
  author-released GQA implementation.
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

The frozen main bitwidths are 4, 3, and 2. Phase 9P fixes `sink_tokens=5` and
a project-defined geometry-aware Key/Value cap of 12 for all three bit widths.
The cap follows `kv_width=8*128=1024`, `tail_fraction=0.005`, six entries per
tail, and twelve total entries; it is not an author-provided default. The paper
discusses retaining the first token, while the repository describes
configurable initial tokens such as five. Phase 9 used without retuning the
five-token and cap-12 compatibility policy.

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

## Frozen calibration state

Phase 9 freezes:

- WikiText-2 train from `Salesforce/wikitext` revision
  `b08601e04326c79dfdd32d625aee71d232d685c3`, conversion revision
  `3f68cd45302c7b4b532d933e71d9e6e54b1c7d5e`, and content SHA-256
  `e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7`;
- the exact 16 ordered 2,048-token windows selected with seed `20260721`;
- BF16 forward, FP32 Fisher, FP16 fitting, and FP32 codebook/threshold
  computation;
- all 32 K and 32 V Fisher tensors in native eight-KV-head geometry;
- safe `kvq4`, `kvq3`, and `kvq2` safetensors;
- five FP16 sink tokens plus shared Key/Value cap 12, six entries per tail,
  `float32` values, `int32` indices, lexicographic ties, and zero fill.

Calibration completion does not admit a reference or Measurement Lane.
Repository-wide license authority remains unresolved under B-006; Phase 10,
the KVQuant Measurement Adapter, G2-KVQ, Pilot, Full Scan, and quality
execution remain closed.

## Phase 9P patched authority

Decision 0020 binds the exact upstream base to one local/private patch with
aggregate SHA-256
`db3b6fb7ec0a72e25001e1c83a5158d86512248db5c3a06c61895598d1d482d6`.
The compact file inventory and tests are in
`docs/evidence/phase9p/patch-manifest.json` and
`docs/evidence/phase9p/test-report.json`; neither record contains source or
patch contents. Those completed Phase 9P records remain unchanged.

### Main-repository patch custody

Decision 0021 records the operator's later instruction to preserve the exact
validated diff in the main repository because the validation server is
ephemeral. The durable files are:

- `third_party/patches/kvquant/0001-llama31-native-gqa.patch`;
- `third_party/patches/kvquant/manifest.json`; and
- `scripts/validate_kvquant_gqa_patch.py`.

The 289,239-byte patch has the same aggregate SHA-256 recorded by Phase 9P.
Static validation requires no separate patched repository. With any local
checkout containing the pinned base commit, optional reconstruction applies
the patch in an ephemeral clone and requires tree
`c4f1490c9c0c4ec46099f1e95c092516df2adb4e`.

The project does not vendor the full upstream source, compiled extension,
model files, or caches. The operator-authorized public patch custody does not
resolve or claim a root KVQuant license or embedded/adapted-source lineage.

The patched path supports exactly the frozen
`meta-llama/Llama-3.1-8B-Instruct` revision
`0e9e39f249a16976918f6564b8830bc894c89659` and the tokenizer at that same
revision. It retains native Llama-3.1 `llama3` RoPE, BF16 model forward, FP32
Fisher accumulation, FP16 fitting activations, 32 layers, 32 query heads, 8 KV
heads, four query groups per KV head, head dimension 128, and native KV width
1024. Legacy linear-RoPE substitution and non-divisible geometry fail closed.

Calibration hooks capture pre-RoPE Keys and Values at native eight-KV-head
geometry. The full Fisher and quantizer CLIs require the same checksum-bound
offline model snapshot and reject changes to the frozen 16-example,
2048-token policy. Focused one-layer smoke evidence covers finite Fisher
collection and a finite 4-bit NUQ dense-and-sparse quantizer; no full Fisher or
4/3/2-bit calibration artifacts were generated in Phase 9P.

The patched deployment stores persistent K/V and sparse metadata by KV head.
For query head `h`, CUDA uses `kv_head = h / 4`; MHA remains the groups=1
special case. Exact eager execution rejects other attention implementations,
does not use `repeat_kv`, physical K/V expansion, a query-head-sized K/V
temporary, or full-prefix dequantization. The first five K/V positions remain
FP16 in native eight-head sink storage and are excluded from dense and sparse
history.

The deterministic sparse policy orders the lower tail by value ascending then
flat index ascending and the upper tail by value descending then flat index
ascending. It emits lower then upper, no duplicates or overlap, `float32`
values, `int32` indices, and zero-filled unused slots. The numeric cap is an
integer policy rather than an enable flag.

The isolated validation image is
`kvbench-phase9p-validation@sha256:16ee632c5ac029deca5859f3da4c74f9e5e55f5080c10745a59653e95d8e5b44`.
The checksum-bound extension is
`280496acd435361245d349b7af92210ea1d9eca7488873523116f7a84b087a71`
and contains native `sm_120` plus `compute_120` PTX. Native execution,
extension-only forced PTX/JIT, CUDA Graph capture/replay, replay-allocation,
MHA/GQA numerical controls, and three representative Compute Sanitizer cases
passed. These are compatibility/correctness results, not timing, HBM,
capacity, speedup, or quality results.

## Phase 9 calibration

The isolated calibration image is fixed by image/config digest
`sha256:127759078f2c70c9e795c7a1bb3408df1eaee8fa019319299d283dc8075b216d`.
It contains no model weights, credentials, source checkout, caches, or
calibration output. All calibration container executions used no network and
received no R2 credentials.

Final calibration
`kvqcal-cdb724c806d64d095c040d2673a987a3` has root SHA-256
`8148306d08205af376994b022f189a0d6837915cd279ca8af6b104e1f4b46ccf`.
Its 68-object read-only bundle contains exact token tensors, 64 finite Fisher
tensors, three complete safe quantizer families, 192-row layer statistics,
replay evidence, inventory, checksum ledger, and `COMPLETE` written last.
The full Fisher SHA-256 is
`a4cd9ad1e28332cc38c0a8bd19c10af079379655baaa2e5066aac6e23472117b`;
the `kvq4`, `kvq3`, and `kvq2` SHA-256 values are respectively
`a8c009633ac4cad952deb2a2fa96c44ef928a1510dadcf11dee29a7a3efe1bf6`,
`97518129cc64ffa445722cb0802b3082631841de50835cbdf2c85c36a0c1579f`,
and `b9bb3a8699aa38fb2a5707ff036814971552462692a180431f6f68df9624560e`.

Token reconstruction is byte exact. Representative layer-0 K/V Fisher replay
is exact. Fresh-process regeneration produced exact values for all 320 tensors
in each quantizer family; safetensors file bytes differ only because JSON
header key order is not canonical, so acceptance uses the tolerance frozen
before the run and records zero absolute and relative tensor differences.
Equal-value outlier ties, no-overlap, cap 12, six-per-tail, fixed dtypes, and
zero-filled unused slots replay exactly.

The first R2 attempt stopped on a transport error before `COMPLETE`; its six
identical small objects were retained. The same conditional publisher then
verified those objects, uploaded the remaining 62, wrote `COMPLETE` last, and
published
`r2://kvbench-artifacts/kvbench/sha256/8148306d08205af376994b022f189a0d6837915cd279ca8af6b104e1f4b46ccf/`.
Independent retrieval into a new empty directory verifies all 68 objects and
the root under indefinite Bucket Lock rule `kvbench-evidence-indefinite`.

## Porting and admission risks

1. The base Python path performs torch.topk and torch.cat while forming capped
   outliers. The patched calibration path replaces unstable top-k tie behavior
   with deterministic lexicographic selection; calibration operations remain
   forbidden in measured decode.
2. Calibration uses NumPy/CPU conversions and data-dependent selection. All of
   it must occur before measurement and produce immutable artifacts.
3. The reference harness contains explicit CUDA synchronization for its own
   timing. Those numbers cannot be mixed with project timing.
4. The repository vendors a large, old Transformers snapshot with no recorded
   upstream commit, increasing compatibility and license risk. Three embedded
   trees and licenses are enumerated in third_party/LOCK.json: deployment/
   transformers and gradients at 4.38.0.dev0, and quant/dbrx at 4.41.0.dev0.
   Their exact upstream revisions and patch deltas remain unresolved.
5. The base CUDA setup has no explicit Blackwell architecture selection.
   Phase 9P adds checksum-bound `sm_120` and `compute_120` validation only for
   the patched private authority.
6. The base reference interface does not establish GQA support. Phase 9P adds
   native patched-upstream 32Q/8KV correctness; unsupported geometry is
   rejected rather than expanded or silently routed elsewhere.
7. Sparse buffers in the reference path are data dependent. The Measurement
   Lane requires fixed-cap preallocation and graph-stable pointers.
8. Phase 9P establishes focused CUDA Graph, replay-allocation, and cap-reached
   correctness for the patched extension. Component-level Measurement Adapter
   byte accounting and G2-KVQ require their own later admission evidence.

## Phase 11R Measurement Adapter admission

Decision 0027 binds execution source
`kvquant_gqa_longctx_deterministic_v3`, corrected commit/tree
`4b8533b29b04f8c4bf55f688a41fefe20487637b` /
`46f2149a0369d5c97d9a6bc77d57b5f3a5a5fb3b`, aggregate patch SHA-256
`bae63bced549479709b10d7f6a8ee35a8f21ec18cc040a7424591cee47c1b0a6`,
and extension SHA-256
`a79644923ba131e56abe95029e669346dbbb11fd210d2b9f8b2086819ffeaad1`.
The q4 Adapter uses its deterministic fixed-order Value decode with one
caller-owned, preallocated FP32 `[1,32,32,128]` workspace. q3 and q2 retain
their corrected paths.

All nine corrected fixtures match for payload, metadata, sparse, sink, store,
append, and byte records; decode agrees within the pre-frozen tolerance. Native
32Q/8KV mapping, fixed-cap sparse state, sink semantics, current-stream
execution, fixed-L CUDA Graph replay, zero replay allocation, and zero-error
memcheck/initcheck pass. The bounded admission grid passes 9/9. The 165-object
inner admission root
`0834410509ea7324a41715e0e84e09617bf9b188b10394a234f9a57e804dd1f2`
was published COMPLETE-last and cleanly retrieved. MethodAdmissionReport
SHA-256 is
`59ef5bfc581a68cdc4d21c4c0a840f046e698633f7475f79906063c6e333ae6a`.

G2-KVQ is PASS only for this exact method/source/container/calibration/oracle
and bounded admission contract. Global G2-G5 remain NOT EVALUATED; no
performance, speedup, HBM, capacity, or quality conclusion follows.
