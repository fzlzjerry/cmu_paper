# Decision 0007: Phase 3 primary model and BF16 backend

- Status: Accepted
- Date: 2026-07-22
- Authority: CODEX_WORKFLOW.md, AGENTS.md, the Phase 3 execution
  instruction, CODEX_POST_PERFORMANCE_QUALITY_VALIDATION.md for the quality
  lock, and Decision 0006
- Supersedes: none
- Superseded by: none

## Context

Phase 2 intentionally left the primary model, tokenizer, BF16 implementation,
and execution backend unresolved. Phase 3 needs one exact baseline identity and
one auditable attention path without introducing a Phase 4 method-adapter
abstraction. The operator has explicitly authorized bounded Phase 3
engineering/admission work on the certified native host while B-010 remains
open. That authorization does not make native-host timing claim-bearing and
does not change the container requirement for later formal measurement.

The supplementary quality protocol names
`meta-llama/Meta-Llama-3.1-8B-Instruct` as the default proposal. The canonical
Hugging Face repository is now `meta-llama/Llama-3.1-8B-Instruct`. The locally
cached `meta-llama/Llama-3.1-8B` snapshot is the base checkpoint, has a
different tokenizer configuration, and is not an admissible substitute.
Credentialed read-only access to the exact Instruct revision was verified
without printing or storing a credential in the repository.

The certified native environment contains PyTorch 2.12.1+cu130 and its bundled
FlashAttention implementation. It does not contain Transformers or a separate
FlashAttention package. Transformers' own eager attention expands GQA K/V with
`repeat_kv`, and its SDPA wrapper can also expand K/V when mask/dispatch
conditions do not permit native GQA. Those paths do not satisfy the baseline
audit. PyTorch's direct Flash SDPA can instead be forced as the sole backend and
fails closed when unsupported.

## Decision

### Model and tokenizer identity

1. The Phase 3 primary checkpoint and the checkpoint reserved for later
   quality validation are both:

   - repository: `meta-llama/Llama-3.1-8B-Instruct`
   - immutable revision:
     `0e9e39f249a16976918f6564b8830bc894c89659`
   - architecture: `LlamaForCausalLM`, decoder-only, full attention
   - parameters: approximately 8B
   - layers: 32
   - hidden size: 4096
   - query heads: 32
   - KV heads: 8
   - head dimension: 128
   - maximum position embeddings: 131072
   - weight dtype: BF16

2. The tokenizer uses the same repository and immutable revision. No tokenizer
   from the cached base model or another Llama revision may be substituted.

3. The frozen configuration artifacts are:

   - `config.json` SHA-256:
     `29e4c210b0d6ac178b16b2a255a568bdb23b581e50ca1ef6a6d071dd85704e6e`
   - `generation_config.json` SHA-256:
     `189fb0c0d7fd8a527db217c0a60a0e013f0394cd8800f9697a666a9e75e5f7fd`
   - `model.safetensors.index.json` SHA-256:
     `146776fce3f6db1103aa6f249e65ee5544c5923ce6f971b092eee79aa6e5d37b`
   - `special_tokens_map.json` SHA-256:
     `6f38c73729248f6c127296386e3cdde96e254636cc58b4169d3fd32328d9a8ec`
   - `tokenizer.json` SHA-256:
     `79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4`
   - `tokenizer_config.json` SHA-256:
     `177c7b61e616fecb84c17ce0591acb92c6c4d60e9ac5ababfb940ff23bbcd424`

4. The expected weight-shard SHA-256 values, in index order, are:

   - `2b1879f356aed350030bb40eb45ad362c89d9891096f79a3ab323d3ba5607668`
   - `09d433f650646834a83c580877bd60c6d1f88f7755305c12576b5c7058f9af15`
   - `fc1cdddd6bfa91128d6e94ee73d0ce62bfcdb7af29e978ddcab30c66ae9ea7fa`
   - `92ecfe1a2414458b4821ac8c13cf8cb70aed66b5eea8dc5ad9eeb4ff309d6d7b`

5. RoPE is frozen to the checkpoint configuration: type `llama3`, factor 8,
   high-frequency factor 4, low-frequency factor 1, original maximum position
   embeddings 8192, and theta 500000. The checkpoint is manually gated under
   the Llama 3.1 Community License. The exact revision must be loaded from a
   content-addressed local cache with network access disabled during runs.

6. Transformers 4.57.6 is frozen as the model-loading and independent trusted
   reference implementation. It is not the SUT attention backend. Its wheel
   SHA-256 is
   `4c9e9de11333ddfe5114bc872c9f370509198acf0b87a832a0ab9458e2bd0550`.
   It and its complete dependency closure will be installed into a separately
   locked Phase 3 dependency directory; the certified E00 `.venv` will not be
   modified.

### BF16 attention backend

7. The SUT backend is direct
   `torch.nn.functional.scaled_dot_product_attention` under
   `torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION)`, with
   `enable_gqa=True`, BF16 Q/K/V, dropout zero, and scale
   `head_dim ** -0.5`. No other SDPA backend is enabled in that context. An
   unsupported shape or device must raise `backend_unsupported`; it must not
   fall back.

8. The backend identity is:

   - PyTorch: `2.12.1+cu130`
   - PyTorch git revision:
     `7269437d655783a26cba32aa88195b741ff496aa`
   - bundled FlashAttention generation/version: FA2 / `2.5.7`
   - CUDA runtime: 13.0
   - cuDNN: 9.20.0
   - Triton: 3.7.1 (recorded environment identity, not the selected attention
     backend)
   - `torch/nn/functional.py` SHA-256:
     `27493186ee22f811b553e31d9c804d4d46716d1be62d034d731537f66f27ef19`
   - `torch/nn/attention/__init__.py` SHA-256:
     `56e10b6f965cc050db782dd4dc472097c9b02ec5b5fe3ab2c8b04055c0b0bbe0`
   - `torch/nn/attention/varlen.py` SHA-256:
     `2f5384e0bc8ce371d00a1c09d38ad019517009798e7cb3434f56cf4b9fa351ea`
   - CUDA `flash_api.h` SHA-256:
     `1474aa79d8aa6ce39984dbc3c0aad9dba283ab819f034370e5cfb70980524ee7`
   - `libtorch_cuda.so` SHA-256:
     `b248fb7e9935440965e4736eea48868b315ba41012734b7ce058fc0a2d0b1984`

9. Native-GQA evidence is the backend C++ API's separate `num_heads` and
   `num_heads_k` geometry, forced-dispatch probes returning backend enum
   `FLASH_ATTENTION`, tensor shapes `[B,32,1,128]` for Q and
   `[B,8,L,128]` for K/V, and an operator audit that found no K/V expansion.
   A same-shape math-SDPA control did dispatch `expand` and `clone` and is
   rejected. A batch-1 synthetic forced-Flash graph probe on SM120 captured and
   replayed with stable pointers and exact eager/graph output agreement.

10. The static cache layout is two contiguous BF16 tensors, K and V, each
    `[layers, batch, kv_heads, capacity, head_dim]`. A layer view passed to
    Flash is `[batch, kv_heads, attended_length, head_dim]`. No layout stores
    K/V with 32 query heads. No `repeat_kv`, `repeat_interleave`,
    `expand(...).reshape(...)`, `torch.cat`, or full-prefix copy is permitted in
    the SUT path.

11. For prefill with a square Q/K sequence, full causal attention uses
    `is_causal=True`. For one-token decode, the cache exposes only positions
    through the current token and uses `is_causal=False`; therefore every
    exposed K/V position is valid past-or-current context and no mask or
    sliding-window path is used.

12. `torch.compile` is disabled. CUDA Graph capture is explicit and separate;
    compile mode and graph mode are not conflated.

### Frozen runner and admission semantics

13. `context_length` means historical prefix length `L`. A fixed-L operation
    attends `L + 1` tokens including the current scratch token. The scratch
    position is index `L`, is overwritten on every iteration, and never
    advances the historical prefix.

14. A growing-context operation reports the historical active length before
    the current append. Its sequence is `L, L+1, ..., L+O-1`; each step writes
    the current token at that index and attends active length plus one.

15. The initial numerical limits are frozen before the admission grid:

    - cache writes and integer/token metadata: exact equality;
    - small-tensor BF16 attention versus FP32 reference: `atol=0.02`,
      `rtol=0.02`;
    - system BF16 attention and eager/graph comparison: `atol=0.02`,
      `rtol=0.02`;
    - full-model BF16 logits versus the trusted reference: `atol=0.125`,
      `rtol=0.02`;
    - every compared output must be finite.

    These tolerances account for BF16 input precision and permitted reduction
    ordering differences. They may not be relaxed after seeing admission
    results. A failure requires a new decision and a new run, not an edit to
    existing evidence.

16. Independent-process stability is evaluated only for the preregistered
    fixed-L `batch=1, L=4096` eager and CUDA Graph points, with three complete
    processes per mode. The engineering criterion is coefficient of variation
    at or below 3% for each mode. This is a Phase 3 G1 check and does not
    evaluate G5.

17. Phase 3 records host-wall and CUDA-event timing only as
    `native_host_admission` engineering evidence. Every run has
    `quality_status=unvalidated`, `claim_eligibility=performance_only`,
    `performance_claim_eligible=false`, and `claim_class=none`. No speedup,
    knee, HBM, capacity, model-comparison, or paper result may be derived.

18. The allocation gate distinguishes immutable cache storage, predeclared
    workspace/reserved-pool capacity, transient framework workspace, and
    persistent growth. Cache or pointer growth, a positive persistent allocated
    or reserved-memory delta after warmup, an unbounded/repeated peak increase,
    or any positive graph-replay delta fails admission. Third-party transient
    workspace observed within a stable pre-reserved pool must be measured,
    bounded, reported as workspace, and separately judged against the
    repository's strict no-dynamic-allocation requirement; it may not be
    silently labeled zero.

19. The local Phase 2 append-only writer may hold Phase 3 non-claim evidence.
    This does not satisfy B-009's durable-store acceptance criteria. B-009 and
    B-010 remain open. The operator's narrow native-host authorization refines
    their former blanket wording only for Phase 3 admission; neither formal
    timing nor later method admission is unlocked.

## Rejected alternatives

- The locally cached base `meta-llama/Llama-3.1-8B` checkpoint: wrong
  checkpoint role and tokenizer; substitution is prohibited.
- A floating model revision: non-reproducible.
- Transformers eager attention: materializes GQA K/V through `repeat_kv`.
- Transformers SDPA as the SUT: conditional `repeat_kv` and a broader dispatch
  surface make fallback/materialization harder to exclude.
- Math SDPA: directly observed GQA expansion through `expand` and `clone`.
- Flex Attention: source contains `repeat_interleave` for this GQA path.
- Memory-efficient SDPA: unsupported for the audited GQA geometry.
- cuDNN attention: not selected because the exact static-capacity and GQA path
  has not received the same source/temporary-allocation audit.
- External FlashAttention 2/3/4: unavailable in the certified environment;
  FA3 is not the SM120 path and the installed Transformers FA4 integration
  rejects SM120.
- DynamicCache: uses concatenation/growth and violates cache semantics.
- `torch.compile`: not required for Phase 3 and would change the execution
  identity.
- Synthetic or randomly initialized full-model weights: invalid substitute.

## Consequences

- B-004 can be resolved only after every downloaded byte matches this record
  and deterministic local loading passes.
- The baseline can share exact checkpoint/tokenizer identity with later
  quality validation while quality execution remains locked.
- The backend has a fail-closed native-GQA route on SM120 and a demonstrated
  graph-capable operator path.
- The observed eager transient-workspace behavior is an explicit G1 risk. If
  the implemented full-model lane cannot meet the frozen allocation audit, G1
  is PARTIAL or FAIL; the result is not reinterpreted after execution.
- No custom CUDA/C++ extension is introduced, so Phase 3 does not claim
  Compute Sanitizer coverage for PyTorch's third-party kernels.
