PHASE 10 REPORT

Status: BLOCKED

Entry:
- Starting HEAD: a873fc93754fa86bfb757fce476388897bee8dca
- Final HEAD: a873fc93754fa86bfb757fce476388897bee8dca
- origin/main: a873fc93754fa86bfb757fce476388897bee8dca
- Working tree: CLEAN
- Phase 9 report: PASS; SHA-256 cd8feff788c4483b8a14bf971400ffff9d69088200cd916fa42b82e0a0d76eb9
- Calibration ID: kvqcal-cdb724c806d64d095c040d2673a987a3
- Calibration root: 8148306d08205af376994b022f189a0d6837915cd279ca8af6b104e1f4b46ccf
- G0: PASS
- G1: PASS
- G2-TQ: PASS
- G2-KIVI: PASS
- G2-KVQ at entry: NOT EVALUATED
- Quality: LOCKED
- Full Scan: CLOSED

Authority:
- Method identifier: kvquant_gqa_upstream_patch_v1
- Upstream base commit/tree: 57a238357f0ffe50084670fcd5781c9848f80ea2 / 094e0f736f77ee327e5350cbd1eefb1c936aa77b
- Patch SHA: db3b6fb7ec0a72e25001e1c83a5158d86512248db5c3a06c61895598d1d482d6
- Patched commit/tree: 4ad80bc8c942d0a05516d2be8f8d443a77a05900 / c4f1490c9c0c4ec46099f1e95c092516df2adb4e
- Decision 0021: Accepted
- Reconstruction: PASS; temporary detached reconstruction verified all 15 changed paths and exact patched tree
- Official author GQA support claimed: No

Minimal scope:
- Plan: Not created; stopped under Step 6 source-semantics rule
- General reference framework: No
- Measurement Adapter started: No
- Calibration rerun: No
- Existing methods changed: No

Reference environment:
- Reused Phase 9 environment: Dependency and semantic probe only; not finalized as the Reference Lane
- Dockerfile, if created: None
- Base digest: sha256:16ee632c5ac029deca5859f3da4c74f9e5e55f5080c10745a59653e95d8e5b44
- Image/config digest: sha256:127759078f2c70c9e795c7a1bb3408df1eaee8fa019319299d283dc8075b216d
- Python: 3.12.3
- PyTorch: 2.12.1+cu130
- CUDA: 13.0.2; NVCC 13.0.88
- Compiler: GCC 13.3.0
- Extension: Not built
- Model weights in image: No
- Credentials in image: No

Calibration binding:
- Calibration ID: kvqcal-cdb724c806d64d095c040d2673a987a3
- Calibration root: 8148306d08205af376994b022f189a0d6837915cd279ca8af6b104e1f4b46ccf
- R2 source: r2://kvbench-artifacts/kvbench/sha256/8148306d08205af376994b022f189a0d6837915cd279ca8af6b104e1f4b46ccf/
- COMPLETE: PASS
- Inventory/checksums: PASS; 68 objects, clean R2 retrieval
- kvq4: PASS; 320/320 tensor identities
- kvq3: PASS; 320/320 tensor identities
- kvq2: PASS; 320/320 tensor identities
- Fisher regenerated: No
- Quantizers regenerated: No

Fixture geometry:
- Batch: 1
- Query heads: 32
- KV heads: 8
- Groups: 4
- Head dimension: 128
- Store context: 17
- Append tokens: 1
- Total context: 18
- Sink tokens: 5
- Key cap: 12
- Value cap: 12
- Seed: 20260729
- Dtype: BF16 interface; source sink boundary FP16

Fixtures:
- Root: Not created
- Total cases: 0
- kvq4 no/few/cap: Not generated
- kvq3 no/few/cap: Not generated
- kvq2 no/few/cap: Not generated
- Store: Not generated
- Append: Not generated
- Decode: Not generated
- Dense K: Not generated
- Dense V: Not generated
- Metadata: Not generated
- Sparse values: Not generated
- Sparse indices: Not generated
- Sink K/V: Not generated
- Checksums: Not generated
- Existing-fixture overwrite: No fixture existed or was written

Outlier behavior:
- No-outlier K/V counts: K not generated; Value requested 0 but patched deployment produces 12 for every non-sink row
- Few-outlier K/V counts: K not generated; Value requested 6 but patched deployment produces 12 for every non-sink row
- Cap-reached K/V counts: K not generated; Value source probe produces 12
- Overflow: No overflow observed
- Duplicate indices: Source selection forbids duplicates; fixtures not generated
- Lower/upper overlap: Source selection forbids overlap; fixtures not generated
- Tie-breaking: Frozen source tests PASS
- Unused slots: Sink rows are zero-count; non-sink Value rows have no unused slots
- Physical capacity: 12

Pre-/post-RoPE:
- Quantized Key: Source authority is pre-RoPE; fixture not generated
- Sink Key: Source authority uses attention-ready FP16 sink storage; fixture not generated
- Attention Key: Native Llama-3.1 RoPE source path verified; fixture not generated
- Value: Immediately after v_proj; fixture not generated
- Position IDs: Not generated
- Head expansion: Source-under-test forbids implementation-side expansion

GQA:
- H_Q: 32
- H_KV: 8
- Groups: 4
- Mapping: query_head // 4
- Dense storage: Source-verified native eight-head geometry
- Sparse storage: Source-verified native KV-width indexing
- Sink storage: Source-verified native eight-head geometry
- repeat_kv: Absent from the patched eager source-under-test path
- Expanded temporary: None identified in the patched eager path

Execution path:
- Dense quantization/packing: Not executed; source contains distinct 4/3/2-bit CUDA pack paths
- Sparse correction: Source uses fused dense-plus-sparse correction
- Sink handling: Separate FP16 native eight-head cache
- Decode/dequantization: Direct compressed-cache kernels in source
- Full-prefix temporary: None identified
- Backend fallback: Not evaluated; no execution fixture
- Reference dynamic allocation: Present in source reference path; not measured
- Timing claim: None

Byte accounting:
- Bit width/case: None
- Dense K bytes: Not generated
- Dense V bytes: Not generated
- K metadata: Not generated
- V metadata: Not generated
- K outlier-value bytes: Not generated
- K outlier-index bytes: Not generated
- V outlier-value bytes: Not generated
- V outlier-index bytes: Not generated
- Sink K bytes: Not generated
- Sink V bytes: Not generated
- Padding: Not generated
- Workspace: Not generated
- Actual allocated total: Not generated
- Active logical total: Not generated
- Logical BF16: Not generated
- rho_alloc: Not generated
- r_alloc: Not generated
- Reciprocal check: Not run
- r_hbm: null

CUDA:
- Native SM120: Not run for Phase 10
- sm_120 cubin: Not produced
- compute_120 PTX: Frozen build flags verified; not built
- Forced PTX/JIT: Not run
- Sanitizer cases: 0
- Sanitizer result: Not run
- Fallback: Not evaluated

Durable publication:
- Local root: None
- R2 URI: None
- Initial publication: Not run
- COMPLETE-last: Not generated
- Clean retrieval: Not run for Phase 10 fixtures
- Checksums: Not generated
- Bucket Lock: Existing exact indefinite lock verified during calibration retrieval
- Credential leakage: None observed; credentials reported only as PRESENT/MISSING

Tests:
- package-lock: PASS
- make test: PASS
- make checks: PASS
- TurboQuant regression: PASS
- KIVI regression: PASS
- Patch reconstruction: PASS
- Calibration validation: PASS
- Reference fixtures: Not run
- Outlier behavior: BLOCKED by Value source semantics
- Pre-/post-RoPE: Not run
- GQA: Static authority checks PASS; fixture execution not run
- Bytes: Not run
- CUDA: Not run
- Sanitizer: Not run
- Governance: PASS; KVQuant factory remains fail-closed
- Historical evidence: PASS; exact hashes unchanged

Gates:
- G0: PASS
- G1: PASS
- G2-TQ: PASS
- G2-KIVI: PASS
- G2-KVQ: NOT EVALUATED
- Global G2: NOT EVALUATED
- G3: NOT EVALUATED
- G4: NOT EVALUATED
- G5: NOT EVALUATED
- Full Scan: CLOSED

Quality governance:
- Quality execution: LOCKED
- PPL/LongBench: Not run
- PERFORMANCE_DATA_FROZEN: absent

Preservation:
- Original Phase 9 BLOCKED report changed: No
- Phase 9 calibration bundle changed: No
- TurboQuant changed: No
- KIVI changed: No
- Measurement Container changed: No
- Historical evidence changed: No
- Existing run overwritten: No
- Performance data: None
- Profiler data: None
- Quality data: None

Changed files:
- None

Commits:
- None

Risks:
- The requested Value no/few/cap fixture contract is incompatible with the exact patched deployment call semantics.
- Adding Value thresholds or a persisted active count would change the frozen compatibility implementation and was not authorized.

Blockers:
- `LlamaAttention.forward` calls `select_fixed_outliers(flattened_values, cap=12, sink_row_mask=...)` without lower or upper thresholds. For finite native-width Value rows, the function unconditionally selects six lowest and six highest indices. The exact-image probe returned non-sink active count 12 for the proposed no-outlier, few-outlier, and cap-reached recipes. A thresholded counterfactual produced 0/6/12, but using it would alter the source-under-test algorithm.

Scientific interpretation:
- The required nine KVQuant-GQA patched-upstream numerical reference fixtures are not reproducible under the frozen source semantics because the required non-sink Value no-outlier and few-outlier states cannot occur. No quality, speedup, memory, HBM, knee, capacity, or official-author-support claim is made.

Next action:
- Minimum remaining technical blocker: in a separate authorized task, either provide a checksum-bound Value-selection implementation that supports source-defined 0/6/12 active counts, or revise the fixture acceptance contract to match the frozen deployment’s fixed 12-entry non-sink Value semantics. Phase 11 must not begin.

<oai-mem-citation>
<citation_entries>
MEMORY.md:26-47|note=[used immutable Phase 9 report custody and fail-closed workflow history]
rollout_summaries/2026-07-27T15-39-40-oA4Y-phase9_entry_blocked_report_custody.md:39-56|note=[verified historical BLOCKED report provenance and preservation expectations]
</citation_entries>
<rollout_ids>
019fa43b-56d6-7e33-a0a3-f027c13b1ab4
</rollout_ids>
</oai-mem-citation>
