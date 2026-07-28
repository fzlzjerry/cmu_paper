PHASE 9 REPORT

Status: BLOCKED

Entry:
- Starting HEAD: f2c6475f09cdf6e9660552eb23c91b03e386aa59
- Final HEAD: f2c6475f09cdf6e9660552eb23c91b03e386aa59
- Working tree: Clean
- Phase 8 report: PASS
- G0: PASS
- G1: PASS
- G2-TQ: PASS
- G2-KIVI: PASS
- G2-KVQ at entry: NOT EVALUATED
- Quality: LOCKED
- Full Scan: CLOSED

Minimal scope:
- Plan: Entry validation, source-authority audit, and primary-model compatibility audit only; stopped on mandatory blockers.
- General calibration framework: No
- Measurement Container changed: No
- Existing adapters changed: No
- Phase 10 started: No

Source authority:
- Repository: [SqueezeAILab/KVQuant](https://github.com/SqueezeAILab/KVQuant)
- Exact commit: 57a238357f0ffe50084670fcd5781c9848f80ea2
- Exact tree: 094e0f736f77ee327e5350cbd1eefb1c936aa77b
- Commit date: 2024-07-09T20:32:36-07:00
- License: BLOCKED; no repository-root license exists in current or historical commits, GitHub reports `license: null`, and nested Apache licenses cover embedded Transformers trees only. Package classifiers are insufficient repository-wide authority.
- Relevant files: `gradients/run-fisher.py` e0d8be98a93ac9a9181076ea29bdfc951e25a77a3ed246fa748ff4d1ed48b8f4; `gradients/datautils.py` d9149f77bca4971171b90ccac1de74922d542a488c2b730c813bc8999b7414ce; `gradients/.../configuration_llama.py` d224032c29e050f1fb9e949e93e06b026be0c3b923b80be1e331dc9ae2ab0db1; `quant/llama_simquant.py` b46f587b0afee1d3a15beb6ee9915c2b3a7f23ca376c6f7c9071831135b704a2; `quant/kvquant/datautils.py` 3c0c0c4527b434974360b5736d6d80c0a6dd73a2de0632e1b4a5759a5e5a2768; `quant/kvquant/simquant_module_quantizer.py` 43513d3abc431759f7c4b53f718f2d87ed79ba88a1bc3df52bff5319fdd60046; `deployment/llama.py` 121bac505a2e7451b18cedaff10c59b606b6f4d4aae4e8030cbb31f6606817a8; deployment Llama integration 890e75f74d30d57aa45e02a53b8cd8c4cee3273e1d8826757ac2bee24faa9b62.
- Source lock: Existing planned record unchanged; remains `unresolved_no_root_license` and `acquisition_status=planned`.
- Compatibility patch: None
- Algorithmic source changed: No

Primary model:
- Model ID: meta-llama/Llama-3.1-8B-Instruct
- Revision: 0e9e39f249a16976918f6564b8830bc894c89659
- Tokenizer: Same ID/revision; frozen class `PreTrainedTokenizerFast`
- Architecture compatibility: BLOCKED; the Fisher path rejects Llama-3.1 `rope_type: llama3`, while quantizer generation forces FP16.
- GQA geometry: Required 32Q/8KV/head_dim=128 is not end-to-end supported; official deployment explicitly rejects GQA.
- Pre-RoPE Key hook: Present in source, but not executed or validated for the frozen model.
- Model substitution: None

Calibration environment:
- Dockerfile: Not created
- Base digest: Not evaluated
- Image/config digest: Not built
- Python: Not frozen
- PyTorch: Not frozen
- CUDA: Not frozen
- Transformers: Not frozen
- Dataset library: Not frozen
- Clustering library: Not frozen
- Source installed: No; only an isolated read-only audit checkout was inspected.
- Credentials/model weights in image: No image created

Dataset:
- Dataset: WikiText-2 proposed only
- Revision/content hash: Not frozen
- Split: Not executed; official loader reads both train and test
- Number of examples: Not generated
- Sequence length: Not generated
- Selection seed: Not frozen; requested 20260721 conflicts with the Fisher source’s hard-coded seed 0
- Selected IDs: None
- Tokenizer revision: Intended exact revision, not exercised
- Token tensor checksum: None
- Preprocessing: Source behavior audited but not frozen
- Test-split overlap: Not evaluated; no samples created

Fisher:
- Command: NOT RUN
- Layers: None
- Dtype: No artifact; official runner requests BF16
- Shapes: None
- Finite values: Not evaluated
- Root/checksum: None
- Representative replay: Not run
- Numerical tolerance: Not established

Quantizers:
- kvq4: Not generated
- kvq3: Not generated
- kvq2: Not generated
- NUQ: Not executed
- Dense-and-sparse: Not executed
- Sparsity threshold: Not frozen
- nsamples: Not executed
- Fisher linkage: None
- Regeneration: Not run
- Serialization: Source-native pickle identified; no artifact produced
- Checksums: None

Sink policy:
- Sink tokens: 5 remains preregistered but not frozen in a bundle
- K/V policy: Source supports an initial K/V FP16 prefix; not validated for the exact model
- Dtype: Source uses FP16
- RoPE interaction: Pre-RoPE Key behavior identified; Llama-3.1 behavior remains unsupported

Outlier caps:
- Source mechanism: Hard-coded top/bottom selection and fixed 42-slot K/V sparse rows
- Key cap: 42 in the MHA deployment path; not authorized for the GQA target
- Value cap: 42 in the MHA deployment path; not authorized for the GQA target
- Scope: Per flattened token vector, not per head
- Shared across bit widths: Source layout is shared for 4/3/2-bit, but no Phase 9 policy was frozen
- Selection rule: At 1% and hidden size 4096, select 21 upper and 21 lower entries; numeric `cap_outliers` is only treated as an enable flag
- Cap-hit behavior: Key excess is truncated; unused value slots are zeroed without an explicit sentinel
- Tie-breaking: Not source-defined deterministically
- Value dtype: Source uses float32
- Index dtype: Source uses int32
- Padding/sentinel: Zero values; no explicit sentinel
- Quality/performance used in selection: No

Layer statistics:
- File: None
- Layer coverage: None
- Bitwidth coverage: None
- Outlier distributions: None
- Cap-hit rates: None
- Invalid values: Not evaluated

Calibration bundle:
- Calibration ID: None
- Local path: None
- Root digest: None
- COMPLETE: Absent
- Inventory: None
- Checksums: None
- Existing-artifact overwrite: No
- Large binaries committed to Git: No

Durable publication:
- R2 URI: None
- Initial publication: Not attempted
- COMPLETE-last: Not attempted
- Clean retrieval: Not attempted
- Checksum result: Not evaluated
- Bucket Lock: Not evaluated for Phase 9
- Credential leakage: None; `.env` was not read and credentials were not printed or passed

Tests:
- package-lock: PASS
- make test: PASS
- make checks: PASS
- TurboQuant regression: PASS
- KIVI reference/admission regression: PASS; local bundle, MethodAdmissionReport, and R2 receipt jointly match root f0c72b5330d2f1f0ab4c6a1594d223fdf068a32cf58cdec63f4e254ef8aed515
- source lock: BLOCKED on license and lineage authority
- dataset: Not run
- Fisher: Not run
- quantizer: Not run
- determinism: Not run
- cap policy: BLOCKED for exact GQA semantics and deterministic tie-breaking
- sink policy: Source-audited only
- governance: PASS; fail-closed boundaries preserved
- historical evidence: PASS; unchanged and checksum-valid

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
- PPL/LongBench run: No
- PERFORMANCE_DATA_FROZEN: Absent

Preservation:
- Measurement Container changed: No
- TurboQuant changed: No
- KIVI changed: No
- Historical evidence changed: No
- Existing run overwritten: No
- Formal performance data: None
- Profiler data: None
- Quality data: None

Changed files:
- None

Commits:
- None

Risks:
- R-003 remains open for KVQuant GQA support.
- R-015 remains open for root licensing and embedded/adapted lineage.
- Official cap semantics are MHA-specific and lack deterministic tie-breaking.

Blockers:
- B-006: repository-wide license authority and file-level lineage remain unresolved.
- The official Fisher path cannot load exact Llama-3.1 RoPE semantics.
- Quantizer generation forces FP16 and does not bind the exact fast tokenizer.
- Native 32Q/8KV support and source-faithful fixed K/V cap semantics are unavailable without non-narrow changes.

Scientific interpretation:
- The dataset, Fisher artifacts, quantizers, sink policy, and fixed outlier-cap policy were not frozen and therefore are not reproducible.
- No KVQuant accuracy, speedup, memory-benefit, HBM, knee, capacity, or quality claim is made.

Next action:
- Obtain explicit repository-wide licensing authority and file-level lineage, then provide an official author-maintained revision supporting Llama-3.1 RoPE, BF16 calibration, the exact tokenizer, native 32Q/8KV geometry, and deterministic fixed K/V cap semantics. Restart Phase 9 entry checks afterward; Phase 10 must remain closed.

<oai-mem-citation>
<citation_entries>
MEMORY.md:148-162|note=[phase boundaries factory and artifact governance reminders]
</citation_entries>
<rollout_ids>
019f972b-4445-7151-a0f5-8e900278aa3c
019f96bd-0e00-7f71-b703-03fb4726d627
</rollout_ids>
</oai-mem-citation>
