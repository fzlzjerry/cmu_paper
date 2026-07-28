# PHASE 9 CALIBRATION REPORT

Status: PASS

## Entry

- Starting HEAD: `b4d253724717076188a38032d6d6204fdf15e191`.
- Final HEAD: calibration execution and reproducibility HEAD
  `37ffdac439ff29df8606c2f61f57157278f321ad`; the report/evidence-only
  descendant containing this file is recorded in the final handoff.
- origin/main: `b4d253724717076188a38032d6d6204fdf15e191`; no push was performed.
- Working tree: clean at entry; final clean state is reported in the handoff
  after the report/evidence commit.
- Prior Phase 9 BLOCKED report: unchanged at
  `docs/phase_reports/phase9-kvquant-calibration-blocked.md`, repository
  storage SHA-256
  `05bbc9d21fe4bff900bd141ddc7f6daec226848178f8c0b78b7ecdaba2c180b7`.
- Phase 9P report: PASS and unchanged.
- G0: PASS.
- G1: PASS.
- G2-TQ: PASS.
- G2-KIVI: PASS.
- G2-KVQ at entry: NOT EVALUATED.
- Quality: execution LOCKED.
- Full Scan: CLOSED.

## Authority

- Method identifier: `kvquant_gqa_upstream_patch_v1`.
- Upstream repository: `https://github.com/SqueezeAILab/KVQuant.git`.
- Base commit/tree:
  `57a238357f0ffe50084670fcd5781c9848f80ea2` /
  `094e0f736f77ee327e5350cbd1eefb1c936aa77b`.
- Patch SHA:
  `db3b6fb7ec0a72e25001e1c83a5158d86512248db5c3a06c61895598d1d482d6`.
- Patched commit/tree:
  `4ad80bc8c942d0a05516d2be8f8d443a77a05900` /
  `c4f1490c9c0c4ec46099f1e95c092516df2adb4e`.
- Decision 0021: Accepted.
- Reconstruction command:
  `make KVQUANT_GQA_SOURCE_ROOT=/home/rockrock/third_party_worktrees/kvquant-gqa validate-kvquant-gqa-patch`.
- Reconstruction result: PASS; exact base commit/tree, patch digest, 15-file
  changed set, before/after hashes, and patched tree match.
- Complete upstream checkout committed: no.
- Official author GQA support claimed: no. The exact authority wording is
  “KVQuant-GQA patched upstream.”

## Minimal scope

- Plan: `docs/plans/phase9-kvquant-calibration.md`.
- General calibration framework: no.
- Existing adapters changed: no.
- Measurement Container changed: no.
- Phase 10 started: no.

## Calibration environment

- Dockerfile: `docker/calibration-kvquant.Dockerfile`.
- Base digest:
  `sha256:16ee632c5ac029deca5859f3da4c74f9e5e55f5080c10745a59653e95d8e5b44`.
- Dockerfile SHA-256:
  `e3ac0933c21c986bed2ca169c8983f6d1e6412e02bed42a282f9c604fd9c4de5`.
- Image ID/config digest:
  `sha256:127759078f2c70c9e795c7a1bb3408df1eaee8fa019319299d283dc8075b216d`.
- Python: 3.12.3.
- PyTorch: 2.12.1+cu130.
- CUDA: userspace 13.0.2; nvcc 13.0.88; driver 595.71.05.
- Transformers: 4.57.6.
- Datasets: 5.0.0.
- Clustering dependencies: NumPy 2.5.1, SciPy 1.16.1, scikit-learn 1.7.1;
  `KMeans(init="k-means++", n_init="auto", max_iter=50)` with one effective
  initialization.
- Patched source identity: exact Decision 0021 patched tree mounted read-only;
  source is absent from the image.
- Model weights in image: no.
- Credentials in image: no.

The small container smoke passed CUDA visibility, exact config/tokenizer and
BF16 model load, patched-source import, pre-RoPE hook availability, one-layer
Fisher, and one small NUQ quantizer.

## Model and tokenizer

- Model: `meta-llama/Llama-3.1-8B-Instruct`.
- Revision: `0e9e39f249a16976918f6564b8830bc894c89659`.
- Tokenizer: `meta-llama/Llama-3.1-8B-Instruct`.
- Tokenizer revision: `0e9e39f249a16976918f6564b8830bc894c89659`.
- Config SHA-256:
  `29e4c210b0d6ac178b16b2a255a568bdb23b581e50ca1ef6a6d071dd85704e6e`.
- Tokenizer hashes:
  `tokenizer.json=79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4`;
  `tokenizer_config.json=177c7b61e616fecb84c17ce0591acb92c6c4d60e9ac5ababfb940ff23bbcd424`;
  `special_tokens_map.json=6f38c73729248f6c127296386e3cdde96e254636cc58b4169d3fd32328d9a8ec`.
- RoPE: native `llama3` fields retained.
- Layers: 32.
- Query heads: 32.
- KV heads: 8.
- Head dimension: 128.
- Pre-RoPE Key: captured immediately after `k_proj` and before RoPE.
- Head expansion: none.

## Dataset

- Source: Hugging Face Hub `Salesforce/wikitext`, config
  `wikitext-2-raw-v1`.
- Revision/content root:
  `b08601e04326c79dfdd32d625aee71d232d685c3`;
  conversion revision `3f68cd45302c7b4b532d933e71d9e6e54b1c7d5e`;
  train object SHA-256
  `e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7`;
  dataset root
  `2d3a6c9877af57e6a3943824cef97ca2c036719c9aab98bfd6735d13bb2a0547`.
- Split: train only.
- Samples: 16.
- Sequence length: 2048.
- Seed: 20260721.
- Selection algorithm: join train rows in source order with `"\n\n"`;
  tokenize once with one tokenizer-default BOS, no appended EOS, padding, or
  truncation; draw 16 ordered
  `random.Random(seed).randint(0, token_count - 2048 - 1)` windows.
- Selected IDs:
  `sample-00@776416`, `sample-01@1584572`, `sample-02@1666780`,
  `sample-03@1692696`, `sample-04@1160405`, `sample-05@1960579`,
  `sample-06@29559`, `sample-07@873733`, `sample-08@1866657`,
  `sample-09@2426841`, `sample-10@527726`, `sample-11@721843`,
  `sample-12@1000843`, `sample-13@1637219`, `sample-14@394364`,
  `sample-15@2341307`; exact row ranges and per-sample checksums are in
  `docs/evidence/phase9/dataset-selection.json`.
- Token tensor root:
  `ebb7a50649f69cde64d74876af3f2e4938405e8cee94d5a2d95e51b911777d47`;
  safe file SHA-256
  `bab01ec45c5a199061c91c39e41634c97fad8faa830d57106f942c2e91074e02`.
- Test split loaded: no.
- Reconstruction: PASS; manifest identity, tensor bits, safe bytes, 16 x 2048
  shape, and checksums are exact.

## Precision and seeds

- Model forward: BF16.
- Fisher: FP32 accumulation.
- Fitting: FP16 activation tensors.
- Codebook computation: FP32.
- Python seed: 20260721.
- NumPy seed: 20260721.
- PyTorch seed: CPU and CUDA 20260721.
- Clustering seed: 20260721.
- Deterministic settings: one thread; deterministic algorithms enabled;
  cuDNN deterministic enabled and benchmark disabled; TF32 disabled; model
  `eval`; `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- Bitwise determinism is not claimed where serializer byte canonicality is not
  defined.

## Fisher

- Command: the exact network-disabled digest-bound Docker argv is recorded in
  `commands.json`; its operation is
  `python3 /opt/phase9/worker.py run-fisher --source-root /source
  --token-tensor /output/tokens/input_ids.safetensors --output-dir
  /output/fisher` with the exact model manifest and offline cache arguments.
- Layer coverage: 32/32.
- K coverage: 32/32 pre-RoPE.
- V coverage: 32/32 post-`v_proj`.
- Shapes: 64 tensors, each `[1, 32768, 1024]`.
- Dtype: FP32.
- Finite values: PASS; NaN 0, Inf 0.
- Root digest:
  `a4cd9ad1e28332cc38c0a8bd19c10af079379655baaa2e5066aac6e23472117b`.
- Representative K replay: layer 0 PASS, exact, max absolute/relative
  difference 0.
- Representative V replay: layer 0 PASS, exact, max absolute/relative
  difference 0.
- Tolerance: `atol=1e-8`, `rtol=1e-5`, frozen before full calibration.

## Quantizers

- kvq4:
  `a8c009633ac4cad952deb2a2fa96c44ef928a1510dadcf11dee29a7a3efe1bf6`.
- kvq3:
  `97518129cc64ffa445722cb0802b3082631841de50835cbdf2c85c36a0c1579f`.
- kvq2:
  `b9bb3a8699aa38fb2a5707ff036814971552462692a180431f6f68df9624560e`.
- Layer coverage: 32 layers x K/V in every family; 320 tensors per family.
- NUQ: yes.
- Dense-and-sparse: yes.
- Fisher linkage: all families bind the same Fisher root.
- Serialization: non-executable safetensors; no source-native pickle was
  retained or published.
- Regeneration: PASS in fresh processes. Every family regenerated all 320
  tensors exactly with max absolute/relative difference 0. File bytes differ
  only in non-canonical JSON header key order, so the pre-frozen fallback rule
  accepts exact tensor identity.
- Checksums: recorded above and in the immutable checksum ledger.

## Policies

- Sink tokens: 5, indices 0-4, stored FP16 and excluded from dense quantized
  history and sparse selection.
- Key cap: 12.
- Value cap: 12.
- Entries per tail: 6 lower and 6 upper.
- Shared across bits: yes.
- Tie-breaking: value then ascending native index; equal-value replay is
  bitwise deterministic.
- Outlier value dtype: float32.
- Outlier index dtype: int32.
- Unused-slot behavior: values and indices zero-filled.
- Quality/performance used for selection: no.

## Layer statistics

- File: `layer_stats.parquet`, SHA-256
  `0c7fccc0dfb707b0457b99f2f6724ce855887e05f025c9930840d05ecca3b8de`.
- Rows: 192.
- Layer coverage: 0-31 complete.
- Role coverage: K and V complete.
- Bit-width coverage: 2, 3, and 4 complete.
- Cap-hit rates: K mean `0.0005200685266764562`, range
  `0.0..0.008229319627998041`, 544 hits per bit width; V mean
  `0.8671167553842388`, range
  `0.8363313754282917..0.9174314733235438`, 907018 hits per bit width.
- Invalid values: NaN 0, Inf 0, clipping/saturation 0; no unusual layer
  omitted.

## Calibration bundle

- Calibration ID: `kvqcal-cdb724c806d64d095c040d2673a987a3`.
- Local path:
  `calibration/kvquant/kvqcal-cdb724c806d64d095c040d2673a987a3`.
- Root digest:
  `8148306d08205af376994b022f189a0d6837915cd279ca8af6b104e1f4b46ccf`.
- COMPLETE: PASS; written last and read-only.
- Inventory: PASS; 68 objects.
- Checksum ledger: PASS.
- Existing bundle overwritten: no.
- Large binaries committed to Git: no.

Two finalized failed calibration IDs remain preserved without replacement:
`kvqcal-e173e987e8b384fb5a78af196d67a032` for the tokenizer-probe failure and
`kvqcal-62c0797d73a7dbd991bd1b14da39abed` for the superseded byte-only
safe-serialization replay rule.

## Durable publication

- R2 URI:
  `r2://kvbench-artifacts/kvbench/sha256/8148306d08205af376994b022f189a0d6837915cd279ca8af6b104e1f4b46ccf/`.
- Initial publication: `TransportError` preserved before `COMPLETE` after six
  identical objects; conditional retry PASS with 62 uploads and six exact
  existing-object verifications.
- COMPLETE-last: PASS.
- Clean retrieval: PASS into a fresh empty directory; 68 expected objects and
  no unexpected objects.
- Checksum result: PASS; inventory, ledger, marker, objects, and root agree.
- Bucket Lock: private bucket `kvbench-artifacts`, exact indefinite rule
  `kvbench-evidence-indefinite` covering `kvbench/sha256/`; verification PASS.
- Credential leakage: none. Credential values were not recorded, `.env` was
  not read, and no R2 credential entered the calibration container or bundle.

## Tests

- package-lock: PASS.
- make test: PASS.
- make checks: PASS.
- TurboQuant regression: PASS.
- KIVI reference/admission regression: PASS.
- Patch reconstruction: PASS, both static and exact-checkout reconstruction.
- Container: PASS.
- Dataset: PASS.
- Fisher: PASS.
- Quantizer: PASS.
- Reproducibility: PASS.
- Policy: PASS.
- Governance: PASS.
- Historical evidence: PASS and unchanged.

The focused Phase 9 governance, scope, R2, and schema set passed 48/48 tests.
The one stale generic test assumption discovered during full regression was
corrected narrowly: it now explicitly nulls a required calibration/runtime
parameter before asserting that a false `resolved` state is rejected.

## Gates

- G0: PASS.
- G1: PASS.
- G2-TQ: PASS.
- G2-KIVI: PASS.
- G2-KVQ: NOT EVALUATED.
- Global G2: NOT EVALUATED.
- G3: NOT EVALUATED.
- G4: NOT EVALUATED.
- G5: NOT EVALUATED.
- Full Scan: CLOSED.

## Quality governance

- Quality execution: LOCKED.
- PPL/LongBench: not run.
- PERFORMANCE_DATA_FROZEN: absent.

## Preservation

- Prior Phase 9 report changed: no.
- Phase 9P evidence changed: no.
- Measurement Container changed: no.
- TurboQuant changed: no.
- KIVI changed: no.
- Historical evidence changed: no.
- Existing run overwritten: no.
- Performance data: none.
- Profiler data: none.
- Quality data: none.

## Changed files

- Calibration contract and environment:
  `.gitignore`;
  `Makefile`;
  `docker/calibration-kvquant.Dockerfile`;
  `docker/calibration-kvquant.Dockerfile.dockerignore`;
  `docker/calibration-kvquant.image.json`;
  `docker/calibration-kvquant.python-freeze.txt`;
  `docker/calibration-kvquant.requirements.txt`;
  `docs/plans/phase9-kvquant-calibration.md`;
  `docs/evidence/phase9/model-snapshot-manifest.json`.
- Calibration implementation and schemas:
  `scripts/phase9_kvquant_calibration.py`;
  `scripts/phase9_kvquant_worker.py`;
  `src/kvbench/runtime/__init__.py`;
  `src/kvbench/runtime/artifacts.py`;
  `src/kvbench/schema/__init__.py`;
  `src/kvbench/schema/config.py`;
  `src/kvbench/schema/phase9.py`;
  `src/kvbench/schema/phase9_config.py`.
- Compact final evidence and publication:
  `docs/evidence/phase9/calibration-checksums.sha256`;
  `docs/evidence/phase9/calibration-manifest.json`;
  `docs/evidence/phase9/dataset-selection.json`;
  `docs/evidence/phase9/layer-stats-summary.json`;
  `docs/evidence/phase9/reproducibility.json`;
  `docs/evidence/phase9/r2-publication.json`;
  `scripts/r2_artifact.py`;
  this report.
- Configuration and governance:
  `configs/methods/kvquant.yaml`;
  `docs/method_notes/kvquant.md`;
  `docs/status.md`;
  `docs/blockers.md`;
  `docs/risk_register.md`;
  `docs/tasks.md`;
  `scripts/validate_phase2.py`.
- Focused tests:
  `tests/schema/test_config_schema.py`;
  `tests/schema/test_phase9_schema.py`;
  `tests/unit/test_phase9_calibration.py`;
  `tests/unit/test_phase9_governance.py`;
  `tests/unit/test_phase9_scope.py`;
  `tests/unit/test_phase9p_governance.py`;
  `tests/unit/test_r2_artifact.py`.

## Commits

- `ebec83d` — `phase9: freeze calibration container contract`.
- `9fd9a20` — `phase9: implement frozen KVQuant calibration lane`.
- `a9b49e5` — `phase9: bind tokenizer defaults to encode path`.
- `37ffdac` — `phase9: verify regenerated quantizer tensors`.
- One report/evidence-only descendant contains the durable receipt, compact
  manifests, governance updates, and this report; its exact SHA is recorded in
  the final handoff.
- No push, tag, or pull request was performed.

## Risks

- B-006 remains open for repository-wide KVQuant license and adapted-source
  lineage. It does not invalidate the private checksum-bound calibration but
  gates Phase 10 reference work, copying, and redistribution.
- Safetensors header key ordering is non-canonical in fresh-process output.
  Tensor names, metadata, shapes, dtypes, and all 320 tensor values per family
  are exact under the rule frozen before the successful fresh ID.
- The first R2 publication transport failure is preserved. Conditional
  creation and authoritative checksum readback prevented replacement or
  conflicted bytes.

## Blockers

- None for completion of this exact Phase 9 calibration.
- B-006 is the minimum remaining blocker before Phase 10/E10 reference work;
  E10 and E11 have not started.

## Scientific interpretation

The exact patched-upstream KVQuant calibration inputs and artifacts are frozen
and reproducible. No official-source, quality, speedup, memory-benefit,
physical-HBM, knee, or capacity claim is made.

## Next action

Phase 10 KVQuant Reference Lane may be proposed only in a separate new task.
