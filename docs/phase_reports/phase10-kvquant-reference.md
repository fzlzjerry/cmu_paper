PHASE 10 REPORT

Status: PASS

Entry:
- Starting HEAD: a873fc93754fa86bfb757fce476388897bee8dca
- Final HEAD: Phase 10 execution/artifact HEAD 4cdbd32806095ba8229f85cfd7f919434006f9e1; the governance/report-only descendant containing this file is recorded in the final handoff
- origin/main: a873fc93754fa86bfb757fce476388897bee8dca; no push was performed
- Working tree: CLEAN after the governance/report commit
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

Contract correction:
- Previous blocker: The original fixture contract required unreachable non-sink Value active counts of zero and fewer than 12.
- Resolution: Decision 0023 corrects the fixture contract to the frozen source-faithful behavior; no algorithm change was made.
- Key fixture counts: 0 / 6 / 12
- Value non-sink count: always 12
- Value sink count: always 0
- Value selection: six lowest plus six highest
- Algorithm/source authority changed: No
- Decision 0021 changed: No
- Calibration bundle changed: No
- Previous blocked report changed: No; repository storage SHA-256 0362ac36f03ec8b92c0b154ad85fd4e810c3d45332f7c063b1da00d7adc94e4d
- G2-KVQ: NOT EVALUATED

Authority:
- Method identifier: kvquant_gqa_upstream_patch_v1
- Upstream base commit/tree: 57a238357f0ffe50084670fcd5781c9848f80ea2 / 094e0f736f77ee327e5350cbd1eefb1c936aa77b
- Patch SHA: db3b6fb7ec0a72e25001e1c83a5158d86512248db5c3a06c61895598d1d482d6
- Patched commit/tree: 4ad80bc8c942d0a05516d2be8f8d443a77a05900 / c4f1490c9c0c4ec46099f1e95c092516df2adb4e
- Decision 0021: Accepted and unchanged
- Reconstruction: PASS; detached clean source matched the exact base commit/tree, patch digest, 15 changed paths with before/after hashes, and patched commit/tree
- Official author GQA support claimed: No

Minimal scope:
- Plan: docs/plans/phase10-kvquant-reference.md; frozen before final generation
- General reference framework: No
- Measurement Adapter started: No
- Calibration rerun: No
- Existing methods changed: No

Reference environment:
- Reused Phase 9 environment: Reused as the exact base; one thin reference image adds only a checksum-locked isolated tokenizers 0.15.2 overlay
- Dockerfile, if created: docker/reference-kvquant.Dockerfile; SHA-256 f1b2f2a6f6f15bf364eb3a8b7a26f01504edbe2dbcfe74b619b1c519120a618e
- Base digest: sha256:127759078f2c70c9e795c7a1bb3408df1eaee8fa019319299d283dc8075b216d
- Image/config digest: sha256:24eb3f6ff39b72f45c353acfbef6ce2d9aaac0860180b4dde8b937593176714b
- Python: 3.12.3
- PyTorch: 2.12.1+cu130
- CUDA: userspace 13.0; NVCC 13.0.88
- Compiler: GCC 13.3.0
- Extension: quant_cuda.cpython-312-x86_64-linux-gnu.so; 12,991,816 bytes; SHA-256 53bee7b4b5a0dead6adb682df1343330963b41149d12c2a876888c1c2ede9597
- Model weights in image: No
- Credentials in image: No

Calibration binding:
- Calibration ID: kvqcal-cdb724c806d64d095c040d2673a987a3
- Calibration root: 8148306d08205af376994b022f189a0d6837915cd279ca8af6b104e1f4b46ccf
- R2 source: r2://kvbench-artifacts/kvbench/sha256/8148306d08205af376994b022f189a0d6837915cd279ca8af6b104e1f4b46ccf/
- COMPLETE: PASS
- Inventory/checksums: PASS; local bundle and new-empty-directory R2 retrieval validated all 68 objects
- kvq4: PASS; 320 exact tensor identities; manifest SHA-256 2b0510d7e2cbcbbed6e0ddafb094e9326be2f3372c853b08ba1a155e5cd38691
- kvq3: PASS; 320 exact tensor identities; manifest SHA-256 94a45ee3e2b64eff69c7e53d0a4afc1949198247ec24b6220bea126ddc7fe614
- kvq2: PASS; 320 exact tensor identities; manifest SHA-256 0cf292aec40f51e2db46b2f601940f36126187e8cc3cf159938dec5d87eb8e57
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
- Dtype: BF16 interface; full-precision source sink boundary FP16; sparse values float32 and indices int32

Fixtures:
- Root: fixture ID kvqref-a50af6511c314b6394e58a7f81ceefb8; local path reference/kvquant/fixtures; root SHA-256 32cdf465a361dd6695b66ccbea0a462bddc075fd9778d0aa8cdaa3f94e6f63ab
- Total cases: 9
- kvq4 no/few/cap: PASS as key_zero_value_fixed12 / key_few_value_fixed12 / key_cap_value_fixed12
- kvq3 no/few/cap: PASS as key_zero_value_fixed12 / key_few_value_fixed12 / key_cap_value_fixed12
- kvq2 no/few/cap: PASS as key_zero_value_fixed12 / key_few_value_fixed12 / key_cap_value_fixed12
- Store: PASS; deterministic cache state for positions 0-16
- Append: PASS; deterministic appended position 17 and resulting cache state
- Decode: PASS; deterministic finite reference output and independent explicit reconstruction control
- Dense K: PASS; source pack order, logical/physical shapes, payload checksum, and bytes recorded
- Dense V: PASS; source pack order, logical/physical shapes, payload checksum, and bytes recorded
- Metadata: PASS; frozen codebooks/LUTs, scales, minima/zeros, ranges, thresholds, dtypes, checksums, and bytes recorded without recalibration
- Sparse values: PASS; fixed physical capacity, float32 dtype, stable ordering, and checksums
- Sparse indices: PASS; fixed physical capacity, int32 dtype, no duplicate/overlap, and checksums
- Sink K/V: PASS; positions 0-4, native eight-head full-precision storage, excluded from dense history and sparse selection
- Checksums: PASS; per-fixture ledgers, 113-object root inventory, root ledger, and COMPLETE validate
- Existing-fixture overwrite: PASS; differing finalized bytes are rejected, byte-identical deterministic regeneration is accepted without replacement

Outlier behavior:
- No-outlier K/V counts: Key non-sink 0; Value non-sink 12; Value sink 0
- Few-outlier K/V counts: Key non-sink 6; Value non-sink 12; Value sink 0
- Cap-reached K/V counts: Key non-sink 12; Value non-sink 12; Value sink 0
- Overflow: None
- Duplicate indices: None
- Lower/upper overlap: None
- Tie-breaking: PASS; stable value then flat-index ordering, including equal-value stress control
- Unused slots: Key unused slots and all sink Value sparse slots are exactly zero; non-sink Value has no unused slots
- Physical capacity: 12 slots for both Key and Value in every case

Pre-/post-RoPE:
- Quantized Key: k_proj output before RoPE
- Sink Key: native attention-ready post-RoPE Key stored separately in full precision
- Attention Key: native Llama-3.1 RoPE is applied for reference decode; quantized pre-RoPE and attention-ready semantics remain distinct
- Value: v_proj output without RoPE
- Position IDs: exact positions 0 through 17
- Head expansion: None in source-under-test; explicit expansion appears only in the independent numerical control

GQA:
- H_Q: 32
- H_KV: 8
- Groups: 4
- Mapping: kv_head = query_head // 4
- Dense storage: Native eight KV heads
- Sparse storage: Native eight KV heads
- Sink storage: Native eight KV heads
- repeat_kv: Absent from source-under-test
- Expanded temporary: None in source-under-test; independent control only

Execution path:
- Dense quantization/packing: Patched CUDA parallel store and append
- Sparse correction: Fused dense-plus-sparse CUDA correction with stable lower-then-upper selection
- Sink handling: Separate native-H_KV FP16 storage and matmul
- Decode/dequantization: Direct compressed-cache consumption
- Full-prefix temporary: No
- Backend fallback: No
- Reference dynamic allocation: Yes; recorded as a reference-only property
- Timing claim: None; traces contain no duration fields

Byte accounting:
- Bit width/case: Exact rows below in zero / few / cap order
- Dense K bytes: kvq4 9,216; kvq3 6,912; kvq2 4,608
- Dense V bytes: kvq4 9,216; kvq3 6,912; kvq2 4,608
- K metadata: kvq4 78,144; kvq3 45,344; kvq2 28,944
- V metadata: kvq4 1,216; kvq3 608; kvq2 304
- K outlier-value bytes: 864 for every fixture
- K outlier-index bytes: 864 for every fixture
- V outlier-value bytes: 864 for every fixture
- V outlier-index bytes: 864 for every fixture
- Sink K bytes: 10,240 for every fixture
- Sink V bytes: 10,240 for every fixture
- Padding: 0
- Workspace: 0 persistent owned bytes
- Actual allocated total: kvq4 121,728; kvq3 83,712; kvq2 62,400
- Active logical total: kvq4 114,080 / 114,704 / 115,328; kvq3 77,504 / 78,128 / 78,752; kvq2 57,552 / 58,176 / 58,800
- Logical BF16: 73,728 bytes for every fixture
- rho_alloc: kvq4 1.6510416666666667; kvq3 1.1354166666666667; kvq2 0.8463541666666666
- r_alloc: kvq4 0.6056782334384858; kvq3 0.8807339449541285; kvq2 1.1815384615384616
- Reciprocal check: PASS; maximum absolute error 2.220446049250313e-16 <= 1e-9
- r_hbm: null for every fixture

CUDA:
- Native SM120: PASS on NVIDIA RTX PRO 6000 Blackwell Workstation Edition, compute capability 12.0, driver 595.71.05
- sm_120 cubin: PASS
- compute_120 PTX: PASS
- Forced PTX/JIT: PASS for 4/3/2-bit dense and sparse GQA/MHA paths
- Sanitizer cases: kvq4 zero/few/cap, kvq3 few, kvq2 zero/cap
- Sanitizer result: PASS; Compute Sanitizer 2025.3.1.0, zero errors, zero leaked allocations, zero leaked bytes
- Fallback: None

Durable publication:
- Local root: reference/kvquant/fixtures; 32cdf465a361dd6695b66ccbea0a462bddc075fd9778d0aa8cdaa3f94e6f63ab
- R2 URI: r2://kvbench-artifacts/kvbench/sha256/32cdf465a361dd6695b66ccbea0a462bddc075fd9778d0aa8cdaa3f94e6f63ab/
- Initial publication: PASS; 113 conditional content-addressed objects uploaded
- COMPLETE-last: PASS; publication-order SHA-256 f0d0f87002d59700298ffa6fc4e67c03c19a88917f9a7bb8f525709f539993ff
- Clean retrieval: PASS into a new empty directory at 2026-07-29T04:31:56.117735Z
- Checksums: PASS; all 113 objects, inventory, ledger, COMPLETE, root, and unexpected-file check
- Bucket Lock: PASS; private bucket kvbench-artifacts, exact enabled indefinite rule kvbench-evidence-indefinite for kvbench/sha256/
- Credential leakage: None; credentials were host-side PRESENT/MISSING checks only, .env was not read, and no credential value entered the image, artifacts, metadata, stdout/stderr evidence, R2, or Git

Tests:
- package-lock: PASS
- make test: PASS
- make checks: PASS
- TurboQuant regression: PASS; make validate-admission-turboquant
- KIVI regression: PASS; make validate-reference-kivi and make validate-admission-kivi
- Patch reconstruction: PASS; make validate-kvquant-gqa-patch
- Calibration validation: PASS; make validate-calibration-kvquant and clean R2 retrieval
- Reference fixtures: PASS; make reference-kvquant and make validate-reference-kvquant; all nine cases and deterministic second generation
- Outlier behavior: PASS; fixed Value extrema, sink zero, Key 0/6/12, ties, no duplicate/overlap/overflow, legacy aliases rejected
- Pre-/post-RoPE: PASS
- GQA: PASS; native 32Q/8KV mapping and no implementation-side expansion
- Bytes: PASS; exact owned totals, fixed-cap allocation, active logical totals, reciprocal ratios, and null r_hbm
- CUDA: PASS; native SM120, cubin, PTX, and forced PTX/JIT
- Sanitizer: PASS; six-case path-family matrix with zero errors and leaks
- Governance: PASS; Measurement Adapter remains absent and factory remains fail-closed; Full Scan and quality remain locked
- Historical evidence: PASS; original Phase 10 BLOCKED report and all earlier phase evidence remain unchanged

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
- PPL/LongBench: NOT RUN
- PERFORMANCE_DATA_FROZEN: absent

Preservation:
- Original Phase 9 BLOCKED report changed: No
- Phase 9 calibration bundle changed: No
- TurboQuant changed: No
- KIVI changed: No
- Measurement Container changed: No
- Historical evidence changed: No; the recovered prior Phase 10 BLOCKED report is preserved byte-for-byte under its custody record
- Existing run overwritten: No; the first unpublished non-canonical-header local finalization remains preserved under its original ID and a fresh ID was used
- Performance data: None
- Profiler data: None; only duration-free reference_trace evidence
- Quality data: None

Changed files:
- Makefile
- docker/reference-kvquant.Dockerfile
- docs/decisions/0023-phase10-kvquant-source-faithful-sparse-fixture-semantics.md
- docs/evidence/phase10/blocked-report-custody.json
- docs/evidence/phase10/cuda-validation.json
- docs/evidence/phase10/r2-publication.json
- docs/phase_reports/phase10-kvquant-reference-blocked.md
- docs/phase_reports/phase10-kvquant-reference.md
- docs/plans/phase10-kvquant-reference.md
- docs/risk_register.md
- docs/status.md
- docs/tasks.md
- reference/kvquant/README.md
- reference/kvquant/generate_fixtures.py
- reference/kvquant/validate_fixtures.py
- reference/kvquant/source_manifest.json
- reference/kvquant/environment.json
- reference/kvquant/calibration_manifest.json
- reference/kvquant/build_manifest.json
- reference/kvquant/fixtures/ (exact 113-object immutable fixture bundle)
- scripts/validate_phase2.py
- tests/cuda/phase10_kvquant_sanitizer_probe.py
- tests/unit/test_measurement_container.py
- tests/unit/test_phase9_governance.py
- tests/unit/test_phase9p_governance.py
- tests/unit/test_phase10_kvquant_reference.py
- tests/unit/test_phase10_scope.py

Commits:
- 3f2569c: phase10r: correct sparse fixture contract
- ff3f040: phase10r: add locked KVQuant reference runner
- 4cdbd32: phase10r: finalize and publish KVQuant fixtures
- Governance and final Phase 10 report: this report-only descendant; exact SHA is recorded in the final handoff

Risks:
- Value fixed sparse occupancy is a method property: every non-sink row owns fixed 12-entry sparse overhead and must not be modeled as data-dependent.
- Reference-only dynamic allocation is recorded and is not admissible Measurement-Lane behavior.
- The frozen 3-bit source packing/decode behavior is preserved exactly and must not be generalized beyond this authority.
- One unpublished local finalized bundle exposed non-canonical safe-format JSON header ordering. It remains immutable; canonical serialization and a fresh fixture ID now reproduce byte-for-byte.

Blockers:
- None.

Scientific interpretation:
- The KVQuant-GQA patched-upstream numerical reference fixtures are reproducible for the exact frozen calibration, model, source authority, and compact geometry. This is not a claim of official author GQA support, quality, speedup, memory benefit, physical HBM traffic, knee location, or capacity improvement.

Next action:
- Phase 11 KVQuant Measurement Adapter may be proposed in a separate new task. It was not started here.
