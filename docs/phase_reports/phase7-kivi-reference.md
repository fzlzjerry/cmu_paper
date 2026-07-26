PHASE 7 REPORT

Status: PASS

Entry:
- Starting HEAD: 3417ea0e7f322369eed21bb787a9a9a19b0a69bd
- Final HEAD: commit containing this report; exact SHA is recorded in the final handoff
- Working tree at entry: clean
- Final working tree: clean after the final Phase 7 commit
- Phase 6 report: PASS
- Phase 6 admission root: outer root 8c4cf76f76bb17e648dfd911f11e268235ed827a9983814b774b9e95405496b0; original root f003bc3dc5de6b67a6d8f1b8bed7fa49b7f90f9d7edc4d1383e2d97c8aa19d6d unchanged
- Complete Phase 6 bundle verified: PASS; 176-object outer bundle, original complete bundle, nine admission runs, MethodAdmissionReport, publication record, final PASS report, inventory, ledger, and clean retrieval
- G0: PASS
- G1: PASS
- G2-TQ: PASS
- Quality execution: LOCKED
- Full Scan: CLOSED

Minimal scope:
- Plan: docs/plans/phase7-kivi-reference.md
- General reference framework added: no
- Measurement Container modified: no
- TurboQuant adapter modified: no
- Phase 8 work started: no

Pinned KIVI source:
- Repository: https://github.com/jy-yuan/KIVI.git
- Exact commit: 876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6
- Exact tree: base c94c31b2cfd44eeb9a18cff9dcdf03adff4ac49b; Decision 0018 patched tree b617493dea5aff1a754cd27ad6be12ac512b2aee
- Commit date: 2025-11-20T12:34:32-08:00
- License: MIT
- Relevant files: 15 upstream files plus one Decision 0018 GQA helper, all bound in third_party/LOCK.json and reference/kivi/source_manifest.json
- Source lock: PASS; Git blob and SHA-256 bindings verified
- Floating branch: no
- Unofficial fork used: no

Reference environment:
- Definition: docker/reference-kivi.Dockerfile
- Base image digest: sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e
- Dockerfile SHA-256: b319d0c15d43d70ce364123d447b820ad1e312d6c60aa737c9707c701da17912
- Image ID/config digest: platform/local image sha256:f27e4cdef6bd15f18ab76b1fe0e4413ede004b42538c74e3dd90d04172406f75; OCI config sha256:0915dc8488fd6c9a150a3b4f56bb4b97b5dbdb7c51d96cda2d431df20e856ce3
- Python: 3.12.3
- PyTorch: 2.12.1+cu130
- CUDA: toolkit/userspace 13.0.2; PyTorch runtime 13.0; nvcc 13.0.88
- Compiler: gcc/g++ 13.3.0
- Extension build command: /opt/kvbench/.venv/bin/python setup.py build_ext --inplace --verbose
- Architecture flags: TORCH_CUDA_ARCH_LIST=12.0+PTX; CUDAARCHS=120; CMAKE_CUDA_ARCHITECTURES=120; compute_120 and sm_120 gencode
- Extension SHA-256: 45d29ec1a3cecc4b253d1d1dd6139ef4f91cff88993db61a9d73685314851aa9
- Measurement Container modified: no
- Credentials/model weights in image: none

CUDA compatibility:
- Native SM120: PASS on NVIDIA RTX PRO 6000 Blackwell, compute capability 12.0
- sm_120 cubin: present
- compute_120 PTX: present
- Forced PTX/JIT: PASS through PTX-only official object pruning, unchanged relink, and fresh-process execution; derived extension SHA-256 bfd9db4f880ff0618214ce5b5029c2299b1b39597a3f792ba7d26a40d9ce28d3
- Compute Sanitizer: PASS; four distinct 2-bit/4-bit key/value families covered; ERROR SUMMARY 0 errors
- Unsupported fallback: absent
- no kernel image: absent

Verified algorithm:
- Key quantization direction: per-channel, grouped along the token dimension
- Value quantization direction: per-token, grouped along the head dimension
- Group semantics: group_size=32; key groups span tokens and value groups span head-dimension elements
- Metadata precision: scale and minimum-offset tensors use the official execution input dtype, FP16 in the half-only CUDA ABI
- Packing: signed quantization metadata plus packed 2-bit/4-bit payloads in int32 words
- Residual policy: recent K/V retained unquantized; older history stored packed
- Rollover trigger: key block quantizes when residual K reaches 32; value moves its oldest token when residual V would exceed 32
- GQA support: PASS under the exact Decision 0018 patch with eight-head K/V operands
- Full-prefix dequantization: absent; packed history is consumed directly by official GEMV and residual storage remains separate

Configurations:
- K4/V4: PASS
- K2/V4: PASS
- K2/V2: PASS
- K4/V2 held-out: PASS
- group_size: 32
- residual_length: 32
- Unsupported behavior: unsupported bits and non-32/8 GQA geometry are rejected

Fixture geometry:
- Batch: 1
- Query heads: 32
- KV heads: 8
- Head dimension: 128
- Basic context: store L=17; append/decode L=18
- Rollover contexts: L=31 before, L=32 boundary, L=33 after; post-rollover decode L=34; static layout check L=64
- Seed: 20260726
- Dtype: BF16 input on an exact BF16-FP16-BF16 grid; unchanged official extension executes FP16 because its CUDA binding is half-only

Fixtures:
- Root: reference/kivi/fixtures
- Mandatory count: 3
- Held-out count: 1
- Store: PASS
- Append: PASS
- Decode: PASS
- Quantized K: payload and identity preserved for all four configurations
- Quantized V: payload and identity preserved for all four configurations
- Metadata: key/value scale and minimum-offset tensors preserved
- Residual K/V: full tensor bytes, shapes, dtypes, token indices, and checksums preserved
- Rollover records: before, boundary, after, and post-rollover decode preserved for every configuration
- Checksums: 9-entry nested fixture ledger PASS
- Existing-fixture overwrite prevention: identical regeneration returns existing_identical; differing regeneration is refused

Byte accounting:
- Configuration/context: exact actual records at L=31/32/33 and source-layout calculation at L=64 for k4v4, k2v4, k2v2, and k4v2
- Quantized K bytes: representative L=33 values 16384, 8192, 8192, 16384
- Quantized V bytes: representative L=33 values 512, 512, 256, 256
- Metadata bytes: representative L=33 value 4224 for every configuration
- Residual K bytes: representative L=33 value 2048
- Residual V bytes: representative L=33 value 65536
- Padding: 0
- Workspace: 0 persistent bytes
- Actual total: representative L=33 values 88704, 80512, 80256, 88448
- Logical BF16 bytes: 135168 at L=33
- r_alloc: representative L=33 values 0.65625, 0.5956439393939394, 0.59375, 0.6543560606060606; values change across L=31/32/33/64
- Storage agreement: PASS; persistent category sum equals actual source-owned tensor storage
- r_hbm: null / not populated

Residual rollover:
- Before: L=31; no quantized history; K/V residual tokens 0-30
- Boundary: L=32; key tokens 0-31 quantized, K residual empty, V residual tokens 0-31
- After: L=33; quantized K tokens 0-31, K residual token 32, quantized V token 0, V residual tokens 1-32
- Tokens moved: K 0-31 at L=32; V token 0 at L=33; V token 1 at L=34
- Missing tokens: none
- Duplicate tokens: none
- Reallocation observed: yes in the dynamic reference path; no Measurement-Lane claim
- Source-faithful: PASS

GQA:
- H_Q: 32
- H_KV: 8
- Storage geometry: persistent K/V and BMM K/V operands retain H_KV=8
- Source audit: PASS for patched tree; original repeat_kv defect remains immutable historical evidence
- Runtime audit: PASS on CPU and SM120 at contexts 17 and 33, plus all fixture states
- repeat_kv: absent from executed patched path
- repeat_interleave: absent
- Expanded temporary: absent
- Head mapping: query_head // 4
- Final verdict: PASS_NATIVE_EIGHT_HEAD_KV_STORAGE

Reference trace:
- Mechanism: torch.profiler CPU/CUDA activities with names and shapes retained and all duration fields discarded
- Quantize/store kernels: _minmax_along_last_dim and _pack_along_last_dim
- Append kernels: aten::bmm and source-owned aten::cat
- Decode/dequant kernels: official bgemv2_kernel_outer_dim and bgemv4_kernel_outer_dim
- Rollover operations: quantize/pack, metadata construction, residual concatenation, and native-eight-head BMM
- Full-prefix temporary: absent
- Backend fallback: absent
- Timing claim: none

Graph information:
- Upstream support: undocumented; dynamic torch.cat and allocations observed
- Reference graph smoke: NOT RUN because it would require non-minimal redesign
- Deferred to Phase 8: static, graph-safe implementation only if separately authorized

Durable publication:
- Local root digest: abd164da0adf9e0c1404e8fba1f6a6e42e57944481cdf060b91e8cef175ed302
- R2 URI: r2://kvbench-artifacts/kvbench/sha256/abd164da0adf9e0c1404e8fba1f6a6e42e57944481cdf060b91e8cef175ed302/
- Initial publication: PASS; two earlier valid append-only roots remain preserved and non-final after successive NOTICE corrections
- COMPLETE-last: PASS
- Clean retrieval: PASS into a new empty temporary directory
- Checksum result: 30/30 objects; inventory, ledger, COMPLETE, and root digest PASS
- Bucket Lock: enabled exact-prefix rule kvbench-evidence-indefinite, Indefinite retention, bucket private
- Credential leakage: none; variables reported only PRESENT/MISSING; .env not read

Commands:
- git status --short; git rev-parse HEAD
- make package-lock-check; make test; make checks
- make validate-reference-turboquant; make validate-admission-turboquant
- make validate-kivi-b019-patch
- make reference-kivi; make validate-reference-kivi
- scripts/r2_artifact.py publish and verify for the final content root

Tests:
- package-lock: PASS
- make test: PASS
- make checks: PASS
- TurboQuant admission regression: PASS
- source-lock: PASS
- build: PASS
- fixture: PASS
- rollover: PASS, including corrupted-boundary rejection
- byte-layout: PASS, including category/storage equality and corruption rejection
- GQA: PASS, including materialization-corruption rejection
- sanitizer: PASS, zero errors
- governance: PASS; adapter fail-closed, Full Scan closed, quality locked, r_hbm absent
- historical evidence: PASS; Measurement Container, TurboQuant, Phase 6, Decision 0017, and earlier artifacts unchanged

Admission gates:
- G0: PASS
- G1: PASS
- G2-TQ: PASS
- G2-KIVI: NOT EVALUATED; Measurement Adapter is not implemented
- Global G2: NOT EVALUATED
- G3: NOT EVALUATED
- G4: NOT EVALUATED
- G5: NOT EVALUATED
- Full Scan: CLOSED

Quality governance:
- Quality execution: LOCKED
- Quality benchmark: NOT RUN
- PERFORMANCE_DATA_FROZEN: absent

Preservation:
- Authorized Measurement Container changed: no
- TurboQuant fixtures changed: no
- TurboQuant adapter changed: no
- Historical evidence changed: no
- Existing run overwritten: no
- Formal performance data: none
- Profiler timing: none retained
- Quality data: none

Changed files:
- Build/reproduction: Makefile and docker/reference-kivi.Dockerfile
- Reference lane: reference/kivi/README.md, environment/build/source manifests, package freeze, generator, validator, and ten compact fixture/control files
- Governance: KIVI plan/note, status, blockers, risks, tasks, NOTICE, R2 receipt, and this report
- Validation: scripts/validate_kivi_b019_patch.py, scripts/validate_phase2.py, three focused Phase 7 unit-test modules, and one narrow Measurement Container regression assertion

Commits:
- One final Phase 7 implementation/evidence commit; exact SHA is recorded in the final handoff
- No push, tag, or PR

Risks:
- The authority is patched official source, not unmodified upstream or paper-era equivalence.
- The upstream CUDA ABI is half-only; the fixture BF16 boundary is exact-grid correctness evidence, not general BF16 adapter support.
- The reference path uses torch.cat and dynamic allocations; it is not Measurement-Lane timing or Graph evidence.
- Two failed local stagings and two earlier valid R2 roots remain preserved; only the final corrected root is authoritative.

Blockers:
- None for Phase 7.
- G2-KIVI remains unavailable until a separately authorized Phase 8 Measurement Adapter exists.

Scientific interpretation:
- The exact official KIVI base plus Decision 0018 patch, locked SM120 build, deterministic fixtures, rollover semantics, actual byte layout, and native-eight-head GQA reference path are reproducible.
- No speedup, HBM, knee, capacity, or quality claim is made.

Next action:
- Phase 8 KIVI Measurement Adapter may be proposed in a separate new task; it was not started here.
