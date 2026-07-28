# Third-party provenance and notices

Phase 0 records exact source pins plus explicit commit-resolution plans for
attributed or embedded code whose upstream lineage is unresolved. Phase 5
executes only the pinned vLLM source described below; the other sources remain
unbuilt and unexecuted. Acquisition must use the commit field in `LOCK.json`;
branch names are informational only.

## vLLM / TurboQuant reference authority

- Repository: https://github.com/vllm-project/vllm
- Pinned release: v0.25.1
- Commit: 752a3a504485790a2e8491cacbb35c137339ad34
- Declared license: Apache License 2.0
- Source tree: 3ec7a4eb00f9bc8fec399bea6cf7de27a7936372
- Commit date: 2026-07-12T16:40:12-07:00
- Intended use: authoritative implementation for the Phase 5
  TurboQuant/vLLM Reference Lane and inspected source candidate for a later,
  separately admitted Measurement Lane adapter.

The TurboQuant paper in the supplied archive does not identify an author-owned
code repository. Phase 5 selects the official vLLM implementation as authority
for the specifically named TurboQuant/vLLM lane; it does not assert that vLLM
is the paper authors' repository or that it implements every paper variant.
The pinned implementation describes its DRIVE/EDEN/HIGGS-style lineage as
predating the TurboQuant paper and omits QJL. Matching preset names is not
evidence of broader paper equivalence.

The official `vllm/vllm-openai:v0.25.1` image is pinned by multi-architecture
manifest digest
`sha256:e4f88a835143cd22aee2397a26ec6bb80b3a4a6fe0c882bcbc63822904766089`;
the linux/amd64 image digest is
`sha256:f0b9a0dc75a9fca3b6811e3279367b2d6a448055a000bfd13859587d74cef268`.
The reference container is separate from the unresolved Measurement Lane
container in B-010. Phase 5 verified the matching vLLM wheel source hashes and
ran compact store, append, and decode fixtures on SM120 in the isolated
reference environment recorded under `reference/turboquant/`. No vLLM source
copy, benchmark timing, or quality result is vendored here.

## KIVI

- Repository: https://github.com/jy-yuan/KIVI
- Commit: 876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6
- Tree: c94c31b2cfd44eeb9a18cff9dcdf03adff4ac49b
- Declared license: MIT
- Intended use: exact Phase 7 reference source with the one Decision 0018
  checksum-bound GQA remediation patch.

The selected commit was authored on 2025-11-20, after arXiv v2 dated 2024-07-25.
Decision 0017 selects the default `main` implementation because it is the
official branch that advertises Llama 3/GQA support; the older official
`develop` and `lmeval` heads are not substituted. The selected source is not
presumed equivalent to the paper-era implementation. Exact relevant-file Git
blobs and SHA-256 values are recorded in `third_party/LOCK.json`.

The unpatched advertised Llama GQA path materializes recent eight-head K/V
storage as a 32-head temporary through Transformers `repeat_kv`; Decision 0017
and its evidence remain the immutable record of that defect. Decision 0018
authorizes one project-maintained patch on the exact official commit. The patch
uses grouped BMM contractions while retaining eight-head K/V operands. It is
not merged upstream and must be described as patched official source, not as an
unmodified author-maintained implementation.

Only the patch, manifest, and validation code are stored here; the upstream
repository is not vendored. The patch passed non-timing BF16 formula and
operand-shape checks on CPU and SM120. The isolated Phase 7 reference lane then
built the unchanged official CUDA extension for native SM120 plus PTX, passed
forced PTX/JIT and minimal Compute Sanitizer controls, generated four
deterministic fixture configurations, and published the checksum-bound bundle
to the existing private, locked R2 prefix. No upstream source tree, model
weights, credentials, source/build caches, or unrelated historical evidence
are included in the fixture bundle.

### KIVI direct Git dependency

- Repository: https://github.com/EleutherAI/lm-evaluation-harness
- Commit: c9bbec6e7de418b9082379da82797522eb173054
- Declared license: MIT
- Evidence: exact requirement in KIVI requirements.txt; commit and license
  verified by a read-only fetch into /tmp.

## KVQuant

- Repository: https://github.com/SqueezeAILab/KVQuant
- Commit: 57a238357f0ffe50084670fcd5781c9848f80ea2
- Tree: 094e0f736f77ee327e5350cbd1eefb1c936aa77b
- License evidence: deployment/pyproject.toml declares an Apache Software License
  classifier.
- License verification: unresolved because the pinned tree has no root license
  file; the classifier alone is not treated as repository-wide authority.
- Phase 9P authority: Decision 0020 binds the upstream commit above plus the
  local/private commit `4ad80bc8c942d0a05516d2be8f8d443a77a05900`, tree
  `c4f1490c9c0c4ec46099f1e95c092516df2adb4e`, and aggregate patch SHA-256
  `db3b6fb7ec0a72e25001e1c83a5158d86512248db5c3a06c61895598d1d482d6`.
- Method label: `kvquant_gqa_upstream_patch_v1` (**KVQuant-GQA patched
  upstream**). This is not an official author-released GQA implementation.
- Intended use: checksum-bound local/private research only. Full Phase 9
  calibration, Phase 10 reference work, and the Measurement Adapter remain
  separate and have not started.

The modified checkout is retained privately at
`/home/rockrock/third_party_worktrees/kvquant-gqa`. The modified source, Git
patch, source archive, Docker source layer, and CUDA extension were not
committed to this project or published to R2. Root-license and adapted-source
lineage questions remain unresolved; Decision 0020 does not grant
redistribution authority.

The pinned outer commit fixes the current bytes of three Transformers-derived
trees, but their exact upstream revisions and local patch deltas are unresolved.
Their nested Apache-2.0 license files are:

- deployment/transformers/LICENSE (observed version 4.38.0.dev0);
- gradients/LICENSE (observed version 4.38.0.dev0);
- quant/dbrx/LICENSE (observed version 4.41.0.dev0).

The README separately attributes reused components to Transformers, GPTQ,
GPTQ-for-LLaMA, and SqueezeLLM. LOCK.json records explicit commit-resolution
plans. No unresolved lineage is accepted for reference execution, copying, or
redistribution.

## Rules for later acquisition

1. Clone or fetch by exact commit and verify the resulting HEAD.
2. Record repository URL, commit, tree hash, license, fetch timestamp, and any
   local patch series.
3. Do not update a pin in an implementation PR.
4. Freeze package and container dependencies independently of source commits.
5. Do not execute third-party setup scripts until the corresponding reference
   container definition has been reviewed.
