# Decision 0021: KVQuant patch custody in the main repository

- Status: Accepted
- Date: 2026-07-28
- Authority: explicit operator instruction to preserve the Phase 9P work in
  `cmu_paper/main` before the current server is released
- Supersedes: Decision 0020 item 11 only for Git custody and publication of the
  checksum-bound patch artifact
- Superseded by: none

## Context

Decision 0020 established the local patched authority:

- upstream commit `57a238357f0ffe50084670fcd5781c9848f80ea2`;
- upstream tree `094e0f736f77ee327e5350cbd1eefb1c936aa77b`;
- patched commit `4ad80bc8c942d0a05516d2be8f8d443a77a05900`;
- patched tree `c4f1490c9c0c4ec46099f1e95c092516df2adb4e`; and
- aggregate diff SHA-256
  `db3b6fb7ec0a72e25001e1c83a5158d86512248db5c3a06c61895598d1d482d6`.

That decision kept the patch only in a server-local checkout because the
KVQuant repository has no verified root license and several adapted-source
lineages remain unresolved. The operator has now stated that the server is
ephemeral and explicitly directed that the work be retained in the main
project repository and pushed to its public `origin/main`.

## Decision

1. The main repository stores the exact 289,239-byte, full-index binary-capable
   Git diff at
   `third_party/patches/kvquant/0001-llama31-native-gqa.patch`.
2. The stored bytes must have SHA-256
   `db3b6fb7ec0a72e25001e1c83a5158d86512248db5c3a06c61895598d1d482d6`.
   This is the same aggregate diff validated in Phase 9P; it is not a regenerated
   or semantically revised patch.
3. The patch plus
   `third_party/patches/kvquant/manifest.json` is the durable project authority.
   A long-lived separate patched checkout is no longer required. Reproduction
   starts from the exact upstream commit, applies the stored patch, and must
   derive the exact patched tree. Patched commit `4ad80bc8...` remains a
   historical validation identity; reconstruction does not require that local
   commit object.
4. The full 13,457-file patched checkout, its `.git` directory, vendored
   caches, compiled extension, model files, and build outputs are not copied
   into this repository.
5. The operator explicitly authorizes publication of this exact patch artifact
   to the public `fzlzjerry/cmu_paper` repository. No push to the official
   `SqueezeAILab/KVQuant` repository and no R2 source publication is authorized.
6. This instruction accepts the custody and redistribution risk; it does not
   resolve, infer, or claim a repository-wide KVQuant license or the exact
   lineage of embedded/adapted code. `third_party/NOTICE.md` and the lock must
   continue to disclose those unresolved facts.
7. Decision 0020 remains authoritative for the method name, source base,
   patched commit/tree, model boundary, GQA geometry, outlier policy, sink
   policy, numerical evidence, and all scientific non-claims.
8. The immutable Phase 9P report and its two evidence records are not rewritten.
   This is a later source-custody action, not a retroactive Phase 9P result.
9. KVQuant remains fail-closed. This action does not start calibration,
   Phase 10, a Measurement Adapter, performance work, profiling, or quality
   evaluation.

## Consequences

A fresh clone of the main project retains every project-authored change needed
to reconstruct the validated patched tree without depending on the current
server. The unresolved-license risk is now visible in public Git history by
explicit operator choice. Any later use must retain the
**KVQuant-GQA patched upstream** label and the exact source/patch identities.
