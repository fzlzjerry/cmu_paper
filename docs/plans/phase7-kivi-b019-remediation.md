# Phase 7 KIVI B-019 remediation plan

Scope is limited to the official Llama GQA residual path. Phase 8 and all
reference-environment, fixture, performance, profiler, and quality work remain
out of scope.

1. Revalidate the clean Phase 7 entry, Phase 6 outer R2 root, and unchanged
   TurboQuant admission.
2. Recheck all author-maintained KIVI branch heads. Prefer a new compliant
   official commit if one exists.
3. If none exists, apply the one Decision 0018 patch to the exact official
   commit and bind the patch, patched files, and resulting tree.
4. Prove BF16 equality with the original `repeat_kv` formula at 32 query heads,
   eight KV heads, head dimension 128, and contexts 17 and 33.
5. Audit source and BMM operands for native eight-head K/V storage; reject
   `repeat_kv`, `repeat_interleave`, expansion, wrong head mapping, and
   unsupported head geometry.
6. Record CPU and SM120 non-timing results, run focused and full repository
   regression tests, commit, and stop before the rest of Phase 7.
