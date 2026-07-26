# Phase 7 KIVI B-019 remediation report

- Status: PASS for B-019 remediation only
- Phase 7 status: PARTIAL
- G2-KIVI: NOT EVALUATED
- Phase 8 started: no
- Date: 2026-07-26

## Entry

- Starting repository commit:
  `755c1bdb87af3e7becda792bd5d300ab877fee7e`
- Working tree at entry: clean
- Phase 6: PASS
- G0 / G1 / G2-TQ: PASS / PASS / PASS
- Global G2-G5: NOT EVALUATED
- Full Scan: CLOSED
- Quality execution: LOCKED
- `PERFORMANCE_DATA_FROZEN`: absent
- Phase 6 outer R2 root:
  `8c4cf76f76bb17e648dfd911f11e268235ed827a9983814b774b9e95405496b0`
- Phase 6 outer object count and clean retrieval: 176 / PASS

All required entry commands passed before remediation. The original Phase 6
root and historical Phase 7 source-audit evidence remain unchanged.

## Official-source recheck

No newer author-maintained KIVI revision was available:

- `main`: `876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6`
- `develop`: `8c3bdf1f83d5b548d1a6ad4fbef8f1bbc5686804`
- `lmeval`: `1d4274d94df708d10d0546c2529c5a62bc2ebc6e`

The unmodified official source still contains the B-019 GQA materialization
identified by Decision 0017. It is not represented as compliant.

## Remediation authority

Decision 0018 authorizes one checksum-bound project patch on the exact official
base:

- Repository: `https://github.com/jy-yuan/KIVI.git`
- Base commit:
  `876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6`
- Base tree: `c94c31b2cfd44eeb9a18cff9dcdf03adff4ac49b`
- Patch:
  `third_party/patches/kivi/0001-preserve-native-gqa-kv-storage.patch`
- Patch SHA-256:
  `c9c2dd52d4c81b844d1d1d7218ad2cd60a5b31574a387f716d466cb01310423d`
- Patched tree: `b617493dea5aff1a754cd27ad6be12ac512b2aee`
- Upstream status: not merged
- Unofficial fork used: no

The resulting authority is the **checksum-bound patched official source**. It
is not an unmodified upstream implementation.

Only `models/llama_kivi.py` and the new `models/kivi_gqa.py` differ from the
official base. Quantization, packing, metadata, cache layout, rollover, and
CUDA extension source are unchanged.

## GQA result

The patched Llama residual path groups query heads under their owning KV head
and uses grouped `torch.bmm` contractions:

- `H_Q`: 32
- `H_KV`: 8
- Head mapping: `kv_head = query_head // 4`
- Persistent K/V shape: `[1, 8, L, 128]`
- `repeat_kv`: absent from patch additions
- `repeat_interleave`: absent from patch additions
- H_Q-sized K/V expansion: absent
- Unsupported non-divisible geometry: rejected

CPU and RTX PRO 6000 Blackwell SM120 BF16 checks passed exactly at:

- query length 3, context length 17
- query length 1, context length 33

Observed query/key BMM K operands retained leading dimension
`batch * H_KV = 8`; attention/value BMM V operands did the same. Outputs were
bitwise equal to the frozen original formula for both cases. No timing was
collected.

Therefore B-019 is:

`RESOLVED_UNDER_PATCHED_SOURCE_AUTHORITY`

## Validation

- `make package-lock-check`: PASS
- `make test`: PASS
- `make checks`: PASS
- focused historical and remediation tests: PASS (16/16)
- `make validate-kivi-b019-patch`: PASS
- SM120 source-tree and BF16 validation: PASS
- `make validate-reference-turboquant`: PASS
- `make validate-admission-turboquant`: PASS (9/9)

## Preservation and boundary

- Authorized Measurement Container modified: no
- TurboQuant adapter or fixtures modified: no
- Common runners modified: no
- Historical artifacts or source-audit evidence modified: no
- KIVI Measurement Adapter created: no
- Reference environment/container created: no
- Official CUDA extension built: no
- Fixtures, rollover evidence, byte accounting, trace, or graph smoke created:
  no
- Compute Sanitizer executed: no
- R2 KIVI fixture publication created: no
- Performance, profiler, HBM, capacity, or quality data created: no
- Phase 8 started: no

## Scientific interpretation

The narrow native-KV-head GQA correction is reproducible from an exact official
base commit plus one checksum-bound patch, and its BF16 semantics and
eight-head K/V operand geometry are reproducible on CPU and SM120. This report
does not establish the remaining Phase 7 reference environment, extension,
fixture, rollover, byte-layout, sanitizer, trace, or durable-publication
requirements and makes no performance or quality claim.

## Next action

Continue the remaining Phase 7 KIVI Reference Lane in a separate clean-entry
task under Decision 0018. Do not begin Phase 8 until Phase 7 independently
passes all remaining acceptance criteria.
