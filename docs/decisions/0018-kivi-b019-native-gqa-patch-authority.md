# Decision 0018: KIVI B-019 native-GQA patch authority

- Status: Accepted
- Date: 2026-07-26
- Authority: explicit B-019 remediation authorization, AGENTS.md,
  CODEX_WORKFLOW.md, and the Phase 7 KIVI Reference Lane contract
- Supersedes: Decision 0017 items 2-4 only for this checksum-bound remediation
- Superseded by: none

## Context

Decision 0017 established positive evidence that the official KIVI `main`
commit materializes recent GQA keys and values from eight KV heads to 32 query
heads through Transformers `repeat_kv`. A fresh remote-ref audit found no newer
author-maintained revision: `main`, `develop`, and `lmeval` remain at the exact
commits recorded by Decision 0017.

Waiting for an upstream revision would preserve an unmodified-source claim but
would not remediate B-019. Substituting an unofficial fork would weaken source
authority. The authorized alternative is one transparent project patch on the
exact official commit, with the base, patch, patched files, and resulting tree
all checksum-bound.

## Decision

1. Retain the official repository, base commit
   `876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6`, and base tree
   `c94c31b2cfd44eeb9a18cff9dcdf03adff4ac49b`.
2. Authorize exactly the patch recorded by
   `third_party/patches/kivi/manifest.json`. No floating branch, fork, or
   additional source delta is permitted.
3. Replace only the Llama residual GQA query/key and attention/value
   contractions. Query heads are grouped by their owning KV head and passed to
   `torch.bmm`; K/V operands remain at `batch * H_KV`.
4. Do not change quantization, packing, metadata, cache layout, rollover, or
   CUDA extension source.
5. Describe the resulting authority as the "checksum-bound patched official
   source", never as an unmodified author-maintained implementation.
6. Resolve B-019 for the Phase 7 source gate only after exact BF16 equivalence,
   source checkout, patched-tree, operator-shape, forbidden-operation, and
   unsupported-geometry checks pass on CPU and SM120.
7. Keep the original Decision 0017 report and source-audit evidence immutable.
   They remain authoritative evidence of the unpatched upstream defect.
8. Do not begin the reference environment, extension build, fixtures,
   sanitizer, trace, byte accounting, R2 publication, or Phase 8 in this
   remediation.

## Consequences

The patched source preserves H_KV=8 K/V inputs and maps query head `h` to KV
head `h // 4` for the frozen 32/8 geometry. Attention scores and outputs retain
their required H_Q=32 geometry. The patch is a source-level correctness
remediation, not a performance optimization or performance claim.

B-019 can close under this explicit project-patch authority, but Phase 7 does
not PASS from this decision alone. The remaining isolated-environment, CUDA
extension, fixture, rollover, layout, trace, sanitizer, and durable-publication
requirements require a separate continuation from a clean entry. If upstream
later merges an equivalent change, a new exact commit/tree audit must replace
this patch authority rather than silently dropping it.
