# Decision 0017: KIVI source authority and GQA materialization

- Status: Accepted
- Date: 2026-07-26
- Authority: AGENTS.md, CODEX_WORKFLOW.md, the experiment and measurement
  contracts, and the Phase 7 KIVI Reference Lane specification
- Supersedes: no method, gate, measurement, or quality requirement
- Superseded by: none

## Context

The paper-provided, author-maintained repository has three official branch
heads:

- `main`: `876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6`, tree
  `c94c31b2cfd44eeb9a18cff9dcdf03adff4ac49b`;
- `develop`: `8c3bdf1f83d5b548d1a6ad4fbef8f1bbc5686804`, tree
  `695f0e36a4add849eb78e689d73fcaa362aafeb2`;
- `lmeval`: `1d4274d94df708d10d0546c2529c5a62bc2ebc6e`, tree
  `f508210ee06c866c2fc8e61d6a01cf2d107c383b`.

The branches are not interchangeable. The default `main` README advertises
the later Llama 3/GQA implementation. The other two branch heads predate that
announcement and do not replace the required primary-model GQA authority.
The selected `main` commit postdates arXiv v2, so this decision grants source
authority only and does not assert paper-era equivalence.

The `main` Llama integration imports `repeat_kv` from the pinned Transformers
4.43.1 model source and invokes it for recent keys and values during GQA
decode. The exact helper states that it is equivalent to
`repeat_interleave`, expands the repetition dimension, and reshapes from
H_KV to H_Q. A non-timing semantic audit at the frozen 8/32 head geometry
produced a contiguous 32-head tensor with distinct storage and four times the
input storage bytes.

The quantized-history CUDA kernel accepts `nh` and `nh_kv` separately, but
that does not cure the required residual-window materialization. Phase 7
forbids any `repeat_kv`, `repeat_interleave`, or equivalent H_Q-sized K/V
temporary.

## Decision

1. Pin the official default `main` implementation at exact commit
   `876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6` for this source audit.
2. Do not substitute `develop`, `lmeval`, or an unofficial fork.
3. Treat the observed Llama GQA residual expansion as a mandatory Phase 7
   failure, not as an optimization deferred to Phase 8.
4. Do not patch the official algorithm, build a reference environment, run
   CUDA, or generate fixtures after this finding.
5. Keep the KIVI Measurement Adapter unimplemented and fail-closed.

## Consequences

Phase 7 is BLOCKED before reference-environment construction. No SM120 build,
PTX/JIT result, sanitizer result, fixture, byte-layout result, trace, graph
smoke, or R2 fixture root exists. G2-KIVI and global G2-G5 remain NOT
EVALUATED; Full Scan remains CLOSED; quality execution remains LOCKED; and
`PERFORMANCE_DATA_FROZEN` remains absent.

The minimum remediation is a new exact author-maintained revision whose
official primary-model path preserves eight-head K/V storage and avoids all
H_Q-sized K/V temporaries. Any proposal to authorize a non-upstream semantic
patch or a different implementation authority requires a separate new
decision and a fresh Phase 7 task.
