# Decision 0001: Initial research scope and semantic freeze

- Status: Accepted
- Date: 2026-07-22
- Authority: CODEX_WORKFLOW.md, treated as the authoritative contract by the
  operator's 2026-07-22 instruction
- Recorder: Codex; no additional operator choice is inferred
- Supersedes: none
- Superseded by: none

## Context

The workspace began with CODEX_WORKFLOW.md, AGENTS.md, and Archive.zip but no
Git repository, implementation, configuration, container, or prior experiment.
The workflow is the authoritative research and engineering contract.

This record freezes the initial study semantics. It does not select a model,
claim hardware readiness, admit a method, or authorize performance claims.

## Decision

1. Study one decoder-only 7B to 9B full-attention GQA model on one NVIDIA RTX
   PRO 6000 Blackwell 96GB GPU with tensor parallel size one.
2. Maintain separate Reference and Measurement lanes. Reference environments
   establish algorithmic fixtures; only the unified Measurement Lane may
   produce cross-method performance data.
3. Compare BF16, TurboQuant, KIVI, and KVQuant under identical model, weight
   dtype, runner, output work, graph mode, cache lifetime, timing boundary,
   process isolation, and randomization protocol.
4. Use fixed-L one-token decode as the primary response-surface runner and a
   growing-context fixed-output runner only for request-latency validation.
5. Maintain eager and CUDA Graph lanes. Never compare graph-on to graph-off.
6. Keep same-work speedup separate from capacity amplification.
7. Record nominal compression, allocated compression, and profiler-measured HBM
   compression separately. Never infer physical traffic from nominal bytes.
8. Preserve the pilot and full grids, repetitions, randomization, QC rules,
   model candidates, and statistical targets stated in CODEX_WORKFLOW.md unless
   a later decision is made before observing the affected result set.
9. Integrate methods sequentially: BF16/common harness, TurboQuant, KIVI, then
   KVQuant. No method enters the full scan before all admission gates pass.
10. Raw experiment outputs are append-only; profiler runs have distinct run
    kinds and cannot enter normal timing analysis.

## Initial source ledger

The supplied archive and all 46 extracted entries are identified by
literature/manifest.csv and literature/checksums.sha256. They are locally
write-protected, not claimed filesystem-immutable. Exact upstream pins and
explicit commit-resolution plans for embedded or attributed source are recorded
in third_party/LOCK.json. Source pinning, parent-tree byte identity, and lineage
plans are separate from accepting algorithmic equivalence.

## Explicitly unresolved

The following are not decided by this record:

- primary model ID and immutable revision;
- model config SHA and maximum supported context;
- final measurement container base digest and dependency lock;
- whether upstream vLLM v0.25.1 is sufficient as TurboQuant's authoritative
  reference, given the source's pre-paper lineage and omission of QJL;
- whether the post-paper KIVI snapshot is semantically equivalent to the paper
  version for every selected fixture;
- KVQuant calibration dataset, revision, preprocessing, seed, outlier cap, and
  artifact;
- exact upstream commits and patch deltas for KVQuant's embedded Transformers
  trees and GPTQ/GPTQ-for-LLaMA/SqueezeLLM-attributed components;
- durable publication and retention backing for ignored append-only artifacts;
- exact tolerances for each method's numerical golden tests;
- any change needed to reconcile a pinned source API with the workflow.

Each semantic choice above requires a new decision record before implementation
or data collection. Source API discrepancies must be evidenced from the pinned
source; they may not be resolved by guessing.

## Excluded from primary claims

Continuous batching, queueing latency, multi-GPU tensor parallelism, offload,
speculative decoding, prefix sharing, cross-model absolute-latency ranking,
sliding-window attention, and microkernel speedups presented as full-model
speedups remain outside the first-stage primary conclusions.

## Consequences

- Phase 0 may create provenance and planning documents but cannot edit a
  benchmark hot path or generate benchmark timing.
- Phase 1 may start only after Phase 0 acceptance checks pass. The Phase 0
  records must receive a reviewed initial commit before E00 produces durable
  gate evidence.
- Any later semantic change and its implementation must be split according to
  the workflow's PR rule.
- Failure of an admission gate produces evidence and a blocker, not a fallback.
