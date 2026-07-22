# Decision 0008: Phase 3 GQA evidence taxonomy

- Status: Accepted
- Date: 2026-07-22
- Authority: AGENTS.md, the Phase 3 remediation instruction, and the immutable
  Phase 3 G1 report
- Supersedes: the fail-closed status interpretation in Decision 0007 only; the
  selected backend, numerical requirements, and G1 proof requirement remain
  unchanged
- Superseded by: none

## Context

The immutable Phase 3 report records 19 runs as
`gqa_materialization_detected`. The underlying evidence did not positively
identify an expanded query-head-sized K/V tensor, a repeat or expand-copy
kernel, a matching allocation, or a source path that replicated K/V. Instead,
the Python dispatch trace exposed only the high-level SDPA operation and could
not identify the CUDA device kernel. The report correctly failed G1, but the
terminal status conflated missing proof with positive materialization.

That conflation prevents a precise scientific interpretation. Absence of
materialization evidence is not proof of non-materialization, and absence of
non-materialization proof is not evidence that materialization occurred.

## Decision

1. Existing Phase 3 artifacts, manifests, lifecycle records, campaign records,
   and the G1 report remain immutable. Their historical
   `gqa_materialization_detected` values are not rewritten or reinterpreted as
   successful evidence.
2. New GQA audits use exactly four mutually exclusive verdicts:
   - `gqa_materialization_detected` requires positive replication evidence;
   - `gqa_dispatch_unverified` means the selected device-kernel path was not
     directly established;
   - `gqa_nonmaterialization_unproven` means dispatch was established and no
     expansion was observed, but at least one other preregistered proof
     component is absent or failed;
   - `gqa_nonmaterialization_verified` requires every proof component to pass.
3. Verdict precedence is positive materialization, then unverified dispatch,
   then incomplete non-materialization proof, then verified
   non-materialization. A negative observation can never produce the positive
   materialization verdict.
4. The complete proof contract consists of forced Flash-only backend
   selection, an identified CUDA attention-kernel family, no preceding
   replication or context-copy kernel, no expanded-K/V allocation, passing
   source audit, and passing input/cache/output shape and stride audit.
5. The first three verdicts are G1 failures. A verified GQA verdict is necessary
   but not sufficient for a run to finish as `completed`; every other Phase 3
   admission condition must also pass.
6. Report derivation must distinguish the new verdicts and must retain a
   compatibility path that independently validates the immutable version-1
   report without changing its bytes or historical conclusions.

## Consequences

- G1 is not weakened: uncertainty and incomplete proof still fail closed.
- New blocker and report text can state whether evidence is positive,
  dispatch-incomplete, or otherwise incomplete without overstating the
  mechanism observed.
- Direct CUDA-kernel and allocation evidence is required before
  `gqa_nonmaterialization_verified` can be emitted.
- A future positive replication observation has an unambiguous terminal
  status and cannot be hidden by a generic proof failure.
