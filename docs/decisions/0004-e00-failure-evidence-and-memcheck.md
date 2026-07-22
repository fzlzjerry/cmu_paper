# Decision 0004: E00 failure evidence and memcheck policy

- Status: Accepted
- Date: 2026-07-22
- Authority: CODEX_WORKFLOW.md failure protocol, AGENTS.md invariants 1-3 and
  6, Decision 0002, and Decision 0003
- Amends: Decision 0003
- Supersedes: none

## Context

The final pre-formal review found that the initial schema could overstate checks
in a completed FAIL manifest by filling unavailable architecture fields and
listing sanitizer tools that had not run. The implementation also requested a
full memcheck leak check, but Decision 0003 did not explicitly make the leak
summary part of the passing protocol.

## Decision

1. E00 capability certification uses independent observations from NVIDIA
   inventory and `torch.cuda.get_device_capability()`. The architecture is
   certified only when the two canonical capabilities agree exactly. Manifest
   provenance describes that agreement rather than attributing the value to
   only one source.
2. A completed FAIL manifest records observations, not intended work. An
   unavailable capability is null, uninspected compiled-target lists are empty,
   `tools_run` contains only sanitizer commands actually invoked, and fields
   that require an unrun test remain null. The initial process snapshot is
   sufficient to finalize an honest FAIL when no supervised CUDA command is
   admitted; before/during/after coverage remains mandatory for a PASS.
   Observed facts are not rewritten to satisfy the schema: an exit-zero command
   with malformed payload retains exit code zero and creates an admission-failing
   audit error, while an observed installed tool remains `installed: true` even
   if its version metadata is unavailable. Incomplete metadata remains nullable,
   records an observation error, and makes the enclosing collection fail.
3. Memcheck is invoked with `--leak-check full`. A passing memcheck requires
   exactly one zero `ERROR SUMMARY` and exactly one zero `LEAK SUMMARY` in the
   captured output. A missing, duplicated, ambiguous, nonzero, or unparsable
   required summary fails G0.
4. Initcheck and synccheck require exactly one zero error summary. Racecheck
   requires exactly one summary with zero hazards, errors, and warnings. Any
   missing, ambiguous, or nonzero summary fails G0.
5. This remains certification-only work. It records no benchmark timing and
   cannot support a performance claim.

## Consequences

- The stricter leak check is now part of the recorded E00 protocol.
- Completed failure evidence remains schema-valid without fabricated success
