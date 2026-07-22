# PHASE 0 REPORT

Status: PASS

## Completed

- Read CODEX_WORKFLOW.md (2,178 lines) and AGENTS.md in full.
- Initialized the non-Git workspace, which had inputs but no implementation, as
  a Git repository on unborn main.
- Audited Archive.zip by path, entry type, encryption, permissions, size,
  duplicate/traversal/symlink indicators, and ZIP integrity.
- Extracted without overwrite into literature/raw/, retained all AppleDouble
  sidecars for provenance, and removed write bits from the raw tree.
- Hashed the archive and every extracted input and created a machine-readable
  manifest plus literature inventory.
- Statically checked all PDFs: pdfinfo reports no JavaScript/encryption and
  pdfdetach reports no embedded files.
- Confirmed no pre-existing source, model config, CUDA/C++ extension,
  Dockerfile, build/CI setup, tests, or experimental artifact existed.
- Pinned exact Phase 0 source candidates for vLLM/TurboQuant, KIVI, KVQuant,
  and KIVI's direct lm-evaluation-harness dependency; inspected source without
  building or executing it; and recorded fixed KVQuant embedded-tree hashes
  plus unresolved-lineage commit plans.
- Recorded method semantics, upstream discrepancies, dependencies, execution
  path risks, initial scope, blockers, risks, E00-E18 tasks, an owned post-pilot
  Graph A/B milestone, and the ignored-artifact audit policy.
- Expanded AGENTS.md to the workflow-required 15 invariants.

## Changed files

- .gitignore
- AGENTS.md
- literature/README.md
- literature/manifest.csv
- literature/checksums.sha256
- third_party/LOCK.json
- third_party/NOTICE.md
- docs/status.md
- docs/risk_register.md
- docs/blockers.md
- docs/artifact_policy.md
- docs/tasks.md
- docs/decisions/0001-initial-scope.md
- docs/method_notes/turboquant.md
- docs/method_notes/kivi.md
- docs/method_notes/kvquant.md
- docs/phase_reports/phase0.md

Operator-provided and extracted raw inputs, intentionally ignored by Git:

- Operator-provided Archive.zip
- Newly extracted literature/raw/ with 46 locally read-only files

No benchmark implementation or completed-run artifact was modified.

## Commands executed

Principal command families:

- wc and contiguous sed ranges to read the full contract and AGENTS.md
- git status/rev-parse/remote inspection, followed by git init -b main
- rg --files and find inventory searches
- file, stat, zipinfo, unzip -Z, unzip -t, and unzip -n
- sha256sum generation and verification
- pdfinfo, pdfdetach, pdftotext, and strings for static literature inspection
- git ls-remote plus an exact-commit read-only fetch for immutable upstream
  revision verification
- read-only shallow git clone into /tmp plus git show/grep/ls-tree for pinned
  source inspection
- chmod -R a-w literature/raw
- patch for Phase 0 documentation because the workspace apply_patch helper
  could not mount the repository
- awk, find, Python JSON parsing, and Git status for acceptance validation

No setup script, package installer, model, CUDA kernel, profiler, benchmark, or
archive-contained program was run.

## Tests and evidence

- ZIP integrity: all 46 entries OK.
- Archive SHA-256:
  20e5b6be5c3060012c48446d1b51067996cd4f13df1d6a73ee8eeb8f855e3ab1.
- Archive inventory: 46 regular unencrypted files; 23 PDFs and 23 AppleDouble
  sidecars; 123,745,542 uncompressed bytes.
- Path safety: no absolute paths, parent traversal, duplicates, symlinks,
  special nodes, executable bits, executable extensions, or macro formats.
- PDF static checks: JavaScript=no, encrypted=no, zero embedded files.
- Checksum verification: 47/47 ledger entries OK.
- Manifest hash verification: 47/47 data rows OK.
- Raw input local-protection check: the archive is non-writable and the
  extracted tree has zero writable entries; chmod is not claimed as immutability.
- Source lock JSON: syntactically valid; four exact source/dependency commits,
  three fixed embedded-tree hashes, and three attributed-repository plans.
- Task ledger: E00 through E18 all present; M14-GRAPH-AB has owner, schedule,
  evidence, and status.
- Required Phase 0 file check: complete.
- Whitespace audit: no trailing whitespace in Phase 0-authored or modified text
  records; the untouched operator-provided contract is outside this lint scope.

## Admission gates

- Phase 0 acceptance: PASS.
- G0: NOT STARTED.
- G1: NOT EVALUATED.
- G2-TQ: NOT EVALUATED.
- G2-KIVI: NOT EVALUATED.
- G2-KVQ: NOT EVALUATED.
- G3-G5: NOT EVALUATED.
- Full-scan admission: CLOSED.

## Observed risks

- Blackwell/toolchain and profiler compatibility remain untested.
- Upstream KIVI uses dynamic cache concatenation and GQA repetition.
- Upstream KVQuant uses data-dependent top-k/concatenation and its GQA path is
  not established.
- vLLM TurboQuant includes a full-prefix dequant continuation-prefill fallback
  and possible decode temporaries requiring allocation/graph audits.
- KIVI and KVQuant depend on legacy or vendored runtime stacks.
- The selected KIVI commit postdates the paper and cannot be assumed paper-era
  equivalent.
- Long-context B/L points may exceed the 96GB safety envelope.
- TurboQuant reference authority is unresolved because the paper gives no code
  repository.
- KVQuant calibration, root-license authority, and embedded/adapted upstream
  lineage are unresolved.
- The Git repository has no initial SHA yet.
- Ignored raw artifacts have no selected durable append-only backing or
  immutable locator mechanism yet.

Full evidence and mitigations are in docs/risk_register.md.

## Blockers

No item blocks Phase 0 acceptance. Open later-phase blockers are:

- reviewed initial Git commit required before E00 durable evidence;
- G0 hardware/toolchain certification;
- TurboQuant reference-authority resolution before E05 fixtures;
- primary model selection and immutable revision;
- KVQuant calibration freeze;
- KVQuant license/embedded-source provenance review before E10;
- E01 append-only writer plus durable artifact-store/locator policy
  implementation before admission or claim-bearing runs.

## Scientific interpretation

Phase 0 produced no timing, memory, profiler, numerical-quality, or hardware
execution evidence. It supports no claim about correctness, performance,
compression, CUDA Graph behavior, GQA traffic, or RTX PRO 6000 compatibility.
It establishes only input identity, source pins and commit plans, observed
source risks, and the ordered research plan.

## Next action

Review and commit the Phase 0 records to close B-001, then begin Phase 1 / E00
hardware certification. Do not implement the BF16 baseline or any quantized
adapter unless G0 passes; do not run admission or claim-bearing work until E01
implements docs/artifact_policy.md and closes B-009.
