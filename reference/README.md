# Reference lane

This directory reserves correctness-only reference-lane integrations for
TurboQuant, KIVI, and KVQuant. Phase 2 does not populate reference
implementations, execute upstream code, build reference containers, or generate
golden fixtures.

Future reference work must:

- use an exact source commit and separately locked environment;
- document licensing, lineage, patches, and any semantic discrepancy;
- emit small, versioned fixtures with configuration, seed, source identity, and
  SHA-256 checksums;
- cover method-specific cache representation, attention output, and byte
  components required by the workflow;
- remain separate from the unified Measurement Lane.

Reference timing is never imported as cross-method benchmark timing. A
reference environment may establish algorithmic equivalence, but it cannot
admit a measurement adapter or close an admission gate by itself.

B-003 and the recorded KIVI/KVQuant authority and lineage risks remain open.
No method adapter or performance runner is implemented here during Phase 2.
