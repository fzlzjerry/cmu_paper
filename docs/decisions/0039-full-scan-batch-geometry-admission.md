# Decision 0039 — Full Scan batch-geometry admission

- Status: Accepted
- Date: 2026-08-30
- Authority: Decision 0030, the current method admissions, and Phase 16G

## Decision

Exact-batch prefix format v3 represents any positive integer batch and binds
the stored token, position, cache, model, tokenizer, method, layout, context,
and checksum identities. The loader validates those stored identities only;
it does not grant execution admission. Legacy v2 B=1/4/8 artifacts remain
unchanged and readable.

After the checksum-bound Phase 16G B=2/B=16 numerical, allocation, path,
prefix-restore, CUDA Graph, and focused sanitizer evidence passes and is
durably published, the Full Scan execution gate admits exactly
`B in {1,2,4,8,16}` for the ten main configurations. Decision 0030 and its
B=1/4/8 records remain immutable. Existing G5 remains the reproducibility
authority; Phase 16 evaluates B=2/B=16 process stability.

This decision changes no adapter, CUDA kernel, quantization, cache layout,
timing boundary, fixture, calibration, MethodAdmissionReport, or performance
claim. Full Scan execution remains deferred to a separate task.
