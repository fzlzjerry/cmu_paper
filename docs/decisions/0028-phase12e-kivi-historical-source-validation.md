# Decision 0028: KIVI historical source validation

- Status: Accepted
- Date: 2026-07-30
- Phase: 12E
- Phase 8 execution commit:
  `462325e9df809d3bcf24a06361bf004bc7383d73`
- Decision 0026 transition commit:
  `781b416748e2bddca8ea5c23cd0f51a63a066276`

## Context

Phase 8 allocation evidence records the endpoint source used by its execution
commit. Decision 0026 later made a backward-compatible pre-RoPE boundary
change, so comparing the historical endpoint digest with the current working
tree rejects intact evidence even though KIVI's adapter and cache authority did
not change.

The Phase 8 endpoint blob is
`8aa48ec285fb9c7853bc19ae10bd8afc07a04d1d6f522f53e67e705a424a27b9`.
Its Git blob object is `853241dd77a6fb7f70cc47894a91c525d7c5f5fe`.
Decision 0026 introduced the current endpoint blob
`9095e9a2a9c01e1ea6afb2f1cefcee46a964a82caae7b819a125757b59244a9b`.
Its Git blob object is `e6967f695540a6a822ffb288f0d9a1f07e905641`;
the transition commit's exact parent is
`1188f1d7c7ddcaa143014b9797b3ddd1e6933597`.
The KIVI adapter and cache digests remain, respectively,
`d47efdb9a9b6e34aaf3f8465a33b6f2bc550680ad369cfb1a3e4d6f0222bccc8`
and
`0c99bb6b6bf9e84074f5e087d545988912285d4c5621c10ee8e7920cac0844a5`.

## Decision

1. Resolve the exact full execution Git SHA from the checksum-validated Phase 8
   run manifests and require the exact Phase 8 execution commit recorded above;
   do not substitute the outer bundle's custody commit or infer it from the
   current checkout.
2. Require that commit to exist locally and verify the recorded endpoint digest
   against the exact Git blob
   `src/kvbench/runtime/bf16_endpoint.py` at that commit.
3. Replay the frozen allocation evidence and operation semantics without
   weakening any semantic check.
4. Verify the recorded KIVI adapter and cache digests against both their exact
   execution-commit blobs and the current files. They must remain unchanged,
   and their paths must have no intervening commit, including a
   modify-then-revert transition.
5. Accept the endpoint transition only through the exact Decision 0026 commit
   above: the execution commit must be its ancestor, Decision 0026 must be an
   ancestor of `HEAD`, and the current endpoint must equal Decision 0026's
   exact blob.
6. Missing commits or blobs, altered evidence or receipts, digest mismatches,
   non-ancestor histories, and any unrecognized endpoint transition fail
   closed. No loose digest allowlist is permitted.
7. This is a validator-only historical-authority correction. It changes no
   method, CUDA source, fixture, calibration, MethodAdmissionReport, historical
   artifact, or receipt, and authorizes no Phase 12 campaign or later phase.
