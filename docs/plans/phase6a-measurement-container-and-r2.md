# Phase 6A: Measurement Container and Cloudflare R2

## Scope and frozen inputs

Phase 6A certifies one Measurement Container and one durable publication path.
It does not implement TurboQuant or authorize Phase 6, performance campaigns,
profiling, fitting, figures, or quality execution.

The container preserves the Phase 3/4 BF16 Measurement Lane:

- `linux/amd64` base
  `nvidia/cuda:13.0.2-devel-ubuntu24.04@sha256:0eee3094c71518ad31d011a594ae6ed6de72959ee07e318cb31cffe71690e90c`;
- CPython 3.12.3, GCC/G++ 13.3.0, CUDA compiler 13.0.88,
  `nvdisasm` 13.0.85, and Compute Sanitizer 2025.3.1;
- the exact hashed `preflight/requirements-e00.txt` closure, including
  PyTorch 2.12.1+cu130 and Triton 3.7.1;
- the exact hashed additive `preflight/requirements-phase3.txt` closure,
  including Transformers 4.57.6;
- `TORCH_CUDA_ARCH_LIST=12.0+PTX`, `CUDAARCHS=120`, and
  `CMAKE_CUDA_ARCHITECTURES=120`;
- the frozen Llama 3.1 8B model and tokenizer revision
  `0e9e39f249a16976918f6564b8830bc894c89659`, mounted read-only and offline;
- the frozen BF16 adapter, eight-KV-head static-cache layout, forced Flash
  SDPA backend, fixed-L runner, and CUDA Graph harness.

The host driver is supplied only through the NVIDIA Container Toolkit. Driver
packages and libraries are not copied into the image. Model bytes, repository
artifacts, `.env`, tokens, and publication credentials are excluded.

## Exact repository changes

Implementation is limited to:

- `.dockerignore`;
- `docker/measurement.Dockerfile`;
- `preflight/run_preflight.py` and `preflight/e00_manifest.schema.json` for one
  explicit container mode while preserving the native-host default;
- `scripts/r2_artifact.py`;
- `scripts/validate_phase2.py`;
- focused tests for the preceding files;
- `Makefile` targets named in this plan;
- this plan and, only after execution, one Phase 6A report, minimum governance
  updates, and one execution-authority decision if B-010 passes.

Generated, append-only local evidence is confined to
`artifacts/phase6a/{container_g0,bf16_parity,r2_acceptance}`. Historical
evidence and Phase 5 fixtures remain outside the writable scope.

## Container identity and G0

The build records the Dockerfile SHA-256, base manifest digest, image
ID/config digest, secret-free full image inspect, exact installed dpkg
inventory, Python freeze, and CUDA/compiler/tool identities. A repository
digest is recorded only if an already-approved registry exists.

`make measurement-container` builds the digest-pinned Dockerfile without
secrets. `make verify-measurement-container` checks the recorded identities.
`make preflight-container` runs the existing E00 implementation, under a new
run ID, with the repository source read-only, a separate writable evidence
mount, `--network=none`, the NVIDIA runtime, the selected GPU UUID, and the
exact image config digest. The explicit container mode requires positive
container detection and repeats every G0 CUDA build, native-SASS, forced-PTX,
Compute Sanitizer, `nvdisasm`, cubin/PTX, exclusivity, schema, checksum,
provenance, and COMPLETE-last check. Any failure is preserved and leaves
B-010 open.

## Bounded BF16 parity

Only after container G0 passes, two new runs use `B=1`, `L=128`: eager and
CUDA Graph. They reuse the frozen adapter and Phase 3 allocation, numerical,
GQA, backend, and graph controls. Both manifests record the exact image config
digest and:

```yaml
quality_status: unvalidated
claim_eligibility: performance_only
performance_claim_eligible: false
measurement_scope: measurement_container_parity
```

No emitted engineering timing is a scientific result. B-010 requires G0 and
both parity runs to pass, exact identity/provenance records to validate, and
an execution-authority decision binding Measurement Lane CUDA execution to
that one image digest. Tags are non-authoritative; a changed digest requires
new parity evidence. The Phase 5 reference environment remains separate and
native-host admission remains non-claim-bearing.

## Cloudflare R2 contract

The host-side tool requires presence of `R2_ACCOUNT_ID`, `R2_BUCKET`,
`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `KVBENCH_R2_PREFIX`, and
`CLOUDFLARE_API_TOKEN`. `R2_ENDPOINT` is optional and otherwise derived
internally. Region is `auto`. Only variable names with `PRESENT` or `MISSING`
may be reported.

Before publication, the Cloudflare REST API must confirm the bucket, direct
public-domain state, and an enabled Bucket Lock rule whose prefix exactly
covers the normalized evidence prefix. The accepted condition is
`type: Indefinite`; a broader rule is accepted only when its intended scope is
already documented. The task never creates, changes, removes, or weakens a
lock rule and never uses AWS Object Lock headers.

For a finalized local artifact, the tool hashes the canonical lexical index of
all files, including `COMPLETE`, and publishes under:

```text
<KVBENCH_R2_PREFIX>/<root_sha256>/<artifact-relative-path>
```

Every PUT uses `If-None-Match: *`. A pre-existing key is downloaded and
accepted as `verified_existing` only when its SHA-256 matches; differing bytes
are rejected. ETag is never authoritative. Payloads precede manifests,
inventory, the checksum ledger, and `COMPLETE` is last. Retries verify existing
objects and add only missing objects.

`make publish-artifact-r2 ARTIFACT=<path>` publishes. A separate empty
temporary directory outside the repository is used by
`make verify-artifact-r2 ROOT_SHA256=<digest>` to list and download the exact
prefix, reject missing or unexpected objects, validate the local inventory and
ledger, and recompute the root digest before the temporary directory is
removed.

B-009 requires read-only Bucket Lock verification, the synthetic
publish/retrieve/republish/conflict test, and publication plus clean retrieval
of the completed container-G0 evidence bundle. No partial prefix is complete
without `COMPLETE`, and locked partial prefixes are never cleaned up.

## Deferred work

Pilot, Full Scan, profiling, fitting, figures, quality evaluation, and the
TurboQuant Measurement Adapter remain locked. Phase 6 may be restarted only
in a separate task after both B-009 and B-010 are directly resolved.
