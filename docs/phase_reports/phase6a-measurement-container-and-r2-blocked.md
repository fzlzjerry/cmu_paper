# Phase 6A Measurement Container and R2 prerequisite report

Status: BLOCKED

Date: 2026-07-24

## Scope and entry

This attempt covered only one Measurement Container definition and one
Cloudflare R2 publication path. Phase 6 was not restarted, no TurboQuant
Measurement Adapter was implemented, and no Pilot, Full Scan, profiler,
fitting, figure, or quality work ran.

The initial checkout was governance recovery commit
`a25a76a052a918428e8eb56cdfde63470cf6a152` with one operator-approved
`.gitignore` change for `.env`. That change was retained in commit
`a9aca47b18cef81681bbf4a757f7a545e5715415`, after which the entry checks were
rerun from a clean descendant. `make package-lock-check`, `make test`, and
`make checks` passed. The authoritative retrospective Phase 6 report remained
byte-identical at SHA-256
`18015eeda156eeb1718d551919605244b6ff21c819182fcd5d66fa316558ed74`.
Phase 5 remained the latest completed PASS phase; native-host G0/G1 remained
PASS; B-009/B-010 were OPEN; and G2-TQ was BLOCKED.

The Phase 6A implementation is recorded by commits
`f877a2a69bef35ca149aad092b5943b382d21b6d` and
`53a15c3af612accba8bbaae1355ff8fa3112567e`.

## Secret safety

`.env` was untracked, Git-ignored, excluded by the deny-by-default Docker
context, and mode `0600`. All five required object-access variables and
`CLOUDFLARE_API_TOKEN` were present. Only variable names and
`PRESENT`/`MISSING` status were emitted. No secret value was printed, committed,
placed in a Docker build argument or image instruction, or included in the
synthetic artifact. The R2 validator rejects `.env` names, credential-shaped
manifest fields, and configured secret bytes.

## Measurement Container definition

`docker/measurement.Dockerfile` pins the linux/amd64 CUDA base manifest
`sha256:0eee3094c71518ad31d011a594ae6ed6de72959ee07e318cb31cffe71690e90c`.
Its SHA-256 is
`333a1e4264e8dc7798c5af06622fc97871371c3fc1e063f4a3b88cfb25389ace`.
The definition pins or requires CPython 3.12.3, PyTorch 2.12.1+cu130, CUDA 13.0,
Triton 3.7.1, Transformers 4.57.6, GCC/G++ 13.3.0, nvcc 13.0.88,
nvdisasm 13.0.85, Compute Sanitizer 2025.3.1, and the SM120 native/PTX target.
Its instructions do not add model weights, credentials, artifacts, or host
driver packages. Because no image was built, actual image contents and package
identities remain unverified.

The existing E00 implementation now has a fail-closed container mode designed
to bind the image/runtime identity, exact package and tool identities, CUDA
lanes, schema, provenance, checksum ledger, inventory, and COMPLETE-last
finalization. It retains the historical native-host default. Synthetic unit
tests cover the Docker build context and saved-image layer scanner, secret
rejection, exact runtime binding, system/Python locks, and historical manifest
compatibility; none is executed-container evidence.

## Container execution outcome

The execution host exposed no `docker`, `podman`, `nerdctl`, `ctr`, `crictl`,
`buildah`, `skopeo`, or `containerd` command and no `nvidia-container-cli` or
`nvidia-ctk`. Phase 6A did not install or alter host NVIDIA/container
packages. Consequently:

- no Measurement Container image ID/config digest exists;
- no observed container package lock or image-layer scan exists;
- no container G0 run exists;
- no BF16 eager or CUDA Graph parity run exists;
- no remote repository digest exists; and
- no execution-authority decision was created.

B-010 remains OPEN.

## Cloudflare R2 implementation and bounded acceptance

`scripts/r2_artifact.py` is a provider-specific `publish`/`verify` tool. It
uses content-addressed keys, signed `If-None-Match: *` creation, the local
SHA-256 ledger as authority, exact existing-byte retrieval, mismatch
rejection, payload/control ordering, and COMPLETE last. Clean retrieval
requires a new empty directory and rejects missing, unexpected, or tampered
objects.

Safe public identity:

- provider: `cloudflare_r2`;
- endpoint: `https://<R2_ACCOUNT_ID>.r2.cloudflarestorage.com`;
- bucket: `kvbench-artifacts`;
- prefix: `kvbench/sha256`;
- region: `auto`.

Synthetic artifact
`phase6a-r2-synthetic-20260724t135642z` has root SHA-256
`bbb80210dc729dedc9dd25a24d61cfbedbbe9d05661b1f95e6af278df3d0c11e`
and locator
`r2://kvbench-artifacts/kvbench/sha256/bbb80210dc729dedc9dd25a24d61cfbedbbe9d05661b1f95e6af278df3d0c11e/`.

- Initial conditional publication uploaded 11 objects and wrote COMPLETE last
  at `2026-07-24T13:57:33.422242Z`.
- Exact republish verified all 11 existing objects and uploaded none. The
  current committed implementation repeated that result at
  `2026-07-24T15:04:10.104433Z`.
- A controlled different-byte conditional write to an existing payload key was
  rejected with HTTP 409. No deletion was attempted.
- An immediate post-conflict diagnostic did not establish byte identity, but a
  subsequent independent GET and clean retrieval verified the retained
  original bytes. No unconditional overwrite or replacement was attempted.
- The first clean-retrieval invocation exposed a local destination-name defect.
  The defect was fixed and regression-tested; a fresh empty-directory
  retrieval then passed. The current committed implementation repeated clean
  retrieval at `2026-07-24T15:03:52.131414Z`: 11 objects, valid COMPLETE,
  inventory, checksum ledger, reconstructed root digest, and no unexpected
  object.

ETag was not used as a scientific checksum. No AWS S3 Object Lock header was
used. Credential leakage into the published R2 objects was not detected.

## Cloudflare certification blocker

The required read-only Cloudflare REST certification stopped on an HTTP 403
with a redacted error; subsequent public-state and lock calls were not
established. Therefore bucket existence through that control plane, direct
public state, Bucket Lock rule identity/prefix/enabled state, and retention
type are NOT VERIFIED. The HTTP 409 object-path observation is supporting
behavior, not a substitute for independent control-plane verification.
Cloudflare documents that object-level R2 tokens do not authorize the REST API
and documents Bucket Lock separately:

- <https://developers.cloudflare.com/r2/platform/troubleshooting/#object-level-api-tokens-fail-against-the-rest-api>
- <https://developers.cloudflare.com/r2/buckets/bucket-locks/>
- <https://developers.cloudflare.com/api/resources/r2/subresources/buckets/subresources/locks/methods/get/>

The second required acceptance artifact, the completed container-G0 bundle,
does not exist and was not published or retrieved. B-009 remains OPEN.

## Validation and preservation

The focused Phase 6A suite passed 86 tests. Repository static checks,
scope validation, existing repository/native package locks, and
immutable-history validation passed.
Phase 5 fixture validation passed. The Phase 5 fixtures, retrospective Phase 6
report, E00 and Phase 3/4/5 evidence, and all completed run directories remained
unchanged. No formal performance, profiler, or quality artifact was created.

Native-host G0 and native-host BF16 G1 remain PASS. G2-TQ remains BLOCKED;
global G2-G5 remain NOT EVALUATED. Full Scan remains CLOSED. Quality execution
remains LOCKED and `PERFORMANCE_DATA_FROZEN` remains absent.

## Minimum remaining action

Make the approved Docker runtime and NVIDIA Container Toolkit available on the
RTX PRO 6000 host. Then build the exact image; record its immutable identity,
full inspect, layer-secret scan, observed system/Python/package/tool identities,
and reviewed package lock; run container G0; run the two bounded BF16 parity
smokes using the frozen Phase 3/4 controls; and create an execution-authority
decision only if all pass.

Provide an account-wide R2 `Admin Read only` Cloudflare API token, not an
`Object Read only` token. Rerun the read-only certification. If and only if it
shows no exact enabled indefinite rule for `kvbench/sha256/`, the operator must
add that rule through the Cloudflare Dashboard or Wrangler in a separately
authorized action. Then publish and clean-retrieve the completed container-G0
artifact. Do not restart Phase 6 until both B-009 and B-010 are resolved.
