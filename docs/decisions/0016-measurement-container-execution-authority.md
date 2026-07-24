# Decision 0016: Measurement Container execution authority

- Status: Accepted
- Date: 2026-07-25
- Authority: AGENTS.md, CODEX_WORKFLOW.md, the experiment and measurement
  contracts, the Phase 6A plan, and Decisions 0005 and 0007
- Supersedes: no method, gate, measurement, or quality requirement
- Superseded by: none

## Context

Phase 6A built and verified the linux/amd64 Measurement Container with Docker
image ID / OCI image-index digest
`sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e`
from the pinned base manifest
`sha256:0eee3094c71518ad31d011a594ae6ed6de72959ee07e318cb31cffe71690e90c`.
The linux/amd64 platform manifest is
`sha256:75b8ee6d204b6af0ab29f05c6b8cdc5b7bcaa214d52f6df13f8b9b37a317eafc`,
whose OCI config descriptor is
`sha256:8f0736d3545e7ee1d79be85feceb462441bb7fe69af1b40144ca28e0a722b44c`.
No approved remote registry or authoritative RepoDigest was introduced.
Docker's local `RepoDigests` metadata reports
`kvbench-measurement@sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e`,
but that local metadata is not registry authority.

Container G0 run
`e00-20260724T195014.679255Z-a6025ae023e1-23dbe853` passed every required
native-SASS, forced-PTX/JIT, sanitizer, Graph, allocation, binary-inspection,
identity, exclusivity, schema, checksum, and finalization control. Its root is
`85e1f49dea76d08b2cba4477d089a71759d529f03b2bc3538da3d15d8639455c`.
The independent BF16 eager and CUDA Graph parity runs passed against that exact
image and G0 artifact. They are untimed, non-claim evidence with
`measurement_scope: measurement_container_parity`.

Cloudflare R2 management verification found the evidence bucket private and
covered by enabled indefinite rule `kvbench-evidence-indefinite` at exact
prefix `kvbench/sha256/`. The existing synthetic root and the container-G0 root
both passed clean retrieval. B-009 and B-010 are therefore resolved.

## Decision

1. Measurement Lane CUDA execution is authorized only inside Docker image ID /
   OCI image-index digest
   `sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e`.
2. A tag, including `kvbench-measurement:phase6a`, is never execution
   authority. The exact digest must be supplied and recorded in every run.
3. Any changed image digest requires a new container G0, both BF16 parity
   smokes, durable publication/retrieval, and a new authority decision before
   Measurement Lane CUDA execution.
4. The Phase 5 vLLM/TurboQuant reference environment remains isolated and is
   not Measurement Lane authority.
5. Native-host G0 and BF16 G1 remain engineering admission evidence only and
   are non-claim-bearing.
6. G2-TQ is `NOT EVALUATED / READY`. This decision does not implement or admit
   TurboQuant and does not restart Phase 6.
7. Pilot, Full Scan, Nsight profiling, fitting, figures, and quality execution
   remain locked. Global G2-G5 remain `NOT EVALUATED`, quality execution
   remains `LOCKED`, and `PERFORMANCE_DATA_FROZEN` remains absent.

## Consequences

B-009 and B-010 may be marked resolved for this exact durable-store and image
identity. A separate new task may restart the Phase 6 TurboQuant Measurement
Adapter. No performance, capacity, memory-traffic, compression, quality, or
method-admission claim follows from this decision.
