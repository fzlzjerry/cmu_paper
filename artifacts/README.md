# Local raw-artifact root

Only this README is tracked. Every other entry below artifacts/ is ignored by
Git and is governed by the append-only artifact policy and experiment contract.
Raw run directories are not normal code-PR content.

The Phase 2 local lifecycle is:

    created -> running -> finalizing -> completed

Terminal failure statuses are finalized and preserved alongside successful
runs. Tooling writes exclusively into a unique same-filesystem staging
directory, validates schemas, generates an artifact inventory and SHA-256
ledger, writes an authenticated completion marker last, and promotes with an
atomic no-replace operation. Existing final, staging, and reserved run IDs are
conflicts and are never silently reused.

Interrupted staging directories remain evidence. Supported APIs provide no
operation that edits a finalized run. Local permission changes are only defense
in depth and do not replace append-only storage governance.

Tests must use caller-provided temporary roots. They must reject the repository
root, docs/evidence, and any ancestor or descendant of the immutable E00
evidence boundary.

The Git-tracked E00 runs under docs/evidence/e00 are a separate Decision-0002
exception and must never be copied, moved, edited, or used as mutable test
fixtures.

This local writer does not provide durable remote retention or an immutable
publication locator. B-009 therefore remains open. B-010 also remains open
because no digest-pinned measurement container or container-parity G0 exists.

Phase 2 creates no formal performance, profiler, or quality run here. Quality
execution is LOCKED, PERFORMANCE_DATA_FROZEN is absent, and this directory
supports no performance or quality claim.
