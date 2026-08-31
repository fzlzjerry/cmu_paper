# Phase 16R Logical-Prefix Remediation

Status: PASS

Decision 0040 makes compact token/position input the canonical prefix evidence. A worker validates one shared logical prefix, allocates a fresh method-owned cache, reconstructs it before timing, performs an untimed validation decode, and writes no materialized cache snapshot.

The catalog contains 59 append-only logical prefix artifacts totaling 54,425,655 bytes. The four-family B=2/L=4096 and B=16/L=16384 matrix passed 16/16 independent CUDA process records with exact output, cache-layout, allocation, kernel-path, pointer, and Graph agreement. A separate BF16 smoke proved identical measured semantics for logical reconstruction and optional exact-layout snapshot restoration. Legacy readable layouts remain restore-incompatible; strict fingerprint validation is unchanged.

The focused 22-test suite and frozen 534-point plan pass. The remediation bundle root is `99cbc3197ba07a6b8ff8dbddad9fe611ca4cc27ad8387d2f362469d4111a7b94`; 119 objects were uploaded COMPLETE-last to `r2://kvbench-artifacts/kvbench/sha256/99cbc3197ba07a6b8ff8dbddad9fe611ca4cc27ad8387d2f362469d4111a7b94/` and one clean retrieval passed.

The stopped campaign `phase16-20260830t172802896531z-08a19c0f-0c0e2f` remains immutable and non-claim-bearing. Adapters, CUDA, cache layouts, method fingerprints, timing boundaries, configurations, and frozen orders are unchanged. Quality remains LOCKED and performance claims remain ineligible.
