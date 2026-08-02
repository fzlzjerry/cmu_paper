# Phase 13F — Prefix feasibility remediation

Recompute the unchanged 810-record Phase 13 design under Decision 0031.  The
formula includes exact model weights, method-owned cache/workspace, endpoint
workspace, prefix token/position/RoPE controls, source-derived attention and
MLP construction peaks, the existing Graph reserve, and the frozen 0.88 GPU
memory limit.

Generation and validation run without timing inside the exact Decision 0016
Measurement Container.  The evidence binds both stopped campaigns unchanged,
the failed `tq_3bit_nc` B=8/L=98304 trace, all source hashes, every corrected
record, and deterministic classification.  Finalization is append-only with
`COMPLETE` last, followed by host-side R2 publication and one clean retrieval.

No Pilot, Phase 14, Full Scan, profiler, or quality work is authorized.  A PASS
only permits proposing a wholly new Phase 13 campaign.

