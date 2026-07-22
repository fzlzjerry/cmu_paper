# Calibration boundary

This directory reserves frozen calibration artifacts required by methods such
as KVQuant. Phase 2 performs no calibration and creates no quantizer, dataset,
layer-statistics, outlier, or model-derived artifact.

Future calibration is a separately reviewed Phase 9 activity. Before any
dependent scan it must freeze and checksum:

- dataset name, exact revision, split, and sample identity;
- preprocessing and tokenizer identity;
- random seed and sample cap;
- quantizer configuration and output bytes;
- sink-token and outlier-cap policy;
- value/index/metadata dtypes and layout;
- complete artifact inventory and reproducibility command.

Calibration never runs inside a measured decode region or during a primary
scan. Completed calibration artifacts are append-only inputs; changing one
creates a new calibration ID and method-configuration fingerprint.

B-005 remains open. No unresolved dataset revision, artifact hash, or outlier
cap is invented by the Phase 2 templates.
