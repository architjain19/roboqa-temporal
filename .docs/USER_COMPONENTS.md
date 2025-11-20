# RoboQA-Temporal: Component Design Document

## 1. Components by Use Case

### Use Case 1 – Validate Temporal Synchronization
- ROS2 bag reader
- Timestamp extractor
- Synchronization analysis module
- Visualization interface
- Temporal sync control logic

### Use Case 2 – Point-Cloud Anomaly Detection
- ROS2 bag reader (shared)
- Point cloud extractor
- Point-cloud anomaly engine
- Visualization interface
- Point-cloud anomaly control logic

### Use Case 3 – Sensor Calibration Quality Assessment
- ROS2 bag reader (shared)
- Calibration data loader
- Calibration evaluator
- Visualization interface
- Calibration quality control logic

### Use Case 4 – Dataset Quality Scoring & Benchmarking
- Metrics aggregator
- Quality scoring engine
- Benchmark comparator
- Report generator
- Dataset quality scoring control logic


## 2. Component Specifications (Contracts)

### Temporal sync control logic
What it does: Coordinates temporal synchronization validation: reads bag, extracts timestamps, computes drift/jitter, generates sync metrics and reports.

Inputs:
- `bag_path`: string (path to ROS2 bag or dataset)
- `topic_config`: dict (mapping sensor names to topics and parsing options)
- `sync_thresholds`: dict (acceptable jitter/drift thresholds)

Outputs:
- `sync_ok`: bool
- `sync_metrics`: dict (e.g., {"drift_ms":..., "jitter_ms":..., "outlier_count":...})
- `sync_report_path`: string (path to written report, JSON/PDF)

Components to be used:
- ROS2 bag reader
- Timestamp extractor
- Synchronization analysis module
- Visualization interface
- Report generator

Side effects: Writes sync reports and logs drift warnings.

Error modes / failures:
- Missing topics: raise descriptive error or return `sync_ok=False` with `sync_metrics` noting missing topics.
- Corrupted bag: abort with error and write error log.
- Very large bag: stream-processing required; avoid loading entire bag into memory.


### Dataset quality scoring control logic
What it does: Runs all quality pipelines, aggregates metrics, computes TFQS (temporal-format quality score), compares against benchmarks, and generates final report.

Inputs:
- `bag_paths`: list[str]
- `scoring_config`: dict (which pipelines to run, weighting, resource limits)
- `benchmark_id`: str (optional; compare against predefined benchmark)

Outputs:
- `overall_quality_score`: float (0-100)
- `dimension_scores`: dict (per-dimension scores: sync, calibration, anomalies, completeness)
- `benchmark_percentiles`: dict
- `quality_report_path`: str

Components to used:
- Temporal sync control logic
- Point-cloud anomaly control logic
- Calibration quality control logic
- Metrics aggregator
- Quality scoring engine
- Benchmark comparator
- Report generator

Side effects: Writes quality reports and can trigger CI pipeline failures when integrated.

Error modes / failures:
- Partial failures in sub-pipelines should not abort entire run; aggregate results and mark sub-pipeline failures in the report.
- Resource exhaustion: implement checkpointing and resume capability.


## 3. Implementation notes & recommended interfaces

- Language: Python 3.10+ (wider ROS2 compatibility, preferred distro is ROS2 Humble). To keep core logic framework-agnostic and provide thin adapters for ROS2, plain bag files, or other dataset formats.
- IO: Use streaming readers for large bags (do not load entire bag into memory). Provide optional chunking and parallel workers (if handlers make it possible)
- Config: YAML/JSON-based `topic_config` and `scoring_config` for reproducibility.
- Outputs: Always write a machine-readable JSON report and/or human-readable summary (Markdown or PDF).
- Logging: Structured logs (JSON) for reproducibility.


## 4. Suggested file layout

components/
- temporal_sync_control.py
- dataset_quality_scoring_control.py
- ros2_reader.py  # adapter for ROS2 bag access
- timestamp_extractor.py
- sync_analysis.py
- pointcloud_anomaly.py
- calibration_evaluator.py
- metrics_aggregator.py
- report_generator.py