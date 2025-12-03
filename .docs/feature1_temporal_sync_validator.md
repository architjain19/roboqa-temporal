# Feature 1 – Multi-Sensor Temporal Synchronization Validator

## Overview

The Temporal Synchronization Validator (TSV) inspects ROS2 MCAP bags that contain multi-sensor streams (camera, LiDAR, IMU, and optional PPS trigger signals). The validator confirms that timestamps, publish frequencies, and phase relationships satisfy ISO 8000-61 data quality expectations and IEEE 1588 (PTP) timing constraints. The validator is fully scriptable via the Python API and is also invokable from the CLI.

## High-Level Architecture

1. **MCAP Ingestion Layer**
   - Uses `rosbag2_py.SequentialReader` with `storage_id="mcap"` to stream messages.
   - Supports folder- and file-based MCAP bags (ROS2 bag layout).
   - Performs type introspection (`get_all_topics_and_types`) to discover actual sensor topic names when not explicitly provided by the user.

2. **Stream Registry**
   - Each sensor stream is represented as a `SensorStream` dataclass that stores timestamps (ns + seconds), approximate publish frequencies, PPS offsets, and per-message metadata (frame ids, seq numbers, QoS profiles when available).
   - The registry tracks expected frequencies and thresholds pulled from configuration (defaults: camera=30 Hz, LiDAR=10 Hz, IMU=200 Hz, PPS=1 Hz).

3. **Parallel Topic Parsing**
   - The ingestion layer pushes raw message tuples `(topic, serialized_data, timestamp_ns)` to worker functions executed by `concurrent.futures.ThreadPoolExecutor`.
   - Workers deserialize only lightweight headers and timestamps, allowing camera/LiDAR/IMU stamp extraction without loading large payloads into RAM.
   - Future results populate `SensorStream` instances in a thread-safe manner (per-topic locks).

4. **Temporal Alignment Analysis Engine**
   - **Message Filter Emulator:** Recreates ApproximateTime and ExactTime logical synchronizers over the collected timestamps, detecting stamp deltas that breach configurable limits (default 10–100 ms depending on sensor pair).
   - **Pairwise Drift Metrics:** Computes signed deltas, rolling statistics (mean/std/max) using pandas `Series.rolling`. Supports pairs: camera↔LiDAR, LiDAR↔IMU, camera↔IMU.
   - **Hardware Trigger Validation:** If PPS or hardware sync topics exist, validates pulse spacing (1 Hz ± 50 ppm) and correlates PPS events with sensor timestamps to ensure edge alignment.
   - **PTP Compliance Checks:** Evaluates offset from master clock, maximum residence time violation, and reports when jitter exceeds IEEE 1588 Class B (<1 µs), Class C (<100 ns) style thresholds (configurable).
   - **Chi-Square Sync Testing:** Uses χ² test on normalized deltas to decide whether a sensor deviates from nominal alignment.
   - **Cross-Correlation:** Employs `scipy.signal.correlate` to estimate best time shift across time windows, highlighting systematic offsets.
   - **Kalman Drift Prediction:** Simple constant-velocity Kalman filter predicts future drift rates; warns users if predicted drift will exceed tolerance within the observed mission duration.

5. **Quality Metrics + Recommendations**
   - `Temporal Offset Score`: 0–1 score aggregated from per-pair offsets (lower drift == higher score).
   - `Drift Rate Detection`: Derivative of rolling deltas (µs/s) with thresholds for warning/critical states.
   - `Frequency Consistency`: Compares measured frequency to expected; flags dropouts/spikes.
   - `ApproximateTime Compliance`: Pass/fail per pair; includes observed max delta vs threshold.
   - `Hardware Recalibration Flags`: Created when PPS alignment, chi-square test, and drift prediction all fail simultaneously.

6. **Outputs**
   - **Structured Report (`TemporalSyncReport` dataclass):** Houses metrics, per-pair stats, recommendations, and compliance flags; serializable to JSON for ISO 8000-61 traceability.
   - **Visualizations:** Matplotlib/Seaborn heatmaps (timestamp delta vs time) and drift plots saved under `reports/temporal_sync/`.
   - **ROS2 Parameter Export:** YAML file containing recommended timestamp offsets or per-sensor correction parameters (usable by downstream launch files).
   - **CLI Summary:** Optional console table summarizing pass/fail for each sensor pair.

## Configuration Surface

```yaml
temporal_sync:
  enabled: true
  storage_id: mcap
  topics:
    camera: /camera/image_raw
    lidar: /lidar/points
    imu: /imu/data
    pps: /hardware/pps
  approximate_time_threshold_ms:
    camera_lidar: 30
    lidar_imu: 15
  ptp:
    max_offset_ns: 1_000_000    # 1 µs
    max_jitter_ns: 100_000
  frequency_hz:
    camera: 30
    lidar: 10
    imu: 200
  heatmap:
    enabled: true
    resolution: 200
  kalman:
    process_noise: 1e-6
    measurement_noise: 1e-5
```

All parameters have sensible defaults so the feature works out of the box while remaining configurable for different robotics stacks.

## Data Flow

`rosbag2_py` → Sequential messages → Topic filters → Parallel timestamp extraction → Sensor streams → Pairwise analysis (rolling stats, χ², cross-correlation, Kalman) → Metrics + visuals + report/params.

## Test Strategy

1. **Synthetic Streams:** Generate deterministic timestamp arrays with known offsets/drifts to assert that metrics capture expected deviations.
2. **Edge Cases:** Empty topics, single-sensor presence, extremely noisy PPS.
3. **Kalman Projection:** Ensure predictor warns when drift trends upward.
4. **Parameter Export:** Validate YAML file contents and schema.

These tests run without ROS dependencies by injecting `SensorStream` fixtures directly into the analysis engine. Bag ingestion is covered via lightweight mocks that exercise MCAP reader setup.
