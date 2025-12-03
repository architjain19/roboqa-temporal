# Multi-Sensor Temporal Synchronization Validation Report

- **Dataset:** KITTI `2011_09_26_drive_0002_sync` (camera_00, Velodyne HDL-64E, OXTS inertial) located under `2011_09_26/`
- **Objective:** Exercise the Multi-Sensor Temporal Synchronization Validator against real-world KITTI data to quantify inter-sensor timing health and capture any actionable violations.
- **Test Date:** 2025-11-23T06:44:55Z
- **Validator Config:** `TemporalSyncValidator.analyze_streams` invoked with expected frequency 10 Hz for all streams, ApproximateTime thresholds fixed at 5 ms for each pair, default PTP budget (1 ms offset / 0.1 ms jitter), visualization output enabled.

## Methodology
1. Parsed KITTI timestamp text files (`image_00/timestamps.txt`, `velodyne_points/timestamps_start.txt`, `oxts/timestamps.txt`) into nanosecond-aligned `SensorStream` objects. Each line was converted to UTC epoch nanoseconds to preserve the original fractional-second precision.
2. Fed the streams directly to `TemporalSyncValidator.analyze_streams` to avoid ROS2 bag ingestion (KITTI provides discrete files rather than MCAP storage).
3. Enabled artifact generation (heatmaps + ISO 8000-61 parameter YAML) and exported a machine-readable JSON snapshot alongside this report.

## Stream Inventory
| Sensor | Messages | Estimated Freq (Hz) | Notes |
| --- | ---: | ---: | --- |
| camera | 77 | 9.70 | Grayscale left camera (`image_00`) timestamps |
| lidar | 77 | 9.69 | Velodyne HDL-64E spin start timestamps |
| imu | 77 | 10.00 | KITTI OXTS packets aligned to camera exposures |

## Global Metrics
| Metric | Value | Interpretation |
| --- | ---: | --- |
| Temporal offset score | 0.1575 | Heavy penalties from >70 ms camera<->LiDAR and LiDAR<->IMU offsets drove the score well below the 0.8 health target. |
| Avg. drift rate | 7.79 ms/s | Sustained drift would accumulate ~39 ms over a 5 s segment even without the large static offsets. |
| Max predicted drift (Kalman) | 67.22 ms | Forecasted divergence within the configured 5 s horizon exceeds the 5 ms ApproximateTime budget. |
| Min chi^2 p-value | 3.45e-22 | Strong statistical evidence of desynchronization; null hypothesis of aligned timestamps is rejected. |
| ISO 8000-61 compliance | **FAIL** | Validator flagged the run as non-compliant due to multiple temporal violations. |

## Sensor Pair Analysis
| Pair | Max Delta t (ms) | Drift Rate (ms/s) | Kalman Drift (ms @5 s) | ApproxTime | PTP | chi^2 p-value |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| camera <-> lidar | 74.30 | +0.19 | 67.22 | **Fail** | **Fail** | 1.73e-11 |
| lidar <-> imu | 79.26 | -14.63 | 66.32 | **Fail** | **Fail** | 3.45e-22 |
| camera <-> imu | 4.96 | -8.55 | 0.90 | Pass | **Fail** | 2.99e-15 |

Notes:
- ApproximateTime was limited to 5 ms to simulate a tight sensor-fusion budget; both lidar-involved pairs violated this margin by >70 ms.
- PTP budget remained at the validator default (1 ms). Even the best-aligned pair (camera<->IMU) exceeded this limit due to a measured 4.96 ms offset.

## Recommendations & Compliance Flags
The validator produced the following actionable items:
- camera<->lidar: Reduce the 74 ms timestamp delta (estimated correction -68.96 ms on LiDAR) and investigate chi-square and IEEE 1588 failures before re-running fusion.
- lidar<->imu: Apply -69.10 ms to LiDAR (or +69.10 ms to IMU) to remove the static offset; high negative drift (-14.63 ms/s) indicates clock slew requiring calibration.
- camera<->imu: Even though ApproximateTime passed, the 4.96 ms hardware offset violates the 1 ms PTP budget. Recalibrate PPS distribution if sub-millisecond sync is required.

Raised compliance flags: `camera_lidar_approximate_time_violation`, `camera_lidar_ptp_violation`, `camera_lidar_chi_square_failure`, `lidar_imu_approximate_time_violation`, `lidar_imu_ptp_violation`, `lidar_imu_chi_square_failure`, `camera_imu_ptp_violation`, `camera_imu_chi_square_failure`.

## Generated Artifacts
- JSON snapshot of the complete validator output: `reports/2011_09_26_drive_0002_sync/temporal_sync_report.json`
- ISO 8000-61 compliant correction params: `reports/2011_09_26_drive_0002_sync/temporal_sync_artifacts/params/2011_09_26_drive_0002_sync_timestamp_corrections.yaml`
- Heatmaps visualizing per-pair delta distributions: files under `reports/2011_09_26_drive_0002_sync/temporal_sync_artifacts/heatmaps/`

These artifacts - along with this report - should be committed to track historical validator performance. The KITTI input data itself remains untouched and uncommitted as required.
