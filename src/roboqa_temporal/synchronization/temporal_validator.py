"""

################################################################

File: roboqa_temporal/synchronization/temporal_validator.py
Created: 2025-12-07
Created by: RoboQA-Temporal Authors
Last Modified: 2025-12-07
Last Modified by: RoboQA-Temporal Authors

#################################################################

Copyright: RoboQA-Temporal Authors
License: MIT License

################################################################

Multi-Sensor Temporal Synchronization Validator.

This module inspects multi-sensor datasets (e.g., KITTI format) with
camera, LiDAR, IMU, and other sensor data stored as image sequences
with timestamp files. It verifies that sensor timestamps are aligned,
frequencies are stable, and long-term drift stays within tolerance.
The validator generates temporal heatmaps and exports comprehensive
reports in multiple formats.

Adapted from ROS2-based validator to work with raw image folders.

################################################################

"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from scipy import signal, stats
except ImportError:
    signal = None
    stats = None


@dataclass
class SensorStream:
    """Container describing a single sensor stream."""

    name: str
    source_path: str
    timestamps_ns: List[int]
    expected_frequency: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamps_sec: np.ndarray = field(init=False, repr=False)
    frequency_estimate_hz: Optional[float] = field(init=False, default=None)

    def __post_init__(self) -> None:
        ts_ns = np.array(self.timestamps_ns, dtype=np.float64)
        self.timestamps_sec = ts_ns / 1e9 if ts_ns.size else np.array([])

        if self.timestamps_sec.size >= 2:
            diffs = np.diff(self.timestamps_sec)
            valid = diffs[diffs > 0]
            if valid.size:
                self.frequency_estimate_hz = float(1.0 / np.median(valid))
        self.metadata["message_count"] = len(self.timestamps_ns)
        self.metadata["missing_frames"] = self._detect_missing_frames()
        self.metadata["duplicate_frames"] = self._detect_duplicates()

    def _detect_missing_frames(self) -> int:
        """Detect missing frames based on expected frequency."""
        if self.timestamps_sec.size < 2:
            return 0
        diffs = np.diff(self.timestamps_sec)
        if self.frequency_estimate_hz is None:
            return 0
        expected_interval = 1.0 / self.frequency_estimate_hz
        # Counting gaps larger than 1.5x expected interval
        missing = np.sum(diffs > expected_interval * 1.5)
        return int(missing)

    def _detect_duplicates(self) -> int:
        """Detect duplicate timestamps."""
        if self.timestamps_sec.size < 2:
            return 0
        diffs = np.diff(self.timestamps_sec)
        # Counting near-zero differences (< 1ms)
        duplicates = np.sum(np.abs(diffs) < 0.001)
        return int(duplicates)


@dataclass
class PairwiseDriftResult:
    """Statistics for a single sensor pair."""

    pair_name: str
    deltas_ms: np.ndarray
    timestamps_sec: np.ndarray
    rolling_mean_ms: np.ndarray
    rolling_std_ms: np.ndarray
    max_delta_ms: float
    temporal_offset_score: float
    drift_rate_ms_per_s: float
    approx_time_pass: bool
    chi_square_pvalue: float
    cross_correlation_lag_ms: float
    kalman_predicted_drift_ms: float
    ptp_pass: bool
    frequency_ok: bool
    recommendations: List[str] = field(default_factory=list)
    compliance_flags: List[str] = field(default_factory=list)
    heatmap_path: Optional[str] = None


@dataclass
class TemporalSyncReport:
    """Aggregate report returned by the validator."""

    streams: Dict[str, SensorStream]
    pair_results: Dict[str, PairwiseDriftResult]
    metrics: Dict[str, float]
    recommendations: List[str]
    compliance_flags: List[str]
    parameter_file: Optional[str] = None
    report_files: Dict[str, str] = field(default_factory=dict)
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serializing reports to a dictionary."""
        return {
            "generated_at": self.generated_at,
            "streams": {name: asdict(stream) for name, stream in self.streams.items()},
            "pair_results": {
                name: {
                    **{
                        key: value
                        for key, value in asdict(result).items()
                        if key not in {"heatmap_path"}
                    },
                    "heatmap_path": result.heatmap_path,
                }
                for name, result in self.pair_results.items()
            },
            "metrics": self.metrics,
            "recommendations": self.recommendations,
            "compliance_flags": self.compliance_flags,
            "parameter_file": self.parameter_file,
            "report_files": self.report_files,
        }


class TemporalSyncValidator:
    """Validates temporal alignment across camera, LiDAR, IMU, and other sensor streams."""

    DEFAULT_SENSOR_FOLDERS = {
        "camera_left": "image_00",
        "camera_right": "image_01",
        "camera_color_left": "image_02",
        "camera_color_right": "image_03",
        "lidar": "velodyne_points",
        "imu": "oxts",
    }

    PAIRS = (
        ("camera_left", "lidar"),
        ("camera_left", "camera_right"),
        ("lidar", "imu"),
        ("camera_left", "imu"),
    )

    def __init__(
        self,
        sensor_folders: Optional[Dict[str, str]] = None,
        expected_frequency_hz: Optional[Dict[str, float]] = None,
        approximate_time_threshold_ms: Optional[Dict[str, float]] = None,
        output_dir: str = "reports/temporal_sync",
        rolling_window: int = 50,
        ptp_config: Optional[Dict[str, float]] = None,
        heatmap_resolution: int = 128,
        kalman_process_noise: float = 1e-6,
        kalman_measurement_noise: float = 1e-4,
        kalman_horizon_s: float = 5.0,
        report_formats: Optional[Iterable[str]] = None,
        auto_export_reports: bool = True,
    ) -> None:
        self.sensor_folders = {
            **TemporalSyncValidator.DEFAULT_SENSOR_FOLDERS,
            **(sensor_folders or {}),
        }
        self.expected_frequency_hz = expected_frequency_hz or {
            "camera_left": 10.0,
            "camera_right": 10.0,
            "camera_color_left": 10.0,
            "camera_color_right": 10.0,
            "lidar": 10.0,
            "imu": 100.0,
        }
        self.approximate_time_threshold_ms = approximate_time_threshold_ms or {
            "camera_left_lidar": 100.0,        # 100ms tolerance for camera-lidar
            "camera_left_camera_right": 50.0,  # 50ms tolerance for multi-camera
            "lidar_imu": 100.0,                # 100ms tolerance for lidar-imu
            "camera_left_imu": 100.0,          # 100ms tolerance for camera-imu
        }
        self.output_dir = Path(output_dir)
        self.heatmap_dir = self.output_dir / "heatmaps"
        self.param_dir = self.output_dir / "params"
        for directory in (self.output_dir, self.heatmap_dir, self.param_dir):
            directory.mkdir(parents=True, exist_ok=True)

        self.rolling_window = rolling_window
        self.ptp_config = ptp_config or {
            "max_offset_ns": 1_000_000,  # 1 millisecond for image data
            "max_jitter_ns": 100_000,  # 100 microseconds
        }
        self.heatmap_resolution = heatmap_resolution
        self.kalman_process_noise = kalman_process_noise
        self.kalman_measurement_noise = kalman_measurement_noise
        self.kalman_horizon_s = kalman_horizon_s
        normalized_formats = None
        if report_formats:
            normalized_formats = tuple(sorted({fmt.lower() for fmt in report_formats}))
        self.report_formats = normalized_formats
        self.auto_export_reports = auto_export_reports

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def validate(
        self,
        dataset_path: str,
        max_frames: Optional[int] = None,
        include_visualizations: bool = True,
    ) -> TemporalSyncReport:
        """
        Running full validation pipeline on a dataset folder.

        Args:
            dataset_path: Path to dataset folder containing sensor subfolders
            max_frames: Maximum number of frames to load per sensor
            include_visualizations: Whether to generate heatmaps

        Returns:
            TemporalSyncReport with validation results
        """
        streams = self._load_streams(Path(dataset_path), max_frames)
        return self.analyze_streams(streams, Path(dataset_path).name, include_visualizations)

    def analyze_streams(
        self,
        streams: Dict[str, SensorStream],
        dataset_name: str = "dataset",
        include_visualizations: bool = True,
    ) -> TemporalSyncReport:
        """
        Perform temporal analysis on already extracted streams.
        Args:
            streams: Dictionary of sensor streams
            dataset_name: Name of the dataset
            include_visualizations: Whether to generate heatmaps
        Returns:
            TemporalSyncReport with validation results
        """
        pair_results: Dict[str, PairwiseDriftResult] = {}
        global_recommendations: List[str] = []
        compliance_flags: List[str] = []

        # Checking individual stream health
        for sensor, stream in streams.items():
            freq_ok = self._check_frequency(stream)
            if not freq_ok:
                msg = f"{sensor} frequency deviation exceeds tolerance"
                global_recommendations.append(msg)
                compliance_flags.append(f"{sensor}_frequency_violation")

            # Checking for data loss/duplication
            if stream.metadata.get("missing_frames", 0) > 0:
                msg = f"{sensor} has {stream.metadata['missing_frames']} missing frame(s)"
                global_recommendations.append(msg)
                compliance_flags.append(f"{sensor}_data_loss")

            if stream.metadata.get("duplicate_frames", 0) > 0:
                msg = f"{sensor} has {stream.metadata['duplicate_frames']} duplicate timestamp(s)"
                global_recommendations.append(msg)
                compliance_flags.append(f"{sensor}_data_duplication")

        # Analyzing sensor pairs
        for a, b in TemporalSyncValidator.PAIRS:
            if a not in streams or b not in streams:
                continue

            pair_key = f"{a}_{b}"
            threshold = self.approximate_time_threshold_ms.get(pair_key, 30.0)
            result = self._analyze_pair(
                streams[a],
                streams[b],
                pair_key,
                threshold,
                dataset_name,
                include_visualizations,
            )
            pair_results[pair_key] = result
            global_recommendations.extend(result.recommendations)
            compliance_flags.extend(result.compliance_flags)

        metrics = self._aggregate_metrics(pair_results, streams)
        param_path = self._export_parameter_file(pair_results, dataset_name)

        report = TemporalSyncReport(
            streams=streams,
            pair_results=pair_results,
            metrics=metrics,
            recommendations=sorted(set(global_recommendations)),
            compliance_flags=sorted(set(compliance_flags)),
            parameter_file=param_path,
        )
        
        if self.auto_export_reports:
            formats = self.report_formats or ("markdown", "html", "csv")
            report.report_files = self._export_summary_reports(
                report, dataset_name, formats, include_visualizations
            )

        return report

    # ------------------------------------------------------------------ #
    # Dataset ingestion
    # ------------------------------------------------------------------ #

    def _load_streams(
        self, dataset_path: Path, max_frames: Optional[int]
    ) -> Dict[str, SensorStream]:
        """Loading sensor streams from dataset folder structure."""
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

        streams: Dict[str, SensorStream] = {}

        for sensor_name, folder_name in self.sensor_folders.items():
            sensor_path = dataset_path / folder_name
            if not sensor_path.exists():
                continue

            timestamps_file = sensor_path / "timestamps.txt"
            if not timestamps_file.exists():
                # Try looking in data subfolder
                timestamps_file = sensor_path / "data" / "timestamps.txt"
                if not timestamps_file.exists():
                    continue

            try:
                timestamps_ns = self._read_timestamps(timestamps_file, max_frames)
                if not timestamps_ns:
                    continue

                # Counting available data files
                data_folder = sensor_path / "data" if (sensor_path / "data").exists() else sensor_path
                data_files = sorted(data_folder.glob("*"))
                data_files = [f for f in data_files if f.is_file() and f.name != "timestamps.txt"]

                metadata = {
                    "source_path": str(sensor_path),
                    "data_files_count": len(data_files),
                    "timestamp_file": str(timestamps_file),
                }

                stream = SensorStream(
                    name=sensor_name,
                    source_path=str(sensor_path),
                    timestamps_ns=timestamps_ns,
                    metadata=metadata,
                    expected_frequency=self.expected_frequency_hz.get(sensor_name),
                )
                streams[sensor_name] = stream

            except Exception as e:
                print(f"Warning: Failed to load {sensor_name} from {sensor_path}: {e}")
                continue

        if not streams:
            raise ValueError(f"No valid sensor streams found in {dataset_path}")

        return streams

    def _read_timestamps(self, timestamp_file: Path, max_frames: Optional[int]) -> List[int]:
        """
        Reading timestamps from a file.

        Supports formats:
        - KITTI format: YYYY-MM-DD HH:MM:SS.nanoseconds
        - Unix timestamps in seconds (float or int)
        - Unix timestamps in nanoseconds (int)
        """
        timestamps_ns: List[int] = []

        with open(timestamp_file, "r") as f:
            for idx, line in enumerate(f):
                if max_frames and idx >= max_frames:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    # Trying KITTI datetime format first
                    if " " in line and "-" in line:
                        # Handling KITTI format: YYYY-MM-DD HH:MM:SS.nanoseconds
                        parts = line.split(".")
                        if len(parts) == 2:
                            dt_part = parts[0]
                            ns_part = parts[1]
                            
                            # Parsing the datetime part
                            dt = datetime.strptime(dt_part, "%Y-%m-%d %H:%M:%S")
                            base_timestamp_ns = int(dt.timestamp() * 1e9)
                            ns_part = ns_part.ljust(9, '0')[:9]
                            nanoseconds = int(ns_part)
                            
                            timestamp_ns = base_timestamp_ns + nanoseconds
                        else:
                            # Fallback if format is unexpected
                            dt = datetime.strptime(line, "%Y-%m-%d %H:%M:%S")
                            timestamp_ns = int(dt.timestamp() * 1e9)
                    else:
                        # Trying numeric format
                        timestamp_float = float(line)
                        # Distinguish between seconds and nanoseconds
                        if timestamp_float > 1e15:
                            timestamp_ns = int(timestamp_float)
                        else:
                            timestamp_ns = int(timestamp_float * 1e9)

                    timestamps_ns.append(timestamp_ns)

                except Exception as e:
                    print(f"Warning: Failed to parse timestamp '{line}': {e}")
                    continue

        return timestamps_ns

    # ------------------------------------------------------------------ #
    # Analysis helpers
    # ------------------------------------------------------------------ #

    def _analyze_pair(
        self,
        stream_a: SensorStream,
        stream_b: SensorStream,
        pair_key: str,
        threshold_ms: float,
        dataset_name: str,
        include_visualizations: bool,
    ) -> PairwiseDriftResult:
        """Analyze synchronization between two sensor streams."""
        threshold_s = threshold_ms / 1000.0
        timestamps_a = stream_a.timestamps_sec
        timestamps_b = stream_b.timestamps_sec

        matches = self._approximate_time_matches(timestamps_a, timestamps_b, threshold_s)
        if not matches:
            # For completely non-overlapping streams, estimating deltas directly
            synthetic_deltas = self._estimate_deltas_without_matches(
                timestamps_a, timestamps_b
            )
            matches = synthetic_deltas

        mid_times = np.array([m[0] for m in matches], dtype=np.float64)
        deltas = np.array([m[1] for m in matches], dtype=np.float64)
        deltas_ms = deltas * 1000.0

        if deltas_ms.size == 0:
            deltas_ms = np.array([0.0])
            mid_times = np.array([0.0])

        series = pd.Series(deltas_ms)
        rolling_mean = (
            series.rolling(window=min(self.rolling_window, len(series)), min_periods=1)
            .mean()
            .to_numpy()
        )
        rolling_std = (
            series.rolling(window=min(self.rolling_window, len(series)), min_periods=1)
            .std()
            .fillna(0.0)
            .to_numpy()
        )

        max_delta = float(np.max(np.abs(deltas_ms)))
        mean_abs_delta = float(np.mean(np.abs(deltas_ms)))
        # If within threshold => score approaches 1.0
        # If beyond threshold => score degrades proportionally
        if mean_abs_delta <= threshold_ms:
            temporal_offset_score = 1.0 - (mean_abs_delta / (threshold_ms * 2.0))
        else:
            temporal_offset_score = max(0.0, 1.0 - (mean_abs_delta / threshold_ms) * 0.5)
        approx_time_pass = max_delta <= threshold_ms * 1.5  # Allowing 1.5x threshold for max

        drift_rate = self._compute_drift_rate(mid_times, deltas_ms)
        chi_square_pvalue = self._chi_square_test(deltas_ms)
        cross_corr_lag = self._cross_correlation_lag(timestamps_a, timestamps_b)
        kalman_projection = self._kalman_predict(mid_times, deltas_ms)
        ptp_pass = (max_delta * 1e6) <= self.ptp_config["max_offset_ns"]

        frequency_ok = (
            stream_a.frequency_estimate_hz is not None
            and stream_b.frequency_estimate_hz is not None
        )

        recommendations: List[str] = []
        compliance_flags: List[str] = []

        if not approx_time_pass:
            recommendations.append(
                f"{pair_key}: timestamp delta ({max_delta:.2f} ms) exceeds threshold ({threshold_ms} ms)"
            )
            compliance_flags.append(f"{pair_key}_time_sync_violation")

        if chi_square_pvalue < 0.01:
            recommendations.append(f"{pair_key}: chi-square test indicates inconsistent sync behavior")
            compliance_flags.append(f"{pair_key}_chi_square_failure")

        if abs(drift_rate) > 1.0:
            recommendations.append(
                f"{pair_key}: drift rate {drift_rate:.2f} ms/s suggests clock desynchronization"
            )

        if kalman_projection > threshold_ms:
            recommendations.append(
                f"{pair_key}: projected drift {kalman_projection:.2f} ms over {self.kalman_horizon_s}s exceeds tolerance"
            )

        if not ptp_pass:
            recommendations.append(
                f"{pair_key}: offset {max_delta:.2f} ms violates timing budget"
            )
            compliance_flags.append(f"{pair_key}_timing_violation")

        heatmap_path = None
        if include_visualizations:
            heatmap_path = self._generate_heatmap(
                pair_key, dataset_name, mid_times, deltas_ms
            )

        return PairwiseDriftResult(
            pair_name=pair_key,
            deltas_ms=deltas_ms,
            timestamps_sec=mid_times,
            rolling_mean_ms=rolling_mean,
            rolling_std_ms=rolling_std,
            max_delta_ms=max_delta,
            temporal_offset_score=temporal_offset_score,
            drift_rate_ms_per_s=drift_rate,
            approx_time_pass=approx_time_pass,
            chi_square_pvalue=chi_square_pvalue,
            cross_correlation_lag_ms=cross_corr_lag,
            kalman_predicted_drift_ms=kalman_projection,
            ptp_pass=ptp_pass,
            frequency_ok=frequency_ok,
            recommendations=recommendations,
            compliance_flags=compliance_flags,
            heatmap_path=heatmap_path,
        )

    def _approximate_time_matches(
        self,
        timestamps_a: np.ndarray,
        timestamps_b: np.ndarray,
        tolerance_s: float,
    ) -> List[Tuple[float, float]]:
        """Emulate approximate time pairing for timestamps."""
        matches: List[Tuple[float, float]] = []
        if not timestamps_a.size or not timestamps_b.size:
            return matches

        i = j = 0
        while i < len(timestamps_a) and j < len(timestamps_b):
            delta = timestamps_a[i] - timestamps_b[j]
            if abs(delta) <= tolerance_s:
                matches.append(((timestamps_a[i] + timestamps_b[j]) / 2.0, delta))
                i += 1
                j += 1
            elif timestamps_a[i] < timestamps_b[j]:
                i += 1
            else:
                j += 1
        return matches

    def _estimate_deltas_without_matches(
        self, timestamps_a: np.ndarray, timestamps_b: np.ndarray
    ) -> List[Tuple[float, float]]:
        """Fallback delta estimation when timestamps never align."""
        if not timestamps_a.size or not timestamps_b.size:
            return []

        min_len = min(len(timestamps_a), len(timestamps_b))
        sampled_a = np.linspace(0, len(timestamps_a) - 1, min_len, dtype=int)
        sampled_b = np.linspace(0, len(timestamps_b) - 1, min_len, dtype=int)
        matches = []
        for idx_a, idx_b in zip(sampled_a, sampled_b):
            time_mid = (timestamps_a[idx_a] + timestamps_b[idx_b]) / 2.0
            delta = timestamps_a[idx_a] - timestamps_b[idx_b]
            matches.append((time_mid, delta))
        return matches

    def _compute_drift_rate(
        self, timestamps: np.ndarray, deltas_ms: np.ndarray
    ) -> float:
        """Compute drift rate in ms/s."""
        if timestamps.size < 2:
            return 0.0
        gradient = np.gradient(deltas_ms, timestamps, edge_order=1)
        return float(np.median(gradient))

    @staticmethod
    def _chi_square_test(deltas_ms: np.ndarray) -> float:
        """Perform chi-square test on deltas."""
        if len(deltas_ms) < 3:
            return 1.0
        observed = np.abs(deltas_ms - np.mean(deltas_ms)) + 1e-6
        expected = np.full_like(observed, np.mean(observed))
        if stats:
            chi_stat, pval = stats.chisquare(observed, expected)
            if math.isnan(pval):
                return 1.0
            return float(pval)
        chi_stat = np.sum((observed - expected) ** 2 / (expected + 1e-6))
        approx_p = math.exp(-0.5 * float(chi_stat))
        return max(0.0, min(1.0, approx_p))

    def _cross_correlation_lag(
        self, timestamps_a: np.ndarray, timestamps_b: np.ndarray
    ) -> float:
        """Compute cross-correlation lag between two timestamp sequences."""
        if timestamps_a.size < 2 or timestamps_b.size < 2:
            return 0.0

        start = min(timestamps_a.min(), timestamps_b.min())
        end = max(timestamps_a.max(), timestamps_b.max())
        if end - start <= 0:
            return 0.0

        bins = min(self.heatmap_resolution, max(len(timestamps_a), len(timestamps_b)))
        hist_a, edges = np.histogram(timestamps_a, bins=bins, range=(start, end))
        hist_b, _ = np.histogram(timestamps_b, bins=bins, range=(start, end))

        if signal:
            corr = signal.correlate(hist_a, hist_b, mode="full")
        else:
            corr = np.correlate(hist_a, hist_b, mode="full")
        lag_index = int(np.argmax(corr) - (len(hist_a) - 1))
        bin_width_s = (edges[1] - edges[0]) if len(edges) > 1 else 0.0
        return float(lag_index * bin_width_s * 1000.0)

    def _kalman_predict(
        self, timestamps: np.ndarray, deltas_ms: np.ndarray
    ) -> float:
        """Predict future drift using Kalman filter."""
        if timestamps.size < 2:
            return float(np.abs(deltas_ms[-1])) if deltas_ms.size else 0.0

        # Initialization of offset and drift rate state
        state = np.array([deltas_ms[0], 0.0])
        covariance = np.eye(2)
        H = np.array([[1.0, 0.0]])
        Q = np.eye(2) * self.kalman_process_noise
        R = self.kalman_measurement_noise

        for idx in range(1, len(deltas_ms)):
            delta_t = max(timestamps[idx] - timestamps[idx - 1], 1e-6)
            F = np.array([[1.0, delta_t], [0.0, 1.0]])
            state = F @ state
            covariance = F @ covariance @ F.T + Q

            y = deltas_ms[idx] - (H @ state)
            S = H @ covariance @ H.T + R
            K = covariance @ H.T / S
            state = state + (K.flatten() * y)
            covariance = (np.eye(2) - K @ H) @ covariance

        future_offset = state[0] + state[1] * self.kalman_horizon_s
        return float(abs(future_offset))

    def _generate_heatmap(
        self,
        pair_key: str,
        dataset_name: str,
        timestamps: np.ndarray,
        deltas_ms: np.ndarray,
    ) -> Optional[str]:
        """Generate temporal alignment heatmap."""
        if timestamps.size < 2:
            return None

        x_min, x_max = float(timestamps.min()), float(timestamps.max())
        y_min, y_max = float(deltas_ms.min()), float(deltas_ms.max())
        
        y_range = y_max - y_min
        if y_range < 1e-6:
            y_min -= 0.5
            y_max += 0.5
        
        heatmap, xedges, yedges = np.histogram2d(
            timestamps,
            deltas_ms,
            bins=self.heatmap_resolution,
            range=[[x_min, x_max], [y_min, y_max]],
        )
        if not heatmap.size or np.allclose(heatmap.sum(), 0):
            return None

        plt.figure(figsize=(10, 5))
        sns.heatmap(
            heatmap.T,
            cmap="coolwarm",
            cbar_kws={"label": "Sample Count"},
            xticklabels=False,
            yticklabels=False,
        )
        plt.title(f"Temporal Alignment Heatmap: {pair_key}")
        plt.xlabel("Time (relative)")
        plt.ylabel("Time Delta (ms)")
        output_path = self.heatmap_dir / f"{dataset_name}_{pair_key}_heatmap.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=100)
        plt.close()
        return str(output_path)

    def _check_frequency(self, stream: SensorStream) -> bool:
        """Check if stream frequency matches expected value."""
        if stream.expected_frequency is None or stream.frequency_estimate_hz is None:
            return True
        freq = stream.frequency_estimate_hz
        target = stream.expected_frequency
        deviation = abs(freq - target) / max(target, 1e-6)
        stream.metadata["frequency_estimate_hz"] = freq
        stream.metadata["frequency_deviation_pct"] = deviation * 100.0
        return deviation <= 0.15

    def _aggregate_metrics(
        self, pair_results: Dict[str, PairwiseDriftResult], streams: Dict[str, SensorStream]
    ) -> Dict[str, float]:
        """Aggregate metrics from all analyses."""
        metrics: Dict[str, float] = {}
        if not pair_results:
            return metrics

        offset_scores = [r.temporal_offset_score for r in pair_results.values()]
        drift_rates = [abs(r.drift_rate_ms_per_s) for r in pair_results.values()]
        chi_square = [r.chi_square_pvalue for r in pair_results.values()]
        predicted = [r.kalman_predicted_drift_ms for r in pair_results.values()]

        metrics["temporal_offset_score"] = float(np.mean(offset_scores))
        metrics["avg_drift_rate_ms_per_s"] = float(np.mean(drift_rates))
        metrics["min_chi_square_pvalue"] = float(np.min(chi_square))
        metrics["max_predicted_drift_ms"] = float(np.max(predicted))
        
        # Data loss metrics
        total_missing = sum(s.metadata.get("missing_frames", 0) for s in streams.values())
        total_duplicates = sum(s.metadata.get("duplicate_frames", 0) for s in streams.values())
        metrics["total_missing_frames"] = float(total_missing)
        metrics["total_duplicate_frames"] = float(total_duplicates)
        
        # Overall quality score
        # Weights: temporal offset (50%), drift rate (30%), chi-square (20%)
        metrics["synchronization_quality_score"] = max(0.0, 1.0 - (
            (1.0 - metrics["temporal_offset_score"]) * 0.5 +
            min(metrics["avg_drift_rate_ms_per_s"] / 20.0, 1.0) * 0.3 +
            (1.0 if metrics["min_chi_square_pvalue"] < 0.01 else 0.0) * 0.2
        ))
        
        return metrics

    def _export_parameter_file(
        self, pair_results: Dict[str, PairwiseDriftResult], dataset_name: str
    ) -> Optional[str]:
        """Export timestamp correction parameters."""
        if not pair_results:
            return None

        corrections: Dict[str, Dict[str, float]] = {}
        for pair_name, result in pair_results.items():
            sensors = pair_name.split("_", 1)
            if result.deltas_ms.size:
                mean_delta = float(np.mean(result.deltas_ms))
            else:
                mean_delta = 0.0
            corrections[pair_name] = {
                "mean_offset_ms": mean_delta,
                "max_offset_ms": result.max_delta_ms,
                "drift_rate_ms_per_s": result.drift_rate_ms_per_s,
            }

        payload = {
            "temporal_corrections": corrections,
            "metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "dataset_name": dataset_name,
            },
        }
        output_path = self.param_dir / f"{dataset_name}_timestamp_corrections.yaml"
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)
        return str(output_path)

    def _export_summary_reports(
        self,
        report: TemporalSyncReport,
        dataset_name: str,
        formats: Iterable[str],
        include_visualizations: bool,
    ) -> Dict[str, str]:
        """Export summary reports in multiple formats."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        base_name = f"{dataset_name}_sync_{timestamp}"
        exported: Dict[str, str] = {}
        pair_df = self._build_pair_dataframe(report)

        for fmt in formats:
            fmt_lower = fmt.lower()
            if fmt_lower == "markdown":
                path = self._write_markdown_report(report, pair_df, base_name)
                exported["markdown"] = str(path)
            elif fmt_lower == "html":
                path = self._write_html_report(report, pair_df, base_name, include_visualizations)
                exported["html"] = str(path)
            elif fmt_lower == "csv":
                path = self._write_csv_report(pair_df, base_name)
                exported["csv"] = str(path)
        return exported

    def _build_pair_dataframe(self, report: TemporalSyncReport) -> pd.DataFrame:
        """Build dataframe from pair results."""
        rows: List[Dict[str, Any]] = []
        for pair in report.pair_results.values():
            rows.append(
                {
                    "pair_name": pair.pair_name,
                    "max_delta_ms": pair.max_delta_ms,
                    "temporal_offset_score": pair.temporal_offset_score,
                    "drift_rate_ms_per_s": pair.drift_rate_ms_per_s,
                    "approx_time_pass": pair.approx_time_pass,
                    "ptp_pass": pair.ptp_pass,
                    "chi_square_pvalue": pair.chi_square_pvalue,
                    "cross_correlation_lag_ms": pair.cross_correlation_lag_ms,
                    "kalman_predicted_drift_ms": pair.kalman_predicted_drift_ms,
                    "frequency_ok": pair.frequency_ok,
                    "recommendations": "; ".join(pair.recommendations),
                    "compliance_flags": "; ".join(pair.compliance_flags),
                }
            )
        columns = [
            "pair_name",
            "max_delta_ms",
            "temporal_offset_score",
            "drift_rate_ms_per_s",
            "approx_time_pass",
            "ptp_pass",
            "chi_square_pvalue",
            "cross_correlation_lag_ms",
            "kalman_predicted_drift_ms",
            "frequency_ok",
            "recommendations",
            "compliance_flags",
        ]
        if rows:
            return pd.DataFrame(rows, columns=columns)
        return pd.DataFrame(columns=columns)

    def _write_markdown_report(
        self, report: TemporalSyncReport, pair_df: pd.DataFrame, base_name: str
    ) -> Path:
        """Write markdown format report."""
        output_path = self.output_dir / f"{base_name}.md"
        with open(output_path, "w", encoding="utf-8") as md_file:
            md_file.write("# Cross-Modal Synchronization Analysis Report\n\n")
            md_file.write(f"**Generated:** {report.generated_at}\n\n")
            md_file.write("---\n\n")

            # Executive Summary
            md_file.write("## Executive Summary\n\n")
            sync_score = report.metrics.get("synchronization_quality_score", 0.0)
            md_file.write(f"**Synchronization Quality Score:** {sync_score:.2%}\n\n")
            md_file.write(f"**Total Sensor Streams:** {len(report.streams)}\n\n")
            md_file.write(f"**Sensor Pairs Analyzed:** {len(report.pair_results)}\n\n")
            md_file.write(f"**Total Missing Frames:** {int(report.metrics.get('total_missing_frames', 0))}\n\n")
            md_file.write(f"**Total Duplicate Timestamps:** {int(report.metrics.get('total_duplicate_frames', 0))}\n\n")

            # Sensor Streams
            md_file.write("## Sensor Streams\n\n")
            md_file.write("| Sensor | Frames | Frequency (Hz) | Missing | Duplicates |\n")
            md_file.write("|--------|--------|----------------|---------|------------|\n")
            for name, stream in sorted(report.streams.items()):
                freq = stream.frequency_estimate_hz or 0.0
                missing = stream.metadata.get("missing_frames", 0)
                dups = stream.metadata.get("duplicate_frames", 0)
                md_file.write(f"| {name} | {stream.metadata['message_count']} | {freq:.2f} | {missing} | {dups} |\n")
            md_file.write("\n")

            # Metrics
            md_file.write("## Synchronization Metrics\n\n")
            md_file.write("| Metric | Value |\n|--------|-------|\n")
            for key, value in sorted(report.metrics.items()):
                if isinstance(value, float):
                    md_file.write(f"| {key} | {value:.6f} |\n")
                else:
                    md_file.write(f"| {key} | {value} |\n")
            md_file.write("\n")

            # Pairwise Results
            if not pair_df.empty:
                md_file.write("## Pairwise Synchronization Results\n\n")
                md_file.write(self._dataframe_to_markdown(pair_df))
                md_file.write("\n\n")

            # Recommendations
            if report.recommendations:
                md_file.write("## Recommendations\n\n")
                for rec in report.recommendations:
                    md_file.write(f"- {rec}\n")
                md_file.write("\n")

            # Compliance Flags
            if report.compliance_flags:
                md_file.write("## Compliance Flags\n\n")
                for flag in report.compliance_flags:
                    md_file.write(f"- {flag}\n")
                md_file.write("\n")

            # Timestamp Corrections
            if report.parameter_file:
                md_file.write("## Timestamp Corrections\n\n")
                md_file.write(f"Parameter file: `{report.parameter_file}`\n\n")

        return output_path

    def _write_html_report(
        self,
        report: TemporalSyncReport,
        pair_df: pd.DataFrame,
        base_name: str,
        include_visualizations: bool,
    ) -> Path:
        """Write HTML format report."""
        output_path = self.output_dir / f"{base_name}.html"
        
        # Building sensor streams table
        stream_rows = []
        for name, stream in sorted(report.streams.items()):
            freq = stream.frequency_estimate_hz or 0.0
            missing = stream.metadata.get("missing_frames", 0)
            dups = stream.metadata.get("duplicate_frames", 0)
            stream_rows.append(
                f"<tr><td>{name}</td><td>{stream.metadata['message_count']}</td>"
                f"<td>{freq:.2f}</td><td>{missing}</td><td>{dups}</td></tr>"
            )
        streams_table = "\n".join(stream_rows)

        # Building metrics table
        metric_entries = []
        for key, value in sorted(report.metrics.items()):
            if isinstance(value, float):
                display = f"{value:.6f}"
            else:
                display = str(value)
            metric_entries.append(f"<tr><td>{key}</td><td>{display}</td></tr>")
        metrics_rows = "\n".join(metric_entries)
        
        pair_table = pair_df.to_html(index=False, classes="pair-table") if not pair_df.empty else "<p>No pair results.</p>"
        
        rec_list = (
            "".join(f"<li>{rec}</li>" for rec in report.recommendations)
            if report.recommendations
            else "<li>No recommendations</li>"
        )
        flag_list = (
            "".join(f"<li>{flag}</li>" for flag in report.compliance_flags)
            if report.compliance_flags
            else "<li>No compliance flags</li>"
        )

        heatmap_section = ""
        if include_visualizations:
            snippets: List[str] = []
            for pair in report.pair_results.values():
                if not pair.heatmap_path:
                    continue
                rel_path = os.path.relpath(pair.heatmap_path, start=output_path.parent)
                snippets.append(
                    f"<div class='heatmap'><h3>{pair.pair_name}</h3>"
                    f"<img src=\"{rel_path}\" alt=\"Heatmap for {pair.pair_name}\" /></div>"
                )
            if snippets:
                heatmap_section = "<section><h2>Temporal Heatmaps</h2>" + "\n".join(snippets) + "</section>"

        param_section = ""
        if report.parameter_file:
            rel_param = os.path.relpath(report.parameter_file, start=output_path.parent)
            param_section = f"<section><h2>Timestamp Corrections</h2><p>Parameter file: <code>{rel_param}</code></p></section>"

        sync_score = report.metrics.get("synchronization_quality_score", 0.0)
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8"/>
    <title>Cross-Modal Synchronization Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; color: #222; background: #f9f9f9; }}
        h1, h2, h3 {{ color: #0b3954; }}
        .summary {{ background: #fff; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .score {{ font-size: 2em; color: #087f5b; font-weight: bold; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; background: #fff; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        th {{ background: #e9ecef; font-weight: bold; }}
        tr:hover {{ background: #f8f9fa; }}
        section {{ background: #fff; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .heatmap {{ margin: 20px 0; }}
        .heatmap img {{ max-width: 100%; border: 1px solid #ddd; padding: 4px; border-radius: 4px; }}
        ul {{ line-height: 1.8; }}
        .pair-table {{ font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>Cross-Modal Synchronization Analysis Report</h1>
    <div class="summary">
        <p><strong>Generated:</strong> {report.generated_at}</p>
        <p class="score">Synchronization Quality Score: {sync_score:.2%}</p>
        <p><strong>Total Sensor Streams:</strong> {len(report.streams)}</p>
        <p><strong>Total Missing Frames:</strong> {int(report.metrics.get('total_missing_frames', 0))}</p>
        <p><strong>Total Duplicate Timestamps:</strong> {int(report.metrics.get('total_duplicate_frames', 0))}</p>
    </div>
    
    <section>
        <h2>Sensor Streams</h2>
        <table>
            <thead>
                <tr>
                    <th>Sensor</th>
                    <th>Frames</th>
                    <th>Frequency (Hz)</th>
                    <th>Missing</th>
                    <th>Duplicates</th>
                </tr>
            </thead>
            <tbody>
                {streams_table}
            </tbody>
        </table>
    </section>
    
    <section>
        <h2>Synchronization Metrics</h2>
        <table>
            <thead><tr><th>Metric</th><th>Value</th></tr></thead>
            <tbody>
                {metrics_rows}
            </tbody>
        </table>
    </section>
    
    <section>
        <h2>Pairwise Synchronization Results</h2>
        {pair_table}
    </section>
    
    <section>
        <h2>Recommendations</h2>
        <ul>{rec_list}</ul>
    </section>
    
    <section>
        <h2>Compliance Flags</h2>
        <ul>{flag_list}</ul>
    </section>
    
    {param_section}
    {heatmap_section}
</body>
</html>
"""
        with open(output_path, "w", encoding="utf-8") as html_file:
            html_file.write(html)
        return output_path

    def _write_csv_report(self, pair_df: pd.DataFrame, base_name: str) -> Path:
        """Write CSV format report."""
        output_path = self.output_dir / f"{base_name}.csv"
        pair_df.to_csv(output_path, index=False)
        return output_path

    def _dataframe_to_markdown(self, dataframe: pd.DataFrame) -> str:
        """Convert dataframe to markdown table."""
        headers = list(dataframe.columns)
        header_line = "| " + " | ".join(headers) + " |"
        separator = "|" + "|".join([" --- " for _ in headers]) + "|"
        rows = [header_line, separator]
        for _, row in dataframe.iterrows():
            formatted_values: List[str] = []
            for value in row:
                if isinstance(value, float):
                    formatted_values.append(f"{value:.4f}")
                elif isinstance(value, bool):
                    formatted_values.append("✓" if value else "✗")
                else:
                    formatted_values.append(str(value))
            rows.append("| " + " | ".join(formatted_values) + " |")
        return "\n".join(rows)
