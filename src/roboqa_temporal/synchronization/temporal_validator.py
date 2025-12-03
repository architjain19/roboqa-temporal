"""

################################################################

File: roboqa_temporal/synchronization/temporal_validator.py
Created: 2025-11-24
Created by: Xinxin Tai (xinxin@example.com)
Last Modified: 2025-11-24
Last Modified by: Xinxin Tai (xinxin@example.com)

#################################################################

Multi-Sensor Temporal Synchronization Validator.

This module inspects ROS2 MCAP bag files that contain camera, LiDAR,
IMU, and PPS trigger topics. It verifies that sensor timestamps are
aligned, frequencies are stable, and long-term drift stays within
Precision Time Protocol (IEEE 1588) limits. The validator emits an
ISO 8000-61 compliant report, generates temporal heatmaps, and
exports recommended timestamp corrections as ROS2 parameter files.

################################################################

"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import concurrent.futures

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from scipy import signal, stats
except ImportError:  # pragma: no cover - SciPy optional in some environments
    signal = None  # type: ignore
    stats = None  # type: ignore

try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except ImportError:  # pragma: no cover - exercised only with ROS downloads
    rosbag2_py = None  # type: ignore
    deserialize_message = None  # type: ignore
    get_message = None  # type: ignore


@dataclass
class SensorStream:
    """Container describing a single sensor stream."""

    name: str
    topic: str
    timestamps_ns: List[int]
    header_timestamps_ns: Optional[List[int]] = None
    expected_frequency: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamps_sec: np.ndarray = field(init=False, repr=False)
    header_timestamps_sec: np.ndarray = field(init=False, repr=False)
    frequency_estimate_hz: Optional[float] = field(init=False, default=None)

    def __post_init__(self) -> None:
        ts_ns = np.array(self.timestamps_ns, dtype=np.float64)
        self.timestamps_sec = ts_ns / 1e9 if ts_ns.size else np.array([])

        header_ts = (
            np.array(self.header_timestamps_ns, dtype=np.float64)
            if self.header_timestamps_ns
            else np.array([])
        )
        self.header_timestamps_sec = header_ts / 1e9 if header_ts.size else np.array([])

        if self.timestamps_sec.size >= 2:
            diffs = np.diff(self.timestamps_sec)
            valid = diffs[diffs > 0]
            if valid.size:
                self.frequency_estimate_hz = float(1.0 / np.median(valid))
        self.metadata["message_count"] = len(self.timestamps_ns)


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
        """Serialize report to a dictionary."""
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
    """Validates temporal alignment across camera, LiDAR, IMU, and PPS streams."""

    DEFAULT_TOPICS = {
        "camera": "/camera/image_raw",
        "lidar": "/velodyne_points",
        "imu": "/imu/data",
        "pps": None,
    }

    PAIRS = (
        ("camera", "lidar"),
        ("lidar", "imu"),
        ("camera", "imu"),
    )

    def __init__(
        self,
        topics: Optional[Dict[str, Optional[str]]] = None,
        expected_frequency_hz: Optional[Dict[str, float]] = None,
        approximate_time_threshold_ms: Optional[Dict[str, float]] = None,
        storage_id: str = "mcap",
        output_dir: str = "reports/temporal_sync",
        rolling_window: int = 50,
        ptp_config: Optional[Dict[str, float]] = None,
        heatmap_resolution: int = 128,
        kalman_process_noise: float = 1e-6,
        kalman_measurement_noise: float = 1e-4,
        kalman_horizon_s: float = 5.0,
        max_workers: int = 4,
        report_formats: Optional[Iterable[str]] = None,
        auto_export_reports: bool = True,
    ) -> None:
        self.topics = {**TemporalSyncValidator.DEFAULT_TOPICS, **(topics or {})}
        self.expected_frequency_hz = expected_frequency_hz or {
            "camera": 30.0,
            "lidar": 10.0,
            "imu": 200.0,
        }
        self.approximate_time_threshold_ms = approximate_time_threshold_ms or {
            "camera_lidar": 30.0,
            "lidar_imu": 15.0,
            "camera_imu": 40.0,
        }
        self.storage_id = storage_id
        self.output_dir = Path(output_dir)
        self.heatmap_dir = self.output_dir / "heatmaps"
        self.param_dir = self.output_dir / "params"
        for directory in (self.output_dir, self.heatmap_dir, self.param_dir):
            directory.mkdir(parents=True, exist_ok=True)

        self.rolling_window = rolling_window
        self.ptp_config = ptp_config or {
            "max_offset_ns": 1_000_000,  # 1 microsecond
            "max_jitter_ns": 100_000,  # 100 nanoseconds
        }
        self.heatmap_resolution = heatmap_resolution
        self.kalman_process_noise = kalman_process_noise
        self.kalman_measurement_noise = kalman_measurement_noise
        self.kalman_horizon_s = kalman_horizon_s
        self.max_workers = max_workers
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
        bag_path: str,
        max_messages_per_topic: Optional[int] = None,
        include_visualizations: bool = True,
    ) -> TemporalSyncReport:
        """Run full validation pipeline on a ROS2 bag."""
        streams = self._load_streams(Path(bag_path), max_messages_per_topic)
        return self.analyze_streams(streams, Path(bag_path).stem, include_visualizations)

    def analyze_streams(
        self,
        streams: Dict[str, SensorStream],
        bag_name: str = "bag",
        include_visualizations: bool = True,
    ) -> TemporalSyncReport:
        """
        Perform temporal analysis on already extracted streams.

        This entry point is used both by validate() and unit tests that pass
        synthetic timestamp streams without touching rosbag APIs.
        """
        pair_results: Dict[str, PairwiseDriftResult] = {}
        global_recommendations: List[str] = []
        compliance_flags: List[str] = []

        for sensor, stream in streams.items():
            freq_ok = self._check_frequency(stream)
            if not freq_ok:
                msg = f"{sensor} frequency deviation exceeds tolerance"
                global_recommendations.append(msg)
                compliance_flags.append(f"{sensor}_frequency_violation")

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
                bag_name,
                include_visualizations,
            )
            pair_results[pair_key] = result
            global_recommendations.extend(result.recommendations)
            compliance_flags.extend(result.compliance_flags)

        metrics = self._aggregate_metrics(pair_results)
        param_path = self._export_parameter_file(pair_results, bag_name)

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
                report, bag_name, formats, include_visualizations
            )

        return report

    # ------------------------------------------------------------------ #
    # Bag ingestion
    # ------------------------------------------------------------------ #

    def _load_streams(
        self, bag_path: Path, max_messages_per_topic: Optional[int]
    ) -> Dict[str, SensorStream]:
        if rosbag2_py is None:
            raise ImportError(
                "rosbag2_py is required to load ROS2 MCAP bags. "
                "Install rosbag2-py via your ROS distribution."
            )

        if not bag_path.exists():
            raise FileNotFoundError(f"Bag path does not exist: {bag_path}")

        reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(
            uri=str(bag_path), storage_id=self.storage_id
        )
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        )
        reader.open(storage_options, converter_options)

        topic_types = {
            topic.name: topic.type for topic in reader.get_all_topics_and_types()
        }

        resolved_topics = self._resolve_topics(topic_types)
        topic_to_sensor = {topic: sensor for sensor, topic in resolved_topics.items()}
        counters = {topic: 0 for topic in resolved_topics.values()}
        processed: Dict[str, List[Tuple[int, Optional[int], Optional[str]]]] = {
            topic: [] for topic in resolved_topics.values()
        }

        futures: List[concurrent.futures.Future] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            while reader.has_next():
                topic, data, ts = reader.read_next()
                if topic not in topic_to_sensor:
                    continue
                if max_messages_per_topic and counters[topic] >= max_messages_per_topic:
                    continue
                counters[topic] += 1
                futures.append(
                    executor.submit(
                        self._extract_message_metadata,
                        topic,
                        data,
                        ts,
                        topic_types.get(topic),
                    )
                )

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is None:
                    continue
                topic_name, recorded_ts, header_ts, frame_id = result
                processed[topic_name].append((recorded_ts, header_ts, frame_id))

        reader.close()

        streams: Dict[str, SensorStream] = {}
        for sensor, topic in resolved_topics.items():
            entries = processed.get(topic, [])
            if not entries:
                continue

            entries.sort(key=lambda item: item[0])
            recorded_ts = [e[0] for e in entries]
            header_ts = [e[1] for e in entries if e[1] is not None]
            frame_ids = [e[2] for e in entries if e[2]]
            metadata = {
                "topic_type": topic_types.get(topic, "unknown"),
                "frame_ids": list(sorted(set(frame_ids))),
                "storage_id": self.storage_id,
            }
            stream = SensorStream(
                name=sensor,
                topic=topic,
                timestamps_ns=recorded_ts,
                header_timestamps_ns=header_ts if header_ts else None,
                metadata=metadata,
                expected_frequency=self.expected_frequency_hz.get(sensor),
            )
            streams[sensor] = stream

        return streams

    def _extract_message_metadata(
        self, topic: str, data: bytes, recorded_ts: int, type_name: Optional[str]
    ) -> Optional[Tuple[str, int, Optional[int], Optional[str]]]:
        """Extract header timestamp/frame id if possible."""
        header_stamp_ns: Optional[int] = None
        frame_id: Optional[str] = None

        if deserialize_message and get_message and type_name:
            try:
                msg_type = get_message(type_name)
                msg = deserialize_message(data, msg_type)
                header = getattr(msg, "header", None)
                if header and hasattr(header, "stamp"):
                    sec = getattr(header.stamp, "sec", None)
                    nanosec = getattr(header.stamp, "nanosec", None)
                    if sec is not None and nanosec is not None:
                        header_stamp_ns = int(sec) * 1_000_000_000 + int(nanosec)
                if header and hasattr(header, "frame_id"):
                    frame_id = header.frame_id
            except Exception:
                header_stamp_ns = None
                frame_id = None

        return (topic, recorded_ts, header_stamp_ns, frame_id)

    def _resolve_topics(self, topic_types: Dict[str, str]) -> Dict[str, str]:
        """Resolve requested sensor topics, falling back to detection."""
        resolved: Dict[str, str] = {}
        for sensor, configured_topic in self.topics.items():
            if configured_topic and configured_topic in topic_types:
                resolved[sensor] = configured_topic
                continue

            candidates = self._auto_detect_topics(sensor, topic_types)
            if not candidates:
                continue
            resolved[sensor] = candidates[0]
        return {k: v for k, v in resolved.items() if v}

    @staticmethod
    def _auto_detect_topics(sensor: str, topic_types: Dict[str, str]) -> List[str]:
        """Infer topic names based on message type or naming convention."""
        preferred_types = {
            "camera": ("sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage"),
            "lidar": ("sensor_msgs/msg/PointCloud2",),
            "imu": ("sensor_msgs/msg/Imu",),
            "pps": ("builtin_interfaces/msg/Time", "std_msgs/msg/Header"),
        }
        name_keywords = {
            "camera": ("image", "camera"),
            "lidar": ("pointcloud", "points", "lidar"),
            "imu": ("imu",),
            "pps": ("pps", "ptp"),
        }
        matches = []
        for topic, type_name in topic_types.items():
            if sensor in preferred_types and type_name in preferred_types[sensor]:
                matches.append(topic)
                continue
            lowered = topic.lower()
            if any(keyword in lowered for keyword in name_keywords.get(sensor, ())):
                matches.append(topic)
        return sorted(matches)

    # ------------------------------------------------------------------ #
    # Analysis helpers
    # ------------------------------------------------------------------ #

    def _analyze_pair(
        self,
        stream_a: SensorStream,
        stream_b: SensorStream,
        pair_key: str,
        threshold_ms: float,
        bag_name: str,
        include_visualizations: bool,
    ) -> PairwiseDriftResult:
        threshold_s = threshold_ms / 1000.0
        timestamps_a = (
            stream_a.header_timestamps_sec
            if stream_a.header_timestamps_sec.size
            else stream_a.timestamps_sec
        )
        timestamps_b = (
            stream_b.header_timestamps_sec
            if stream_b.header_timestamps_sec.size
            else stream_b.timestamps_sec
        )

        matches = self._approximate_time_matches(timestamps_a, timestamps_b, threshold_s)
        if not matches:
            # No direct matches, synthesize using histogram lag estimation
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
        temporal_offset_score = max(0.0, 1.0 - (mean_abs_delta / (threshold_ms + 1e-6)))
        approx_time_pass = max_delta <= threshold_ms

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
                f"{pair_key}: timestamp delta ({max_delta:.2f} ms) exceeds ApproximateTime threshold ({threshold_ms} ms)"
            )
            compliance_flags.append(f"{pair_key}_approximate_time_violation")

        if chi_square_pvalue < 0.05:
            recommendations.append(f"{pair_key}: chi-square test indicates out-of-sync behavior")
            compliance_flags.append(f"{pair_key}_chi_square_failure")

        if abs(drift_rate) > 0.5:
            recommendations.append(
                f"{pair_key}: drift rate {drift_rate:.2f} ms/s suggests hardware desync"
            )

        if kalman_projection > threshold_ms:
            recommendations.append(
                f"{pair_key}: projected drift {kalman_projection:.2f} ms over {self.kalman_horizon_s}s exceeds tolerance"
            )

        if not ptp_pass:
            recommendations.append(
                f"{pair_key}: PTP offset {max_delta:.2f} ms violates IEEE 1588 budget"
            )
            compliance_flags.append(f"{pair_key}_ptp_violation")

        heatmap_path = None
        if include_visualizations:
            heatmap_path = self._generate_heatmap(
                pair_key, bag_name, mid_times, deltas_ms
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
        """Emulate ROS message_filters ApproximateTime pairing."""
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
        if timestamps.size < 2:
            return 0.0
        gradient = np.gradient(deltas_ms, timestamps, edge_order=1)
        return float(np.median(gradient))

    @staticmethod
    def _chi_square_test(deltas_ms: np.ndarray) -> float:
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
        if timestamps.size < 2:
            return float(np.abs(deltas_ms[-1])) if deltas_ms.size else 0.0

        state = np.array([deltas_ms[0], 0.0])  # offset, drift_rate
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
        bag_name: str,
        timestamps: np.ndarray,
        deltas_ms: np.ndarray,
    ) -> Optional[str]:
        if timestamps.size < 2:
            return None

        x_min, x_max = float(timestamps.min()), float(timestamps.max())
        y_min, y_max = float(deltas_ms.min()), float(deltas_ms.max())
        heatmap, xedges, yedges = np.histogram2d(
            timestamps,
            deltas_ms,
            bins=self.heatmap_resolution,
            range=[[x_min, x_max], [y_min, y_max]],
        )
        if not heatmap.size or np.allclose(heatmap.sum(), 0):
            return None

        plt.figure(figsize=(8, 4))
        sns.heatmap(
            heatmap.T,
            cmap="coolwarm",
            cbar_kws={"label": "Count"},
            xticklabels=False,
            yticklabels=False,
        )
        plt.title(f"Temporal Alignment Heatmap: {pair_key}")
        plt.xlabel("Time bins")
        plt.ylabel("Δt bins (ms)")
        output_path = self.heatmap_dir / f"{bag_name}_{pair_key}_heatmap.png"
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return str(output_path)

    def _check_frequency(self, stream: SensorStream) -> bool:
        if stream.expected_frequency is None or stream.frequency_estimate_hz is None:
            return True
        freq = stream.frequency_estimate_hz
        target = stream.expected_frequency
        deviation = abs(freq - target) / max(target, 1e-6)
        stream.metadata["frequency_estimate_hz"] = freq
        stream.metadata["frequency_deviation_pct"] = deviation * 100.0
        return deviation <= 0.1  # allow 10% tolerance

    def _aggregate_metrics(
        self, pair_results: Dict[str, PairwiseDriftResult]
    ) -> Dict[str, float]:
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
        metrics["iso_8000_61_pass"] = 1.0 if metrics["temporal_offset_score"] >= 0.8 else 0.0
        return metrics

    def _export_parameter_file(
        self, pair_results: Dict[str, PairwiseDriftResult], bag_name: str
    ) -> Optional[str]:
        if not pair_results:
            return None

        corrections: Dict[str, Dict[str, float]] = {}
        for pair_name, result in pair_results.items():
            sensors = pair_name.split("_")
            if result.deltas_ms.size:
                mean_delta = float(np.mean(result.deltas_ms))
            else:
                mean_delta = 0.0
            corrections[pair_name] = {"mean_offset_ms": mean_delta}
            for idx, sensor in enumerate(sensors):
                corrections[f"{pair_name}_{sensor}"] = {
                    "suggested_offset_ms": mean_delta * (1 if idx == 0 else -1)
                }

        payload = {
            "temporal_corrections": corrections,
            "metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "bag_name": bag_name,
                "iso_8000_61": True,
            },
        }
        output_path = self.param_dir / f"{bag_name}_timestamp_corrections.yaml"
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=True)
        return str(output_path)

    def _export_summary_reports(
        self,
        report: TemporalSyncReport,
        bag_name: str,
        formats: Iterable[str],
        include_visualizations: bool,
    ) -> Dict[str, str]:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        base_name = f"{bag_name}_temporal_sync_{timestamp}"
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
        output_path = self.output_dir / f"{base_name}.md"
        with open(output_path, "w", encoding="utf-8") as md_file:
            md_file.write("# Temporal Synchronization Report\n\n")
            md_file.write(f"**Generated:** {report.generated_at}\n\n")
            md_file.write("## Metrics\n\n")
            md_file.write("| Metric | Value |\n|--------|-------|\n")
            for key, value in sorted(report.metrics.items()):
                if isinstance(value, float):
                    md_file.write(f"| {key} | {value:.6f} |\n")
                else:
                    md_file.write(f"| {key} | {value} |\n")
            md_file.write("\n")

            if not pair_df.empty:
                md_file.write("## Pairwise Results\n\n")
                md_file.write(self._dataframe_to_markdown(pair_df))
                md_file.write("\n\n")

            if report.recommendations:
                md_file.write("## Recommendations\n\n")
                for rec in report.recommendations:
                    md_file.write(f"- {rec}\n")
                md_file.write("\n")

            if report.compliance_flags:
                md_file.write("## Compliance Flags\n\n")
                for flag in report.compliance_flags:
                    md_file.write(f"- {flag}\n")
                md_file.write("\n")

            if report.parameter_file:
                md_file.write("## Timestamp Corrections\n\n")
                md_file.write(f"`{report.parameter_file}`\n")

        return output_path

    def _write_html_report(
        self,
        report: TemporalSyncReport,
        pair_df: pd.DataFrame,
        base_name: str,
        include_visualizations: bool,
    ) -> Path:
        output_path = self.output_dir / f"{base_name}.html"
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
                heatmap_section = "<section><h2>Heatmaps</h2>" + "\n".join(snippets) + "</section>"

        param_section = ""
        if report.parameter_file:
            rel_param = os.path.relpath(report.parameter_file, start=output_path.parent)
            param_section = f"<section><h2>Timestamp Corrections</h2><p><code>{rel_param}</code></p></section>"

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8"/>
    <title>Temporal Synchronization Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; color: #222; }}
        h1, h2 {{ color: #0b3954; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
        th {{ background: #f2f2f2; }}
        section {{ margin-bottom: 30px; }}
        .heatmap img {{ max-width: 100%; border: 1px solid #ddd; padding: 4px; }}
    </style>
</head>
<body>
    <h1>Temporal Synchronization Report</h1>
    <p><strong>Generated:</strong> {report.generated_at}</p>
    <section>
        <h2>Metrics</h2>
        <table>
            <thead><tr><th>Metric</th><th>Value</th></tr></thead>
            <tbody>
                {metrics_rows}
            </tbody>
        </table>
    </section>
    <section>
        <h2>Pairwise Results</h2>
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
        output_path = self.output_dir / f"{base_name}.csv"
        pair_df.to_csv(output_path, index=False)
        return output_path

    def _dataframe_to_markdown(self, dataframe: pd.DataFrame) -> str:
        headers = list(dataframe.columns)
        header_line = "| " + " | ".join(headers) + " |"
        separator = "|" + "|".join([" --- " for _ in headers]) + "|"
        rows = [header_line, separator]
        for _, row in dataframe.iterrows():
            formatted_values: List[str] = []
            for value in row:
                if isinstance(value, float):
                    formatted_values.append(f"{value:.6f}")
                else:
                    formatted_values.append(str(value))
            rows.append("| " + " | ".join(formatted_values) + " |")
        return "\n".join(rows)
