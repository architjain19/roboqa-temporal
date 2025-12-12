"""

################################################################

File: roboqa_temporal/detection/detector.py
Created: 2025-11-20
Created by: Archit Jain (architj@uw.edu)
Last Modified: 2025-11-20
Last Modified by: Archit Jain (architj@uw.edu)

#################################################################

Copyright: RoboQA-Temporal Authors
License: MIT License

################################################################

Main anomaly detection module for RoboQA-Temporal. This module
provides the AnomalyDetector class which orchestrates multiple
detection algorithms to identify anomalies in point cloud data
collected in ROS2 bag files. It includes detectors for density
drops, spatial discontinuities, ghost points, and temporal
inconsistencies.

################################################################

"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import numpy as np

from roboqa_temporal.loader.bag_loader import PointCloudFrame
from roboqa_temporal.constants import MIN_SEVERITY, MAX_SEVERITY
from roboqa_temporal.detection.detectors import (
    DensityDropDetector,
    SpatialDiscontinuityDetector,
    GhostPointDetector,
    TemporalConsistencyDetector,
)


@dataclass
class Anomaly:
    """
    Represents a detected anomaly in the point cloud data. This
    includes information about the frame index, timestamp, type
    of anomaly, severity score, and additional metadata.
    """

    frame_index: int
    timestamp: float
    anomaly_type: str
    severity: float  # 0.0 to 1.0
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (MIN_SEVERITY <= self.severity <= MAX_SEVERITY):
            raise ValueError(
                f"severity {self.severity} outside bounds [{MIN_SEVERITY}, {MAX_SEVERITY}]"
            )


@dataclass
class DetectionResult:
    """
    Aggregated results from the anomaly detection process. Includes a list
    of detected anomalies, overall health metrics, and per-frame statistics.
    """

    anomalies: List[Anomaly]
    health_metrics: Dict[str, float]
    frame_statistics: List[Dict[str, Any]]
    detector_results: Dict[str, Any] = field(default_factory=dict)


class AnomalyDetector:
    """
    Main anomaly detection engine.

    Orchestrates multiple detection algorithms:
    - Density drop & occlusion detection
    - Spatial discontinuity analysis
    - Ghost point identification
    - Temporal consistency scoring
    """

    def __init__(
        self,
        enable_density_detection: bool = True,
        enable_spatial_detection: bool = True,
        enable_ghost_detection: bool = True,
        enable_temporal_detection: bool = True,
        density_threshold: float = 0.5,
        spatial_threshold: float = 0.3,
        ghost_threshold: float = 0.7,
        temporal_threshold: float = 0.4,
    ):
        """
        Initialize anomaly detector.

        Args:
            enable_density_detection: Enable density drop detection
            enable_spatial_detection: Enable spatial discontinuity detection
            enable_ghost_detection: Enable ghost point detection
            enable_temporal_detection: Enable temporal consistency detection
            density_threshold: Threshold for density anomaly severity (0-1)
            spatial_threshold: Threshold for spatial anomaly severity (0-1)
            ghost_threshold: Threshold for ghost point severity (0-1)
            temporal_threshold: Threshold for temporal anomaly severity (0-1)
        """
        self.detectors = {}

        if enable_density_detection:
            self.detectors["density"] = DensityDropDetector(threshold=density_threshold)

        if enable_spatial_detection:
            self.detectors["spatial"] = SpatialDiscontinuityDetector(
                threshold=spatial_threshold
            )

        if enable_ghost_detection:
            self.detectors["ghost"] = GhostPointDetector(threshold=ghost_threshold)

        if enable_temporal_detection:
            self.detectors["temporal"] = TemporalConsistencyDetector(
                threshold=temporal_threshold
            )

        if not self.detectors:
            raise ValueError(
                "No detectors enabled. Enable at least one anomaly detector or adjust configuration."
            )

    def detect(self, frames: List[PointCloudFrame]) -> DetectionResult:
        """
        Run anomaly detection on a sequence of frames.

        Args:
            frames: List of point cloud frames

        Returns:
            DetectionResult with anomalies and metrics
        """
        if not frames:
            return DetectionResult(
                anomalies=[],
                health_metrics={},
                frame_statistics=[],
            )

        all_anomalies = []
        detector_results = {}
        frame_stats = []

        # Run each detector
        for detector_name, detector in self.detectors.items():
            try:
                print(f"  Running {detector_name} detector...")
                result = detector.detect(frames)
                detector_results[detector_name] = result

                # Convert detector-specific results to Anomaly objects
                anomalies = self._convert_to_anomalies(
                    result, frames, detector_name
                )
                all_anomalies.extend(anomalies)
                print(f"    Found {len(anomalies)} {detector_name} anomalies")
            except Exception as e:
                print(f"Warning: Detector {detector_name} failed: {e}")

        # Compute frame statistics
        for i, frame in enumerate(frames):
            stats = {
                "frame_index": i,
                "timestamp": frame.timestamp,
                "num_points": frame.num_points,
                "density": frame.num_points,
            }
            frame_stats.append(stats)

        # Compute overall health metrics
        health_metrics = self._compute_health_metrics(
            frames, all_anomalies, detector_results
        )

        return DetectionResult(
            anomalies=all_anomalies,
            health_metrics=health_metrics,
            frame_statistics=frame_stats,
            detector_results=detector_results,
        )

    def _convert_to_anomalies(
        self, detector_result: Dict[str, Any], frames: List[PointCloudFrame], detector_type: str
    ) -> List[Anomaly]:
        """Convert detector-specific results to Anomaly objects."""
        anomalies = []

        if "anomalies" in detector_result:
            for anomaly_data in detector_result["anomalies"]:
                frame_idx = anomaly_data.get("frame_index", 0)
                if 0 <= frame_idx < len(frames):
                    anomaly = Anomaly(
                        frame_index=frame_idx,
                        timestamp=frames[frame_idx].timestamp,
                        anomaly_type=detector_type,
                        severity=anomaly_data.get("severity", 0.5),
                        description=anomaly_data.get("description", ""),
                        metadata=anomaly_data.get("metadata", {}),
                    )
                    anomalies.append(anomaly)

        return anomalies

    def _compute_health_metrics(
        self,
        frames: List[PointCloudFrame],
        anomalies: List[Anomaly],
        detector_results: Dict[str, Any],
    ) -> Dict[str, float]:
        """Compute overall health metrics for the dataset."""
        if not frames:
            return {}

        metrics = {}

        # Basic statistics
        num_points = [f.num_points for f in frames]
        metrics["avg_points_per_frame"] = float(np.mean(num_points))
        metrics["min_points"] = float(np.min(num_points))
        metrics["max_points"] = float(np.max(num_points))
        metrics["points_std"] = float(np.std(num_points))

        # Anomaly statistics
        metrics["total_anomalies"] = float(len(anomalies))
        metrics["anomaly_rate"] = len(anomalies) / len(frames) if frames else 0.0
        metrics["avg_severity"] = (
            float(np.mean([a.severity for a in anomalies])) if anomalies else 0.0
        )

        # Per-detector metrics
        for detector_name, result in detector_results.items():
            if "metrics" in result:
                for key, value in result["metrics"].items():
                    metrics[f"{detector_name}_{key}"] = float(value)

        # Overall health score (0-1, higher is better)
        health_score = 1.0
        if anomalies:
            # Penalize based on anomaly rate and severity
            anomaly_penalty = min(metrics["anomaly_rate"] * 0.5, 0.5)
            severity_penalty = min(metrics["avg_severity"] * 0.3, 0.3)
            health_score = max(0.0, 1.0 - anomaly_penalty - severity_penalty)

        metrics["overall_health_score"] = health_score

        return metrics
