"""

################################################################

File: roboqa_temporal/detection/detectors.py
Created: 2025-11-20
Created by: Archit Jain (architj@uw.edu)
Last Modified: 2025-11-20
Last Modified by: Archit Jain (architj@uw.edu)

#################################################################

Copyright: RoboQA-Temporal Authors
License: MIT License

################################################################

Implementations of various anomaly detection algorithms for
RoboQA-Temporal. This includes detectors for density drops,
spatial discontinuities, ghost points, and temporal consistency
analysis.

################################################################

"""

from typing import List, Dict, Any, Tuple
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from scipy.stats import zscore
from sklearn.covariance import EllipticEnvelope

from roboqa_temporal.loader.bag_loader import PointCloudFrame


class DensityDropDetector:
    """
    Detects density drops and occlusions in point cloud sequences.

    Identifies frames with sudden drops in point density that may indicate:
    - Sensor occlusions
    - Hardware faults
    - Environmental changes
    """

    def __init__(self, threshold: float = 0.5, window_size: int = 5):
        """
        Initialize density drop detector.

        Args:
            threshold: Severity threshold (0-1) for flagging anomalies
            window_size: Window size for moving average
        """
        self.threshold = threshold
        self.window_size = window_size

    def detect(self, frames: List[PointCloudFrame]) -> Dict[str, Any]:
        """
        Detect density drops in frame sequence.

        Args:
            frames: List of point cloud frames

        Returns:
            Dictionary with anomalies and metrics
        """
        if len(frames) < 2:
            return {"anomalies": [], "metrics": {}}

        densities = np.array([f.num_points for f in frames])

        # Compute moving average
        if len(densities) >= self.window_size:
            kernel = np.ones(self.window_size) / self.window_size
            moving_avg = np.convolve(densities, kernel, mode="same")
        else:
            moving_avg = np.full_like(densities, np.mean(densities))

        # Compute relative density drops
        relative_drops = (moving_avg - densities) / (moving_avg + 1e-6)
        relative_drops = np.clip(relative_drops, 0, 1)

        # Detect anomalies using z-score
        z_scores = np.abs(zscore(densities))
        anomaly_mask = (relative_drops > self.threshold) | (z_scores > 2.0)

        anomalies = []
        for i, is_anomaly in enumerate(anomaly_mask):
            if is_anomaly:
                severity = min(1.0, relative_drops[i] * 1.5)
                anomalies.append(
                    {
                        "frame_index": i,
                        "severity": float(severity),
                        "description": f"Density drop detected: {densities[i]:.0f} points (avg: {moving_avg[i]:.1f})",
                        "metadata": {
                            "density": int(densities[i]),
                            "expected_density": float(moving_avg[i]),
                            "drop_ratio": float(relative_drops[i]),
                        },
                    }
                )

        metrics = {
            "avg_density": float(np.mean(densities)),
            "density_std": float(np.std(densities)),
            "density_cv": float(np.std(densities) / (np.mean(densities) + 1e-6)),
            "num_density_drops": len(anomalies),
        }

        return {"anomalies": anomalies, "metrics": metrics}


class SpatialDiscontinuityDetector:
    """
    Detects spatial discontinuities and irregular motion.

    Analyzes geometric changes and frame-to-frame transformations
    to spot irregular motion or environmental shifts.
    """

    def __init__(self, threshold: float = 0.3):
        """
        Initialize spatial discontinuity detector.

        Args:
            threshold: Severity threshold for flagging anomalies
        """
        self.threshold = threshold

    def detect(self, frames: List[PointCloudFrame]) -> Dict[str, Any]:
        """
        Detect spatial discontinuities.

        Args:
            frames: List of point cloud frames

        Returns:
            Dictionary with anomalies and metrics
        """
        if len(frames) < 2:
            return {"anomalies": [], "metrics": {}}

        discontinuities = []
        transformations = []

        for i in range(1, len(frames)):
            prev_frame = frames[i - 1]
            curr_frame = frames[i]

            if prev_frame.num_points == 0 or curr_frame.num_points == 0:
                continue

            # Compute centroids
            prev_centroid = np.mean(prev_frame.points, axis=0)
            curr_centroid = np.mean(curr_frame.points, axis=0)

            # Compute translation
            translation = np.linalg.norm(curr_centroid - prev_centroid)
            transformations.append(translation)

            # Compute point cloud spread (standard deviation)
            prev_spread = np.std(prev_frame.points, axis=0)
            curr_spread = np.std(curr_frame.points, axis=0)
            spread_change = np.linalg.norm(curr_spread - prev_spread)

            # Compute overlap using nearest neighbor distances
            if len(prev_frame.points) > 0 and len(curr_frame.points) > 0:
                distances = cdist(
                    prev_frame.points[: min(100, len(prev_frame.points))],
                    curr_frame.points[: min(100, len(curr_frame.points))],
                )
                min_distances = np.min(distances, axis=1)
                avg_min_distance = np.mean(min_distances)
            else:
                avg_min_distance = float("inf")

            # Normalize discontinuity score
            translation_norm = translation / (np.linalg.norm(prev_spread) + 1e-6)
            spread_norm = spread_change / (np.linalg.norm(prev_spread) + 1e-6)
            distance_norm = avg_min_distance / (np.linalg.norm(prev_spread) + 1e-6)

            discontinuity_score = (
                translation_norm * 0.4 + spread_norm * 0.3 + distance_norm * 0.3
            )

            if discontinuity_score > self.threshold:
                severity = min(1.0, discontinuity_score)
                discontinuities.append(
                    {
                        "frame_index": i,
                        "severity": float(severity),
                        "description": f"Spatial discontinuity: translation={translation:.3f}m, spread_change={spread_change:.3f}",
                        "metadata": {
                            "translation": float(translation),
                            "spread_change": float(spread_change),
                            "avg_min_distance": float(avg_min_distance),
                            "discontinuity_score": float(discontinuity_score),
                        },
                    }
                )

        metrics = {
            "avg_translation": float(np.mean(transformations)) if transformations else 0.0,
            "max_translation": float(np.max(transformations)) if transformations else 0.0,
            "num_discontinuities": len(discontinuities),
        }

        return {"anomalies": discontinuities, "metrics": metrics}


class GhostPointDetector:
    """
    Identifies ghost points likely due to reflections, multi-path returns,
    or hardware artifacts using statistical heuristics.
    """

    def __init__(self, threshold: float = 0.7):
        """
        Initialize ghost point detector.

        Args:
            threshold: Severity threshold for flagging anomalies
        """
        self.threshold = threshold

    def detect(self, frames: List[PointCloudFrame]) -> Dict[str, Any]:
        """
        Detect ghost points in frames.

        Args:
            frames: List of point cloud frames

        Returns:
            Dictionary with anomalies and metrics
        """
        anomalies = []
        ghost_ratios = []

        for i, frame in enumerate(frames):
            if frame.num_points < 10:
                continue

            points = frame.points

            # Method 1: Statistical outlier detection (elliptic envelope)
            try:
                if len(points) > 20:
                    # Sample points for efficiency
                    sample_size = min(1000, len(points))
                    sample_indices = np.random.choice(
                        len(points), sample_size, replace=False
                    )
                    sample_points = points[sample_indices]

                    # Fit elliptic envelope
                    envelope = EllipticEnvelope(contamination=0.1, random_state=42)
                    envelope.fit(sample_points)
                    labels = envelope.predict(points)

                    # Count outliers
                    num_outliers = np.sum(labels == -1)
                    ghost_ratio = num_outliers / len(points)
                    ghost_ratios.append(ghost_ratio)

                    if ghost_ratio > self.threshold:
                        severity = min(1.0, ghost_ratio)
                        anomalies.append(
                            {
                                "frame_index": i,
                                "severity": float(severity),
                                "description": f"Ghost points detected: {num_outliers}/{len(points)} points ({ghost_ratio*100:.1f}%)",
                                "metadata": {
                                    "num_ghost_points": int(num_outliers),
                                    "total_points": int(len(points)),
                                    "ghost_ratio": float(ghost_ratio),
                                },
                            }
                        )
            except Exception:
                # If detection fails, skip this frame
                continue

            # Method 2: Distance-based heuristic (points far from neighbors)
            try:
                if len(points) > 50:
                    # Using KD-tree for computing memory-efficient k-nearest neighbor search
                    k = min(10, len(points) - 1)
                    tree = KDTree(points)
                    distances, _ = tree.query(points, k=k + 1)
                    k_distances = distances[:, 1:]
                    mean_k_distances = np.mean(k_distances, axis=1)

                    # Points with unusually large neighbor distances
                    threshold_dist = np.percentile(mean_k_distances, 95)
                    isolated_points = np.sum(mean_k_distances > threshold_dist)
                    isolation_ratio = isolated_points / len(points)

                    if isolation_ratio > 0.2:  # More than 20% isolated points
                        severity = min(1.0, isolation_ratio * 2.0)
                        anomalies.append(
                            {
                                "frame_index": i,
                                "severity": float(severity),
                                "description": f"Isolated/ghost points: {isolated_points}/{len(points)} points",
                                "metadata": {
                                    "num_isolated_points": int(isolated_points),
                                    "isolation_ratio": float(isolation_ratio),
                                },
                            }
                        )
            except Exception:
                continue

        metrics = {
            "avg_ghost_ratio": float(np.mean(ghost_ratios)) if ghost_ratios else 0.0,
            "max_ghost_ratio": float(np.max(ghost_ratios)) if ghost_ratios else 0.0,
            "num_ghost_frames": len(anomalies),
        }

        return {"anomalies": anomalies, "metrics": metrics}


class TemporalConsistencyDetector:
    """
    Calculates temporal consistency and smoothness metrics.

    Quantifies naturalness and consistency in spatio-temporal evolution
    of point cloud sequences.
    """

    def __init__(self, threshold: float = 0.4):
        """
        Initialize temporal consistency detector.

        Args:
            threshold: Severity threshold for flagging anomalies
        """
        self.threshold = threshold

    def detect(self, frames: List[PointCloudFrame]) -> Dict[str, Any]:
        """
        Detect temporal inconsistencies.

        Args:
            frames: List of point cloud frames

        Returns:
            Dictionary with anomalies and metrics
        """
        if len(frames) < 3:
            return {"anomalies": [], "metrics": {}}

        inconsistencies = []
        smoothness_scores = []

        # Compute frame-to-frame differences
        for i in range(1, len(frames)):
            prev_frame = frames[i - 1]
            curr_frame = frames[i]

            if prev_frame.num_points == 0 or curr_frame.num_points == 0:
                continue

            # Compute temporal smoothness metrics
            # 1. Point count change
            count_change = abs(curr_frame.num_points - prev_frame.num_points)
            count_change_ratio = count_change / (prev_frame.num_points + 1e-6)

            # 2. Centroid movement
            prev_centroid = np.mean(prev_frame.points, axis=0)
            curr_centroid = np.mean(curr_frame.points, axis=0)
            centroid_velocity = np.linalg.norm(curr_centroid - prev_centroid)

            # 3. Bounding box change
            prev_bbox_size = np.max(prev_frame.points, axis=0) - np.min(
                prev_frame.points, axis=0
            )
            curr_bbox_size = np.max(curr_frame.points, axis=0) - np.min(
                curr_frame.points, axis=0
            )
            bbox_change = np.linalg.norm(curr_bbox_size - prev_bbox_size)

            # 4. Compute acceleration (second derivative)
            if i > 1:
                prev_prev_frame = frames[i - 2]
                prev_prev_centroid = np.mean(prev_prev_frame.points, axis=0)
                prev_velocity = np.linalg.norm(prev_centroid - prev_prev_centroid)
                curr_velocity = centroid_velocity
                acceleration = abs(curr_velocity - prev_velocity)
            else:
                acceleration = 0.0

            # Normalize and combine metrics
            count_score = min(1.0, count_change_ratio)
            velocity_score = min(1.0, centroid_velocity / 1.0)  # Normalize by 1m
            bbox_score = min(1.0, bbox_change / (np.linalg.norm(prev_bbox_size) + 1e-6))
            accel_score = min(1.0, acceleration / 0.5)  # Normalize by 0.5 m/s^2

            # Inconsistency score (higher = more inconsistent)
            inconsistency_score = (
                count_score * 0.3
                + velocity_score * 0.3
                + bbox_score * 0.2
                + accel_score * 0.2
            )

            smoothness_scores.append(1.0 - inconsistency_score)

            if inconsistency_score > self.threshold:
                severity = min(1.0, inconsistency_score)
                inconsistencies.append(
                    {
                        "frame_index": i,
                        "severity": float(severity),
                        "description": f"Temporal inconsistency: count_change={count_change_ratio:.2f}, velocity={centroid_velocity:.3f}m/s",
                        "metadata": {
                            "count_change_ratio": float(count_change_ratio),
                            "centroid_velocity": float(centroid_velocity),
                            "bbox_change": float(bbox_change),
                            "acceleration": float(acceleration),
                            "inconsistency_score": float(inconsistency_score),
                        },
                    }
                )

        metrics = {
            "avg_smoothness": float(np.mean(smoothness_scores)) if smoothness_scores else 1.0,
            "min_smoothness": float(np.min(smoothness_scores)) if smoothness_scores else 1.0,
            "num_inconsistencies": len(inconsistencies),
        }

        return {"anomalies": inconsistencies, "metrics": metrics}
