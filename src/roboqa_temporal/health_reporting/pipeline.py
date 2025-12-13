"""
################################################################

File: roboqa_temporal/health_reporting/pipeline.py
Created: 2025-12-08
Created by: Sayali Nehul
Last Modified: 2025-12-08
Last Modified by: Sayali Nehul

#################################################################

Copyright: RoboQA-Temporal Authors
License: MIT License

################################################################

Dataset Health Scoring & Quality Assessment Pipeline.

Scans KITTI-style sequences, loads per-sensor timestamps (camera,
LiDAR, OXTS), and computes dataset quality metrics:

- temporal_score: Regularity of frame spacing
- anomaly_score: Fraction of outlier inter-frame gaps
- completeness_metrics: Message availability and dropout rates
- health_tier: Overall quality classification (excellent/good/fair/poor)

################################################################

"""

from __future__ import annotations

import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


def to_datetime_ns(arr: np.ndarray) -> np.ndarray:
    """
    Ensure array is numpy datetime64[ns].
    """
    return arr.astype("datetime64[ns]")

def inter_frame_diffs_seconds(ts_np: np.ndarray) -> np.ndarray:
    """
    Return inter-frame time differences in seconds.
    """
    return np.diff(ts_np.astype("int64") / 1e9)

def compute_temporal_score(ts_np: np.ndarray) -> float:
    """
    Timeliness sub-metric: regularity of frame spacing.
    Higher = more regular.
    
    Args:
        ts_np: numpy array of timestamps (datetime64[ns])
        
    Returns:
        Score between 0.0 and 1.0
    """
    if ts_np.size < 2:
        return 0.0
    diffs = inter_frame_diffs_seconds(ts_np)
    mean = np.mean(diffs)
    std = np.std(diffs)
    score = np.exp(-std / (mean + 1e-6))
    return float(np.clip(score, 0.0, 1.0))


def compute_anomaly_score(ts_np: np.ndarray) -> float:
    """
    Timeliness sub-metric: fraction of inter-frame gaps that are strong outliers (|z| > 3).
    Lower anomaly count -> higher score.
    
    Args:
        ts_np: numpy array of timestamps (datetime64[ns])
        
    Returns:
        Score between 0.0 and 1.0
    """
    if ts_np.size < 2:
        return 0.0
    diffs = inter_frame_diffs_seconds(ts_np)
    mean = diffs.mean()
    std = diffs.std() + 1e-6
    z = (diffs - mean) / std
    anomalies = np.sum(np.abs(z) > 3.0)
    score = 1.0 - anomalies / len(diffs)
    return float(np.clip(score, 0.0, 1.0))


def compute_lidar_scan_duration_score(
    start_ts: np.ndarray,
    end_ts: np.ndarray,
) -> float:
    """
    LiDAR-specific timeliness sub-metric for KITTI:
    Stability of scan duration (end - start) across frames.

    Score ~ 1.0 if all scans have similar duration; lower if durations vary wildly.
    
    Args:
        start_ts: numpy array of scan start timestamps (datetime64[ns])
        end_ts: numpy array of scan end timestamps (datetime64[ns])
        
    Returns:
        Score between 0.0 and 1.0
    """
    if start_ts.size < 2 or end_ts.size < 2:
        return 0.0

    start_ns = start_ts.astype("int64")
    end_ns = end_ts.astype("int64")
    durations = (end_ns - start_ns) / 1e9  # seconds

    mean = np.mean(durations)
    std = np.std(durations)
    if mean <= 0:
        return 0.0

    score = np.exp(-std / (mean + 1e-6))
    return float(np.clip(score, 0.0, 1.0))


def compute_completeness_metrics(
    ts_np: np.ndarray,
    max_frames_in_sequence: int,
) -> Dict[str, float]:
    """
    Completeness dimension for one sensor/topic in a sequence.

    - message_availability: N_frames / max_frames_in_sequence
    - dropout_rate: fraction of gaps that are "too large"
    - dim_completeness: combines both into [0,1]
    
    Args:
        ts_np: numpy array of timestamps (datetime64[ns])
        max_frames_in_sequence: maximum frames in any sensor in this sequence
        
    Returns:
        Dictionary with completeness metrics
    """
    n_frames = ts_np.size
    if n_frames < 2 or max_frames_in_sequence <= 0:
        return {
            "message_availability": 0.0,
            "dropout_rate": 1.0,
            "dim_completeness": 0.0,
        }

    # Availability = frames vs best sensor in sequence
    message_availability = n_frames / max_frames_in_sequence

    diffs = inter_frame_diffs_seconds(ts_np)
    median_dt = np.median(diffs) if diffs.size > 0 else 0.0
    if median_dt <= 0:
        dropout_rate = 0.0
    else:
        dropout_mask = diffs > (2.0 * median_dt)
        dropout_rate = float(np.sum(dropout_mask) / len(diffs))

    # set to 1 if no big gaps
    dropout_score = 1.0 - dropout_rate

    dim_completeness = float(
        np.clip(0.5 * message_availability + 0.5 * dropout_score, 0.0, 1.0)
    )

    return {
        "message_availability": float(np.clip(message_availability, 0.0, 1.0)),
        "dropout_rate": float(np.clip(dropout_rate, 0.0, 1.0)),
        "dim_completeness": dim_completeness,
    }


def combine_timeliness_dimension(
    temporal_score: float,
    anomaly_score: float,
    extra_lidar_scan_score: Optional[float] = None,
) -> float:
    """
    Combine temporal regularity, anomaly score, and optional LiDAR scan-duration
    stability into a single timeliness dimension.
    
    Args:
        temporal_score: Score for temporal regularity
        anomaly_score: Score for anomaly absence
        extra_lidar_scan_score: Optional LiDAR scan duration score
        
    Returns:
        Combined timeliness score (0.0 to 1.0)
    """
    scores = [temporal_score, anomaly_score]
    if extra_lidar_scan_score is not None:
        scores.append(extra_lidar_scan_score)
    arr = np.array(scores, dtype=float)
    return float(np.clip(np.mean(arr), 0.0, 1.0))


def health_tier_from_overall(overall_score: float) -> str:
    """
    Map overall score (0-1) to qualitative tier.
    
    Args:
        overall_score: Overall quality score (0.0 to 1.0)
        
    Returns:
        Health tier classification: "excellent", "good", "fair", or "poor"
    """
    if overall_score >= 0.85:
        return "excellent"
    if overall_score >= 0.70:
        return "good"
    if overall_score >= 0.50:
        return "fair"
    return "poor"


def find_timestamp_files_for_sequence(seq_path: str, max_depth: int = 4) -> List[str]:
    """
    Recursively search for timestamps.txt under a KITTI sequence folder.
    
    Args:
        seq_path: Path to KITTI sequence directory
        max_depth: Maximum recursion depth
        
    Returns:
        List of paths to timestamps.txt files
    """
    timestamp_files: List[str] = []
    base_depth = seq_path.rstrip(os.sep).count(os.sep)

    for root, dirs, files in os.walk(seq_path):
        depth = root.rstrip(os.sep).count(os.sep) - base_depth
        if depth > max_depth:
            dirs[:] = []
            continue

        if "timestamps.txt" in files:
            ts_file = os.path.join(root, "timestamps.txt")
            timestamp_files.append(ts_file)

    timestamp_files = sorted(timestamp_files)
    return timestamp_files


def load_kitti_timestamps(ts_file: str) -> np.ndarray:
    """
    Load KITTI timestamps.txt into numpy datetime64[ns].
    
    Args:
        ts_file: Path to KITTI timestamps.txt file
        
    Returns:
        numpy array of datetime64[ns] timestamps
    """
    try:
        df = pd.read_csv(ts_file, header=None, names=["datetime"])
        df["dt"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna()
        return df["dt"].values.astype("datetime64[ns]")
    except Exception as e:
        print(f"    [WARN] Failed to load timestamps from {ts_file}: {e}")
        return np.array([], dtype="datetime64[ns]")


def load_kitti_lidar_start_end(
    velodyne_dir: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load LiDAR timestamps_start.txt and timestamps_end.txt if present.

    Returns (start_ts, end_ts) as numpy datetime64[ns] arrays, or (None, None) if missing.
    
    Args:
        velodyne_dir: Path to velodyne_points directory
        
    Returns:
        Tuple of (start_ts, end_ts) or (None, None)
    """
    start_file = os.path.join(velodyne_dir, "timestamps_start.txt")
    end_file = os.path.join(velodyne_dir, "timestamps_end.txt")

    if not (os.path.exists(start_file) and os.path.exists(end_file)):
        return None, None

    start_ts = load_kitti_timestamps(start_file)
    end_ts = load_kitti_timestamps(end_file)

    n = min(start_ts.size, end_ts.size)
    if n == 0:
        return None, None
    return start_ts[:n], end_ts[:n]


def run_health_check(
    sequences_root: str,
    output_dir: str = "health_reports",
) -> pd.DataFrame:
    """
    Compute health metrics for KITTI sequences in a directory.
    
    Args:
        sequences_root: Root directory containing KITTI sequences
        output_dir: Directory to save reports (created if not exists)
        
    Returns:
        DataFrame with per-sensor metrics
    """
    abs_root = os.path.abspath(sequences_root)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[INFO] Scanning sequences under {abs_root}")

    sequences = [
        d for d in sorted(os.listdir(sequences_root))
        if os.path.isdir(os.path.join(sequences_root, d))
    ]
    print(f"[INFO] Found {len(sequences)} sequences")

    rows: List[Dict] = []

    for seq_name in sequences:
        seq_path = os.path.join(sequences_root, seq_name)
        print(f"\n[INFO] Processing sequence: {seq_name}")

        ts_files = find_timestamp_files_for_sequence(seq_path, max_depth=4)
        if not ts_files:
            print(f"[WARN] No timestamps.txt found under {seq_name}, skipping.")
            continue

        # Loading all sensors & count frames
        frame_counts = []
        ts_per_sensor: Dict[str, np.ndarray] = {}
        lidar_start_end_per_sensor: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        for ts_file in ts_files:
            sensor_name = os.path.basename(os.path.dirname(ts_file))
            ts_np = load_kitti_timestamps(ts_file)
            if ts_np.size == 0:
                continue
            ts_per_sensor[sensor_name] = ts_np
            frame_counts.append(ts_np.size)

            # For LiDAR data, trying to load start/end timestamps
            if sensor_name == "velodyne_points":
                velodyne_dir = os.path.dirname(ts_file)
                start_ts, end_ts = load_kitti_lidar_start_end(velodyne_dir)
                if start_ts is not None and end_ts is not None:
                    lidar_start_end_per_sensor[sensor_name] = (start_ts, end_ts)

        if not frame_counts:
            print(f"[WARN] No valid timestamps in any sensor for {seq_name}, skipping.")
            continue

        max_frames_in_sequence = max(frame_counts)

        # Computing metrics per sensor
        for sensor_name, ts_np in ts_per_sensor.items():
            print(f"  [INFO] Sensor: {sensor_name} (frames={ts_np.size})")

            temporal_score = compute_temporal_score(ts_np)
            anomaly_score = compute_anomaly_score(ts_np)

            lidar_scan_score = None
            if sensor_name in lidar_start_end_per_sensor:
                start_ts, end_ts = lidar_start_end_per_sensor[sensor_name]
                n = min(len(start_ts), len(ts_np))
                if n > 1:
                    lidar_scan_score = compute_lidar_scan_duration_score(
                        start_ts[:n],
                        end_ts[:n],
                    )

            dim_timeliness = combine_timeliness_dimension(
                temporal_score,
                anomaly_score,
                extra_lidar_scan_score=lidar_scan_score,
            )

            completeness_stats = compute_completeness_metrics(
                ts_np, max_frames_in_sequence
            )

            dim_accuracy = np.nan
            dim_consistency = np.nan
            dim_relevance = np.nan
            dim_sensor_fusion = np.nan

            # Overall quality = average of main implemented dimensions
            dims_for_overall = [
                dim_timeliness,
                completeness_stats["dim_completeness"],
            ]
            overall_quality_score = float(np.mean(dims_for_overall))
            overall_quality_score_0_100 = float(overall_quality_score * 100.0)
            tier = health_tier_from_overall(overall_quality_score)

            print(
                f"    [INFO] Timeliness={dim_timeliness:.3f}, "
                f"Completeness={completeness_stats['dim_completeness']:.3f}, "
                f"Overall={overall_quality_score:.3f} ({overall_quality_score_0_100:.1f}), "
                f"Tier={tier}"
            )

            rows.append(
                {
                    "sequence": seq_name,
                    "sensor_or_topic": sensor_name,
                    # raw scores
                    "temporal_score": temporal_score,
                    "anomaly_score": anomaly_score,
                    "lidar_scan_score": lidar_scan_score,
                    # completeness sub-metrics
                    "message_availability": completeness_stats["message_availability"],
                    "dropout_rate": completeness_stats["dropout_rate"],
                    # dimensions
                    "dim_timeliness": dim_timeliness,
                    "dim_completeness": completeness_stats["dim_completeness"],
                    "dim_accuracy": dim_accuracy,
                    "dim_consistency": dim_consistency,
                    "dim_relevance": dim_relevance,
                    "dim_sensor_fusion": dim_sensor_fusion,
                    # overall
                    "overall_quality_score": overall_quality_score,
                    "overall_quality_score_0_100": overall_quality_score_0_100,
                    "health_tier": tier,
                }
            )

    return pd.DataFrame(rows)


def percentile_of_scores(scores: np.ndarray, x: float) -> float:
    """
    Compute percentile of x within scores (0-1).
    
    Args:
        scores: Array of scores
        x: Score value to find percentile of
        
    Returns:
        Percentile value (0.0 to 1.0)
    """
    sorted_scores = np.sort(scores)
    rank = np.searchsorted(sorted_scores, x, side="right")
    return float(rank / len(sorted_scores))
