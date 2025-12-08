"""
################################################################
File: roboqa_temporal/dataset_quality/feature4_pipeline.py
Author: Sayali Nehul
################################################################

Feature 4 – Dataset Quality Scoring & Cross-Benchmarking (KITTI).

Scans a KITTI-style sequences root directory, loads per-sensor
timestamps (camera, LiDAR, OXTS), and computes simple Feature-4-like
metrics for each sequence:

- temporal_score
- anomaly_score
- multimodal_health_score
- health_tier  (e.g., excellent / good / fair / poor)

Outputs:
- feature4_kitti_metrics.csv   (tabular metrics per sequence)
- feature4_health_scores.png   (visual summary / bar chart)

This pipeline is independent of Feature 1 and Feature 2, and is
designed to run directly on KITTI raw data folders.

################################################################
"""

from __future__ import annotations

import os
import argparse
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plotly.subplots import make_subplots
import plotly.graph_objects as go


try:
    import yaml
except ImportError:  
    yaml = None


# =========================================================
# Shared timestamp utilities
# =========================================================

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


# =========================================================
# Core metrics (dimensions)
# =========================================================

def compute_temporal_score(ts_np: np.ndarray) -> float:
    """
    Timeliness sub-metric: regularity of frame spacing.
    Higher = more regular.
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


def compute_liar_scan_duration_score(
    start_ts: np.ndarray,
    end_ts: np.ndarray,
) -> float:
    """
    LiDAR-specific timeliness sub-metric for KITTI:
    Stability of scan duration (end - start) across frames.

    Score ~ 1.0 if all scans have similar duration; lower if durations vary wildly.
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
    """
    n_frames = ts_np.size
    if n_frames < 2 or max_frames_in_sequence <= 0:
        return {
            "message_availability": 0.0,
            "dropout_rate": 1.0,
            "dim_completeness": 0.0,
        }

    # Availability: frames vs best sensor in sequence
    message_availability = n_frames / max_frames_in_sequence

    diffs = inter_frame_diffs_seconds(ts_np)
    median_dt = np.median(diffs) if diffs.size > 0 else 0.0
    if median_dt <= 0:
        dropout_rate = 0.0
    else:
        dropout_mask = diffs > (2.0 * median_dt)
        dropout_rate = float(np.sum(dropout_mask) / len(diffs))

    dropout_score = 1.0 - dropout_rate  # 1 if no big gaps

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
    """
    scores = [temporal_score, anomaly_score]
    if extra_lidar_scan_score is not None:
        scores.append(extra_lidar_scan_score)
    arr = np.array(scores, dtype=float)
    return float(np.clip(np.mean(arr), 0.0, 1.0))


def health_tier_from_overall(overall_score: float) -> str:
    """
    Map overall score (0–1) to qualitative tier.
    """
    if overall_score >= 0.85:
        return "excellent"
    if overall_score >= 0.70:
        return "good"
    if overall_score >= 0.50:
        return "fair"
    return "poor"


# =========================================================
# KITTI loader
# =========================================================

def find_timestamp_files_for_sequence(seq_path: str, max_depth: int = 4) -> List[str]:
    """
    Recursively search for timestamps.txt under a KITTI sequence folder.
    """
    timestamp_files: List[str] = []
    base_depth = seq_path.rstrip(os.sep).count(os.sep)

    print(f"  [DEBUG] Walking under: {seq_path}")
    for root, dirs, files in os.walk(seq_path):
        depth = root.rstrip(os.sep).count(os.sep) - base_depth
        if depth > max_depth:
            dirs[:] = []
            continue

        if "timestamps.txt" in files:
            ts_file = os.path.join(root, "timestamps.txt")
            timestamp_files.append(ts_file)

    timestamp_files = sorted(timestamp_files)
    if not timestamp_files:
        print(f"  [DEBUG] No timestamps.txt found under: {seq_path}")
    else:
        print(f"  [DEBUG] Found {len(timestamp_files)} timestamps.txt files:")
        for p in timestamp_files:
            print(f"    - {p}")

    return timestamp_files


def load_kitti_timestamps(ts_file: str) -> np.ndarray:
    """
    Load KITTI timestamps.txt into numpy datetime64[ns].
    """
    print(f"    [DEBUG] Loading timestamps from: {ts_file}")
    df = pd.read_csv(ts_file, header=None, names=["datetime"])
    df["dt"] = pd.to_datetime(df["datetime"], errors="coerce")
    before = len(df)
    df = df.dropna()
    after = len(df)
    print(f"    [DEBUG] Loaded {after} valid timestamps (dropped {before - after}).")
    return df["dt"].values.astype("datetime64[ns]")


def load_kitti_lidar_start_end(
    velodyne_dir: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load LiDAR timestamps_start.txt and timestamps_end.txt if present.

    Returns (start_ts, end_ts) as numpy datetime64[ns] arrays, or (None, None) if missing.
    """
    start_file = os.path.join(velodyne_dir, "timestamps_start.txt")
    end_file = os.path.join(velodyne_dir, "timestamps_end.txt")

    if not (os.path.exists(start_file) and os.path.exists(end_file)):
        print(f"    [DEBUG] No timestamps_start/end.txt in {velodyne_dir}")
        return None, None

    print(f"    [DEBUG] Loading LiDAR timestamps_start/end from: {velodyne_dir}")
    start_ts = load_kitti_timestamps(start_file)
    end_ts = load_kitti_timestamps(end_file)

    # Ensure same length
    n = min(start_ts.size, end_ts.size)
    if n == 0:
        return None, None
    return start_ts[:n], end_ts[:n]


def run_feature4_kitti(sequences_root: str) -> pd.DataFrame:
    """
    Compute Feature 4 metrics for KITTI sequences.
    """
    abs_root = os.path.abspath(sequences_root)
    print(f"[INFO] (KITTI) Scanning sequences under {abs_root}")

    sequences = [
        d for d in sorted(os.listdir(sequences_root))
        if os.path.isdir(os.path.join(sequences_root, d))
    ]
    print(f"[INFO] Found {len(sequences)} sequences under {abs_root}")

    rows: List[Dict] = []

    for seq_name in sequences:
        seq_path = os.path.join(sequences_root, seq_name)
        print(f"[INFO] Processing sequence: {seq_name}")

        ts_files = find_timestamp_files_for_sequence(seq_path, max_depth=4)
        if not ts_files:
            print(f"[WARN] No timestamps.txt found under {seq_name}, skipping.")
            continue

        # First pass: load all sensors & count frames
        frame_counts = []
        ts_per_sensor: Dict[str, np.ndarray] = {}
        lidar_start_end_per_sensor: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        for ts_file in ts_files:
            sensor_name = os.path.basename(os.path.dirname(ts_file))  # image_00, velodyne_points, oxts
            ts_np = load_kitti_timestamps(ts_file)
            if ts_np.size == 0:
                continue
            ts_per_sensor[sensor_name] = ts_np
            frame_counts.append(ts_np.size)

            # For LiDAR, try to load start/end timestamps
            if sensor_name == "velodyne_points":
                velodyne_dir = os.path.dirname(ts_file)
                start_ts, end_ts = load_kitti_lidar_start_end(velodyne_dir)
                if start_ts is not None and end_ts is not None:
                    lidar_start_end_per_sensor[sensor_name] = (start_ts, end_ts)

        if not frame_counts:
            print(f"[WARN] No valid timestamps in any sensor for {seq_name}, skipping.")
            continue

        max_frames_in_sequence = max(frame_counts)

        # Second pass: compute metrics per sensor
        for sensor_name, ts_np in ts_per_sensor.items():
            print(f"  [INFO] Sensor: {sensor_name}")

            temporal_score = compute_temporal_score(ts_np)
            anomaly_score = compute_anomaly_score(ts_np)

            # LiDAR-specific scan-duration score if available
            lidar_scan_score = None
            if sensor_name in lidar_start_end_per_sensor:
                start_ts, end_ts = lidar_start_end_per_sensor[sensor_name]
                n = min(len(start_ts), len(ts_np))
                if n > 1:
                    lidar_scan_score = compute_liar_scan_duration_score(
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

            # Placeholder for future advanced dimensions
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
                    "source": "kitti",
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


# =========================================================
# ROS2 loader (rosbag2_py)
# =========================================================

def run_feature4_ros2_bag(bag_path: str) -> pd.DataFrame:
    """
    Compute Feature 4 metrics for a single ROS2 bag using rosbag2_py.

    We treat each topic as a "sensor" and use message timestamps.
    """
    try:
        import rosbag2_py  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "rosbag2_py is not available. Activate your ROS2 environment "
            "before running in --mode ros2."
        ) from exc

    bag_path = os.path.abspath(bag_path)
    bag_name = os.path.basename(bag_path.rstrip(os.sep))
    print(f"[INFO] (ROS2) Processing bag: {bag_name} at {bag_path}")

    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    topics = [t.name for t in topic_types]

    # Collect timestamps per topic
    ts_per_topic: Dict[str, List[int]] = {t: [] for t in topics}

    while reader.has_next():
        topic, data, t = reader.read_next()
        ts_per_topic[topic].append(t)  # nanoseconds

    # Convert to numpy datetime64[ns]
    ts_array_per_topic: Dict[str, np.ndarray] = {}
    frame_counts = []

    for topic, ts_list in ts_per_topic.items():
        if not ts_list:
            continue
        ts_ns = np.array(ts_list, dtype=np.int64)
        ts_dt = ts_ns.astype("datetime64[ns]")
        ts_array_per_topic[topic] = ts_dt
        frame_counts.append(ts_dt.size)

    if not frame_counts:
        print(f"[WARN] Bag {bag_name}: no messages, skipping.")
        return pd.DataFrame()

    max_frames_in_sequence = max(frame_counts)
    rows: List[Dict] = []

    for topic, ts_np in ts_array_per_topic.items():
        print(f"  [INFO] Topic: {topic} (frames={ts_np.size})")

        temporal_score = compute_temporal_score(ts_np)
        anomaly_score = compute_anomaly_score(ts_np)
        dim_timeliness = combine_timeliness_dimension(
            temporal_score,
            anomaly_score,
            extra_lidar_scan_score=None,
        )

        completeness_stats = compute_completeness_metrics(
            ts_np, max_frames_in_sequence
        )

        # Placeholders for future ROS2-level metrics
        dim_accuracy = np.nan
        dim_consistency = np.nan
        dim_relevance = np.nan
        dim_sensor_fusion = np.nan

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
                "source": "ros2",
                "sequence": bag_name,
                "sensor_or_topic": topic,
                "temporal_score": temporal_score,
                "anomaly_score": anomaly_score,
                "lidar_scan_score": None,
                "message_availability": completeness_stats["message_availability"],
                "dropout_rate": completeness_stats["dropout_rate"],
                "dim_timeliness": dim_timeliness,
                "dim_completeness": completeness_stats["dim_completeness"],
                "dim_accuracy": dim_accuracy,
                "dim_consistency": dim_consistency,
                "dim_relevance": dim_relevance,
                "dim_sensor_fusion": dim_sensor_fusion,
                "overall_quality_score": overall_quality_score,
                "overall_quality_score_0_100": overall_quality_score_0_100,
                "health_tier": tier,
            }
        )

    return pd.DataFrame(rows)


# =========================================================
# Dashboard & report builder
# =========================================================

def build_dashboard_html(
    df_per_sensor: pd.DataFrame,
    df_per_sequence: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Build an HTML dashboard summarizing quality metrics.
    """

    seq_order = df_per_sequence["sequence"].tolist()

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Overall Quality per Sequence",
            "Timeliness Dimension per Sequence",
            "Completeness Dimension per Sequence",
            "Temporal vs Anomaly Scores",
        ],
    )

    # 1) Overall quality
    fig.add_trace(
        go.Bar(
            x=seq_order,
            y=df_per_sequence["overall_quality_score"],
            name="Overall quality",
        ),
        row=1,
        col=1,
    )

    # 2) Timeliness
    fig.add_trace(
        go.Scatter(
            x=seq_order,
            y=df_per_sequence["dim_timeliness"],
            mode="lines+markers",
            name="Timeliness",
        ),
        row=1,
        col=2,
    )

    # 3) Completeness
    fig.add_trace(
        go.Scatter(
            x=seq_order,
            y=df_per_sequence["dim_completeness"],
            mode="lines+markers",
            name="Completeness",
        ),
        row=2,
        col=1,
    )

    # 4) Temporal vs anomaly
    fig.add_trace(
        go.Scatter(
            x=seq_order,
            y=df_per_sequence["temporal_score"],
            mode="lines+markers",
            name="Temporal score",
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=seq_order,
            y=df_per_sequence["anomaly_score"],
            mode="lines+markers",
            name="Anomaly score",
        ),
        row=2,
        col=2,
    )

    # Axis labels
    fig.update_yaxes(title_text="Health (0–1)", row=1, col=1)
    fig.update_yaxes(title_text="Dim. score (0–1)", row=1, col=2)
    fig.update_yaxes(title_text="Dim. score (0–1)", row=2, col=1)
    fig.update_yaxes(title_text="Score (0–1)", row=2, col=2)

    for r in [1, 2]:
        for c in [1, 2]:
            fig.update_xaxes(title_text="Sequence", row=r, col=c)

    fig.update_layout(
        title="Dataset Quality Scoring & Cross-Benchmarking",
        height=900,
        showlegend=True,
        font=dict(color="black", size=12),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#dddddd", linecolor="black")
    fig.update_yaxes(showgrid=True, gridcolor="#dddddd", linecolor="black")

    fig_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

    # Aggregated metrics table
    metrics_cols = [
        "sequence",
        "overall_quality_score_0_100",
        "health_tier",
        "dim_timeliness",
        "dim_completeness",
        "temporal_score",
        "anomaly_score",
        "percentile_vs_run",
    ]
    table_df = df_per_sequence[metrics_cols].round(3)
    table_html = table_df.to_html(index=False, classes="metrics-table")

    # Dimension status table (based on our discussion)
    dims_rows = [
        {
            "name": "Accuracy",
            "sub": (
                "Sensor measurement accuracy; positional accuracy (GPS/RTK); "
                "annotation / inter-rater accuracy."
            ),
            "status": (
                "Not evaluated in this run – requires ground-truth labels or "
                "reference trajectories beyond timestamps."
            ),
        },
        {
            "name": "Completeness",
            "sub": (
                "Topic/message availability; temporal coverage gaps; "
                "sensor dropout rate; imputation feasibility."
            ),
            "status": (
                "PARTIALLY IMPLEMENTED – message availability and dropout rate "
                "computed from timestamps; imputation feasibility planned."
            ),
        },
        {
            "name": "Consistency",
            "sub": (
                "Cross-sensor consistency (fusion residuals); temporal consistency "
                "via motion model; semantic consistency."
            ),
            "status": (
                "PARTIALLY IMPLEMENTED – temporal consistency from timestamps; "
                "full sensor-fusion and semantic checks planned."
            ),
        },
        {
            "name": "Timeliness",
            "sub": (
                "Timestamp validity; latency; real-time constraint satisfaction."
            ),
            "status": (
                "IMPLEMENTED – temporal regularity and anomaly score for all sensors; "
                "LiDAR scan-duration stability from KITTI timestamps_start/end when available."
            ),
        },
        {
            "name": "Relevance",
            "sub": (
                "Scene diversity; edge-case coverage (night/rain/occlusion); "
                "distribution similarity to deployment environment."
            ),
            "status": (
                "Not evaluated in this run – requires labels or image/content analysis."
            ),
        },
        {
            "name": "Sensor Fusion Quality",
            "sub": (
                "Data fusion confidence; redundancy utilization; "
                "complementarity score for each sensor."
            ),
            "status": (
                "Not evaluated – requires fusion pipeline outputs beyond raw timestamps."
            ),
        },
        {
            "name": "Comparative Benchmarking",
            "sub": (
                "Percentile ranks vs KITTI / nuScenes / Waymo; radar charts; "
                "KS tests for distribution similarity."
            ),
            "status": (
                "PARTIALLY IMPLEMENTED – internal percentile ranks computed "
                "within this run; external dataset stats planned."
            ),
        },
        {
            "name": "Active Learning / Performance Modeling",
            "sub": (
                "Predict downstream model performance f(quality_metrics); "
                "identify high-value sequences; estimate required dataset size."
            ),
            "status": (
                "Not evaluated – requires downstream ML training signals."
            ),
        },
        {
            "name": "Integration & Curation",
            "sub": (
                "Export metadata (JSON/YAML, ROS2 bag tags); interface with MLflow, "
                "Label Studio/Labelbox; curation recommendations."
            ),
            "status": (
                "IMPLEMENTED IN PROJECT – this pipeline exports JSON/YAML quality "
                "metadata and supports ROS2 bags; this dashboard focuses on metrics "
                "visualization for the current run."
            ),
        },
    ]

    dims_rows_html = "\n".join(
        f"""
        <tr>
          <td><strong>{row['name']}</strong></td>
          <td>{row['sub']}</td>
          <td>{row['status']}</td>
        </tr>
        """
        for row in dims_rows
    )

    dims_table_html = f"""
    <table class="dim-table">
      <thead>
        <tr>
          <th>Dimension / Component</th>
          <th>Sub-metrics</th>
          <th>Status in this run</th>
        </tr>
      </thead>
      <tbody>
        {dims_rows_html}
      </tbody>
    </table>
    """

    # HTML wrapper
    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Feature 4 – Dataset Quality Scoring & Cross-Benchmarking</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 0;
      padding: 0;
      background-color: #ffffff;
      color: #000000;
    }}
    .header {{
      background-color: #f2f2f2;
      color: #000000;
      padding: 16px 24px;
      border-bottom: 1px solid #dddddd;
    }}
    .container {{
      padding: 16px 24px 40px 24px;
    }}
    h1 {{
      margin: 0;
      font-size: 24px;
    }}
    h2 {{
      margin-top: 32px;
      font-size: 20px;
    }}
    .summary-card {{
      background: #ffffff;
      border-radius: 8px;
      padding: 16px 20px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.08);
      margin-top: 16px;
      margin-bottom: 24px;
      border: 1px solid #e0e0e0;
    }}
    .summary-grid {{
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
    }}
    .summary-item {{
      flex: 1 1 160px;
    }}
    .summary-label {{
      font-size: 12px;
      color: #555;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .summary-value {{
      font-size: 20px;
      font-weight: bold;
      margin-top: 4px;
      color: #000;
    }}
    table.metrics-table,
    table.dim-table {{
      border-collapse: collapse;
      width: 100%;
      background: #ffffff;
      margin-top: 8px;
      font-size: 13px;
    }}
    table.metrics-table th,
    table.metrics-table td,
    table.dim-table th,
    table.dim-table td {{
      border: 1px solid #ddd;
      padding: 6px 8px;
    }}
    table.metrics-table th,
    table.dim-table th {{
      background: #f5f5f5;
      font-weight: 600;
    }}
  </style>
</head>
<body>
  <div class="header">
    <h1>Feature 4 – Dataset Quality Scoring & Cross-Benchmarking</h1>
  </div>
  <div class="container">
    <div class="summary-card">
      <div class="summary-grid">
        <div class="summary-item">
          <div class="summary-label">Sequences</div>
          <div class="summary-value">{len(df_per_sequence)}</div>
        </div>
        <div class="summary-item">
          <div class="summary-label">Sensors / Topics</div>
          <div class="summary-value">{len(df_per_sensor)}</div>
        </div>
        <div class="summary-item">
          <div class="summary-label">Mean Quality Score</div>
          <div class="summary-value">
            {df_per_sequence['overall_quality_score_0_100'].mean():.1f} / 100
          </div>
        </div>
        <div class="summary-item">
          <div class="summary-label">Best Sequence</div>
          <div class="summary-value">
            {df_per_sequence.loc[df_per_sequence['overall_quality_score'].idxmax(), 'sequence']}
          </div>
        </div>
      </div>
    </div>

    <h2>Multi-Modal Health Overview</h2>
    {fig_html}

    <h2>Aggregated Metrics per Sequence</h2>
    {table_html}

    <h2>Quality Dimensions & Implementation Status</h2>
    <p>
      This section summarizes the full Feature 4 design based on ISO/IEC 25012-style
      data quality characteristics and your project specification. The
      <strong>Status</strong> column indicates what is implemented in this run,
      which currently focuses on timestamp-based metrics for KITTI and/or ROS2 bags.
    </p>
    {dims_table_html}
  </div>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[INFO] Saved dashboard HTML: {output_path}")


def percentile_of_scores(scores: np.ndarray, x: float) -> float:
    """
    Compute percentile of x within scores (0–1).
    """
    sorted_scores = np.sort(scores)
    rank = np.searchsorted(sorted_scores, x, side="right")
    return float(rank / len(sorted_scores))


def export_json_yaml(
    df_per_sequence: pd.DataFrame,
    output_dir: str,
) -> None:
    """
    Export a compact JSON + YAML quality report.
    """
    records = df_per_sequence.to_dict(orient="records")
    report = {
        "feature4_version": "unified_kitti_ros2_v1",
        "num_sequences": len(df_per_sequence),
        "sequences": records,
    }

    json_path = os.path.join(output_dir, "feature4_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        import json
        json.dump(report, f, indent=2)
    print(f"[INFO] Saved JSON report: {json_path}")

    if yaml is not None:
        yaml_path = os.path.join(output_dir, "feature4_report.yaml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(report, f, sort_keys=False)
        print(f"[INFO] Saved YAML report: {yaml_path}")
    else:
        print("[WARN] PyYAML not installed; skipping YAML export.")


# =========================================================
# Main orchestration
# =========================================================

def run_feature4(
    mode: str,
    sequences_root: Optional[str],
    ros2_bag: Optional[str],
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    if mode == "kitti":
        if sequences_root is None:
            raise SystemExit("--sequences-root is required when --mode kitti")
        df = run_feature4_kitti(sequences_root)
    elif mode == "ros2":
        if ros2_bag is None:
            raise SystemExit("--ros2-bag is required when --mode ros2")
        df = run_feature4_ros2_bag(ros2_bag)
    else:
        raise SystemExit(f"Unknown mode: {mode}")

    if df.empty:
        print("[ERROR] No metrics accumulated; nothing to export.")
        return

    # Percentile vs run
    scores = df["overall_quality_score"].values
    df["percentile_vs_run"] = df["overall_quality_score"].apply(
        lambda x: percentile_of_scores(scores, x)
    )

    # Save per-sensor CSV
    csv_path = os.path.join(output_dir, "feature4_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Saved metrics CSV: {csv_path}")

    # Aggregate per sequence
    agg_cols = [
        "temporal_score",
        "anomaly_score",
        "dim_timeliness",
        "dim_completeness",
        "overall_quality_score",
        "overall_quality_score_0_100",
        "percentile_vs_run",
    ]
    df_seq = df.groupby("sequence", as_index=False)[agg_cols].mean()
    df_seq["health_tier"] = df_seq["overall_quality_score"].apply(
        health_tier_from_overall
    )

    agg_csv = os.path.join(output_dir, "feature4_metrics_by_sequence.csv")
    df_seq.to_csv(agg_csv, index=False)
    print(f"[INFO] Saved aggregated per-sequence CSV: {agg_csv}")

    # Simple bar plot of overall quality scores (per sensor/topic)
    labels = df["sequence"] + " / " + df["sensor_or_topic"]
    scores_plot = df["overall_quality_score_0_100"]

    plt.figure(figsize=(10, 6))
    plt.title("Feature 4 – Overall Quality Score (per sensor/topic)")
    plt.bar(labels, scores_plot)
    plt.xticks(rotation=90)
    plt.ylabel("overall_quality_score (0–100)")
    plt.ylim(0.0, 100.0)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "feature4_health_scores.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"[INFO] Saved plot: {plot_path}")

    # Dashboard HTML
    dash_path = os.path.join(output_dir, "feature4_dashboard.html")
    build_dashboard_html(df, df_seq, dash_path)

    # JSON/YAML export
    export_json_yaml(df_seq, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Unified Feature 4 pipeline (KITTI + ROS2)")
    parser.add_argument(
        "--mode",
        choices=["kitti", "ros2"],
        required=True,
        help="Input mode: 'kitti' for KITTI sequences, 'ros2' for ROS2 bag.",
    )
    parser.add_argument(
        "--sequences-root",
        help="Root folder of KITTI sequences (e.g. ./data/sequences) when --mode kitti",
    )
    parser.add_argument(
        "--ros2-bag",
        help="Path to ROS2 bag directory when --mode ros2",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to store CSV, plots, dashboard, and reports",
    )
    args = parser.parse_args()

    run_feature4(args.mode, args.sequences_root, args.ros2_bag, args.output_dir)


if __name__ == "__main__":
    main()

