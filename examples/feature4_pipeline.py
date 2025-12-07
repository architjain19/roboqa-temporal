from __future__ import annotations

import argparse
import json
import os
import glob
import importlib.util
from typing import Dict, List

import numpy as np

# ---------------------------------------------------------------------
# Dynamic loading of Feature 4 modules 
# ---------------------------------------------------------------------
def _load_module(name: str, path: str):
    """Load a Python module from a specific file path (no package import)."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Locate project root and paths to Feature 4 code
ROOT = os.path.dirname(os.path.abspath(__file__))  # .../roboqa-temporal/examples
ROOT = os.path.dirname(ROOT)                       # .../roboqa-temporal

BENCH_PATH = os.path.join(ROOT, "src", "roboqa_temporal", "benchmarking.py")
METRICS_PATH = os.path.join(
    ROOT, "src", "roboqa_temporal", "reporting", "feature4", "metrics_to_df.py"
)
EXPORTERS_PATH = os.path.join(
    ROOT, "src", "roboqa_temporal", "reporting", "feature4", "exporters.py"
)
DASHBOARDS_PATH = os.path.join(
    ROOT, "src", "roboqa_temporal", "reporting", "feature4", "dashboards.py"
)
HTML_UTILS_PATH = os.path.join(
    ROOT, "src", "roboqa_temporal", "reporting", "feature4", "html_utils.py"
)

# Load Feature 4 modules directly by path
bench_mod = _load_module("benchmarking_mod", BENCH_PATH)
metrics_mod = _load_module("metrics_to_df_mod", METRICS_PATH)
exporters_mod = _load_module("exporters_mod", EXPORTERS_PATH)
dashboards_mod = _load_module("dashboards_mod", DASHBOARDS_PATH)
html_utils_mod = _load_module("html_utils_mod", HTML_UTILS_PATH)

# Pull the functions we need
add_temporal_score = bench_mod.add_temporal_score
add_anomaly_score = bench_mod.add_anomaly_score
add_multimodal_health_score = bench_mod.add_multimodal_health_score
add_health_tiers = bench_mod.add_health_tiers

metrics_list_to_dataframe = metrics_mod.metrics_list_to_dataframe
export_dataframe = exporters_mod.export_dataframe
build_quality_dashboard = dashboards_mod.build_quality_dashboard
save_dashboard_html = html_utils_mod.save_dashboard_html


# ---------------------------------------------------------------------
# Feature 1 temporal metrics + TXT fallback
# ---------------------------------------------------------------------
def load_temporal_metrics_from_feature1(
    seq_name: str,
    temporal_root: str = "reports/temporal_sync",
) -> Dict[str, float]:
    """
    Try to load temporal metrics produced by Feature 1 (TemporalSyncValidator).

    Expected file:
      {temporal_root}/{seq_name}_temporal_metrics.json

    Structure:
      {
        "sequence": "<seq_name>",
        "metrics": { ... float metrics ... }
      }
    """
    path = os.path.join(temporal_root, f"{seq_name}_temporal_metrics.json")
    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read temporal metrics for {seq_name}: {e}")
        return {}

    raw = data.get("metrics", {})
    if not isinstance(raw, dict):
        return {}

    out: Dict[str, float] = {}
    for k, v in raw.items():
        try:
            out[k] = float(v)
        except Exception:
            continue
    return out


def compute_temporal_metrics_from_txt(
    timestamps_by_topic: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """
    Lightweight temporal metrics based only on timestamp TXT files.

    Used as a fallback when Feature 1 JSON is not present.
    """
    if not timestamps_by_topic:
        return {
            "global_mean_sync_error": 0.0,
            "matched_fraction": 0.0,
        }

    topics = sorted(timestamps_by_topic.keys())
    if len(topics) < 2:
        ts = np.sort(np.asarray(timestamps_by_topic[topics[0]], dtype=float))
        return {
            "global_mean_sync_error": 0.0,
            "matched_fraction": 1.0 if ts.size > 0 else 0.0,
        }

    # Simple pairwise alignment between first two topics
    t0, t1 = topics[:2]
    ts0 = np.sort(np.asarray(timestamps_by_topic[t0], dtype=float))
    ts1 = np.sort(np.asarray(timestamps_by_topic[t1], dtype=float))

    n = min(len(ts0), len(ts1))
    if n == 0:
        return {
            "global_mean_sync_error": 0.0,
            "matched_fraction": 0.0,
        }

    diffs = np.abs(ts0[:n] - ts1[:n])
    global_mean_sync_error = float(np.mean(diffs))
    matched_fraction = n / max(len(ts0), len(ts1))

    return {
        "global_mean_sync_error": global_mean_sync_error,
        "matched_fraction": float(matched_fraction),
    }


# ---------------------------------------------------------------------
# Feature 2 anomaly metrics + dummy fallback
# ---------------------------------------------------------------------
def load_anomaly_metrics_for_sequence(
    seq_name: str,
    feature2_root: str = "reports/feature2",
) -> Dict[str, float]:
    """
    Load anomaly / health metrics exported by Feature 2.

    Expected file:
      {feature2_root}/{seq_name}_anomaly_metrics.json

    We accept either:
      { "sequence": ..., "health_metrics": { ... } }
    or:
      { "metrics": { ... } }
    or even a flat dict of floats.
    """
    path = os.path.join(feature2_root, f"{seq_name}_anomaly_metrics.json")
    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read anomaly metrics for {seq_name}: {e}")
        return {}

    if "health_metrics" in data and isinstance(data["health_metrics"], dict):
        raw = data["health_metrics"]
    elif "metrics" in data and isinstance(data["metrics"], dict):
        raw = data["metrics"]
    else:
        raw = data

    out: Dict[str, float] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            try:
                out[k] = float(v)
            except Exception:
                continue
    return out


def compute_dummy_anomaly_metrics(
    timestamps_by_topic: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """
    Very simple dummy anomaly metric so Feature 4 has something to use.
    Only used when Feature 2 JSON is missing.
    """
    # Cheap heuristic based on variance in message counts across topics
    counts = [len(ts) for ts in timestamps_by_topic.values()]
    if not counts:
        return {"mean_anomaly_ratio": 0.0}

    mean = float(np.mean(counts))
    std = float(np.std(counts))
    if mean <= 0:
        ratio = 0.0
    else:
        ratio = min(1.0, std / (mean + 1e-6))

    return {
        "mean_anomaly_ratio": ratio,
    }


# ---------------------------------------------------------------------
# Data loading helper (timestamp TXT files)
# ---------------------------------------------------------------------
def load_sequence_timestamps(sequence_dir: str) -> Dict[str, np.ndarray]:
    """
    Each *.txt file in sequence_dir is treated as one topic's timestamps.

    This is only used:
      - For small synthetic tests, or
      - As a fallback when Feature 1 metrics are not available.
    """
    timestamps_by_topic: Dict[str, np.ndarray] = {}

    for entry in os.listdir(sequence_dir):
        full = os.path.join(sequence_dir, entry)
        if not os.path.isfile(full) or not entry.endswith(".txt"):
            continue

        topic = os.path.splitext(entry)[0]
        ts = np.loadtxt(full, dtype=float)
        ts = np.atleast_1d(ts)
        timestamps_by_topic[topic] = ts

    return timestamps_by_topic


# ---------------------------------------------------------------------
# Main Feature 4 pipeline
# ---------------------------------------------------------------------
def run_feature4_pipeline(
    sequences_root: str,
    output_dir: str,
    feature2_root: str = "reports/feature2",
    temporal_root: str = "reports/temporal_sync",
) -> None:
    """
    Feature 4 pipeline.

    For each sequence:
      1. Prefer temporal metrics from Feature 1 JSON (tiny, safe for 10 GB bags).
      2. Prefer anomaly metrics from Feature 2 JSON (tiny).
      3. Only fall back to TXT timestamps and dummy metrics if JSON is missing.
    """
    sequences_root = os.path.abspath(sequences_root)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    sequence_dirs: List[str] = sorted(
        d for d in glob.glob(os.path.join(sequences_root, "*")) if os.path.isdir(d)
    )
    if not sequence_dirs:
        print(f"[ERROR] No sequences found under {sequences_root}")
        return

    print(f"[INFO] Found {len(sequence_dirs)} sequences under {sequences_root}")

    all_metrics: List[dict] = []

    for seq_dir in sequence_dirs:
        seq_name = os.path.basename(seq_dir)
        print(f"[INFO] Processing sequence: {seq_name}")

        timestamps_by_topic: Dict[str, np.ndarray] = {}

        # -------- Temporal metrics (Feature 1 preferred) --------
        temporal_metrics = load_temporal_metrics_from_feature1(seq_name, temporal_root)

        if not temporal_metrics:
            # Fallback: load *.txt timestamps and compute a simple sync metric
            timestamps_by_topic = load_sequence_timestamps(seq_dir)
            if not timestamps_by_topic:
                print(f"[WARN] No timestamp topics in {seq_name}, skipping.")
                continue
            temporal_metrics = compute_temporal_metrics_from_txt(timestamps_by_topic)

        # -------- Anomaly metrics (Feature 2 preferred) --------
        anomaly_metrics = load_anomaly_metrics_for_sequence(seq_name, feature2_root)

        if not anomaly_metrics:
            # Fallback: if we didn't already load timestamps, load them now
            if not timestamps_by_topic:
                timestamps_by_topic = load_sequence_timestamps(seq_dir)
            if not timestamps_by_topic:
                anomaly_metrics = {"mean_anomaly_ratio": 0.0}
            else:
                anomaly_metrics = compute_dummy_anomaly_metrics(timestamps_by_topic)

        merged = {
            "sequence": seq_name,
            **temporal_metrics,
            **anomaly_metrics,
        }
        all_metrics.append(merged)

    if not all_metrics:
        print("[ERROR] No metrics accumulated; nothing to benchmark.")
        return

    # -------- Feature 4: DataFrame + benchmarking --------
    df = metrics_list_to_dataframe(all_metrics)

    # Let benchmarking.add_temporal_score pick sensible defaults that know about
    # both the TXT fallback metrics and the TemporalSyncValidator metrics.
    df = add_temporal_score(df)

    # Similar for anomaly score
    df = add_anomaly_score(df)

    # Combine into a single multimodal health score
    df = add_multimodal_health_score(
        df,
        temporal_col="temporal_score",
        anomaly_col="anomaly_score",
        alpha_temporal=0.5,
    )

    df = add_health_tiers(df)

    # -------- Feature 4: export --------
    csv_path = os.path.join(output_dir, "metrics.csv")
    json_path = os.path.join(output_dir, "metrics.json")
    export_dataframe(df, csv_path, fmt="csv")
    export_dataframe(df, json_path, fmt="json")
    print(f"[INFO] Metrics written to:\n  {csv_path}\n  {json_path}")

    # -------- Feature 4: dashboard --------
    fig = build_quality_dashboard(df)
    html_path = os.path.join(output_dir, "dashboard.html")
    save_dashboard_html(fig, html_path)
    print(f"[INFO] Dashboard saved to: {html_path}")
    print("[INFO] Feature 4 pipeline complete ✅")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Feature 4 (multi-modal benchmarking + reporting).",
    )
    parser.add_argument(
        "--sequences-root",
        type=str,
        required=True,
        help="Root directory containing sequence subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for metrics + dashboard.",
    )
    parser.add_argument(
        "--feature2-root",
        type=str,
        default="reports/feature2",
        help="Directory containing <seq>_anomaly_metrics.json from Feature 2.",
    )
    parser.add_argument(
        "--temporal-root",
        type=str,
        default="reports/temporal_sync",
        help="Directory containing <seq>_temporal_metrics.json from Feature 1.",
    )

    args = parser.parse_args()
    run_feature4_pipeline(
        sequences_root=args.sequences_root,
        output_dir=args.output_dir,
        feature2_root=args.feature2_root,
        temporal_root=args.temporal_root,
    )
