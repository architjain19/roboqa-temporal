"""
Tests for Feature 4 reporting + benchmarking.

We only use small synthetic data here and do NOT touch ROS2 / bags.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List
import importlib.util

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helper: load modules directly from src/ paths (avoid roboqa_temporal.__init__)
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]  # repo root: .../roboqa-temporal
FEATURE4_DIR = ROOT / "src" / "roboqa_temporal" / "reporting" / "feature4"
BENCH_PATH = ROOT / "src" / "roboqa_temporal" / "benchmarking.py"
METRICS_PATH = FEATURE4_DIR / "metrics_to_df.py"
EXPORTERS_PATH = FEATURE4_DIR / "exporters.py"
DASHBOARDS_PATH = FEATURE4_DIR / "dashboards.py"
HTML_UTILS_PATH = FEATURE4_DIR / "html_utils.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench_mod = _load_module("benchmarking_mod", BENCH_PATH)
metrics_mod = _load_module("metrics_to_df_mod", METRICS_PATH)
exporters_mod = _load_module("exporters_mod", EXPORTERS_PATH)
dashboards_mod = _load_module("dashboards_mod", DASHBOARDS_PATH)
html_utils_mod = _load_module("html_utils_mod", HTML_UTILS_PATH)

# Functions under test
add_temporal_score = bench_mod.add_temporal_score
add_anomaly_score = bench_mod.add_anomaly_score
add_multimodal_health_score = bench_mod.add_multimodal_health_score
add_health_tiers = bench_mod.add_health_tiers

metrics_list_to_dataframe = metrics_mod.metrics_list_to_dataframe
export_dataframe = exporters_mod.export_dataframe
build_quality_dashboard = dashboards_mod.build_quality_dashboard
save_dashboard_html = html_utils_mod.save_dashboard_html


# ---------------------------------------------------------------------------
# Smoke test: does the basic reporting + benchmarking pipeline run end-to-end?
# ---------------------------------------------------------------------------

def test_feature4_smoke_end_to_end():
    """
    author: sayali
    reviewer: Xinxin
    category: smoke test
    """

    # Fake metrics for two sequences mimicking Feature 1 + 2 outputs
    metrics: List[dict] = [
        {
            "sequence": "seq_0001",
            # Feature 1-style metrics
            "temporal_offset_score": 0.95,
            "avg_drift_rate_ms_per_s": 0.1,
            "max_predicted_drift_ms": 1.2,
            "iso_8000_61_pass": 1.0,
            # Feature 2-style metrics
            "overall_health_score": 0.9,
            "mean_anomaly_ratio": 0.05,
        },
        {
            "sequence": "seq_0002",
            "temporal_offset_score": 0.7,
            "avg_drift_rate_ms_per_s": 0.8,
            "max_predicted_drift_ms": 5.0,
            "iso_8000_61_pass": 0.0,
            "overall_health_score": 0.6,
            "mean_anomaly_ratio": 0.3,
        },
    ]

    df = metrics_list_to_dataframe(metrics)
    assert list(df.columns)[0] == "sequence"
    assert len(df) == 2

    df = add_temporal_score(df)
    df = add_anomaly_score(df)
    df = add_multimodal_health_score(df)
    df = add_health_tiers(df)

    # Make sure key columns exist
    for col in [
        "temporal_score",
        "anomaly_score",
        "multimodal_health_score",
        "health_tier",
    ]:
        assert col in df.columns

    # Scores should be finite and within [0, 1]
    for col in ["temporal_score", "anomaly_score", "multimodal_health_score"]:
        assert np.all(np.isfinite(df[col]))
        assert np.all(df[col] >= 0.0)
        assert np.all(df[col] <= 1.0)

    # Health tier should be one of the allowed labels
    valid_tiers = {"excellent", "good", "fair", "poor"}
    assert set(df["health_tier"]).issubset(valid_tiers)


# ---------------------------------------------------------------------------
# One-shot test: single row → still produces reasonable scores
# ---------------------------------------------------------------------------

def test_add_temporal_score_one_shot():
    """
    author: sayali
    reviewer: Xinxin
    category: one-shot test
    """

    df = pd.DataFrame(
        [
            {
                "sequence": "single_seq",
                "temporal_offset_score": 0.8,
                "avg_drift_rate_ms_per_s": 0.2,
                "max_predicted_drift_ms": 2.0,
                "iso_8000_61_pass": 1.0,
            }
        ]
    )

    df = add_temporal_score(df)

    assert "temporal_score" in df.columns
    # With one row, normalization degenerates to neutral-ish value, but must be finite and in [0, 1]
    val = float(df.loc[0, "temporal_score"])
    assert np.isfinite(val)
    assert 0.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# Edge test: identical metrics across sequences (no variance)
# ---------------------------------------------------------------------------

def test_add_anomaly_score_edge_identical_rows():
    """
    author: sayali
    reviewer: Xinxin
    category: edge test
    """

    # Two identical rows => variance zero => our normalization should not blow up
    df = pd.DataFrame(
        [
            {
                "sequence": "seq_a",
                "overall_health_score": 0.7,
                "mean_anomaly_ratio": 0.2,
            },
            {
                "sequence": "seq_b",
                "overall_health_score": 0.7,
                "mean_anomaly_ratio": 0.2,
            },
        ]
    )

    df = add_anomaly_score(df)

    assert "anomaly_score" in df.columns
    # Both rows should end up with the same finite score
    scores = df["anomaly_score"].to_numpy()
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)
    assert np.allclose(scores[0], scores[1])


# ---------------------------------------------------------------------------
# Pattern test: ordering + reporting artifacts
# ---------------------------------------------------------------------------

def test_metrics_export_and_dashboard_pattern(tmp_path):
    """
    author: sayali
    reviewer: Xinxin
    category: pattern test
    justification: We want to ensure reporting follows a stable pattern:
                   sorted sequences, CSV + JSON being created, and
                   dashboard HTML file produced.
    """

    # Construct a small dataframe intentionally unsorted by "sequence"
    df = pd.DataFrame(
        [
            {"sequence": "seq_10", "multimodal_health_score": 0.2},
            {"sequence": "seq_02", "multimodal_health_score": 0.9},
            {"sequence": "seq_01", "multimodal_health_score": 0.7},
        ]
    )

    # Add dummy scores required by the dashboard
    df["temporal_score"] = [0.4, 0.9, 0.6]
    df["anomaly_score"] = [0.3, 0.2, 0.8]

    # For this pattern test we are not testing tier logic,
    # so we can assign a simple dummy tier.
    df["health_tier"] = ["good", "excellent", "poor"]

    # Enforce a consistent pattern in reporting: sort by sequence
    df_sorted = df.sort_values("sequence").reset_index(drop=True)
    assert list(df_sorted["sequence"]) == ["seq_01", "seq_02", "seq_10"]

    # Export CSV + JSON and check files exist and non-empty
    csv_path = tmp_path / "metrics.csv"
    json_path = tmp_path / "metrics.json"

    export_dataframe(df_sorted, str(csv_path), fmt="csv")
    export_dataframe(df_sorted, str(json_path), fmt="json")

    assert csv_path.exists()
    assert json_path.exists()
    assert csv_path.stat().st_size > 0
    assert json_path.stat().st_size > 0

    # Build dashboard and save HTML
    fig = build_quality_dashboard(df_sorted)
    html_path = tmp_path / "dashboard.html"
    save_dashboard_html(fig, str(html_path))

    assert html_path.exists()
    assert html_path.stat().st_size > 0
