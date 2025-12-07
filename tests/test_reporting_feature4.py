"""
################################################################
File: roboqa_temporal/tests/test_reporting_feature4.py
Author: Sayali Nehul
Reviewer: Xinxin 
################################################################

Feature 4 — Dataset Quality Scoring & Cross-Benchmarking Tests

This test module validates the correctness, stability, and robustness of the
Feature 4 reporting pipeline:

- Converting raw metrics (list[dict]) into a pandas.DataFrame
- Exporting metrics to CSV / JSON
- Building a quality dashboard figure
- Saving the dashboard to an HTML file

All inputs are fully synthetic and are constructed directly inside the tests.
No Feature 1 (temporal synchronization), Feature 2 (anomaly detection), ROS2,
bag files, or real datasets are imported or executed.

Test coverage includes:
- Unit-level smoke tests for the core Feature 4 reporting helpers
- One-shot tests for minimal valid inputs
- Edge tests for heterogeneous / missing metrics fields
- Pattern tests for reporting artifacts (sorted output, CSV/JSON export,
  and dashboard HTML generation)

################################################################
"""

from __future__ import annotations

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


metrics_mod = _load_module("metrics_to_df_mod", METRICS_PATH)
exporters_mod = _load_module("exporters_mod", EXPORTERS_PATH)
dashboards_mod = _load_module("dashboards_mod", DASHBOARDS_PATH)
html_utils_mod = _load_module("html_utils_mod", HTML_UTILS_PATH)

# Functions under test (reporting-only)
metrics_list_to_dataframe = metrics_mod.metrics_list_to_dataframe
export_dataframe = exporters_mod.export_dataframe
build_quality_dashboard = dashboards_mod.build_quality_dashboard
save_dashboard_html = html_utils_mod.save_dashboard_html


# ---------------------------------------------------------------------------
# Smoke test: minimal metrics → DataFrame
# ---------------------------------------------------------------------------

def test_metrics_list_to_dataframe_smoke():
    """
    author: sayali
    reviewer: Xinxin
    category: smoke test
    justification: Unit-level smoke test for converting a small metrics
                   list into a well-formed DataFrame.
    """
    metrics: List[dict] = [
        {"sequence": "seq_0001", "multimodal_health_score": 0.9},
        {"sequence": "seq_0002", "multimodal_health_score": 0.6},
    ]

    df = metrics_list_to_dataframe(metrics)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns)[0] == "sequence"
    assert len(df) == 2
    assert "multimodal_health_score" in df.columns


# ---------------------------------------------------------------------------
# One-shot test: single-row metrics list
# ---------------------------------------------------------------------------

def test_metrics_list_to_dataframe_one_shot():
    """
    author: sayali
    reviewer: Xinxin
    category: one-shot test
    justification: Ensure a single-metric input still results in a valid
                   DataFrame with expected columns and finite values.
    """
    metrics: List[dict] = [
        {
            "sequence": "single_seq",
            "multimodal_health_score": 0.75,
            "temporal_score": 0.8,
            "anomaly_score": 0.6,
        }
    ]

    df = metrics_list_to_dataframe(metrics)

    assert isinstance(df, pd.DataFrame)
    assert list(df["sequence"]) == ["single_seq"]
    for col in ["multimodal_health_score", "temporal_score", "anomaly_score"]:
        assert col in df.columns
        val = float(df.loc[0, col])
        assert np.isfinite(val)


# ---------------------------------------------------------------------------
# Edge test: heterogeneous keys / missing fields
# ---------------------------------------------------------------------------

def test_metrics_list_to_dataframe_edge_missing_fields():
    """
    author: sayali
    reviewer: Xinxin
    category: edge test
    justification: Mixed metrics dictionaries (different keys) should still
                   produce a DataFrame without crashing; missing values can
                   appear as NaN but the structure must be consistent.
    """
    metrics: List[dict] = [
        {
            "sequence": "seq_a",
            "multimodal_health_score": 0.9,
            "temporal_score": 0.8,
        },
        {
            "sequence": "seq_b",
            # no temporal_score here on purpose
            "multimodal_health_score": 0.4,
            "anomaly_score": 0.3,
        },
    ]

    df = metrics_list_to_dataframe(metrics)

    assert isinstance(df, pd.DataFrame)
    # Both sequences present
    assert set(df["sequence"]) == {"seq_a", "seq_b"}
    # Columns should be the union of keys
    for col in [
        "sequence",
        "multimodal_health_score",
        "temporal_score",
        "anomaly_score",
    ]:
        assert col in df.columns

    # At least one NaN is expected due to heterogeneous keys
    assert df.isna().sum().sum() >= 1


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
