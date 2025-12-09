"""
################################################################
File: roboqa_temporal/reporting/dashboards.py
Created: 2025-12-07
Created by: Sayali Nehul
Last Modified: 2025-12-07
Last Modified by: Sayali Nehul
#################################################################
4-Panel Interactive Quality Dashboard Generator.
Builds a 2×2 Plotly dashboard used by Feature 4 to visualize dataset
quality across multiple dimensions. The dashboard includes:
(1) Dataset-level radar chart of core quality dimensions
(2) Per-sequence overall quality score bar chart
(3) Cross-benchmark percentile comparison
(4) Sensor-fusion quality metric summary
This tool supports dataset triage, curation prioritization,
and ISO-8000-aligned quality reporting for multi-sensor systems.
################################################################
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_quality_dashboard(df: pd.DataFrame) -> go.Figure:
    """
    Build a multi-panel dashboard for Feature 4.

    Required columns:
      - sequence
      - temporal_score
      - anomaly_score
      - multimodal_health_score
      - health_tier
    """
    required = [
        "sequence",
        "temporal_score",
        "anomaly_score",
        "multimodal_health_score",
        "health_tier",
    ]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column {col}")

    # Make sure sequences are ordered nicely
    df = df.sort_values("sequence").reset_index(drop=True)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Multi-Modal Health per Sequence",
            "Temporal Score per Sequence",
            "Anomaly Score per Sequence",
            "Temporal vs Anomaly Scores",
        ),
        horizontal_spacing=0.12,
        vertical_spacing=0.18,
    )

    # ------------------------------------------------------------------ #
    # 1) Multi-modal health (bar)
    # ------------------------------------------------------------------ #
    fig.add_trace(
        go.Bar(
            x=df["sequence"],
            y=df["multimodal_health_score"],
            name="Multi-Modal Health",
            showlegend=False,  # legend only in bottom-right plot
        ),
        row=1,
        col=1,
    )

    # ------------------------------------------------------------------ #
    # 2) Temporal score (line)
    # ------------------------------------------------------------------ #
    fig.add_trace(
        go.Scatter(
            x=df["sequence"],
            y=df["temporal_score"],
            mode="lines+markers",
            name="Temporal score",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # ------------------------------------------------------------------ #
    # 3) Anomaly score (line)
    # ------------------------------------------------------------------ #
    fig.add_trace(
        go.Scatter(
            x=df["sequence"],
            y=df["anomaly_score"],
            mode="lines+markers",
            name="Anomaly score",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # ------------------------------------------------------------------ #
    # 4) Combined temporal vs anomaly (with legend)
    # ------------------------------------------------------------------ #
    fig.add_trace(
        go.Scatter(
            x=df["sequence"],
            y=df["temporal_score"],
            mode="lines+markers",
            name="Temporal score",
            showlegend=True,
        ),
        row=2,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=df["sequence"],
            y=df["anomaly_score"],
            mode="lines+markers",
            name="Anomaly score",
            showlegend=True,
        ),
        row=2,
        col=2,
    )

    # ------------------------------------------------------------------ #
    # Layout
    # ------------------------------------------------------------------ #
    fig.update_xaxes(title_text="Sequence", row=1, col=1)
    fig.update_xaxes(title_text="Sequence", row=1, col=2)
    fig.update_xaxes(title_text="Sequence", row=2, col=1)
    fig.update_xaxes(title_text="Sequence", row=2, col=2)

    fig.update_yaxes(title_text="Health (0–1)", row=1, col=1)
    fig.update_yaxes(title_text="Temporal score", row=1, col=2)
    fig.update_yaxes(title_text="Anomaly score", row=2, col=1)
    fig.update_yaxes(title_text="Score (0–1)", row=2, col=2)

    fig.update_layout(
        title="Feature 4 – Multi-Modal Benchmarking Dashboard",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=40, r=40, t=80, b=80),
    )

    return fig
