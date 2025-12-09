"""
################################################################

File: roboqa_temporal/health_reporting/dashboard.py
Created: 2025-12-08
Created by: RoboQA-Temporal Authors
Last Modified: 2025-12-08
Last Modified by: RoboQA-Temporal Authors

#################################################################

Copyright: RoboQA-Temporal Authors
License: MIT License

################################################################

Dashboard and visualization builder for dataset health metrics.

Generates:
- Interactive Plotly HTML dashboards
- PNG bar charts of quality scores
- Quality dimension status tables

################################################################

"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def build_dashboard_html(
    df_per_sensor: pd.DataFrame,
    df_per_sequence: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Build an interactive HTML dashboard summarizing quality metrics.
    
    Args:
        df_per_sensor: DataFrame with per-sensor metrics
        df_per_sequence: DataFrame with per-sequence aggregated metrics
        output_path: Path to save HTML dashboard
    """
    if not HAS_PLOTLY:
        print("[WARN] plotly not available; skipping interactive dashboard")
        return

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
            marker=dict(color="rgb(99, 110, 250)"),
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
            marker=dict(color="rgb(239, 85, 59)"),
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
            marker=dict(color="rgb(0, 204, 150)"),
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
            marker=dict(color="rgb(171, 99, 250)"),
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
            marker=dict(color="rgb(255, 161, 90)"),
        ),
        row=2,
        col=2,
    )

    # Axis labels
    fig.update_yaxes(title_text="Health (0-1)", row=1, col=1)
    fig.update_yaxes(title_text="Dim. score (0-1)", row=1, col=2)
    fig.update_yaxes(title_text="Dim. score (0-1)", row=2, col=1)
    fig.update_yaxes(title_text="Score (0-1)", row=2, col=2)

    for r in [1, 2]:
        for c in [1, 2]:
            fig.update_xaxes(title_text="Sequence", row=r, col=c)

    fig.update_layout(
        title="Dataset Health Scoring & Quality Assessment",
        height=900,
        showlegend=True,
        font=dict(color="black", size=12),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
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
    ]
    
    metrics_cols = [col for col in metrics_cols if col in df_per_sequence.columns]
    
    table_df = df_per_sequence[metrics_cols].round(3)
    table_html = table_df.to_html(index=False, classes="metrics-table", border=0)

    # Dimension status table
    dims_rows = [
        {
            "name": "Timeliness",
            "sub": (
                "Timestamp validity; temporal regularity; latency; "
                "inter-frame gap consistency."
            ),
            "status": (
                "IMPLEMENTED - temporal regularity and anomaly score for all sensors; "
                "LiDAR scan-duration stability from KITTI timestamps_start/end when available."
            ),
        },
        {
            "name": "Completeness",
            "sub": (
                "Topic/message availability; temporal coverage gaps; "
                "sensor dropout rate; frame availability vs best sensor."
            ),
            "status": (
                "IMPLEMENTED - message availability and dropout rate "
                "computed from timestamps; availability vs best sensor in sequence."
            ),
        },
        {
            "name": "Accuracy",
            "sub": (
                "Sensor measurement accuracy; positional accuracy (GPS/RTK); "
                "annotation / inter-rater accuracy."
            ),
            "status": (
                "Not evaluated in this run - requires ground-truth labels or "
                "reference trajectories beyond timestamps."
            ),
        },
        {
            "name": "Consistency",
            "sub": (
                "Cross-sensor consistency; temporal consistency via motion model; "
                "semantic consistency."
            ),
            "status": (
                "PARTIALLY IMPLEMENTED - temporal consistency from timestamps; "
                "full sensor-fusion and semantic checks planned."
            ),
        },
        {
            "name": "Relevance",
            "sub": (
                "Scene diversity; edge-case coverage (night/rain/occlusion); "
                "distribution similarity to deployment environment."
            ),
            "status": (
                "Not evaluated in this run - requires labels or image/content analysis."
            ),
        },
        {
            "name": "Sensor Fusion Quality",
            "sub": (
                "Data fusion confidence; redundancy utilization; "
                "complementarity score for each sensor."
            ),
            "status": (
                "Not evaluated - requires fusion pipeline outputs beyond raw timestamps."
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
  <title>Dataset Health Scoring & Quality Assessment</title>
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
      text-align: left;
    }}
    table.metrics-table th,
    table.dim-table th {{
      background: #f5f5f5;
      font-weight: 600;
    }}
    table.metrics-table tr:nth-child(even),
    table.dim-table tr:nth-child(even) {{
      background-color: #f9f9f9;
    }}
    .plotly-container {{
      margin-top: 16px;
      background: white;
      padding: 10px;
      border-radius: 4px;
    }}
  </style>
</head>
<body>
  <div class="header">
    <h1>Dataset Health Scoring & Quality Assessment</h1>
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
    <div class="plotly-container">
      {fig_html}
    </div>

    <h2>Aggregated Metrics per Sequence</h2>
    {table_html}

    <h2>Quality Dimensions & Implementation Status</h2>
    <p>
      This section summarizes the dataset quality dimensions being assessed.
      The <strong>Status</strong> column indicates what is implemented in this run,
      which currently focuses on timestamp-based metrics for KITTI sequences.
    </p>
    {dims_table_html}
  </div>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[INFO] Saved dashboard HTML: {output_path}")


def plot_quality_scores(
    df_per_sensor: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Create a simple bar plot of overall quality scores per sensor/topic.
    
    Args:
        df_per_sensor: DataFrame with per-sensor metrics
        output_path: Path to save PNG plot
    """
    labels = df_per_sensor["sequence"] + " / " + df_per_sensor["sensor_or_topic"]
    scores_plot = df_per_sensor["overall_quality_score_0_100"]

    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))
    
    colors = ["#2ecc71" if s >= 85 else "#f39c12" if s >= 70 else "#e74c3c" for s in scores_plot]
    
    plt.title("Dataset Health Scores (per sensor/topic)", fontsize=16, fontweight="bold")
    bars = plt.bar(range(len(labels)), scores_plot, color=colors, edgecolor="black", linewidth=0.5)
    
    for i, (bar, score) in enumerate(zip(bars, scores_plot)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.0f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    plt.ylabel("Overall Quality Score (0-100)", fontsize=12)
    plt.ylim(0.0, 105.0)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', edgecolor='black', label='Excellent (≥85)'),
        Patch(facecolor='#f39c12', edgecolor='black', label='Good/Fair (70-85)'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='Poor (<70)'),
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved plot: {output_path}")


def plot_dimension_comparison(
    df_per_sequence: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Create a dimension comparison plot showing timeliness and completeness per sequence.
    
    Args:
        df_per_sequence: DataFrame with per-sequence aggregated metrics
        output_path: Path to save PNG plot
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(df_per_sequence))
    width = 0.35
    
    timeliness = df_per_sequence["dim_timeliness"]
    completeness = df_per_sequence["dim_completeness"]
    
    bars1 = ax.bar(x - width/2, timeliness, width, label="Timeliness", 
                   color="#3498db", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width/2, completeness, width, label="Completeness",
                   color="#e74c3c", edgecolor="black", linewidth=0.5)
    
    ax.set_ylabel("Dimension Score (0-1)", fontsize=12)
    ax.set_title("Quality Dimensions Comparison per Sequence", fontsize=16, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(df_per_sequence["sequence"], rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved dimension comparison plot: {output_path}")


def plot_health_distribution(
    df_per_sequence: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Create a distribution plot of health tiers across sequences.
    
    Args:
        df_per_sequence: DataFrame with per-sequence aggregated metrics
        output_path: Path to save PNG plot
    """
    sns.set_style("whitegrid")
    
    # Count health tiers
    tier_counts = df_per_sequence["health_tier"].value_counts()
    tier_order = ["excellent", "good", "fair", "poor"]
    tier_counts = tier_counts.reindex(tier_order, fill_value=0)
    
    colors_map = {
        "excellent": "#2ecc71",
        "good": "#3498db",
        "fair": "#f39c12",
        "poor": "#e74c3c",
    }
    colors = [colors_map.get(tier, "#95a5a6") for tier in tier_counts.index]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(tier_counts.index, tier_counts.values, color=colors, edgecolor="black", linewidth=1)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}',
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel("Number of Sequences", fontsize=12)
    ax.set_title("Health Tier Distribution Across Sequences", fontsize=16, fontweight="bold")
    ax.set_ylim(0, max(tier_counts.values) * 1.15)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved health distribution plot: {output_path}")
