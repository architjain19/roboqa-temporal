"""
################################################################

File: roboqa_temporal/health_reporting/exporters.py
Created: 2025-12-08
Created by: RoboQA-Temporal Authors
Last Modified: 2025-12-08
Last Modified by: RoboQA-Temporal Authors

#################################################################

Copyright: RoboQA-Temporal Authors
License: MIT License

################################################################

Export functionality for dataset health metrics.

Supports exporting metrics to:
- CSV (tabular format)
- JSON (structured format)
- YAML (human-readable format)

################################################################

"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any

import pandas as pd

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def export_csv(
    df_per_sensor: pd.DataFrame,
    df_per_sequence: pd.DataFrame,
    output_dir: str,
) -> Dict[str, str]:
    """
    Export metrics to CSV files.
    
    Args:
        df_per_sensor: DataFrame with per-sensor metrics
        df_per_sequence: DataFrame with per-sequence aggregated metrics
        output_dir: Directory to save CSV files
        
    Returns:
        Dictionary mapping format names to file paths
    """
    output_files = {}
    
    # Per-sensor CSV
    csv_path = os.path.join(output_dir, "health_metrics.csv")
    df_per_sensor.to_csv(csv_path, index=False)
    output_files["csv"] = str(csv_path)
    print(f"[INFO] Saved metrics CSV: {csv_path}")

    # Per-sequence aggregated CSV
    agg_csv = os.path.join(output_dir, "health_metrics_by_sequence.csv")
    df_per_sequence.to_csv(agg_csv, index=False)
    output_files["csv_aggregated"] = str(agg_csv)
    print(f"[INFO] Saved aggregated per-sequence CSV: {agg_csv}")
    
    return output_files


def export_json(
    df_per_sequence: pd.DataFrame,
    output_dir: str,
) -> str:
    """
    Export metrics to JSON file.
    
    Args:
        df_per_sequence: DataFrame with per-sequence aggregated metrics
        output_dir: Directory to save JSON file
        
    Returns:
        Path to JSON file
    """
    records = df_per_sequence.to_dict(orient="records")
    
    # Converting NaN to None for JSON serialization
    for record in records:
        for key, value in record.items():
            if pd.isna(value):
                record[key] = None
    
    report = {
        "feature": "dataset_health_assessment",
        "version": "1.0.0",
        "num_sequences": len(df_per_sequence),
        "sequences": records,
    }

    json_path = os.path.join(output_dir, "health_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    
    print(f"[INFO] Saved JSON report: {json_path}")
    return json_path


def export_yaml(
    df_per_sequence: pd.DataFrame,
    output_dir: str,
) -> str | None:
    """
    Export metrics to YAML file.
    
    Args:
        df_per_sequence: DataFrame with per-sequence aggregated metrics
        output_dir: Directory to save YAML file
        
    Returns:
        Path to YAML file, or None if PyYAML not available
    """
    if not HAS_YAML:
        print("[WARN] PyYAML not installed; skipping YAML export.")
        return None
    
    records = df_per_sequence.to_dict(orient="records")
    
    # Converting NaN to None for YAML serialization
    for record in records:
        for key, value in record.items():
            if pd.isna(value):
                record[key] = None
    
    report = {
        "feature": "dataset_health_assessment",
        "version": "1.0.0",
        "num_sequences": len(df_per_sequence),
        "sequences": records,
    }

    yaml_path = os.path.join(output_dir, "health_report.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(report, f, sort_keys=False, default_flow_style=False)
    
    print(f"[INFO] Saved YAML report: {yaml_path}")
    return yaml_path


def export_all(
    df_per_sensor: pd.DataFrame,
    df_per_sequence: pd.DataFrame,
    output_dir: str,
) -> Dict[str, str]:
    """
    Export metrics to all supported formats (CSV, JSON, YAML).
    
    Args:
        df_per_sensor: DataFrame with per-sensor metrics
        df_per_sequence: DataFrame with per-sequence aggregated metrics
        output_dir: Directory to save all reports
        
    Returns:
        Dictionary mapping format names to file paths
    """
    output_files = {}
    
    # CSV export
    csv_files = export_csv(df_per_sensor, df_per_sequence, output_dir)
    output_files.update(csv_files)
    
    # JSON export
    json_file = export_json(df_per_sequence, output_dir)
    output_files["json"] = json_file
    
    # YAML export
    yaml_file = export_yaml(df_per_sequence, output_dir)
    if yaml_file:
        output_files["yaml"] = yaml_file
    
    return output_files


def create_summary_report(
    df_per_sequence: pd.DataFrame,
    output_dir: str,
) -> str:
    """
    Create a human-readable text summary report.
    
    Args:
        df_per_sequence: DataFrame with per-sequence aggregated metrics
        output_dir: Directory to save summary report
        
    Returns:
        Path to summary report file
    """
    summary_path = os.path.join(output_dir, "health_summary.txt")
    
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("DATASET HEALTH ASSESSMENT SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Sequences: {len(df_per_sequence)}\n")
        f.write(f"Mean Quality Score: {df_per_sequence['overall_quality_score_0_100'].mean():.1f} / 100\n")
        f.write(f"Min Quality Score: {df_per_sequence['overall_quality_score_0_100'].min():.1f} / 100\n")
        f.write(f"Max Quality Score: {df_per_sequence['overall_quality_score_0_100'].max():.1f} / 100\n")
        f.write("\n")
        
        # Health tier distribution
        f.write("HEALTH TIER DISTRIBUTION\n")
        f.write("-" * 80 + "\n")
        tier_counts = df_per_sequence["health_tier"].value_counts()
        for tier in ["excellent", "good", "fair", "poor"]:
            count = tier_counts.get(tier, 0)
            pct = 100.0 * count / len(df_per_sequence)
            f.write(f"  {tier.upper():12s}: {count:3d} sequences ({pct:5.1f}%)\n")
        f.write("\n")
        
        # Best and worst sequences
        f.write("BEST AND WORST SEQUENCES\n")
        f.write("-" * 80 + "\n")
        best_idx = df_per_sequence["overall_quality_score"].idxmax()
        worst_idx = df_per_sequence["overall_quality_score"].idxmin()
        f.write(f"Best:  {df_per_sequence.loc[best_idx, 'sequence']} "
                f"({df_per_sequence.loc[best_idx, 'overall_quality_score_0_100']:.1f}/100)\n")
        f.write(f"Worst: {df_per_sequence.loc[worst_idx, 'sequence']} "
                f"({df_per_sequence.loc[worst_idx, 'overall_quality_score_0_100']:.1f}/100)\n")
        f.write("\n")
        
        # Dimension statistics
        f.write("DIMENSION STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Mean Timeliness Score: {df_per_sequence['dim_timeliness'].mean():.3f}\n")
        f.write(f"Mean Completeness Score: {df_per_sequence['dim_completeness'].mean():.3f}\n")
        f.write("\n")
        
        # Detailed per-sequence breakdown
        f.write("DETAILED METRICS PER SEQUENCE\n")
        f.write("-" * 80 + "\n")
        for _, row in df_per_sequence.iterrows():
            f.write(f"\nSequence: {row['sequence']}\n")
            f.write(f"  Overall Quality:     {row['overall_quality_score_0_100']:.1f}/100\n")
            f.write(f"  Health Tier:         {row['health_tier'].upper()}\n")
            f.write(f"  Timeliness:          {row['dim_timeliness']:.3f}\n")
            f.write(f"  Completeness:        {row['dim_completeness']:.3f}\n")
            f.write(f"  Temporal Score:      {row['temporal_score']:.3f}\n")
            f.write(f"  Anomaly Score:       {row['anomaly_score']:.3f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"[INFO] Saved summary report: {summary_path}")
    return summary_path
