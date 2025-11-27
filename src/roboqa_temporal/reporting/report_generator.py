"""

################################################################

File: roboqa_temporal/reporting/report_generator.py
Created: 2025-11-20
Created by: Archit Jain (architj@uw.edu)
Last Modified: 2025-11-20
Last Modified by: Archit Jain (architj@uw.edu)

#################################################################

Copyright: RoboQA-Temporal Authors
License: MIT License

################################################################

Report generation module for RoboQA-Temporal. Implements
functionality to create quality assessment reports in various
formats including Markdown, HTML, and CSV. Includes visualizations
of detected anomalies and frame statistics.

################################################################

"""

import os
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import json

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from roboqa_temporal.detection.detector import DetectionResult, Anomaly


class ReportGenerator:
    """
    Generates quality assessment reports in multiple formats.

    Supports:
    - Markdown reports
    - HTML reports with visualizations
    - CSV exports
    """

    def __init__(self, output_dir: str = "reports"):
        """
        Initialize report generator.

        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        result: DetectionResult,
        bag_path: str,
        output_format: str = "all",
        include_plots: bool = True,
    ) -> Dict[str, str]:
        """
        Generate reports in specified format(s).

        Args:
            result: DetectionResult from anomaly detection
            bag_path: Path to input bag file
            output_format: 'markdown', 'html', 'csv', or 'all'
            include_plots: Whether to include visualizations

        Returns:
            Dictionary mapping format names to output file paths
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(bag_path).stem

        output_files = {}

        if output_format in ["markdown", "all"]:
            md_path = self._generate_markdown(result, bag_path, timestamp, base_name)
            output_files["markdown"] = str(md_path)

        if output_format in ["html", "all"]:
            html_path = self._generate_html(
                result, bag_path, timestamp, base_name, include_plots
            )
            output_files["html"] = str(html_path)

        if output_format in ["csv", "all"]:
            csv_path = self._generate_csv(result, timestamp, base_name)
            output_files["csv"] = str(csv_path)

        return output_files

    def _generate_markdown(
        self, result: DetectionResult, bag_path: str, timestamp: str, base_name: str
    ) -> Path:
        """Generate markdown report."""
        output_path = self.output_dir / f"{base_name}_report_{timestamp}.md"

        with open(output_path, "w") as f:
            f.write(f"# Quality Assessment Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Bag File:** `{bag_path}`\n\n")
            f.write("---\n\n")

            # Executive Summary
            f.write("## Executive Summary\n\n")
            health_score = result.health_metrics.get("overall_health_score", 0.0)
            f.write(f"**Overall Health Score:** {health_score:.2%}\n\n")
            f.write(f"**Total Anomalies Detected:** {len(result.anomalies)}\n\n")
            f.write(f"**Total Frames Analyzed:** {len(result.frame_statistics)}\n\n")

            # Health Metrics
            f.write("## Health Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            for key, value in sorted(result.health_metrics.items()):
                if isinstance(value, float):
                    f.write(f"| {key} | {value:.4f} |\n")
                else:
                    f.write(f"| {key} | {value} |\n")
            f.write("\n")

            # Anomalies
            f.write("## Detected Anomalies\n\n")
            if result.anomalies:
                f.write("| Frame | Timestamp | Type | Severity | Description |\n")
                f.write("|-------|-----------|------|----------|-------------|\n")
                for anomaly in sorted(result.anomalies, key=lambda x: x.frame_index):
                    f.write(
                        f"| {anomaly.frame_index} | {anomaly.timestamp:.3f}s | "
                        f"{anomaly.anomaly_type} | {anomaly.severity:.2f} | "
                        f"{anomaly.description} |\n"
                    )
            else:
                f.write("No anomalies detected.\n")
            f.write("\n")

            # Frame Statistics
            f.write("## Frame Statistics\n\n")
            if result.frame_statistics:
                df = pd.DataFrame(result.frame_statistics)
                f.write(df.to_markdown(index=False))
                f.write("\n\n")

        return output_path

    def _generate_html(
        self,
        result: DetectionResult,
        bag_path: str,
        timestamp: str,
        base_name: str,
        include_plots: bool,
    ) -> Path:
        """Generate HTML report with visualizations."""
        output_path = self.output_dir / f"{base_name}_report_{timestamp}.html"

        # Generate plots if requested
        plot_paths = {}
        if include_plots:
            plot_paths = self._generate_plots(result, timestamp, base_name)

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Quality Assessment Report - {base_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; border-bottom: 2px solid #ddd; padding-bottom: 5px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #f9f9f9; border-radius: 5px; }}
        .severity-high {{ color: #d32f2f; font-weight: bold; }}
        .severity-medium {{ color: #f57c00; }}
        .severity-low {{ color: #388e3c; }}
        img {{ max-width: 100%; height: auto; margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>Quality Assessment Report</h1>
    <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><strong>Bag File:</strong> <code>{bag_path}</code></p>
    <hr>
    
    <h2>Executive Summary</h2>
    <div class="metric">
        <strong>Overall Health Score:</strong> 
        <span class="{'severity-high' if result.health_metrics.get('overall_health_score', 0) < 0.5 else 'severity-medium' if result.health_metrics.get('overall_health_score', 0) < 0.7 else 'severity-low'}">
            {result.health_metrics.get('overall_health_score', 0):.2%}
        </span>
    </div>
    <div class="metric">
        <strong>Total Anomalies:</strong> {len(result.anomalies)}
    </div>
    <div class="metric">
        <strong>Frames Analyzed:</strong> {len(result.frame_statistics)}
    </div>
    
    <h2>Health Metrics</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
"""

        for key, value in sorted(result.health_metrics.items()):
            if isinstance(value, float):
                html_content += f"<tr><td>{key}</td><td>{value:.4f}</td></tr>\n"
            else:
                html_content += f"<tr><td>{key}</td><td>{value}</td></tr>\n"

        html_content += """
    </table>
    
    <h2>Detected Anomalies</h2>
    <table>
        <tr><th>Frame</th><th>Timestamp</th><th>Type</th><th>Severity</th><th>Description</th></tr>
"""

        if result.anomalies:
            for anomaly in sorted(result.anomalies, key=lambda x: x.frame_index):
                severity_class = (
                    "severity-high"
                    if anomaly.severity > 0.7
                    else "severity-medium"
                    if anomaly.severity > 0.4
                    else "severity-low"
                )
                html_content += f"""
        <tr>
            <td>{anomaly.frame_index}</td>
            <td>{anomaly.timestamp:.3f}s</td>
            <td>{anomaly.anomaly_type}</td>
            <td class="{severity_class}">{anomaly.severity:.2f}</td>
            <td>{anomaly.description}</td>
        </tr>
"""
        else:
            html_content += "<tr><td colspan='5'>No anomalies detected.</td></tr>"

        html_content += """
    </table>
"""

        # Add plots if available
        if plot_paths:
            html_content += "<h2>Visualizations</h2>\n"
            for plot_name, plot_path in plot_paths.items():
                rel_path = os.path.relpath(plot_path, output_path.parent)
                html_content += f'<h3>{plot_name}</h3>\n<img src="{rel_path}" alt="{plot_name}">\n'

        html_content += """
</body>
</html>
"""

        with open(output_path, "w") as f:
            f.write(html_content)

        return output_path

    def _generate_csv(self, result: DetectionResult, timestamp: str, base_name: str) -> Path:
        """Generate CSV export."""
        output_path = self.output_dir / f"{base_name}_anomalies_{timestamp}.csv"

        # Export anomalies
        if result.anomalies:
            anomalies_data = []
            for anomaly in result.anomalies:
                row = {
                    "frame_index": anomaly.frame_index,
                    "timestamp": anomaly.timestamp,
                    "anomaly_type": anomaly.anomaly_type,
                    "severity": anomaly.severity,
                    "description": anomaly.description,
                }
                row.update(anomaly.metadata)
                anomalies_data.append(row)

            df = pd.DataFrame(anomalies_data)
            df.to_csv(output_path, index=False)
        else:
            # Create empty CSV with headers
            pd.DataFrame(columns=["frame_index", "timestamp", "anomaly_type", "severity", "description"]).to_csv(
                output_path, index=False
            )

        return output_path

    def _generate_plots(
        self, result: DetectionResult, timestamp: str, base_name: str
    ) -> Dict[str, str]:
        """Generate visualization plots."""
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        plot_paths = {}

        # Plot 1: Point count over time
        if result.frame_statistics:
            fig, ax = plt.subplots(figsize=(12, 6))
            df = pd.DataFrame(result.frame_statistics)
            ax.plot(df["frame_index"], df["num_points"], "b-", alpha=0.7)
            ax.set_xlabel("Frame Index")
            ax.set_ylabel("Number of Points")
            ax.set_title("Point Count Over Time")
            ax.grid(True, alpha=0.3)

            # Highlight anomalies
            if result.anomalies:
                anomaly_frames = [a.frame_index for a in result.anomalies]
                anomaly_points = [
                    df[df["frame_index"] == idx]["num_points"].values[0]
                    if len(df[df["frame_index"] == idx]) > 0
                    else 0
                    for idx in anomaly_frames
                ]
                ax.scatter(anomaly_frames, anomaly_points, c="red", s=100, alpha=0.7, label="Anomalies")

            # sns.histplot(df["num_points"], bins=30, kde=False, color="gray", alpha=0.3, ax=ax, label="Point Count Distribution")
            ax.legend()
            plt.tight_layout()
            plot_path = plots_dir / f"{base_name}_point_count_{timestamp}.png"
            plt.savefig(plot_path, dpi=150)
            plt.close()
            plot_paths["Point Count Over Time"] = str(plot_path)

        # Plot 2: Anomaly severity distribution
        if result.anomalies:
            fig, ax = plt.subplots(figsize=(10, 6))
            severities = [a.severity for a in result.anomalies]
            types = [a.anomaly_type for a in result.anomalies]

            df_anomalies = pd.DataFrame({"severity": severities, "type": types})
            df_anomalies.groupby("type")["severity"].plot(kind="hist", alpha=0.7, ax=ax, legend=True)
            ax.set_xlabel("Severity")
            ax.set_ylabel("Frequency")
            ax.set_title("Anomaly Severity Distribution by Type")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_path = plots_dir / f"{base_name}_severity_{timestamp}.png"
            plt.savefig(plot_path, dpi=150)
            plt.close()
            plot_paths["Anomaly Severity Distribution"] = str(plot_path)

        return plot_paths

