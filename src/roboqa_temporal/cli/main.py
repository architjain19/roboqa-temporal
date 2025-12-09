"""

################################################################

File: roboqa_temporal/cli/main.py
Created: 2025-11-20
Created by: Archit Jain (architj@uw.edu)
Last Modified: 2025-12-08
Last Modified by: Archit Jain (architj@uw.edu)

#################################################################

Copyright: RoboQA-Temporal Authors
License: MIT License

################################################################

Basic usage example for RoboQA-Temporal.

################################################################

"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Optional
import yaml
import numpy as np

from roboqa_temporal.loader import BagLoader
from roboqa_temporal.preprocessing import Preprocessor
from roboqa_temporal.detection import AnomalyDetector
from roboqa_temporal.reporting import ReportGenerator
from roboqa_temporal.synchronization import TemporalSyncValidator
from roboqa_temporal.fusion import CalibrationQualityValidator
from roboqa_temporal.health_reporting import (
    run_health_check,
    generate_curation_recommendations,
)
from roboqa_temporal.health_reporting.dashboard import (
    build_dashboard_html,
    plot_quality_scores,
    plot_dimension_comparison,
    plot_health_distribution,
)
from roboqa_temporal.health_reporting.exporters import (
    export_csv,
    export_json,
    export_yaml,
    create_summary_report,
)
from roboqa_temporal.health_reporting.curation import (
    generate_curation_report,
    generate_curation_json,
    get_sequences_to_exclude,
    get_sequences_for_review,
)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RoboQA-Temporal: Automated quality assessment for ROS2 bag files and multi-sensor datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Anomaly detection on ROS2 bag
  roboqa anomaly bag_file.db3

  # Synchronization analysis on KITTI dataset
  roboqa sync dataset/2011_09_26_drive_0005_sync/

  # Camera-LiDAR fusion quality assessment
  roboqa fusion dataset/2011_09_26_drive_0005_sync/

  # Dataset health assessment and quality dashboard
  roboqa health dataset/

  # With configuration file
  roboqa anomaly bag_file.db3 --config config.yaml

  # Specify output format
  roboqa sync dataset/ --output html --output-dir reports/

  # Limit number of frames
  roboqa anomaly bag_file.db3 --max-frames 1000
        """,
    )

    parser.add_argument(
        "mode",
        type=str,
        choices=["anomaly", "sync", "fusion", "health"],
        help="Operation mode: 'anomaly' for anomaly detection, 'sync' for synchronization analysis, 'fusion' for camera-LiDAR fusion quality, 'health' for dataset quality assessment",
    )

    parser.add_argument(
        "input_path",
        type=str,
        help="Path to ROS2 bag file (for anomaly mode) or dataset folder (for sync mode)",
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration YAML file",
    )

    parser.add_argument(
        "--topics",
        nargs="+",
        help="Point cloud topics to analyze (default: auto-detect)",
    )

    parser.add_argument(
        "--output",
        choices=["markdown", "html", "csv", "all"],
        default="all",
        help="Output format (default: all)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Output directory for reports (default: reports)",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        help="Maximum number of frames to process",
    )

    parser.add_argument(
        "--voxel-size",
        type=float,
        help="Voxel size for downsampling (default: no downsampling). "
             "Recommended for large point clouds (e.g., 0.1 for KITTI)",
    )

    parser.add_argument(
        "--no-outlier-removal",
        action="store_true",
        help="Disable outlier removal",
    )

    parser.add_argument(
        "--max-points-for-outliers",
        type=int,
        default=50000,
        help="Maximum number of points for outlier removal (default: 50000). "
             "Prevents memory issues with large point clouds",
    )

    parser.add_argument(
        "--disable-detector",
        nargs="+",
        choices=["density", "spatial", "ghost", "temporal"],
        help="Disable specific detectors",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        help="Global threshold for anomaly detection (0-1)",
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plot generation in HTML reports",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Load configuration if provided
    config = {}
    if args.config:
        if not Path(args.config).exists():
            print(f"Error: Configuration file not found: {args.config}")
            sys.exit(1)
        config = load_config(args.config)

    # Merge CLI arguments with config (CLI takes precedence)
    topics = args.topics or config.get("topics")
    max_frames = args.max_frames or config.get("max_frames")
    voxel_size = args.voxel_size or config.get("preprocessing", {}).get("voxel_size")
    remove_outliers = not args.no_outlier_removal and config.get("preprocessing", {}).get("remove_outliers", True)
    max_points_for_outliers = args.max_points_for_outliers or config.get("preprocessing", {}).get("max_points_for_outliers", 50000)
    output_format = args.output or config.get("output", "all")
    output_dir = args.output_dir or config.get("output_dir", "reports")
    include_plots = not args.no_plots

    # Detector configuration
    disabled_detectors = args.disable_detector or config.get("detection", {}).get("disabled", [])
    threshold = args.threshold or config.get("detection", {}).get("threshold", 0.5)

    enable_density = "density" not in disabled_detectors
    enable_spatial = "spatial" not in disabled_detectors
    enable_ghost = "ghost" not in disabled_detectors
    enable_temporal = "temporal" not in disabled_detectors

    # Validate input path
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: Input path not found: {args.input_path}")
        sys.exit(1)

    mode = args.mode
    print("=" * 60)
    print("RoboQA-Temporal: Quality Assessment Tool")
    print("=" * 60)
    print(f"Mode: {mode.upper()}")
    print(f"Input: {input_path}")
    print(f"Output directory: {output_dir}")
    print()

    try:
        if mode == "health":
            run_health_analysis(
                input_path,
                output_dir,
                output_format,
                include_plots,
                args.verbose,
            )
        elif mode == "sync":
            run_sync_analysis(
                input_path,
                output_dir,
                output_format,
                max_frames,
                include_plots,
                args.verbose,
            )
        elif mode == "fusion":
            run_fusion_analysis(
                input_path,
                output_dir,
                output_format,
                max_frames,
                include_plots,
                args.verbose,
            )
        else:  # anomaly mode
            run_anomaly_detection(
                input_path,
                output_dir,
                output_format,
                topics,
                max_frames,
                voxel_size,
                remove_outliers,
                max_points_for_outliers,
                enable_density,
                enable_spatial,
                enable_ghost,
                enable_temporal,
                threshold,
                include_plots,
                args.verbose,
            )

    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def run_sync_analysis(
    dataset_path: Path,
    output_dir: str,
    output_format: str,
    max_frames: Optional[int],
    include_plots: bool,
    verbose: bool,
) -> None:
    """Run synchronization analysis on a multi-sensor dataset."""
    print("Running Cross-Modal Synchronization Analysis...")
    print()

    # Initialize validator
    validator = TemporalSyncValidator(
        output_dir=output_dir,
        report_formats=(output_format,) if output_format != "all" else ("markdown", "html", "csv"),
        auto_export_reports=True,
    )

    if verbose:
        print(f"Loading sensor streams from: {dataset_path}")

    # Run validation
    report = validator.validate(
        str(dataset_path),
        max_frames=max_frames,
        include_visualizations=include_plots,
    )

    print()
    print("=" * 60)
    print("Synchronization Analysis Complete!")
    print("=" * 60)
    
    # Display summary
    print("\nSynchronization Quality:")
    sync_score = report.metrics.get("synchronization_quality_score", 0.0)
    print(f"  Overall Score: {sync_score:.2%}")
    print(f"  Temporal Offset Score: {report.metrics.get('temporal_offset_score', 0.0):.2%}")
    print(f"  Avg Drift Rate: {report.metrics.get('avg_drift_rate_ms_per_s', 0.0):.4f} ms/s")
    
    print("\nData Quality:")
    print(f"  Total Sensor Streams: {len(report.streams)}")
    print(f"  Missing Frames: {int(report.metrics.get('total_missing_frames', 0))}")
    print(f"  Duplicate Timestamps: {int(report.metrics.get('total_duplicate_frames', 0))}")
    
    print("\nGenerated Reports:")
    for format_name, file_path in report.report_files.items():
        print(f"  {format_name.upper()}: {file_path}")
    
    if report.parameter_file:
        print(f"\nTimestamp Corrections: {report.parameter_file}")
    
    if report.recommendations:
        print(f"\nRecommendations: {len(report.recommendations)} issue(s) found")
        if verbose:
            for rec in report.recommendations[:5]:
                print(f"  - {rec}")
            if len(report.recommendations) > 5:
                print(f"  ... and {len(report.recommendations) - 5} more (see report)")


def run_fusion_analysis(
    dataset_path: Path,
    output_dir: str,
    output_format: str,
    max_frames: Optional[int],
    include_plots: bool,
    verbose: bool,
) -> None:
    """Run camera-LiDAR fusion quality analysis on a multi-sensor dataset."""
    print("Running Camera-LiDAR Fusion Quality Assessment...")
    print()

    # Initializing validator
    validator = CalibrationQualityValidator(output_dir=output_dir)

    if verbose:
        print(f"Loading dataset from: {dataset_path}")

    # Running validation
    report = validator.analyze_dataset(
        str(dataset_path),
        max_frames=max_frames,
        include_visualizations=include_plots,
    )

    print()
    print("=" * 60)
    print("Fusion Quality Assessment Complete!")
    print("=" * 60)

    # Displaying summary
    print("\nCalibration Quality:")
    for pair_name, result in report.pair_results.items():
        status = "PASS" if result.overall_pass else "FAIL"
        print(f"  {pair_name}: {status}")
        print(f"    Edge Alignment: {result.geom_edge_score:.3f}")
        print(f"    Mutual Information: {result.mutual_information:.3f}")

    print("\nProjection Error Analysis:")
    if report.projection_errors:
        mean_error = np.mean([e.reprojection_error for e in report.projection_errors])
        max_error = np.max([e.reprojection_error for e in report.projection_errors])
        increasing = sum(1 for e in report.projection_errors if e.error_trend == "increasing")
        print(f"  Mean Error: {mean_error:.3f}")
        print(f"  Max Error: {max_error:.3f}")
        print(f"  Frames with Increasing Error: {increasing}")

    print("\nIllumination Changes:")
    if report.illumination_changes:
        mean_brightness = np.mean([i.brightness_mean for i in report.illumination_changes])
        light_changes = sum(1 for i in report.illumination_changes if i.light_source_change)
        print(f"  Mean Brightness: {mean_brightness:.1f}")
        print(f"  Light Source Changes: {light_changes}")

    print("\nMoving Object Detection:")
    if report.moving_objects:
        mean_objects = np.mean([o.detected_objects for o in report.moving_objects])
        mean_confidence = np.mean([o.detection_confidence for o in report.moving_objects])
        mean_fusion_quality = np.mean([o.fusion_quality_score for o in report.moving_objects])
        print(f"  Mean Detected Objects: {mean_objects:.1f}")
        print(f"  Mean Detection Confidence: {mean_confidence:.2f}")
        print(f"  Mean Fusion Quality Score: {mean_fusion_quality:.2f}")

    print("\nGenerated Reports:")
    if report.parameter_file:
        print(f"  YAML Parameters: {report.parameter_file}")
    if report.html_report_file:
        print(f"  HTML Report: {report.html_report_file}")

    if report.recommendations:
        print(f"\nRecommendations: {len(report.recommendations)} issue(s) found")
        if verbose:
            for rec in report.recommendations[:5]:  # Show first 5
                print(f"  - {rec}")
            if len(report.recommendations) > 5:
                print(f"  ... and {len(report.recommendations) - 5} more (see report)")


def run_anomaly_detection(
    bag_path: Path,
    output_dir: str,
    output_format: str,
    topics: Optional[List[str]],
    max_frames: Optional[int],
    voxel_size: Optional[float],
    remove_outliers: bool,
    max_points_for_outliers: int,
    enable_density: bool,
    enable_spatial: bool,
    enable_ghost: bool,
    enable_temporal: bool,
    threshold: float,
    include_plots: bool,
    verbose: bool,
) -> None:
    """Run anomaly detection on a ROS2 bag file."""
    if not bag_path.is_file() and not (bag_path.is_dir() and (bag_path / "metadata.yaml").exists()):
        print(f"Error: Invalid bag file or directory: {bag_path}")
        sys.exit(1)

    try:
        # Initialize components
        print("Loading bag file...")
        loader = BagLoader(str(bag_path), topics=topics)

        if verbose:
            topic_info = loader.get_topic_info()
            print(f"Topics: {list(topic_info.keys())}")

        print("Reading point cloud frames...")
        frames = loader.read_all_frames(max_frames=max_frames)
        loader.close()

        if not frames:
            print("Error: No point cloud frames found in bag file.")
            sys.exit(1)

        print(f"Loaded {len(frames)} frames")
        print()

        # Preprocessing
        if voxel_size is not None or remove_outliers:
            print("Preprocessing point clouds...")
            preprocessor = Preprocessor(
                voxel_size=voxel_size,
                remove_outliers=remove_outliers,
                max_points_for_outliers=max_points_for_outliers,
            )
            frames = preprocessor.process_sequence(frames)
            print(f"Processed {len(frames)} frames")
            print()

        # Anomaly detection
        print("Running anomaly detection...")
        detector = AnomalyDetector(
            enable_density_detection=enable_density,
            enable_spatial_detection=enable_spatial,
            enable_ghost_detection=enable_ghost,
            enable_temporal_detection=enable_temporal,
            density_threshold=threshold,
            spatial_threshold=threshold,
            ghost_threshold=threshold,
            temporal_threshold=threshold,
        )

        result = detector.detect(frames)
        print(f"Detected {len(result.anomalies)} anomalies")
        print(f"Overall health score: {result.health_metrics.get('overall_health_score', 0):.2%}")
        print()

        # Generate reports
        print("Generating reports...")
        report_generator = ReportGenerator(output_dir=output_dir)
        output_files = report_generator.generate(
            result=result,
            bag_path=str(bag_path),
            output_format=output_format,
            include_plots=include_plots,
        )

        print()
        print("=" * 60)
        print("Anomaly Detection Complete!")
        print("=" * 60)
        print("Generated reports:")
        for format_name, file_path in output_files.items():
            print(f"  {format_name.upper()}: {file_path}")

        print()
        print("Summary:")
        print(f"  Total frames: {len(frames)}")
        print(f"  Anomalies detected: {len(result.anomalies)}")
        print(f"  Health score: {result.health_metrics.get('overall_health_score', 0):.2%}")

    except Exception as e:
        raise


def run_health_analysis(
    dataset_path: Path,
    output_dir: str,
    output_format: str,
    include_plots: bool,
    verbose: bool,
) -> None:
    """Run dataset health assessment on multi-sensor dataset folders."""
    print("Running Dataset Health Assessment & Quality Dashboard...")
    print()

    # Initializing output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Scanning dataset sequences from: {dataset_path}")
        print()

    # Running health check
    print("Computing quality metrics...")
    df_per_sensor = run_health_check(str(dataset_path), output_dir)

    if df_per_sensor.empty:
        print("[ERROR] No metrics accumulated; nothing to export.")
        return

    print()
    print("=" * 60)
    print("Dataset Health Assessment Complete!")
    print("=" * 60)
    
    # Aggregating per sequence
    agg_cols = [
        "temporal_score",
        "anomaly_score",
        "dim_timeliness",
        "dim_completeness",
        "overall_quality_score",
        "overall_quality_score_0_100",
    ]
    df_per_sequence = df_per_sensor.groupby("sequence", as_index=False)[agg_cols].mean()
    df_per_sequence["health_tier"] = df_per_sequence["overall_quality_score"].apply(
        lambda x: "excellent" if x >= 0.85 else "good" if x >= 0.70 else "fair" if x >= 0.50 else "poor"
    )

    # Exporting metrics
    print("\nExporting metrics...")
    if output_format in ["csv", "all"]:
        export_csv(df_per_sensor, df_per_sequence, output_dir)
    
    if output_format in ["json", "all"]:
        export_json(df_per_sequence, output_dir)
    
    if output_format in ["yaml", "all"]:
        export_yaml(df_per_sequence, output_dir)

    # Creating visualizations if requested
    if include_plots:
        print("Generating visualizations...")
        plot_quality_scores(df_per_sensor, os.path.join(output_dir, "health_scores.png"))
        plot_dimension_comparison(df_per_sequence, os.path.join(output_dir, "dimension_comparison.png"))
        plot_health_distribution(df_per_sequence, os.path.join(output_dir, "health_distribution.png"))

    # Building interactive dashboard
    print("Building interactive dashboard...")
    build_dashboard_html(
        df_per_sensor,
        df_per_sequence,
        os.path.join(output_dir, "health_dashboard.html"),
    )

    # Creating summary report
    print("Creating summary report...")
    create_summary_report(df_per_sequence, output_dir)

    # Generating curation recommendations
    print("Generating curation recommendations...")
    recommendations = generate_curation_recommendations(
        df_per_sensor,
        df_per_sequence,
        temporal_threshold=0.6,
        completeness_threshold=0.6,
        quality_threshold=0.5,
    )
    
    if recommendations:
        generate_curation_report(recommendations, os.path.join(output_dir, "curation_recommendations.txt"))
        generate_curation_json(recommendations, os.path.join(output_dir, "curation_recommendations.json"))

    # Displaying results summary
    print()
    print("Quality Assessment Summary:")
    print(f"  Total Sequences: {len(df_per_sequence)}")
    print(f"  Mean Quality Score: {df_per_sequence['overall_quality_score_0_100'].mean():.1f} / 100")
    
    print("\nHealth Tier Distribution:")
    tier_counts = df_per_sequence["health_tier"].value_counts()
    for tier in ["excellent", "good", "fair", "poor"]:
        count = tier_counts.get(tier, 0)
        pct = 100.0 * count / len(df_per_sequence)
        print(f"  {tier.upper():10s}: {count:3d} sequences ({pct:5.1f}%)")
    
    if recommendations:
        print(f"\nCuration Issues Found: {len(recommendations)}")
        exclude_list = get_sequences_to_exclude(recommendations)
        review_list = get_sequences_for_review(recommendations)
        if exclude_list:
            print(f"  Sequences to EXCLUDE: {len(exclude_list)}")
            if verbose:
                for seq in exclude_list[:5]:
                    print(f"    - {seq}")
                if len(exclude_list) > 5:
                    print(f"    ... and {len(exclude_list) - 5} more")
        if review_list:
            print(f"  Sequences for REVIEW: {len(review_list)}")
            if verbose:
                for seq in review_list[:5]:
                    print(f"    - {seq}")
                if len(review_list) > 5:
                    print(f"    ... and {len(review_list) - 5} more")
    else:
        print("\nNo curation issues found. All sequences meet quality thresholds.")
    
    print(f"\nReports saved to: {output_dir}")
    print("  - health_dashboard.html (interactive dashboard)")
    print("  - health_metrics.csv (detailed per-sensor metrics)")
    print("  - health_metrics_by_sequence.csv (aggregated per-sequence metrics)")
    print("  - health_summary.txt (summary report)")
    if include_plots:
        print("  - health_scores.png (quality scores bar chart)")
        print("  - dimension_comparison.png (dimension scores comparison)")
        print("  - health_distribution.png (health tier distribution)")
    if recommendations:
        print("  - curation_recommendations.txt (detailed curation recommendations)")
        print("  - curation_recommendations.json (recommendations in JSON format)")


if __name__ == "__main__":
    main()

