"""

################################################################

File: roboqa_temporal/cli/main.py
Created: 2025-11-20
Created by: Archit Jain (architj@uw.edu)
Last Modified: 2025-11-20
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
from pathlib import Path
import yaml

from roboqa_temporal.loader import BagLoader
from roboqa_temporal.preprocessing import Preprocessor
from roboqa_temporal.detection import AnomalyDetector
from roboqa_temporal.reporting import ReportGenerator


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RoboQA-Temporal: Automated quality assessment for ROS2 bag files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  roboqa bag_file.db3

  # With configuration file
  roboqa bag_file.db3 --config config.yaml

  # Specify output format
  roboqa bag_file.db3 --output html --output-dir reports/

  # Limit number of frames
  roboqa bag_file.db3 --max-frames 1000
        """,
    )

    parser.add_argument(
        "bag_file",
        type=str,
        help="Path to ROS2 bag file or directory",
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
        help="Voxel size for downsampling (default: no downsampling)",
    )

    parser.add_argument(
        "--no-outlier-removal",
        action="store_true",
        help="Disable outlier removal",
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

    # Validate bag file
    bag_path = Path(args.bag_file)
    if not bag_path.exists():
        print(f"Error: Bag file not found: {args.bag_file}")
        sys.exit(1)

    print("=" * 60)
    print("RoboQA-Temporal: Quality Assessment Tool")
    print("=" * 60)
    print(f"Bag file: {bag_path}")
    print(f"Output directory: {output_dir}")
    print()

    try:
        # Initialize components
        print("Loading bag file...")
        loader = BagLoader(str(bag_path), topics=topics)

        if args.verbose:
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
        print("Analysis Complete!")
        print("=" * 60)
        print("Generated reports:")
        for format_name, file_path in output_files.items():
            print(f"  {format_name.upper()}: {file_path}")

        print()
        print("Summary:")
        print(f"  Total frames: {len(frames)}")
        print(f"  Anomalies detected: {len(result.anomalies)}")
        print(f"  Health score: {result.health_metrics.get('overall_health_score', 0):.2%}")

    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

