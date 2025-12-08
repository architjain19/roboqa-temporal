"""

################################################################

File: examples/sync_example.py
Created: 2025-12-07
Created by: Archit Jain (architj@uw.edu)
Last Modified: 2025-12-07
Last Modified by: Archit Jain (architj@uw.edu)

#################################################################

Copyright: RoboQA-Temporal Authors
License: MIT License

################################################################

Cross-Modal Synchronization Analysis example for RoboQA-Temporal.
This script demonstrates how to use the RoboQA-Temporal API
to analyze temporal synchronization in multi-sensor datasets,
such as the KITTI dataset, and generate synchronization quality reports.
Example usage of Cross-Modal Synchronization Analysis feature.

Demonstrates how to:
1. Load multi-sensor datasets (e.g., KITTI format)
2. Analyze temporal synchronization between sensors
3. Detect timestamp drift, missing frames, and duplicates
4. Generate comprehensive reports

################################################################

"""

from pathlib import Path
from roboqa_temporal.synchronization import TemporalSyncValidator

def main():
    """Run synchronization analysis on KITTI dataset."""
    
    # Path to dataset folder
    dataset_path = "dataset/2011_09_26_drive_0005_sync"
    
    print("=" * 70)
    print("Cross-Modal Synchronization Analysis Example")
    print("=" * 70)
    print()
    
    # Initialize validator with custom configuration
    validator = TemporalSyncValidator(
        # Optional: customize sensor folder names
        sensor_folders={
            "camera_left": "image_00",
            "camera_right": "image_01",
            "camera_color_left": "image_02",
            "camera_color_right": "image_03",
            "lidar": "velodyne_points",
            "imu": "oxts",
        },
        # Optional: set expected frequencies (Hz)
        expected_frequency_hz={
            "camera_left": 10.0,
            "camera_right": 10.0,
            "lidar": 10.0,
            "imu": 100.0,
        },
        # Optional: set synchronization thresholds (ms)
        approximate_time_threshold_ms={
            "camera_left_lidar": 50.0,
            "camera_left_camera_right": 10.0,
            "lidar_imu": 20.0,
        },
        # Output configuration
        output_dir="reports/temporal_sync",
        report_formats=("markdown", "html", "csv"),
        auto_export_reports=True,
    )
    
    print(f"Analyzing dataset: {dataset_path}")
    print()
    
    # Run validation
    report = validator.validate(
        dataset_path=dataset_path,
        max_frames=None,  # Process all frames, or set limit (e.g., 100)
        include_visualizations=True,
    )
    
    # Display results
    print("=" * 70)
    print("Analysis Results")
    print("=" * 70)
    print()
    
    print("Synchronization Quality:")
    sync_score = report.metrics.get("synchronization_quality_score", 0.0)
    print(f"  Overall Score: {sync_score:.2%}")
    print(f"  Temporal Offset Score: {report.metrics.get('temporal_offset_score', 0.0):.2%}")
    print(f"  Average Drift Rate: {report.metrics.get('avg_drift_rate_ms_per_s', 0.0):.4f} ms/s")
    print()
    
    print("Sensor Streams:")
    for name, stream in report.streams.items():
        freq = stream.frequency_estimate_hz or 0.0
        missing = stream.metadata.get('missing_frames', 0)
        duplicates = stream.metadata.get('duplicate_frames', 0)
        print(f"  {name}:")
        print(f"    Frames: {stream.metadata['message_count']}")
        print(f"    Frequency: {freq:.2f} Hz")
        print(f"    Missing: {missing}")
        print(f"    Duplicates: {duplicates}")
    print()
    
    print("Data Quality Issues:")
    print(f"  Total Missing Frames: {int(report.metrics.get('total_missing_frames', 0))}")
    print(f"  Total Duplicate Timestamps: {int(report.metrics.get('total_duplicate_frames', 0))}")
    print()
    
    if report.recommendations:
        print(f"Recommendations ({len(report.recommendations)} issues):")
        for i, rec in enumerate(report.recommendations[:5], 1):
            print(f"  {i}. {rec}")
        if len(report.recommendations) > 5:
            print(f"  ... and {len(report.recommendations) - 5} more (see report)")
        print()
    
    print("Generated Reports:")
    for format_name, file_path in report.report_files.items():
        print(f"  {format_name.upper()}: {file_path}")
    print()
    
    if report.parameter_file:
        print(f"Timestamp Corrections: {report.parameter_file}")
        print()
    
    print("=" * 70)
    print("Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
