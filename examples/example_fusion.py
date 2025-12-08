"""

################################################################

File: examples/example_fusion.py
Created: 2025-12-08
Created by: RoboQA-Temporal Authors
Last Modified: 2025-12-08
Last Modified by: RoboQA-Temporal Authors

#################################################################

Copyright: RoboQA-Temporal Authors
License: MIT License

################################################################

Example script demonstrating Camera-LiDAR fusion quality assessment.

This script shows how to use the CalibrationQualityValidator to analyze
fusion quality in a KITTI-format dataset.

################################################################

"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from roboqa_temporal.fusion import CalibrationQualityValidator


def main():
    """Run fusion quality assessment on example dataset."""
    
    # Path to example dataset (KITTI format)
    dataset_path = Path(__file__).parent.parent / "dataset" / "2011_09_26_drive_0005_sync"
    
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        print("Please download or create a KITTI-format dataset first.")
        return
    
    # Initialize validator
    output_dir = "reports/fusion"
    validator = CalibrationQualityValidator(output_dir=output_dir)
    
    print("=" * 70)
    print("Camera-LiDAR Fusion Quality Assessment")
    print("=" * 70)
    print(f"Dataset: {dataset_path}")
    print(f"Output Directory: {output_dir}")
    print()
    
    # Run analysis
    print("Analyzing fusion quality...")
    report = validator.analyze_dataset(
        str(dataset_path),
        camera_id="image_02",
        lidar_id="velodyne_points",
        max_frames=None,  # Use all frames
        include_visualizations=True,
    )
    
    # Display results
    print()
    print("=" * 70)
    print("Fusion Quality Assessment Results")
    print("=" * 70)
    
    # Calibration quality
    print("\nCalibration Quality Metrics:")
    for pair_name, result in report.pair_results.items():
        status = "PASS" if result.overall_pass else "FAIL"
        print(f"  {pair_name}: {status}")
        print(f"    - Edge Alignment Score: {result.geom_edge_score:.3f}")
        print(f"    - Mutual Information: {result.mutual_information:.3f}")
        print(f"    - Contrastive Score: {result.contrastive_score:.3f}")
    
    # Projection error analysis
    print("\nProjection Error Analysis:")
    if report.projection_errors:
        mean_error = sum(e.reprojection_error for e in report.projection_errors) / len(report.projection_errors)
        max_error = max(e.reprojection_error for e in report.projection_errors)
        increasing_count = sum(1 for e in report.projection_errors if e.error_trend == "increasing")
        print(f"  - Mean Error: {mean_error:.3f}")
        print(f"  - Max Error: {max_error:.3f}")
        print(f"  - Frames with Increasing Error: {increasing_count}")
    
    # Illumination analysis
    print("\nIllumination Analysis:")
    if report.illumination_changes:
        mean_brightness = sum(i.brightness_mean for i in report.illumination_changes) / len(report.illumination_changes)
        light_changes = sum(1 for i in report.illumination_changes if i.light_source_change)
        print(f"  - Mean Brightness: {mean_brightness:.1f}")
        print(f"  - Detected Light Source Changes: {light_changes}")
    
    # Moving object detection
    print("\nMoving Object Detection Quality:")
    if report.moving_objects:
        mean_objects = sum(o.detected_objects for o in report.moving_objects) / len(report.moving_objects)
        mean_confidence = sum(o.detection_confidence for o in report.moving_objects) / len(report.moving_objects)
        mean_fusion = sum(o.fusion_quality_score for o in report.moving_objects) / len(report.moving_objects)
        print(f"  - Average Detected Objects: {mean_objects:.1f}")
        print(f"  - Average Detection Confidence: {mean_confidence:.3f}")
        print(f"  - Average Fusion Quality Score: {mean_fusion:.3f}")
    
    # Overall metrics
    print("\nOverall Metrics:")
    print(f"  - Edge Alignment Score: {report.metrics['calibration_quality']['edge_alignment_score']:.3f}")
    print(f"  - Mutual Information: {report.metrics['calibration_quality']['mutual_information']:.3f}")
    print(f"  - Projection Error (Mean): {report.metrics['projection_error']['mean_error']:.3f}")
    print(f"  - Illumination Stability: {report.metrics['illumination']['mean_brightness']:.1f}")
    print(f"  - Fusion Quality (Objects): {report.metrics['moving_objects']['mean_fusion_quality']:.3f}")
    
    # Recommendations
    if report.recommendations:
        print(f"\nRecommendations ({len(report.recommendations)} issues found):")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"  {i}. {rec}")
    else:
        print("\nRecommendations: No issues found - fusion quality is good!")
    
    # Output files
    print()
    print("Generated Reports:")
    if report.parameter_file:
        print(f"  - Parameters: {report.parameter_file}")
    if report.html_report_file:
        print(f"  - HTML Report: {report.html_report_file}")
    
    print()
    print("=" * 70)
    print("Assessment complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
