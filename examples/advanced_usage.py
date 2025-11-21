"""

################################################################

File: examples/advanced_usage.py
Created: 2025-11-20
Created by: Archit Jain (architj@uw.edu)
Last Modified: 2025-11-21
Last Modified by: Archit Jain (architj@uw.edu)

#################################################################

Copyright: RoboQA-Temporal Authors
License: MIT License

################################################################

Advanced usage example for RoboQA-Temporal.

Features:
- Custom preprocessing
- Selective detector usage
- Accessing detailed results
- Custom report generation

################################################################

"""

import sys
from roboqa_temporal import BagLoader, Preprocessor, AnomalyDetector, ReportGenerator
from roboqa_temporal.detection.detectors import (
    DensityDropDetector,
    SpatialDiscontinuityDetector,
)


def main(bag_path="path/to/your/bag_file.db3", topics=None, frame_id=None):

    # Advanced loader usage
    print("Loading with specific topics...")
    loader = BagLoader(
        bag_path,
        topics=None,  # Specific topics
        frame_id=None,  # Optional frame ID filter
    )

    # Iterate through frames (memory efficient for large bags)
    print("Processing frames iteratively...")
    frames = []
    for frame in loader.read_point_clouds(max_frames=200, progress=True):
        frames.append(frame)
    loader.close()

    print(f"Loaded {len(frames)} frames")

    # Custom preprocessing pipeline
    print("\nCustom preprocessing...")
    preprocessor = Preprocessor(
        voxel_size=0.1,  # Larger voxel for faster processing
        remove_outliers=True,
        outlier_method="radius",  # Use radius-based outlier removal
        outlier_params={"radius": 0.2, "min_neighbors": 5},
    )

    processed_frames = preprocessor.process_sequence(frames)

    # Use individual detectors
    print("\nRunning individual detectors...")
    density_detector = DensityDropDetector(threshold=0.4, window_size=10)
    spatial_detector = SpatialDiscontinuityDetector(threshold=0.25)

    density_result = density_detector.detect(processed_frames)
    spatial_result = spatial_detector.detect(processed_frames)

    print(f"Density anomalies: {len(density_result['anomalies'])}")
    print(f"Spatial anomalies: {len(spatial_result['anomalies'])}")

    # Access detailed metrics
    print("\nDetailed metrics:")
    print(f"  Average density: {density_result['metrics'].get('avg_density', 0):.0f}")
    print(f"  Density CV: {density_result['metrics'].get('density_cv', 0):.3f}")  # Coefficient of variation
    print(f"  Average translation: {spatial_result['metrics'].get('avg_translation', 0):.3f}m")

    # Full detector with custom configuration
    print("\nRunning full detector...")
    detector = AnomalyDetector(
        enable_density_detection=True,
        enable_spatial_detection=True,
        enable_ghost_detection=False,  # Disable ghost detection
        enable_temporal_detection=True,
        density_threshold=0.4,
        spatial_threshold=0.25,
        temporal_threshold=0.35,
    )

    result = detector.detect(processed_frames)

    # Analyze results programmatically
    print("\nAnalyzing results...")
    anomalies_by_type = {}
    for anomaly in result.anomalies:
        anomaly_type = anomaly.anomaly_type
        if anomaly_type not in anomalies_by_type:
            anomalies_by_type[anomaly_type] = []
        anomalies_by_type[anomaly_type].append(anomaly)

    print("Anomalies by type:")
    for anomaly_type, anomalies in anomalies_by_type.items():
        avg_severity = sum(a.severity for a in anomalies) / len(anomalies)
        print(f"  {anomaly_type}: {len(anomalies)} (avg severity: {avg_severity:.2f})")

    # High-severity anomalies
    high_severity = [a for a in result.anomalies if a.severity > 0.7]
    print(f"\nHigh-severity anomalies (>0.7): {len(high_severity)}")

    # Generate custom report
    print("\nGenerating custom report...")
    report_generator = ReportGenerator(output_dir="reports")
    output_files = report_generator.generate(
        result=result,
        bag_path=bag_path,
        output_format="html",  # Only HTML
        include_plots=True,
    )

    print(f"Report generated: {output_files.get('html', 'N/A')}")


if __name__ == "__main__":
    if len(sys.argv) > 3:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print("Please provide the path to the bag file, topics, and frame_id as command-line arguments. Ex: python3 examples/advanced_usage.py path/to/bag.db3 '/lidar_points' 'map'")