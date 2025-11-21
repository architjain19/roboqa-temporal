"""

################################################################

File: examples/basic_usage.py
Created: 2025-11-20
Created by: Archit Jain (architj@uw.edu)
Last Modified: 2025-11-21
Last Modified by: Archit Jain (architj@uw.edu)

#################################################################

Copyright: RoboQA-Temporal Authors
License: MIT License

################################################################

Basic usage example for RoboQA-Temporal.

This script demonstrates how to use the RoboQA-Temporal API
to analyze a ROS2 bag file and generate quality assessment reports.

################################################################

"""

import sys
from roboqa_temporal import BagLoader, Preprocessor, AnomalyDetector, ReportGenerator

def main(bag_path="path/to/your/bag_file.db3"):

    print("Loading bag file...")

    # Initialize bag loader
    # Optionally specify topics or frame_id
    loader = BagLoader(bag_path, topics=None)  # None = auto-detect point cloud topics

    # Get bag information
    topic_info = loader.get_topic_info()
    print(f"Found topics: {list(topic_info.keys())}")

    # Read point cloud frames
    # Use max_frames to limit processing for testing
    frames = loader.read_all_frames(max_frames=100)
    loader.close()

    print(f"Loaded {len(frames)} frames")

    if not frames:
        print("No point cloud frames found!")
        return

    # Preprocessing
    print("\nPreprocessing point clouds...")
    preprocessor = Preprocessor(
        voxel_size=0.05,  # 5cm voxel size for downsampling
        remove_outliers=True,
        outlier_method="statistical",
        outlier_params={"nb_neighbors": 20, "std_ratio": 2.0},
    )

    processed_frames = preprocessor.process_sequence(frames)
    print(f"Processed {len(processed_frames)} frames")

    # Anomaly Detection
    print("\nRunning anomaly detection...")
    detector = AnomalyDetector(
        enable_density_detection=True,
        enable_spatial_detection=True,
        enable_ghost_detection=True,
        enable_temporal_detection=True,
        density_threshold=0.5,
        spatial_threshold=0.3,
        ghost_threshold=0.7,
        temporal_threshold=0.4,
    )

    result = detector.detect(processed_frames)

    print(f"\nDetection Results:")
    print(f"  Total anomalies: {len(result.anomalies)}")
    print(f"  Overall health score: {result.health_metrics.get('overall_health_score', 0):.2%}")

    # Print some anomalies
    if result.anomalies:
        print(f"\nSample anomalies:")
        for anomaly in result.anomalies[:5]:  # Show first 5
            print(f"  Frame {anomaly.frame_index}: {anomaly.anomaly_type} (severity: {anomaly.severity:.2f}) - {anomaly.description}")

    # Generate Reports
    print("\nGenerating reports...")
    report_generator = ReportGenerator(output_dir="reports")
    output_files = report_generator.generate(
        result=result,
        bag_path=bag_path,
        output_format="all",  # 'markdown', 'html', 'csv', or 'all'
        include_plots=True,
    )

    print("\nGenerated reports:")
    for format_name, file_path in output_files.items():
        print(f"  {format_name.upper()}: {file_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Please provide the path to the bag file as a command-line argument. Ex: python3 examples/basic_usage.py path/to/bag.db3")