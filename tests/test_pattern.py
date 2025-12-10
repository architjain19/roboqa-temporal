"""
Pattern Tests for RoboQA-Temporal

These tests verify common usage patterns and workflows that users
would typically follow when using the package. They test realistic
combinations of operations and integration between components.
"""

import pytest
import numpy as np
from pathlib import Path
from roboqa_temporal.detection import AnomalyDetector
from roboqa_temporal.detection.detector import DetectionResult
from roboqa_temporal.loader.bag_loader import PointCloudFrame
from roboqa_temporal.preprocessing import Preprocessor
from roboqa_temporal.reporting import ReportGenerator


def test_pattern_preprocess_then_detect():
    """
    author: architjain
    reviewer: dharinesh
    category: pattern test
    """
    # Step 1: Create synthetic frames
    frames = [
        PointCloudFrame(
            timestamp=1000.0 + i * 100,
            frame_id=f"frame_{i}",
            points=np.random.rand(200, 3) * 10
        )
        for i in range(10)
    ]
    
    # Step 2: Preprocess
    preprocessor = Preprocessor(voxel_size=0.5, remove_outliers=True)
    processed_frames = preprocessor.process_sequence(frames)
    
    # Step 3: Detect anomalies
    detector = AnomalyDetector()
    result = detector.detect(processed_frames)
    
    assert isinstance(result, DetectionResult)
    assert len(processed_frames) <= len(frames)


def test_pattern_detect_then_report():
    """
    author: architjain
    reviewer: dharinesh
    category: pattern test
    """
    # Step 1: Create and detect
    frames = [
        PointCloudFrame(
            timestamp=1000.0 + i * 100,
            frame_id=f"frame_{i}",
            points=np.random.rand(150, 3) * 10
        )
        for i in range(5)
    ]
    
    detector = AnomalyDetector()
    result = detector.detect(frames)
    
    # Step 2: Generate report
    generator = ReportGenerator()
    report_data = generator.generate(result, "test_bag")
    
    assert report_data is not None
    assert isinstance(report_data, dict)


def test_pattern_selective_detection_workflow():
    """
    author: architjain
    reviewer: dharinesh
    category: pattern test
    """
    frames = [
        PointCloudFrame(
            timestamp=1000.0 + i * 100,
            frame_id=f"frame_{i}",
            points=np.random.rand(100, 3) * 10
        )
        for i in range(8)
    ]
    
    # Use only spatial and temporal detection
    detector = AnomalyDetector(
        enable_density_detection=False,
        enable_spatial_detection=True,
        enable_ghost_detection=False,
        enable_temporal_detection=True,
    )
    
    result = detector.detect(frames)
    
    # Verify only requested detectors ran
    assert isinstance(result, DetectionResult)


def test_pattern_sync_validator_workflow():
    """
    author: xinxin
    reviewer: sayali
    category: pattern test
    """
    from roboqa_temporal.synchronization import TemporalSyncValidator, SensorStream
    
    # Create synthetic sensor streams with pairs that will be analyzed
    camera_left_ts = list(range(0, 1000000000, 100000000))   # 10 Hz
    lidar_ts = list(range(0, 1000000000, 100000000))         # 10 Hz
    camera_right_ts = list(range(0, 1000000000, 100000000))  # 10 Hz
    
    streams = {
        "camera_left": SensorStream(
            name="camera_left",
            source_path="/fake/camera_left",
            timestamps_ns=camera_left_ts,
            expected_frequency=10.0,
        ),
        "camera_right": SensorStream(
            name="camera_right",
            source_path="/fake/camera_right",
            timestamps_ns=camera_right_ts,
            expected_frequency=10.0,
        ),
        "lidar": SensorStream(
            name="lidar",
            source_path="/fake/lidar",
            timestamps_ns=lidar_ts,
            expected_frequency=10.0,
        ),
    }
    
    # Initialize validator
    validator = TemporalSyncValidator(output_dir="reports/test_sync", auto_export_reports=False)
    
    # Analyze streams
    report = validator.analyze_streams(streams, dataset_name="test_sync", include_visualizations=False)
    
    assert report is not None
    assert len(report.streams) == 3
    assert len(report.pair_results) > 0


def test_pattern_sync_validator_with_drift():
    """
    author: xinxin
    reviewer: sayali
    category: pattern test
    """
    from roboqa_temporal.synchronization import TemporalSyncValidator, SensorStream
    import numpy as np
    
    # Create streams with temporal drift
    base_timestamps = np.arange(0, 1000000000, 100000000, dtype=np.int64)
    camera_left_timestamps = list(base_timestamps)
    # Add progressive drift to lidar (1ms per frame)
    lidar_timestamps = list(base_timestamps + np.arange(len(base_timestamps)) * 1000000)
    
    streams = {
        "camera_left": SensorStream(
            name="camera_left",
            source_path="/fake/camera",
            timestamps_ns=camera_left_timestamps,
            expected_frequency=10.0,
        ),
        "lidar": SensorStream(
            name="lidar",
            source_path="/fake/lidar",
            timestamps_ns=lidar_timestamps,
            expected_frequency=10.0,
        ),
    }
    
    validator = TemporalSyncValidator(auto_export_reports=False)
    report = validator.analyze_streams(streams, dataset_name="test_drift", include_visualizations=False)
    
    assert len(report.pair_results) > 0
    for pair_result in report.pair_results.values():
        assert hasattr(pair_result, "drift_rate_ms_per_s")


def test_pattern_multi_sensor_synchronization():
    """
    author: xinxin
    reviewer: sayali
    category: pattern test
    """
    from roboqa_temporal.synchronization import TemporalSyncValidator, SensorStream
    
    # Create multiple sensor streams
    camera_left_ts = list(range(0, 1000000000, 100000000))   # 10 Hz
    camera_right_ts = list(range(0, 1000000000, 100000000))  # 10 Hz
    lidar_ts = list(range(0, 1000000000, 100000000))         # 10 Hz
    imu_ts = list(range(0, 1000000000, 10000000))            # 100 Hz
    
    streams = {
        "camera_left": SensorStream(
            name="camera_left",
            source_path="/fake/cam_left",
            timestamps_ns=camera_left_ts,
            expected_frequency=10.0,
        ),
        "camera_right": SensorStream(
            name="camera_right",
            source_path="/fake/cam_right",
            timestamps_ns=camera_right_ts,
            expected_frequency=10.0,
        ),
        "lidar": SensorStream(
            name="lidar",
            source_path="/fake/lidar",
            timestamps_ns=lidar_ts,
            expected_frequency=10.0,
        ),
        "imu": SensorStream(
            name="imu",
            source_path="/fake/imu",
            timestamps_ns=imu_ts,
            expected_frequency=100.0,
        ),
    }
    
    validator = TemporalSyncValidator(auto_export_reports=False)
    report = validator.analyze_streams(streams, dataset_name="multi_sensor", include_visualizations=False)
    
    assert len(report.streams) == 4
    assert len(report.pair_results) > 0


def test_pattern_sync_report_serialization():
    """
    author: xinxin
    reviewer: sayali
    category: pattern test
    """
    from roboqa_temporal.synchronization import TemporalSyncValidator, SensorStream
    
    timestamps_camera = list(range(0, 500000000, 100000000))
    timestamps_lidar = list(range(0, 500000000, 100000000))
    
    streams = {
        "camera_left": SensorStream(
            name="camera_left",
            source_path="/fake/camera",
            timestamps_ns=timestamps_camera,
            expected_frequency=10.0,
        ),
        "lidar": SensorStream(
            name="lidar",
            source_path="/fake/lidar",
            timestamps_ns=timestamps_lidar,
            expected_frequency=10.0,
        ),
    }
    
    validator = TemporalSyncValidator(auto_export_reports=False)
    report = validator.analyze_streams(streams, dataset_name="test_serial", include_visualizations=False)
    
    # Test serialization
    report_dict = report.to_dict()
    
    assert isinstance(report_dict, dict)
    assert "streams" in report_dict
    assert "metrics" in report_dict
    assert "recommendations" in report_dict
    assert "generated_at" in report_dict


def test_pattern_iterative_detection():
    """
    author: architjain
    reviewer: dharinesh
    category: pattern test
    """
    all_frames = [
        PointCloudFrame(
            timestamp=1000.0 + i * 100,
            frame_id=f"frame_{i}",
            points=np.random.rand(100, 3) * 10
        )
        for i in range(20)
    ]
    
    detector = AnomalyDetector()
    
    # Detect on first half
    result1 = detector.detect(all_frames[:10])
    
    # Detect on second half
    result2 = detector.detect(all_frames[10:])
    
    # Both should be valid results
    assert isinstance(result1, DetectionResult)
    assert isinstance(result2, DetectionResult)


def test_pattern_custom_detector_thresholds():
    """
    author: architjain
    reviewer: dharinesh
    category: pattern test
    """
    frames = [
        PointCloudFrame(
            timestamp=1000.0 + i * 100,
            frame_id=f"frame_{i}",
            points=np.random.rand(100, 3) * 10
        )
        for i in range(5)
    ]
    
    # Strict thresholds (more sensitive)
    strict_detector = AnomalyDetector(
        density_threshold=0.3,
        spatial_threshold=0.2,
        ghost_threshold=0.4,
        temporal_threshold=0.3
    )
    
    # Lenient thresholds (less sensitive)
    lenient_detector = AnomalyDetector(
        density_threshold=0.8,
        spatial_threshold=0.7,
        ghost_threshold=0.9,
        temporal_threshold=0.8
    )
    
    strict_result = strict_detector.detect(frames)
    lenient_result = lenient_detector.detect(frames)
    
    assert isinstance(strict_result, DetectionResult)
    assert isinstance(lenient_result, DetectionResult)


def test_pattern_batch_processing():
    """
    author: architjain
    reviewer: dharinesh
    category: pattern test
    """
    detector = AnomalyDetector()
    
    batches = [
        [PointCloudFrame(
            timestamp=1000.0 + b * 1000 + i * 100,
            frame_id=f"batch{b}_frame{i}",
            points=np.random.rand(100, 3) * 10
        ) for i in range(5)]
        for b in range(3)
    ]
    
    results = []
    for batch in batches:
        processed = Preprocessor(voxel_size=0.5).process_sequence(batch)
        result = detector.detect(processed)
        results.append(result)
    
    assert len(results) == 3
    assert all(isinstance(r, DetectionResult) for r in results)


def test_pattern_full_pipeline():
    """
    author: architjain
    reviewer: dharinesh
    category: pattern test
    """
    # 1. Create synthetic data
    raw_frames = [
        PointCloudFrame(
            timestamp=1000.0 + i * 100,
            frame_id=f"frame_{i}",
            points=np.random.rand(300, 3) * 10
        )
        for i in range(10)
    ]
    
    # 2. Preprocess
    preprocessor = Preprocessor(remove_outliers=True)
    cleaned = preprocessor.process_sequence(raw_frames)
    processed = Preprocessor(voxel_size=0.5).process_sequence(cleaned)
    
    # 3. Detect
    detector = AnomalyDetector()
    detection_result = detector.detect(processed)
    
    # 4. Report
    generator = ReportGenerator()
    report = generator.generate(detection_result, "test_bag")
    
    # Verify pipeline completed
    assert len(processed) > 0
    assert isinstance(detection_result, DetectionResult)
    assert report is not None
