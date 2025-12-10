"""
One-Shot Tests for RoboQA-Temporal

These tests verify single execution paths with minimal setup,
focusing on core functionality with simple, synthetic inputs.
No external dependencies like ROS2 bags required.
"""

import pytest
import numpy as np
from roboqa_temporal.detection import AnomalyDetector
from roboqa_temporal.detection.detector import Anomaly, DetectionResult
from roboqa_temporal.loader.bag_loader import PointCloudFrame
from roboqa_temporal.preprocessing import Preprocessor


def test_anomaly_detector_with_empty_frames():
    """
    author: architjain
    reviewer: dharinesh
    category: one-shot test
    """
    detector = AnomalyDetector()
    result = detector.detect([])
    
    assert isinstance(result, DetectionResult)
    assert result.anomalies == []
    assert result.health_metrics == {}
    assert result.frame_statistics == []


def test_anomaly_detector_with_single_frame():
    """
    author: architjain
    reviewer: dharinesh
    category: one-shot test
    """
    detector = AnomalyDetector()
    
    # Create a simple synthetic frame
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    frame = PointCloudFrame(
        timestamp=1000.0,
        frame_id="test_frame",
        points=points
    )
    
    result = detector.detect([frame])
    
    assert isinstance(result, DetectionResult)
    assert isinstance(result.anomalies, list)
    assert isinstance(result.health_metrics, dict)


def test_anomaly_detector_with_multiple_frames():
    """
    author: architjain
    reviewer: dharinesh
    category: one-shot test
    """
    detector = AnomalyDetector()
    
    # Create multiple synthetic frames
    frames = []
    for i in range(5):
        points = np.random.rand(100, 3) * 10  # 100 random points
        frame = PointCloudFrame(
            timestamp=1000.0 + i * 100,
            frame_id=f"frame_{i}",
            points=points
        )
        frames.append(frame)
    
    result = detector.detect(frames)
    
    assert isinstance(result, DetectionResult)
    assert len(result.frame_statistics) <= len(frames)
    assert "overall_health_score" in result.health_metrics


def test_preprocessor_downsample():
    """
    author: architjain
    reviewer: dharinesh
    category: one-shot test
    """
    preprocessor = Preprocessor(voxel_size=0.5)
    
    # Create a frame with many points
    points = np.random.rand(1000, 3) * 10
    frame = PointCloudFrame(timestamp=1000.0, frame_id="test", points=points)
    
    downsampled = preprocessor.process_sequence([frame])
    
    assert len(downsampled) == 1
    assert downsampled[0].num_points <= frame.num_points


def test_temporal_sync_validator_with_single_stream():
    """
    author: xinxin
    reviewer: sayali
    category: one-shot test
    """
    preprocessor = Preprocessor(remove_outliers=True, max_points_for_outliers=500)
    
    # Create a frame with some outliers
    points = np.random.rand(200, 3) * 10
    # Add some outliers
    outliers = np.array([[100.0, 100.0, 100.0], [-100.0, -100.0, -100.0]])
    all_points = np.vstack([points, outliers])
    
    frame = PointCloudFrame(timestamp=1000.0, frame_id="test", points=all_points)
    processed = preprocessor.process_sequence([frame])
    
    assert len(processed) == 1
    assert processed[0].num_points < frame.num_points


def test_sensor_stream_creation():
    """
    author: xinxin
    reviewer: sayali
    category: one-shot test
    """
    from roboqa_temporal.synchronization import SensorStream
    
    # Create a simple sensor stream with synthetic timestamps
    timestamps_ns = [1000000000, 1100000000, 1200000000, 1300000000]  # 100ms intervals
    stream = SensorStream(
        name="test_camera",
        source_path="/fake/path",
        timestamps_ns=timestamps_ns,
        expected_frequency=10.0,
    )
    
    assert stream.name == "test_camera"
    assert len(stream.timestamps_ns) == 4
    assert stream.frequency_estimate_hz is not None
    assert abs(stream.frequency_estimate_hz - 10.0) < 1.0


def test_sensor_stream_with_missing_frames():
    """
    author: xinxin
    reviewer: sayali
    category: one-shot test
    """
    from roboqa_temporal.synchronization import SensorStream
    
    # Create timestamps with a missing frame (gap)
    timestamps_ns = [
        1000000000,  # t=0
        1100000000,  # t=0.1s
        1200000000,  # t=0.2s
        1400000000,  # t=0.4s (missing frame at 0.3s)
        1500000000,  # t=0.5s
    ]
    stream = SensorStream(
        name="test_lidar",
        source_path="/fake/path",
        timestamps_ns=timestamps_ns,
        expected_frequency=10.0,
    )
    
    assert stream.metadata["missing_frames"] > 0


def test_sensor_stream_with_duplicate_timestamps():
    """
    author: xinxin
    reviewer: sayali
    category: one-shot test
    """
    from roboqa_temporal.synchronization import SensorStream
    
    # Create timestamps with duplicates
    timestamps_ns = [
        1000000000,
        1100000000,
        1100000000,  # Duplicate
        1200000000,
    ]
    stream = SensorStream(
        name="test_imu",
        source_path="/fake/path",
        timestamps_ns=timestamps_ns,
        expected_frequency=10.0,
    )
    
    assert stream.metadata["duplicate_frames"] > 0


def test_temporal_sync_validator_with_empty_streams():
    """
    author: xinxin
    reviewer: sayali
    category: one-shot test
    """
    from roboqa_temporal.synchronization import TemporalSyncValidator, SensorStream
    
    validator = TemporalSyncValidator(auto_export_reports=False)
    streams = {}
    
    report = validator.analyze_streams(streams, dataset_name="empty_test", include_visualizations=False)
    
    assert report is not None
    assert len(report.streams) == 0
    assert len(report.pair_results) == 0


def test_preprocessor_remove_outliers():
    """
    author: architjain
    reviewer: dharinesh
    category: one-shot test
    """
    preprocessor = Preprocessor(remove_outliers=True)
    
    # Create points with some outliers
    points = np.random.rand(100, 3) * 10
    # Add obvious outliers
    outliers = np.array([[100.0, 100.0, 100.0], [-100.0, -100.0, -100.0]])
    all_points = np.vstack([points, outliers])
    
    frame = PointCloudFrame(timestamp=1000.0, frame_id="test", points=all_points)
    cleaned = preprocessor.process_sequence([frame])
    
    assert len(cleaned) == 1
    assert cleaned[0].num_points <= frame.num_points


def test_point_cloud_frame_creation():
    """
    author: architjain
    reviewer: dharinesh
    category: one-shot test
    """
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    
    frame = PointCloudFrame(
        timestamp=1000.0,
        frame_id="test_frame",
        points=points
    )
    
    assert frame.timestamp == 1000.0
    assert frame.frame_id == "test_frame"
    assert frame.num_points == 2
    assert np.array_equal(frame.points, points)


def test_detection_result_creation():
    """
    author: architjain
    reviewer: dharinesh
    category: one-shot test
    """
    anomaly = Anomaly(
        frame_index=0,
        timestamp=1000.0,
        anomaly_type="test",
        severity=0.5,
        description="Test anomaly"
    )
    
    result = DetectionResult(
        anomalies=[anomaly],
        health_metrics={"overall_health_score": 0.8},
        frame_statistics=[{"frame": 0, "points": 100}]
    )
    
    assert len(result.anomalies) == 1
    assert result.health_metrics["overall_health_score"] == 0.8
    assert len(result.frame_statistics) == 1


def test_calibration_stream_creation():
    """
    author: dharinesh
    reviewer: architjain
    category: one-shot test
    """
    from roboqa_temporal.fusion import CalibrationStream
    
    stream = CalibrationStream(
        name="camera_lidar_pair",
        image_paths=["/fake/img1.png", "/fake/img2.png"],
        pointcloud_paths=["/fake/pc1.bin", "/fake/pc2.bin"],
        calibration_file="/fake/calib.txt",
        camera_id="cam_02",
        lidar_id="velodyne"
    )
    
    assert stream.name == "camera_lidar_pair"
    assert len(stream.image_paths) == 2
    assert len(stream.pointcloud_paths) == 2
    assert stream.camera_id == "cam_02"


def test_calibration_pair_result_creation():
    """
    author: dharinesh
    reviewer: architjain
    category: one-shot test
    """
    from roboqa_temporal.fusion import CalibrationPairResult
    
    result = CalibrationPairResult(
        geom_edge_score=0.85,
        mutual_information=0.75,
        contrastive_score=0.80,
        pass_geom_edge=True,
        pass_mi=True,
        pass_contrastive=True,
        overall_pass=True,
        details={"frames_analyzed": 10}
    )
    
    assert result.geom_edge_score == 0.85
    assert result.overall_pass is True
    assert result.details["frames_analyzed"] == 10


def test_projection_error_frame_creation():
    """
    author: dharinesh
    reviewer: architjain
    category: one-shot test
    """
    from roboqa_temporal.fusion import ProjectionErrorFrame
    
    error_frame = ProjectionErrorFrame(
        frame_index=5,
        timestamp=1500.0,
        reprojection_error=2.5,
        max_error_point=(100.0, 200.0),
        projected_points_count=150,
        error_trend="increasing"
    )
    
    assert error_frame.frame_index == 5
    assert error_frame.reprojection_error == 2.5
    assert error_frame.error_trend == "increasing"
    assert error_frame.projected_points_count == 150


def test_illumination_frame_creation():
    """
    author: dharinesh
    reviewer: architjain
    category: one-shot test
    """
    from roboqa_temporal.fusion import IlluminationFrame
    
    illum_frame = IlluminationFrame(
        frame_index=10,
        timestamp=2000.0,
        brightness_mean=128.5,
        brightness_std=45.2,
        contrast=0.65,
        scene_change_score=0.3,
        light_source_change=False
    )
    
    assert illum_frame.frame_index == 10
    assert illum_frame.brightness_mean == 128.5
    assert illum_frame.light_source_change is False


def test_moving_object_frame_creation():
    """
    author: dharinesh
    reviewer: architjain
    category: one-shot test
    """
    from roboqa_temporal.fusion import MovingObjectFrame
    
    obj_frame = MovingObjectFrame(
        frame_index=15,
        timestamp=2500.0,
        detected_objects=3,
        detection_confidence=0.92,
        consistency_score=0.88,
        fusion_quality_score=0.85
    )
    
    assert obj_frame.frame_index == 15
    assert obj_frame.detected_objects == 3
    assert obj_frame.detection_confidence == 0.92
    assert obj_frame.fusion_quality_score == 0.85


def test_calibration_quality_validator_initialization():
    """
    author: dharinesh
    reviewer: architjain
    category: one-shot test
    """
    from roboqa_temporal.fusion import CalibrationQualityValidator
    
    validator = CalibrationQualityValidator(
        output_dir="reports/test_fusion",
        config={"edge_threshold": 0.7}
    )
    
    assert validator.output_dir.name == "test_fusion"
    assert validator.config["edge_threshold"] == 0.7
