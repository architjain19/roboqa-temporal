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


def test_calibration_validator_one_shot(tmp_path):
    """author: Dharineesh Somisetty
    reviewer: <buddy name>
    category: one-shot test
    """
    import math
    from roboqa_temporal.calibration import (
        CalibrationQualityValidator,
        CalibrationStream,
    )

    validator = CalibrationQualityValidator(output_dir=str(tmp_path))
    
    # Create synthetic calibration stream with 10px miscalibration
    miscalibration_pixels = 10.0
    image_paths = [f"/synthetic/one_shot/image_{i:06d}.png" for i in range(50)]
    pointcloud_paths = [f"/synthetic/one_shot/cloud_{i:06d}.bin" for i in range(50)]
    calib_tag = f"miscalib_{miscalibration_pixels:.1f}px"
    calibration_file = f"/synthetic/calib/one_shot_{calib_tag}.txt"
    
    pair = CalibrationStream(
        name="one_shot",
        image_paths=image_paths,
        pointcloud_paths=pointcloud_paths,
        calibration_file=calibration_file,
        camera_id="image_02",
        lidar_id="velodyne",
    )
    
    report = validator.analyze_sequences({"one_shot": pair}, bag_name="one_shot")

    expected_quality = max(0.0, 1.0 - 10.0 / 20.0)
    assert math.isclose(
        report.pair_results["one_shot"].geom_edge_score,
        expected_quality,
        rel_tol=1e-6,
    )
