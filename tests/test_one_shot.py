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
    """Test detector handles empty frame list gracefully."""
    detector = AnomalyDetector()
    result = detector.detect([])
    
    assert isinstance(result, DetectionResult)
    assert result.anomalies == []
    assert result.health_metrics == {}
    assert result.frame_statistics == []


def test_anomaly_detector_with_single_frame():
    """Test detector processes a single frame without crashing."""
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
    """Test detector processes multiple frames."""
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


def test_anomaly_detector_selective_detectors():
    """Test detector with only specific detectors enabled."""
    detector = AnomalyDetector(
        enable_density_detection=True,
        enable_spatial_detection=False,
        enable_ghost_detection=False,
        enable_temporal_detection=False,
    )
    
    points = np.random.rand(50, 3) * 10
    frame = PointCloudFrame(timestamp=1000.0, frame_id="test", points=points)
    
    result = detector.detect([frame])
    
    assert isinstance(result, DetectionResult)
    # Should only have density detector results
    assert "density" in result.detector_results or len(result.detector_results) == 1


def test_preprocessor_downsample():
    """Test preprocessor downsampling with synthetic data."""
    preprocessor = Preprocessor(voxel_size=0.5)
    
    # Create a frame with many points
    points = np.random.rand(1000, 3) * 10
    frame = PointCloudFrame(timestamp=1000.0, frame_id="test", points=points)
    
    downsampled = preprocessor.process_sequence([frame])
    
    assert len(downsampled) == 1
    assert downsampled[0].num_points <= frame.num_points


def test_preprocessor_remove_outliers():
    """Test outlier removal with synthetic data."""
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
    """Test PointCloudFrame dataclass creation and initialization."""
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


def test_point_cloud_frame_with_intensities():
    """Test PointCloudFrame with intensity data."""
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    intensities = np.array([0.5, 0.8])
    
    frame = PointCloudFrame(
        timestamp=1000.0,
        frame_id="test_frame",
        points=points,
        intensities=intensities
    )
    
    assert frame.intensities is not None
    assert len(frame.intensities) == 2


def test_anomaly_dataclass_creation():
    """Test Anomaly dataclass creation."""
    anomaly = Anomaly(
        frame_index=0,
        timestamp=1000.0,
        anomaly_type="density_drop",
        severity=0.75,
        description="Significant density drop detected",
        metadata={"drop_percentage": 0.5}
    )
    
    assert anomaly.frame_index == 0
    assert anomaly.severity == 0.75
    assert anomaly.anomaly_type == "density_drop"
    assert "drop_percentage" in anomaly.metadata


def test_detection_result_creation():
    """Test DetectionResult dataclass creation."""
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


def test_preprocessor_with_empty_frames():
    """Test preprocessor handles empty frame list."""
    preprocessor = Preprocessor()
    
    result = preprocessor.process_sequence([])
    assert result == []


def test_detector_custom_thresholds():
    """Test detector with custom threshold values."""
    detector = AnomalyDetector(
        density_threshold=0.8,
        spatial_threshold=0.6,
        ghost_threshold=0.9,
        temporal_threshold=0.5
    )
    
    points = np.random.rand(50, 3) * 10
    frame = PointCloudFrame(timestamp=1000.0, frame_id="test", points=points)
    
    result = detector.detect([frame])
    assert isinstance(result, DetectionResult)
