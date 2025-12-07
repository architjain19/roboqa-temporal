"""
Edge Tests for RoboQA-Temporal

These tests verify behavior at boundary conditions and with
unusual inputs: empty data, null values, extreme parameters,
malformed inputs, etc.
"""

import pytest
import numpy as np
from roboqa_temporal.detection import AnomalyDetector
from roboqa_temporal.detection.detector import Anomaly, DetectionResult
from roboqa_temporal.loader.bag_loader import PointCloudFrame, BagLoader
from roboqa_temporal.preprocessing import Preprocessor


def test_frame_with_zero_points():
    """
    author: architjain
    reviewer: dharinesh
    category: edge test
    """
    points = np.array([]).reshape(0, 3)
    frame = PointCloudFrame(timestamp=1000.0, frame_id="empty", points=points)
    
    assert frame.num_points == 0
    assert frame.points.shape[0] == 0


def test_frame_with_nan_values():
    """
    author: architjain
    reviewer: dharinesh
    category: edge test
    """
    points = np.array([[1.0, 2.0, np.nan], [4.0, 5.0, 6.0]])
    frame = PointCloudFrame(timestamp=1000.0, frame_id="nan", points=points)
    
    assert frame.num_points == 2
    assert np.isnan(frame.points[0, 2])


def test_frame_with_inf_values():
    """
    author: architjain
    reviewer: dharinesh
    category: edge test
    """
    points = np.array([[1.0, 2.0, np.inf], [4.0, -np.inf, 6.0]])
    frame = PointCloudFrame(timestamp=1000.0, frame_id="inf", points=points)
    
    assert frame.num_points == 2
    assert np.isinf(frame.points[0, 2])


def test_detector_with_all_detectors_disabled():
    """
    author: architjain
    reviewer: dharinesh
    category: edge test
    """
    detector = AnomalyDetector(
        enable_density_detection=False,
        enable_spatial_detection=False,
        enable_ghost_detection=False,
        enable_temporal_detection=False,
    )
    
    points = np.random.rand(50, 3)
    frame = PointCloudFrame(timestamp=1000.0, frame_id="test", points=points)
    result = detector.detect([frame])
    
    assert isinstance(result, DetectionResult)
    assert len(result.detector_results) == 0


def test_detector_with_extreme_thresholds():
    """
    author: architjain
    reviewer: dharinesh
    category: edge test
    """
    # All thresholds at 0.0
    detector = AnomalyDetector(
        density_threshold=0.0,
        spatial_threshold=0.0,
        ghost_threshold=0.0,
        temporal_threshold=0.0
    )
    
    points = np.random.rand(50, 3)
    frame = PointCloudFrame(timestamp=1000.0, frame_id="test", points=points)
    result = detector.detect([frame])
    assert isinstance(result, DetectionResult)


def test_preprocessor_downsample_with_zero_voxel_size():
    """
    author: architjain
    reviewer: dharinesh
    category: edge test
    """
    preprocessor = Preprocessor(voxel_size=0.0)
    
    points = np.random.rand(100, 3) * 10
    frame = PointCloudFrame(timestamp=1000.0, frame_id="test", points=points)
    
    # Should handle gracefully or raise appropriate error
    with pytest.raises((ValueError, ZeroDivisionError, RuntimeError)):
        preprocessor.process_sequence([frame])


def test_preprocessor_downsample_with_negative_voxel_size():
    """
    author: architjain
    reviewer: dharinesh
    category: edge test
    """
    preprocessor = Preprocessor(voxel_size=-1.0)
    
    points = np.random.rand(100, 3) * 10
    frame = PointCloudFrame(timestamp=1000.0, frame_id="test", points=points)
    
    # Should handle gracefully or raise appropriate error
    with pytest.raises((ValueError, RuntimeError)):
        preprocessor.process_sequence([frame])


def test_bag_loader_with_nonexistent_path():
    """
    author: architjain
    reviewer: dharinesh
    category: edge test
    """
    with pytest.raises(FileNotFoundError):
        BagLoader("/nonexistent/path/to/bag")


def test_anomaly_with_zero_severity():
    """
    author: architjain
    reviewer: dharinesh
    category: edge test
    """
    anomaly = Anomaly(
        frame_index=0,
        timestamp=1000.0,
        anomaly_type="test",
        severity=0.0,
        description="Zero severity"
    )
    
    assert anomaly.severity == 0.0


def test_anomaly_with_max_severity():
    """
    author: architjain
    reviewer: dharinesh
    category: edge test
    """
    anomaly = Anomaly(
        frame_index=0,
        timestamp=1000.0,
        anomaly_type="test",
        severity=1.0,
        description="Max severity"
    )
    
    assert anomaly.severity == 1.0


def test_calibration_validator_edge_case_extreme_miscalibration(tmp_path):
    """author: Dharineesh Somisetty
    reviewer: Archit Jain
    category: edge test
    """
    import math
    from roboqa_temporal.calibration import (
        CalibrationQualityValidator,
        CalibrationStream,
    )

    validator = CalibrationQualityValidator(output_dir=str(tmp_path))
    
    # Create synthetic calibration stream with extreme miscalibration (1000px)
    extreme_miscalibration = 1000.0
    image_paths = [f"/synthetic/extreme/image_{i:06d}.png" for i in range(20)]
    pointcloud_paths = [f"/synthetic/extreme/cloud_{i:06d}.bin" for i in range(20)]
    calib_tag = f"miscalib_{extreme_miscalibration:.1f}px"
    calibration_file = f"/synthetic/calib/extreme_{calib_tag}.txt"
    
    pair = CalibrationStream(
        name="extreme_test",
        image_paths=image_paths,
        pointcloud_paths=pointcloud_paths,
        calibration_file=calibration_file,
        camera_id="image_02",
        lidar_id="velodyne",
    )
    
    report = validator.analyze_sequences(
        {"extreme_test": pair},
        bag_name="extreme",
        include_visualizations=False,
    )

    # With extreme miscalibration, quality should be clamped at 0.0
    actual_score = report.pair_results["extreme_test"].geom_edge_score
    assert math.isclose(actual_score, 0.0, abs_tol=1e-9)
    assert not report.pair_results["extreme_test"].overall_pass
    assert len(report.recommendations) > 0


def test_calibration_validator_edge_case_beyond_max(tmp_path):
    """author: Dharineesh Somisetty
    reviewer: Archit Jain
    category: edge test
    
    Verify score clamping at 0.0 for miscalib > max_px (25px > 20px default)
    """
    import math
    from roboqa_temporal.calibration import (
        CalibrationQualityValidator,
        CalibrationStream,
    )

    validator = CalibrationQualityValidator(output_dir=str(tmp_path))
    
    # Create calibration with miscalibration slightly beyond max_px
    miscalibration_pixels = 25.0  # Greater than default max_px of 20.0
    image_paths = [f"/synthetic/beyond/image_{i:06d}.png" for i in range(15)]
    pointcloud_paths = [f"/synthetic/beyond/cloud_{i:06d}.bin" for i in range(15)]
    calib_tag = f"miscalib_{miscalibration_pixels:.1f}px"
    calibration_file = f"/synthetic/calib/beyond_{calib_tag}.txt"
    
    pair = CalibrationStream(
        name="beyond_test",
        image_paths=image_paths,
        pointcloud_paths=pointcloud_paths,
        calibration_file=calibration_file,
        camera_id="image_02",
        lidar_id="velodyne",
    )

    report = validator.analyze_sequences(
        {"beyond_test": pair},
        bag_name="beyond_max",
        include_visualizations=False,
    )

    # Score should be clamped at 0.0
    actual_score = report.pair_results["beyond_test"].geom_edge_score
    assert math.isclose(actual_score, 0.0, abs_tol=1e-9)
