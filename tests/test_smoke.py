"""
Smoke Tests for RoboQA-Temporal

These tests verify that the package can be imported and basic
functionality is accessible without errors. They serve as a first
line of defense to catch major issues.
"""

import pytest


def test_package_import():
    """
    author: architjain
    reviewer: dharinesh
    category: smoke test
    """
    import roboqa_temporal
    assert roboqa_temporal.__version__ is not None


def test_main_classes_importable():
    """
    author: architjain
    reviewer: dharinesh
    category: smoke test
    """
    from roboqa_temporal import (
        BagLoader,
        Preprocessor,
        AnomalyDetector,
        ReportGenerator,
    )
    assert BagLoader is not None
    assert Preprocessor is not None
    assert AnomalyDetector is not None
    assert ReportGenerator is not None


def test_anomaly_detector_instantiation():
    """
    author: architjain
    reviewer: dharinesh
    category: smoke test
    """
    from roboqa_temporal.detection import AnomalyDetector
    detector = AnomalyDetector()
    assert detector is not None
    assert hasattr(detector, 'detect')


def test_preprocessor_instantiation():
    """
    author: architjain
    reviewer: dharinesh
    category: smoke test
    """
    from roboqa_temporal.preprocessing import Preprocessor
    preprocessor = Preprocessor()
    assert preprocessor is not None


def test_report_generator_instantiation():
    """
    author: architjain
    reviewer: dharinesh
    category: smoke test
    """
    from roboqa_temporal.reporting import ReportGenerator
    generator = ReportGenerator()
    assert generator is not None


def test_calibration_validator_smoke(tmp_path):
    """author: Dharineesh Somisetty
    reviewer: Archit Jain
    category: smoke test
    """
    import math
    from roboqa_temporal.calibration import (
        CalibrationQualityValidator,
        CalibrationStream,
    )

    validator = CalibrationQualityValidator(output_dir=str(tmp_path))
    
    # Create synthetic calibration stream with minimal miscalibration
    image_paths = [f"/synthetic/cam_lidar/image_{i:06d}.png" for i in range(10)]
    pointcloud_paths = [f"/synthetic/cam_lidar/cloud_{i:06d}.bin" for i in range(10)]
    calibration_file = "/synthetic/calib/cam_lidar_miscalib_0.5px.txt"
    
    cam_lidar = CalibrationStream(
        name="cam_lidar",
        image_paths=image_paths,
        pointcloud_paths=pointcloud_paths,
        calibration_file=calibration_file,
        camera_id="image_02",
        lidar_id="velodyne",
    )

    report = validator.analyze_sequences(
        {"cam_lidar": cam_lidar},
        bag_name="smoke_test",
        include_visualizations=False,
    )

    # Check exact expected score (0.975) with precision tolerance
    # quality = 1.0 - (0.5 / 20.0) = 0.975
    assert math.isclose(report.metrics["edge_alignment_score"], 0.975, rel_tol=1e-6)
    assert report.parameter_file is not None
