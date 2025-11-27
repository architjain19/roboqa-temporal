"""
Smoke Tests for RoboQA-Temporal

These tests verify that the package can be imported and basic
functionality is accessible without errors. They serve as a first
line of defense to catch major issues.
"""

import pytest


def test_package_import():
    """Verify the main package can be imported."""
    import roboqa_temporal
    assert roboqa_temporal.__version__ is not None


def test_main_classes_importable():
    """Verify all main classes can be imported."""
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


def test_detection_classes_importable():
    """Verify detection module classes can be imported."""
    from roboqa_temporal.detection import AnomalyDetector
    from roboqa_temporal.detection.detector import Anomaly, DetectionResult
    assert AnomalyDetector is not None
    assert Anomaly is not None
    assert DetectionResult is not None


def test_loader_classes_importable():
    """Verify loader module classes can be imported."""
    from roboqa_temporal.loader import BagLoader
    from roboqa_temporal.loader.bag_loader import PointCloudFrame
    assert BagLoader is not None
    assert PointCloudFrame is not None


def test_preprocessing_classes_importable():
    """Verify preprocessing module classes can be imported."""
    from roboqa_temporal.preprocessing import Preprocessor
    assert Preprocessor is not None


def test_reporting_classes_importable():
    """Verify reporting module classes can be imported."""
    from roboqa_temporal.reporting import ReportGenerator
    assert ReportGenerator is not None


def test_anomaly_detector_instantiation():
    """Verify AnomalyDetector can be instantiated with default settings."""
    from roboqa_temporal.detection import AnomalyDetector
    detector = AnomalyDetector()
    assert detector is not None
    assert hasattr(detector, 'detect')


def test_preprocessor_instantiation():
    """Verify Preprocessor can be instantiated."""
    from roboqa_temporal.preprocessing import Preprocessor
    preprocessor = Preprocessor()
    assert preprocessor is not None


def test_report_generator_instantiation():
    """Verify ReportGenerator can be instantiated."""
    from roboqa_temporal.reporting import ReportGenerator
    generator = ReportGenerator()
    assert generator is not None


def test_package_metadata():
    """Verify package metadata is accessible."""
    import roboqa_temporal
    assert hasattr(roboqa_temporal, '__version__')
    assert hasattr(roboqa_temporal, '__author__')
    assert hasattr(roboqa_temporal, '__all__')
    assert len(roboqa_temporal.__all__) == 4


def test_detector_has_expected_methods():
    """Verify AnomalyDetector has expected public methods."""
    from roboqa_temporal.detection import AnomalyDetector
    detector = AnomalyDetector()
    assert hasattr(detector, 'detect')
    assert callable(detector.detect)


def test_preprocessor_has_expected_methods():
    """Verify Preprocessor has expected public methods."""
    from roboqa_temporal.preprocessing import Preprocessor
    preprocessor = Preprocessor()
    # Check for common preprocessing methods
    assert hasattr(preprocessor, 'process_frame')
    assert hasattr(preprocessor, 'process_sequence')
