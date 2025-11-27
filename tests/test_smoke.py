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
