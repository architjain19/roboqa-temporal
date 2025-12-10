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


def test_temporal_sync_validator_importable():
    """
    author: xinxin
    reviewer: sayali
    category: smoke test
    """
    from roboqa_temporal.synchronization import TemporalSyncValidator
    assert TemporalSyncValidator is not None


def test_temporal_sync_validator_instantiation():
    """
    author: xinxin
    reviewer: sayali
    category: smoke test
    """
    from roboqa_temporal.synchronization import TemporalSyncValidator
    validator = TemporalSyncValidator()
    assert validator is not None
    assert hasattr(validator, 'validate')
    assert hasattr(validator, 'analyze_streams')


def test_sensor_stream_importable():
    """
    author: xinxin
    reviewer: sayali
    category: smoke test
    """
    from roboqa_temporal.synchronization import SensorStream
    assert SensorStream is not None


def test_temporal_sync_report_importable():
    """
    author: xinxin
    reviewer: sayali
    category: smoke test
    """
    from roboqa_temporal.synchronization import TemporalSyncReport
    assert TemporalSyncReport is not None


def test_pairwise_drift_result_importable():
    """
    author: xinxin
    reviewer: sayali
    category: smoke test
    """
    from roboqa_temporal.synchronization import PairwiseDriftResult
    assert PairwiseDriftResult is not None


def test_calibration_quality_validator_importable():
    """
    author: dharinesh
    reviewer: xinxin
    category: smoke test
    """
    from roboqa_temporal.fusion import CalibrationQualityValidator
    assert CalibrationQualityValidator is not None


def test_calibration_quality_validator_instantiation():
    """
    author: dharinesh
    reviewer: xinxin
    category: smoke test
    """
    from roboqa_temporal.fusion import CalibrationQualityValidator
    validator = CalibrationQualityValidator(output_dir="reports/test_fusion")
    assert validator is not None
    assert hasattr(validator, 'analyze_dataset')


def test_calibration_stream_importable():
    """
    author: dharinesh
    reviewer: xinxin
    category: smoke test
    """
    from roboqa_temporal.fusion import CalibrationStream
    assert CalibrationStream is not None


def test_calibration_quality_report_importable():
    """
    author: dharinesh
    reviewer: xinxin
    category: smoke test
    """
    from roboqa_temporal.fusion import CalibrationQualityReport
    assert CalibrationQualityReport is not None


def test_calibration_pair_result_importable():
    """
    author: dharinesh
    reviewer: xinxin
    category: smoke test
    """
    from roboqa_temporal.fusion import CalibrationPairResult
    assert CalibrationPairResult is not None


def test_projection_error_frame_importable():
    """
    author: dharinesh
    reviewer: xinxin
    category: smoke test
    """
    from roboqa_temporal.fusion import ProjectionErrorFrame
    assert ProjectionErrorFrame is not None


def test_illumination_frame_importable():
    """
    author: dharinesh
    reviewer: xinxin
    category: smoke test
    """
    from roboqa_temporal.fusion import IlluminationFrame
    assert IlluminationFrame is not None


def test_moving_object_frame_importable():
    """
    author: dharinesh
    reviewer: xinxin
    category: smoke test
    """
    from roboqa_temporal.fusion import MovingObjectFrame
    assert MovingObjectFrame is not None


def test_health_check_importable():
    """
    author: sayali
    reviewer: xinxin
    category: smoke test
    """
    from roboqa_temporal.health_reporting import run_health_check
    assert run_health_check is not None


def test_temporal_score_computation_importable():
    """
    author: sayali
    reviewer: xinxin
    category: smoke test
    """
    from roboqa_temporal.health_reporting import compute_temporal_score
    assert compute_temporal_score is not None


def test_anomaly_score_computation_importable():
    """
    author: sayali
    reviewer: xinxin
    category: smoke test
    """
    from roboqa_temporal.health_reporting import compute_anomaly_score
    assert compute_anomaly_score is not None


def test_completeness_metrics_computation_importable():
    """
    author: sayali
    reviewer: xinxin
    category: smoke test
    """
    from roboqa_temporal.health_reporting import compute_completeness_metrics
    assert compute_completeness_metrics is not None


def test_curation_recommendation_importable():
    """
    author: sayali
    reviewer: xinxin
    category: smoke test
    """
    from roboqa_temporal.health_reporting.curation import CurationRecommendation
    assert CurationRecommendation is not None


def test_curation_recommendations_function_importable():
    """
    author: sayali
    reviewer: xinxin
    category: smoke test
    """
    from roboqa_temporal.health_reporting import generate_curation_recommendations
    assert generate_curation_recommendations is not None


def test_export_functions_importable():
    """
    author: sayali
    reviewer: xinxin
    category: smoke test
    """
    from roboqa_temporal.health_reporting import export_csv, export_json, export_yaml
    assert export_csv is not None
    assert export_json is not None
    assert export_yaml is not None
