from __future__ import annotations

import numpy as np

from roboqa_temporal.detection.detector import AnomalyDetector, DetectionResult


def test_anomaly_detector_returns_health_metrics(sample_frames):
    np.random.seed(0)  # deterministic sampling inside ghost detector

    detector = AnomalyDetector()
    result = detector.detect(sample_frames)

    assert isinstance(result, DetectionResult)
    assert isinstance(result.anomalies, list)
    assert "overall_health_score" in result.health_metrics
    assert result.health_metrics["avg_points_per_frame"] > 0


def test_anomaly_detector_respects_disabled_detectors(sample_frames):
    detector = AnomalyDetector(
        enable_density_detection=False,
        enable_spatial_detection=True,
        enable_ghost_detection=False,
        enable_temporal_detection=True,
    )
    result = detector.detect(sample_frames)

    assert set(result.detector_results.keys()).issubset({"spatial", "temporal"})
    assert "density" not in result.detector_results
    assert "ghost" not in result.detector_results

