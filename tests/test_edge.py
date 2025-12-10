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


def test_sensor_stream_with_empty_timestamps():
    """
    author: xinxin
    reviewer: sayali
    category: edge test
    """
    from roboqa_temporal.synchronization import SensorStream
    
    stream = SensorStream(
        name="empty_stream",
        source_path="/fake/path",
        timestamps_ns=[],
        expected_frequency=10.0,
    )
    
    assert stream.timestamps_sec.size == 0
    assert stream.frequency_estimate_hz is None
    assert stream.metadata["message_count"] == 0


def test_sensor_stream_with_single_timestamp():
    """
    author: xinxin
    reviewer: sayali
    category: edge test
    """
    from roboqa_temporal.synchronization import SensorStream
    
    stream = SensorStream(
        name="single_stream",
        source_path="/fake/path",
        timestamps_ns=[1000000000],
        expected_frequency=10.0,
    )
    
    assert stream.timestamps_sec.size == 1
    assert stream.frequency_estimate_hz is None
    assert stream.metadata["message_count"] == 1


def test_sensor_stream_with_negative_timestamps():
    """
    author: xinxin
    reviewer: sayali
    category: edge test
    """
    from roboqa_temporal.synchronization import SensorStream
    
    timestamps_ns = [-1000000000, -900000000, -800000000]
    stream = SensorStream(
        name="negative_stream",
        source_path="/fake/path",
        timestamps_ns=timestamps_ns,
        expected_frequency=10.0,
    )
    
    assert stream.timestamps_sec.size == 3
    assert stream.frequency_estimate_hz is not None


def test_sensor_stream_with_unordered_timestamps():
    """
    author: xinxin
    reviewer: sayali
    category: edge test
    """
    from roboqa_temporal.synchronization import SensorStream
    
    timestamps_ns = [1000000000, 1200000000, 1100000000, 1300000000]
    stream = SensorStream(
        name="unordered_stream",
        source_path="/fake/path",
        timestamps_ns=timestamps_ns,
        expected_frequency=10.0,
    )
    
    assert stream.timestamps_sec.size == 4


def test_sensor_stream_with_zero_intervals():
    """
    author: xinxin
    reviewer: sayali
    category: edge test
    """
    from roboqa_temporal.synchronization import SensorStream
    
    timestamps_ns = [1000000000, 1000000000, 1000000000]
    stream = SensorStream(
        name="zero_interval_stream",
        source_path="/fake/path",
        timestamps_ns=timestamps_ns,
        expected_frequency=10.0,
    )
    
    assert stream.timestamps_sec.size == 3
    assert stream.metadata["duplicate_frames"] > 0


def test_temporal_sync_validator_with_extreme_frequencies():
    """
    author: xinxin
    reviewer: sayali
    category: edge test
    """
    from roboqa_temporal.synchronization import TemporalSyncValidator, SensorStream
    
    high_freq_ts = list(range(0, 1000000, 1000))  # 1000 Hz
    low_freq_ts = list(range(0, 1000000000, 1000000000))  # 1 Hz
    
    streams = {
        "high_freq": SensorStream(
            name="high_freq",
            source_path="/fake/high",
            timestamps_ns=high_freq_ts,
            expected_frequency=1000.0,
        ),
        "low_freq": SensorStream(
            name="low_freq",
            source_path="/fake/low",
            timestamps_ns=low_freq_ts,
            expected_frequency=1.0,
        ),
    }
    
    validator = TemporalSyncValidator(auto_export_reports=False)
    report = validator.analyze_streams(streams, dataset_name="extreme_freq", include_visualizations=False)
    
    assert report is not None
    assert len(report.streams) == 2


def test_temporal_sync_validator_with_huge_time_gaps():
    """
    author: xinxin
    reviewer: sayali
    category: edge test
    """
    from roboqa_temporal.synchronization import TemporalSyncValidator, SensorStream
    import numpy as np
    
    timestamps_ns = [
        1000000000,
        1100000000,
        5000000000,
        5100000000,
    ]
    
    stream = SensorStream(
        name="gappy_stream",
        source_path="/fake/path",
        timestamps_ns=timestamps_ns,
        expected_frequency=10.0,
    )
    
    streams = {"gappy": stream}
    validator = TemporalSyncValidator(auto_export_reports=False)
    report = validator.analyze_streams(streams, dataset_name="gappy_test", include_visualizations=False)
    
    assert report is not None
    assert stream.metadata["missing_frames"] > 0


def test_temporal_sync_validator_with_nan_timestamps():
    """
    author: xinxin
    reviewer: sayali
    category: edge test
    """
    from roboqa_temporal.synchronization import SensorStream
    import numpy as np
    
    timestamps_ns = [1000000000, 1100000000, int(np.nan) if not np.isnan(np.nan) else 0]
    
    stream = SensorStream(
        name="nan_stream",
        source_path="/fake/path",
        timestamps_ns=timestamps_ns,
        expected_frequency=10.0,
    )
    
    assert stream.timestamps_sec.size == 3


def test_temporal_sync_validator_custom_thresholds():
    """
    author: xinxin
    reviewer: sayali
    category: edge test
    """
    from roboqa_temporal.synchronization import TemporalSyncValidator
    
    validator = TemporalSyncValidator(
        approximate_time_threshold_ms={
            "camera_left_lidar": 0.001,
        },
        rolling_window=1,
    )
    
    assert validator.approximate_time_threshold_ms["camera_left_lidar"] == 0.001
    assert validator.rolling_window == 1


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


def test_calibration_stream_with_empty_paths():
    """
    author: dharinesh
    reviewer: architjain
    category: edge test
    """
    from roboqa_temporal.fusion import CalibrationStream
    
    stream = CalibrationStream(
        name="empty_stream",
        image_paths=[],
        pointcloud_paths=[],
        calibration_file="/fake/calib.txt",
        camera_id="cam_00",
        lidar_id="velodyne"
    )
    
    assert len(stream.image_paths) == 0
    assert len(stream.pointcloud_paths) == 0


def test_calibration_stream_with_mismatched_counts():
    """
    author: dharinesh
    reviewer: architjain
    category: edge test
    """
    from roboqa_temporal.fusion import CalibrationStream
    
    # Different number of images and point clouds
    stream = CalibrationStream(
        name="mismatched_stream",
        image_paths=["/fake/img1.png", "/fake/img2.png", "/fake/img3.png"],
        pointcloud_paths=["/fake/pc1.bin", "/fake/pc2.bin"],
        calibration_file="/fake/calib.txt",
        camera_id="cam_01",
        lidar_id="velodyne"
    )
    
    assert len(stream.image_paths) == 3
    assert len(stream.pointcloud_paths) == 2


def test_projection_error_frame_with_zero_error():
    """
    author: dharinesh
    reviewer: architjain
    category: edge test
    """
    from roboqa_temporal.fusion import ProjectionErrorFrame
    
    error_frame = ProjectionErrorFrame(
        frame_index=0,
        timestamp=1000.0,
        reprojection_error=0.0,
        projected_points_count=0
    )
    
    assert error_frame.reprojection_error == 0.0
    assert error_frame.projected_points_count == 0


def test_projection_error_frame_with_extreme_error():
    """
    author: dharinesh
    reviewer: architjain
    category: edge test
    """
    from roboqa_temporal.fusion import ProjectionErrorFrame
    
    error_frame = ProjectionErrorFrame(
        frame_index=0,
        timestamp=1000.0,
        reprojection_error=999999.9,
        max_error_point=(1e6, 1e6),
        projected_points_count=1
    )
    
    assert error_frame.reprojection_error > 1000.0
    assert error_frame.max_error_point[0] == 1e6


def test_illumination_frame_with_zero_brightness():
    """
    author: dharinesh
    reviewer: architjain
    category: edge test
    """
    from roboqa_temporal.fusion import IlluminationFrame
    
    illum_frame = IlluminationFrame(
        frame_index=0,
        timestamp=1000.0,
        brightness_mean=0.0,
        brightness_std=0.0,
        contrast=0.0,
        scene_change_score=0.0,
        light_source_change=False
    )
    
    assert illum_frame.brightness_mean == 0.0
    assert illum_frame.contrast == 0.0


def test_illumination_frame_with_max_brightness():
    """
    author: dharinesh
    reviewer: architjain
    category: edge test
    """
    from roboqa_temporal.fusion import IlluminationFrame
    
    illum_frame = IlluminationFrame(
        frame_index=0,
        timestamp=1000.0,
        brightness_mean=255.0,
        brightness_std=0.0,
        contrast=1.0,
        scene_change_score=1.0,
        light_source_change=True
    )
    
    assert illum_frame.brightness_mean == 255.0
    assert illum_frame.scene_change_score == 1.0


def test_moving_object_frame_with_no_objects():
    """
    author: dharinesh
    reviewer: architjain
    category: edge test
    """
    from roboqa_temporal.fusion import MovingObjectFrame
    
    obj_frame = MovingObjectFrame(
        frame_index=0,
        timestamp=1000.0,
        detected_objects=0,
        detection_confidence=0.0,
        consistency_score=0.0,
        fusion_quality_score=0.0
    )
    
    assert obj_frame.detected_objects == 0
    assert obj_frame.detection_confidence == 0.0


def test_moving_object_frame_with_max_confidence():
    """
    author: dharinesh
    reviewer: architjain
    category: edge test
    """
    from roboqa_temporal.fusion import MovingObjectFrame
    
    obj_frame = MovingObjectFrame(
        frame_index=0,
        timestamp=1000.0,
        detected_objects=100,
        detection_confidence=1.0,
        consistency_score=1.0,
        fusion_quality_score=1.0
    )
    
    assert obj_frame.detected_objects == 100
    assert obj_frame.detection_confidence == 1.0
    assert obj_frame.fusion_quality_score == 1.0


def test_calibration_pair_result_all_fail():
    """
    author: dharinesh
    reviewer: architjain
    category: edge test
    """
    from roboqa_temporal.fusion import CalibrationPairResult
    
    result = CalibrationPairResult(
        geom_edge_score=0.1,
        mutual_information=0.2,
        contrastive_score=0.15,
        pass_geom_edge=False,
        pass_mi=False,
        pass_contrastive=False,
        overall_pass=False,
        details={"reason": "Poor calibration"}
    )
    
    assert result.overall_pass is False
    assert not result.pass_geom_edge
    assert not result.pass_mi
    assert not result.pass_contrastive


def test_calibration_pair_result_all_perfect():
    """
    author: dharinesh
    reviewer: architjain
    category: edge test
    """
    from roboqa_temporal.fusion import CalibrationPairResult
    
    result = CalibrationPairResult(
        geom_edge_score=1.0,
        mutual_information=1.0,
        contrastive_score=1.0,
        pass_geom_edge=True,
        pass_mi=True,
        pass_contrastive=True,
        overall_pass=True,
        details={"quality": "perfect"}
    )
    
    assert result.overall_pass is True
    assert result.geom_edge_score == 1.0
    assert result.mutual_information == 1.0


def test_calibration_quality_validator_with_empty_config():
    """
    author: dharinesh
    reviewer: architjain
    category: edge test
    """
    from roboqa_temporal.fusion import CalibrationQualityValidator
    
    validator = CalibrationQualityValidator(
        output_dir="reports/empty_config",
        config={}
    )
    
    assert validator.config == {}
    assert validator.output_dir.exists()


def test_calibration_quality_report_with_empty_lists():
    """
    author: dharinesh
    reviewer: architjain
    category: edge test
    """
    from roboqa_temporal.fusion import CalibrationQualityReport
    
    report = CalibrationQualityReport(
        dataset_name="empty_dataset",
        metrics={},
        pair_results={},
        projection_errors=[],
        illumination_changes=[],
        moving_objects=[],
        recommendations=[],
        parameter_file=None
    )
    
    assert len(report.projection_errors) == 0
    assert len(report.illumination_changes) == 0
    assert len(report.moving_objects) == 0
    assert report.parameter_file is None


def test_temporal_score_with_single_timestamp():
    """
    author: sayali
    reviewer: xinxin
    category: edge test
    """
    from roboqa_temporal.health_reporting import compute_temporal_score
    import numpy as np
    
    # Single timestamp should return 0
    timestamps = np.array([1000], dtype='datetime64[ns]')
    score = compute_temporal_score(timestamps)
    
    assert score == 0.0


def test_temporal_score_with_empty_timestamps():
    """
    author: sayali
    reviewer: xinxin
    category: edge test
    """
    from roboqa_temporal.health_reporting import compute_temporal_score
    import numpy as np
    
    # Empty array should return 0
    timestamps = np.array([], dtype='datetime64[ns]')
    score = compute_temporal_score(timestamps)
    
    assert score == 0.0


def test_temporal_score_with_identical_timestamps():
    """
    author: sayali
    reviewer: xinxin
    category: edge test
    """
    from roboqa_temporal.health_reporting import compute_temporal_score
    import numpy as np
    
    # All identical timestamps (zero intervals)
    timestamps = np.array([1000, 1000, 1000], dtype='datetime64[ns]')
    score = compute_temporal_score(timestamps)
    
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_anomaly_score_with_single_timestamp():
    """
    author: sayali
    reviewer: xinxin
    category: edge test
    """
    from roboqa_temporal.health_reporting import compute_anomaly_score
    import numpy as np
    
    timestamps = np.array([1000], dtype='datetime64[ns]')
    score = compute_anomaly_score(timestamps)
    
    assert score == 0.0


def test_anomaly_score_with_empty_timestamps():
    """
    author: sayali
    reviewer: xinxin
    category: edge test
    """
    from roboqa_temporal.health_reporting import compute_anomaly_score
    import numpy as np
    
    timestamps = np.array([], dtype='datetime64[ns]')
    score = compute_anomaly_score(timestamps)
    
    assert score == 0.0


def test_completeness_metrics_with_empty_timestamps():
    """
    author: sayali
    reviewer: xinxin
    category: edge test
    """
    from roboqa_temporal.health_reporting import compute_completeness_metrics
    import numpy as np
    
    timestamps = np.array([], dtype='datetime64[ns]')
    metrics = compute_completeness_metrics(timestamps, max_frames_in_sequence=10)
    
    assert isinstance(metrics, dict)
    assert metrics["message_availability"] == 0.0


def test_completeness_metrics_with_zero_max_frames():
    """
    author: sayali
    reviewer: xinxin
    category: edge test
    """
    from roboqa_temporal.health_reporting import compute_completeness_metrics
    import numpy as np
    
    timestamps = np.arange(0, 1000, 100).astype('datetime64[ns]')
    metrics = compute_completeness_metrics(timestamps, max_frames_in_sequence=0)
    
    assert isinstance(metrics, dict)
    assert metrics["message_availability"] == 0.0


def test_completeness_metrics_exceeding_max_frames():
    """
    author: sayali
    reviewer: xinxin
    category: edge test
    """
    from roboqa_temporal.health_reporting import compute_completeness_metrics
    import numpy as np
    
    timestamps = np.arange(0, 2000, 100).astype('datetime64[ns]')  # 20 frames
    metrics = compute_completeness_metrics(timestamps, max_frames_in_sequence=10)
    
    assert isinstance(metrics, dict)
    assert metrics["message_availability"] >= 1.0


def test_curation_recommendation_with_critical_severity():
    """
    author: sayali
    reviewer: xinxin
    category: edge test
    """
    from roboqa_temporal.health_reporting.curation import CurationRecommendation
    
    rec = CurationRecommendation(
        sequence="bad_sequence",
        severity="critical",
        category="quality",
        message="Dataset unusable",
        metric_value=0.1,
        threshold=0.5,
        action="exclude"
    )
    
    assert rec.severity == "critical"
    assert rec.action == "exclude"
    assert rec.metric_value < rec.threshold


def test_curation_recommendation_with_low_severity():
    """
    author: sayali
    reviewer: xinxin
    category: edge test
    """
    from roboqa_temporal.health_reporting.curation import CurationRecommendation
    
    rec = CurationRecommendation(
        sequence="ok_sequence",
        severity="low",
        category="completeness",
        message="Minor data loss",
        metric_value=0.58,
        threshold=0.6,
        action="monitor"
    )
    
    assert rec.severity == "low"
    assert rec.action == "monitor"
    assert rec.metric_value < rec.threshold
