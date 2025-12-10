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
