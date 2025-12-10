"""
Pattern Tests for RoboQA-Temporal

These tests verify common usage patterns and workflows that users
would typically follow when using the package. They test realistic
combinations of operations and integration between components.
"""

import pytest
import numpy as np
from pathlib import Path
from roboqa_temporal.detection import AnomalyDetector
from roboqa_temporal.detection.detector import DetectionResult
from roboqa_temporal.loader.bag_loader import PointCloudFrame
from roboqa_temporal.preprocessing import Preprocessor
from roboqa_temporal.reporting import ReportGenerator


def test_pattern_preprocess_then_detect():
    """
    author: architjain
    reviewer: dharinesh
    category: pattern test
    """
    # Step 1: Create synthetic frames
    frames = [
        PointCloudFrame(
            timestamp=1000.0 + i * 100,
            frame_id=f"frame_{i}",
            points=np.random.rand(200, 3) * 10
        )
        for i in range(10)
    ]
    
    # Step 2: Preprocess
    preprocessor = Preprocessor(voxel_size=0.5, remove_outliers=True)
    processed_frames = preprocessor.process_sequence(frames)
    
    # Step 3: Detect anomalies
    detector = AnomalyDetector()
    result = detector.detect(processed_frames)
    
    assert isinstance(result, DetectionResult)
    assert len(processed_frames) <= len(frames)


def test_pattern_detect_then_report():
    """
    author: architjain
    reviewer: dharinesh
    category: pattern test
    """
    # Step 1: Create and detect
    frames = [
        PointCloudFrame(
            timestamp=1000.0 + i * 100,
            frame_id=f"frame_{i}",
            points=np.random.rand(150, 3) * 10
        )
        for i in range(5)
    ]
    
    detector = AnomalyDetector()
    result = detector.detect(frames)
    
    # Step 2: Generate report
    generator = ReportGenerator()
    report_data = generator.generate(result, "test_bag")
    
    assert report_data is not None
    assert isinstance(report_data, dict)


def test_pattern_selective_detection_workflow():
    """
    author: architjain
    reviewer: dharinesh
    category: pattern test
    """
    frames = [
        PointCloudFrame(
            timestamp=1000.0 + i * 100,
            frame_id=f"frame_{i}",
            points=np.random.rand(100, 3) * 10
        )
        for i in range(8)
    ]
    
    # Use only spatial and temporal detection
    detector = AnomalyDetector(
        enable_density_detection=False,
        enable_spatial_detection=True,
        enable_ghost_detection=False,
        enable_temporal_detection=True,
    )
    
    result = detector.detect(frames)
    
    # Verify only requested detectors ran
    assert isinstance(result, DetectionResult)


def test_pattern_sync_validator_workflow():
    """
    author: xinxin
    reviewer: sayali
    category: pattern test
    """
    from roboqa_temporal.synchronization import TemporalSyncValidator, SensorStream
    
    # Create synthetic sensor streams with pairs that will be analyzed
    camera_left_ts = list(range(0, 1000000000, 100000000))   # 10 Hz
    lidar_ts = list(range(0, 1000000000, 100000000))         # 10 Hz
    camera_right_ts = list(range(0, 1000000000, 100000000))  # 10 Hz
    
    streams = {
        "camera_left": SensorStream(
            name="camera_left",
            source_path="/fake/camera_left",
            timestamps_ns=camera_left_ts,
            expected_frequency=10.0,
        ),
        "camera_right": SensorStream(
            name="camera_right",
            source_path="/fake/camera_right",
            timestamps_ns=camera_right_ts,
            expected_frequency=10.0,
        ),
        "lidar": SensorStream(
            name="lidar",
            source_path="/fake/lidar",
            timestamps_ns=lidar_ts,
            expected_frequency=10.0,
        ),
    }
    
    # Initialize validator
    validator = TemporalSyncValidator(output_dir="reports/test_sync", auto_export_reports=False)
    
    # Analyze streams
    report = validator.analyze_streams(streams, dataset_name="test_sync", include_visualizations=False)
    
    assert report is not None
    assert len(report.streams) == 3
    assert len(report.pair_results) > 0


def test_pattern_sync_validator_with_drift():
    """
    author: xinxin
    reviewer: sayali
    category: pattern test
    """
    from roboqa_temporal.synchronization import TemporalSyncValidator, SensorStream
    import numpy as np
    
    # Create streams with temporal drift
    base_timestamps = np.arange(0, 1000000000, 100000000, dtype=np.int64)
    camera_left_timestamps = list(base_timestamps)
    # Add progressive drift to lidar (1ms per frame)
    lidar_timestamps = list(base_timestamps + np.arange(len(base_timestamps)) * 1000000)
    
    streams = {
        "camera_left": SensorStream(
            name="camera_left",
            source_path="/fake/camera",
            timestamps_ns=camera_left_timestamps,
            expected_frequency=10.0,
        ),
        "lidar": SensorStream(
            name="lidar",
            source_path="/fake/lidar",
            timestamps_ns=lidar_timestamps,
            expected_frequency=10.0,
        ),
    }
    
    validator = TemporalSyncValidator(auto_export_reports=False)
    report = validator.analyze_streams(streams, dataset_name="test_drift", include_visualizations=False)
    
    assert len(report.pair_results) > 0
    for pair_result in report.pair_results.values():
        assert hasattr(pair_result, "drift_rate_ms_per_s")


def test_pattern_multi_sensor_synchronization():
    """
    author: xinxin
    reviewer: sayali
    category: pattern test
    """
    from roboqa_temporal.synchronization import TemporalSyncValidator, SensorStream
    
    # Create multiple sensor streams
    camera_left_ts = list(range(0, 1000000000, 100000000))   # 10 Hz
    camera_right_ts = list(range(0, 1000000000, 100000000))  # 10 Hz
    lidar_ts = list(range(0, 1000000000, 100000000))         # 10 Hz
    imu_ts = list(range(0, 1000000000, 10000000))            # 100 Hz
    
    streams = {
        "camera_left": SensorStream(
            name="camera_left",
            source_path="/fake/cam_left",
            timestamps_ns=camera_left_ts,
            expected_frequency=10.0,
        ),
        "camera_right": SensorStream(
            name="camera_right",
            source_path="/fake/cam_right",
            timestamps_ns=camera_right_ts,
            expected_frequency=10.0,
        ),
        "lidar": SensorStream(
            name="lidar",
            source_path="/fake/lidar",
            timestamps_ns=lidar_ts,
            expected_frequency=10.0,
        ),
        "imu": SensorStream(
            name="imu",
            source_path="/fake/imu",
            timestamps_ns=imu_ts,
            expected_frequency=100.0,
        ),
    }
    
    validator = TemporalSyncValidator(auto_export_reports=False)
    report = validator.analyze_streams(streams, dataset_name="multi_sensor", include_visualizations=False)
    
    assert len(report.streams) == 4
    assert len(report.pair_results) > 0


def test_pattern_sync_report_serialization():
    """
    author: xinxin
    reviewer: sayali
    category: pattern test
    """
    from roboqa_temporal.synchronization import TemporalSyncValidator, SensorStream
    
    timestamps_camera = list(range(0, 500000000, 100000000))
    timestamps_lidar = list(range(0, 500000000, 100000000))
    
    streams = {
        "camera_left": SensorStream(
            name="camera_left",
            source_path="/fake/camera",
            timestamps_ns=timestamps_camera,
            expected_frequency=10.0,
        ),
        "lidar": SensorStream(
            name="lidar",
            source_path="/fake/lidar",
            timestamps_ns=timestamps_lidar,
            expected_frequency=10.0,
        ),
    }
    
    validator = TemporalSyncValidator(auto_export_reports=False)
    report = validator.analyze_streams(streams, dataset_name="test_serial", include_visualizations=False)
    
    # Test serialization
    report_dict = report.to_dict()
    
    assert isinstance(report_dict, dict)
    assert "streams" in report_dict
    assert "metrics" in report_dict
    assert "recommendations" in report_dict
    assert "generated_at" in report_dict


def test_pattern_iterative_detection():
    """
    author: architjain
    reviewer: dharinesh
    category: pattern test
    """
    all_frames = [
        PointCloudFrame(
            timestamp=1000.0 + i * 100,
            frame_id=f"frame_{i}",
            points=np.random.rand(100, 3) * 10
        )
        for i in range(20)
    ]
    
    detector = AnomalyDetector()
    
    # Detect on first half
    result1 = detector.detect(all_frames[:10])
    
    # Detect on second half
    result2 = detector.detect(all_frames[10:])
    
    # Both should be valid results
    assert isinstance(result1, DetectionResult)
    assert isinstance(result2, DetectionResult)


def test_pattern_custom_detector_thresholds():
    """
    author: architjain
    reviewer: dharinesh
    category: pattern test
    """
    frames = [
        PointCloudFrame(
            timestamp=1000.0 + i * 100,
            frame_id=f"frame_{i}",
            points=np.random.rand(100, 3) * 10
        )
        for i in range(5)
    ]
    
    # Strict thresholds (more sensitive)
    strict_detector = AnomalyDetector(
        density_threshold=0.3,
        spatial_threshold=0.2,
        ghost_threshold=0.4,
        temporal_threshold=0.3
    )
    
    # Lenient thresholds (less sensitive)
    lenient_detector = AnomalyDetector(
        density_threshold=0.8,
        spatial_threshold=0.7,
        ghost_threshold=0.9,
        temporal_threshold=0.8
    )
    
    strict_result = strict_detector.detect(frames)
    lenient_result = lenient_detector.detect(frames)
    
    assert isinstance(strict_result, DetectionResult)
    assert isinstance(lenient_result, DetectionResult)


def test_pattern_batch_processing():
    """
    author: architjain
    reviewer: dharinesh
    category: pattern test
    """
    detector = AnomalyDetector()
    
    batches = [
        [PointCloudFrame(
            timestamp=1000.0 + b * 1000 + i * 100,
            frame_id=f"batch{b}_frame{i}",
            points=np.random.rand(100, 3) * 10
        ) for i in range(5)]
        for b in range(3)
    ]
    
    results = []
    for batch in batches:
        processed = Preprocessor(voxel_size=0.5).process_sequence(batch)
        result = detector.detect(processed)
        results.append(result)
    
    assert len(results) == 3
    assert all(isinstance(r, DetectionResult) for r in results)


def test_pattern_full_pipeline():
    """
    author: architjain
    reviewer: dharinesh
    category: pattern test
    """
    # 1. Create synthetic data
    raw_frames = [
        PointCloudFrame(
            timestamp=1000.0 + i * 100,
            frame_id=f"frame_{i}",
            points=np.random.rand(300, 3) * 10
        )
        for i in range(10)
    ]
    
    # 2. Preprocess
    preprocessor = Preprocessor(remove_outliers=True)
    cleaned = preprocessor.process_sequence(raw_frames)
    processed = Preprocessor(voxel_size=0.5).process_sequence(cleaned)
    
    # 3. Detect
    detector = AnomalyDetector()
    detection_result = detector.detect(processed)
    
    # 4. Report
    generator = ReportGenerator()
    report = generator.generate(detection_result, "test_bag")
    
    # Verify pipeline completed
    assert len(processed) > 0
    assert isinstance(detection_result, DetectionResult)
    assert report is not None


def test_pattern_fusion_quality_assessment():
    """
    author: dharinesh
    reviewer: architjain
    category: pattern test
    """
    from roboqa_temporal.fusion import (
        CalibrationQualityValidator,
        CalibrationStream,
        CalibrationPairResult
    )
    
    # Create a validator instance
    validator = CalibrationQualityValidator(
        output_dir="reports/test_fusion",
        config={"edge_threshold": 0.7}
    )
    
    assert validator is not None
    assert validator.output_dir.exists()


def test_pattern_calibration_stream_workflow():
    """
    author: dharinesh
    reviewer: architjain
    category: pattern test
    """
    from roboqa_temporal.fusion import CalibrationStream
    
    # Create multiple calibration streams
    streams = []
    for i in range(3):
        stream = CalibrationStream(
            name=f"pair_{i}",
            image_paths=[f"/fake/img_{i}_{j}.png" for j in range(5)],
            pointcloud_paths=[f"/fake/pc_{i}_{j}.bin" for j in range(5)],
            calibration_file=f"/fake/calib_{i}.txt",
            camera_id=f"cam_0{i}",
            lidar_id="velodyne"
        )
        streams.append(stream)
    
    # Verify streams
    assert len(streams) == 3
    for stream in streams:
        assert len(stream.image_paths) == 5
        assert len(stream.pointcloud_paths) == 5


def test_pattern_projection_error_tracking():
    """
    author: dharinesh
    reviewer: architjain
    category: pattern test
    """
    from roboqa_temporal.fusion import ProjectionErrorFrame
    
    # Simulate projection errors over time
    error_frames = []
    for i in range(10):
        # Simulate increasing error trend
        error = 1.0 + i * 0.5
        error_frame = ProjectionErrorFrame(
            frame_index=i,
            timestamp=1000.0 + i * 100,
            reprojection_error=error,
            max_error_point=(float(i * 10), float(i * 20)),
            projected_points_count=100 - i * 5,
            error_trend="increasing" if i > 5 else "stable"
        )
        error_frames.append(error_frame)
    
    assert len(error_frames) == 10
    assert error_frames[0].reprojection_error < error_frames[-1].reprojection_error
    assert sum(1 for f in error_frames if f.error_trend == "increasing") > 0


def test_pattern_illumination_detection():
    """
    author: dharinesh
    reviewer: architjain
    category: pattern test
    """
    from roboqa_temporal.fusion import IlluminationFrame
    
    # Simulate illumination changes
    illum_frames = []
    for i in range(8):
        brightness = 100.0 + np.random.randn() * 20.0
        illum_frame = IlluminationFrame(
            frame_index=i,
            timestamp=1000.0 + i * 100,
            brightness_mean=brightness,
            brightness_std=15.0 + np.random.randn() * 5.0,
            contrast=0.6 + np.random.rand() * 0.2,
            scene_change_score=0.1 if i < 4 else 0.8,
            light_source_change=(i == 4)
        )
        illum_frames.append(illum_frame)
    
    assert len(illum_frames) == 8
    assert sum(1 for f in illum_frames if f.light_source_change) == 1
    assert any(f.scene_change_score > 0.5 for f in illum_frames)


def test_pattern_moving_object_consistency():
    """
    author: dharinesh
    reviewer: architjain
    category: pattern test
    """
    from roboqa_temporal.fusion import MovingObjectFrame
    
    # Simulate moving object detection
    obj_frames = []
    for i in range(12):
        obj_frame = MovingObjectFrame(
            frame_index=i,
            timestamp=1000.0 + i * 100,
            detected_objects=2 + (i % 3),
            detection_confidence=0.8 + np.random.rand() * 0.15,
            consistency_score=0.85 if i < 6 else 0.70,
            fusion_quality_score=0.80 + np.random.rand() * 0.1
        )
        obj_frames.append(obj_frame)
    
    assert len(obj_frames) == 12
    avg_confidence = np.mean([f.detection_confidence for f in obj_frames])
    assert 0.8 <= avg_confidence <= 1.0
    assert all(f.detected_objects > 0 for f in obj_frames)


def test_pattern_fusion_quality_report_generation():
    """
    author: dharinesh
    reviewer: architjain
    category: pattern test
    """
    from roboqa_temporal.fusion import (
        CalibrationQualityReport,
        CalibrationPairResult,
        ProjectionErrorFrame
    )
    
    # Create sample pair result
    pair_result = CalibrationPairResult(
        geom_edge_score=0.82,
        mutual_information=0.78,
        contrastive_score=0.80,
        pass_geom_edge=True,
        pass_mi=True,
        pass_contrastive=True,
        overall_pass=True,
        details={"frames": 10}
    )
    
    # Create sample projection errors
    proj_errors = [
        ProjectionErrorFrame(
            frame_index=i,
            timestamp=1000.0 + i * 100,
            reprojection_error=2.0 + i * 0.1,
            projected_points_count=100
        )
        for i in range(5)
    ]
    
    # Create report
    report = CalibrationQualityReport(
        dataset_name="test_dataset",
        metrics={"overall_score": 0.85},
        pair_results={"cam_lidar": pair_result},
        projection_errors=proj_errors,
        illumination_changes=[],
        moving_objects=[],
        recommendations=["Calibration is stable"],
        parameter_file="/fake/params.yaml"
    )
    
    assert report.dataset_name == "test_dataset"
    assert len(report.pair_results) == 1
    assert len(report.projection_errors) == 5
    assert len(report.recommendations) == 1


def test_pattern_health_scoring_workflow():
    """
    author: sayali
    reviewer: xinxin
    category: pattern test
    """
    from roboqa_temporal.health_reporting import (
        compute_temporal_score,
        compute_anomaly_score,
        compute_completeness_metrics
    )
    import numpy as np
    
    # Create sensor timestamps
    camera_ts = np.arange(0, 2000, 100).astype('datetime64[ns]')
    lidar_ts = np.arange(0, 2000, 100).astype('datetime64[ns]')
    imu_ts = np.arange(0, 2000, 10).astype('datetime64[ns]')
    
    # Compute temporal health
    camera_temporal = compute_temporal_score(camera_ts)
    lidar_temporal = compute_temporal_score(lidar_ts)
    imu_temporal = compute_temporal_score(imu_ts)
    
    # Compute anomaly detection
    camera_anomaly = compute_anomaly_score(camera_ts)
    lidar_anomaly = compute_anomaly_score(lidar_ts)
    
    # Verify scores
    assert 0.0 <= camera_temporal <= 1.0
    assert 0.0 <= lidar_temporal <= 1.0
    assert 0.0 <= camera_anomaly <= 1.0
    assert camera_temporal > 0.8


def test_pattern_completeness_tracking():
    """
    author: sayali
    reviewer: xinxin
    category: pattern test
    """
    from roboqa_temporal.health_reporting import compute_completeness_metrics
    import numpy as np
    
    # Simulate multiple sequences with varying completeness
    sequences_data = []
    for seq_id in range(3):
        # Each sequence has different frame counts
        max_frames = 100
        n_frames = 100 - (seq_id * 20)
        timestamps = np.arange(0, n_frames * 100, 100).astype('datetime64[ns]')
        metrics = compute_completeness_metrics(timestamps, max_frames)
        sequences_data.append(metrics)
    
    assert len(sequences_data) == 3
    assert sequences_data[0]["message_availability"] == 1.0
    assert sequences_data[1]["message_availability"] == 0.8
    assert sequences_data[2]["message_availability"] == 0.6


def test_pattern_curation_recommendation_generation():
    """
    author: sayali
    reviewer: xinxin
    category: pattern test
    """
    from roboqa_temporal.health_reporting import generate_curation_recommendations
    import pandas as pd
    
    # Create sample per-sequence data
    df_per_sequence = pd.DataFrame({
        "sequence": ["seq_1", "seq_2", "seq_3"],
        "temporal_score": [0.9, 0.5, 0.2],
        "anomaly_score": [0.95, 0.7, 0.3],
        "dim_timeliness": [0.92, 0.6, 0.25],
        "dim_completeness": [0.95, 0.85, 0.4],
        "overall_quality_score": [0.92, 0.65, 0.3],
    })
    
    df_per_sensor = pd.DataFrame()
    
    # Generate recommendations
    recs = generate_curation_recommendations(
        df_per_sensor,
        df_per_sequence,
        temporal_threshold=0.6,
        completeness_threshold=0.6,
        quality_threshold=0.5
    )
    
    assert isinstance(recs, list)
    assert len(recs) > 0
    seq_names = [r.sequence for r in recs]
    assert "seq_2" in seq_names or "seq_3" in seq_names


def test_pattern_multi_sensor_health_assessment():
    """
    author: sayali
    reviewer: xinxin
    category: pattern test
    """
    from roboqa_temporal.health_reporting import (
        compute_temporal_score,
        compute_completeness_metrics
    )
    import numpy as np
    
    # Simulate multi-sensor sequence
    sensors = {
        "camera_left": np.arange(0, 2000, 100).astype('datetime64[ns]'),
        "camera_right": np.arange(0, 2000, 100).astype('datetime64[ns]'),
        "lidar": np.arange(0, 2000, 100).astype('datetime64[ns]'),
        "imu": np.arange(0, 2000, 10).astype('datetime64[ns]'),
    }
    
    results = {}
    max_frames = 20
    
    for sensor_name, timestamps in sensors.items():
        temporal = compute_temporal_score(timestamps)
        completeness = compute_completeness_metrics(timestamps, max_frames)
        results[sensor_name] = {
            "temporal": temporal,
            "completeness": completeness["message_availability"]
        }
    
    assert len(results) == 4
    for sensor_name, metrics in results.items():
        assert 0.0 <= metrics["temporal"] <= 1.0
        assert 0.0 <= metrics["completeness"] <= 1.0
