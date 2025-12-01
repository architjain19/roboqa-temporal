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
    if result.detector_results:
        for key in result.detector_results.keys():
            assert key in ["spatial", "temporal"]


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


def test_calibration_validator_pattern_quality_decreases_with_miscalibration(tmp_path):
    """author: Dharineesh Somisetty
    reviewer: <buddy name>
    category: pattern test
    """
    from roboqa_temporal.calibration import (
        CalibrationQualityValidator,
        CalibrationStream,
    )

    validator = CalibrationQualityValidator(output_dir=str(tmp_path))

    miscalibrations = [0.0, 2.0, 5.0, 10.0, 15.0]
    previous_score = float("inf")

    for idx, mis_px in enumerate(miscalibrations):
        pair_name = f"pattern_{idx}"
        
        # Create synthetic calibration stream
        image_paths = [f"/synthetic/{pair_name}/image_{i:06d}.png" for i in range(50)]
        pointcloud_paths = [f"/synthetic/{pair_name}/cloud_{i:06d}.bin" for i in range(50)]
        calib_tag = f"miscalib_{mis_px:.1f}px"
        calibration_file = f"/synthetic/calib/{pair_name}_{calib_tag}.txt"
        
        pair = CalibrationStream(
            name=pair_name,
            image_paths=image_paths,
            pointcloud_paths=pointcloud_paths,
            calibration_file=calibration_file,
            camera_id="image_02",
            lidar_id="velodyne",
        )
        
        report = validator.analyze_sequences({pair_name: pair}, bag_name=f"pattern_{idx}")

        score = report.pair_results[pair_name].geom_edge_score
        assert score <= previous_score + 1e-9
        previous_score = score
