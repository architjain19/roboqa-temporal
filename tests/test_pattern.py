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


class TestBasicWorkflowPatterns:
    """Test basic usage patterns and workflows."""
    
    def test_pattern_preprocess_then_detect(self):
        """Test common pattern: preprocess data then run detection."""
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
    
    def test_pattern_detect_then_report(self):
        """Test common pattern: detect anomalies then generate report."""
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
    
    def test_pattern_selective_detection_workflow(self):
        """Test pattern: selective detection with specific detectors."""
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
    
    def test_pattern_iterative_detection(self):
        """Test pattern: detecting on different subsets of data."""
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
    
    def test_pattern_progressive_preprocessing(self):
        """Test pattern: applying preprocessing steps progressively."""
        frames = [
            PointCloudFrame(
                timestamp=1000.0 + i * 100,
                frame_id=f"frame_{i}",
                points=np.random.rand(500, 3) * 10
            )
            for i in range(5)
        ]
        
        preprocessor = Preprocessor()
        
        # Progressive preprocessing
        step1 = Preprocessor(remove_outliers=True).process_sequence(frames)
        step2 = Preprocessor(voxel_size=0.5).process_sequence(step1)
        
        # Each step should reduce or maintain size
        assert len(step1) <= len(frames)
        assert len(step2) <= len(step1)
        
        for s2_frame, s1_frame in zip(step2, step1):
            assert s2_frame.num_points <= s1_frame.num_points


class TestConfigurationPatterns:
    """Test patterns related to configuration and customization."""
    
    def test_pattern_custom_detector_thresholds(self):
        """Test pattern: customizing detector sensitivity."""
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
    
    def test_pattern_detector_combination_variations(self):
        """Test pattern: trying different detector combinations."""
        frames = [
            PointCloudFrame(
                timestamp=1000.0 + i * 100,
                frame_id=f"frame_{i}",
                points=np.random.rand(100, 3) * 10
            )
            for i in range(5)
        ]
        
        # Test different combinations
        configs = [
            {"enable_density_detection": True, "enable_spatial_detection": True,
             "enable_ghost_detection": False, "enable_temporal_detection": False},
            {"enable_density_detection": False, "enable_spatial_detection": False,
             "enable_ghost_detection": True, "enable_temporal_detection": True},
            {"enable_density_detection": True, "enable_spatial_detection": False,
             "enable_ghost_detection": True, "enable_temporal_detection": False},
        ]
        
        for config in configs:
            detector = AnomalyDetector(**config)
            result = detector.detect(frames)
            assert isinstance(result, DetectionResult)


class TestDataFlowPatterns:
    """Test patterns related to data flow and transformations."""
    
    def test_pattern_frame_filtering_workflow(self):
        """Test pattern: filtering frames based on criteria before detection."""
        all_frames = [
            PointCloudFrame(
                timestamp=1000.0 + i * 100,
                frame_id=f"frame_{i}",
                points=np.random.rand(50 + i * 10, 3) * 10
            )
            for i in range(10)
        ]
        
        # Filter frames with sufficient points
        min_points = 80
        filtered_frames = [f for f in all_frames if f.num_points >= min_points]
        
        detector = AnomalyDetector()
        result = detector.detect(filtered_frames)
        
        assert isinstance(result, DetectionResult)
        assert len(filtered_frames) < len(all_frames)
    
    def test_pattern_temporal_windowing(self):
        """Test pattern: processing data in temporal windows."""
        frames = [
            PointCloudFrame(
                timestamp=1000.0 + i * 100,
                frame_id=f"frame_{i}",
                points=np.random.rand(100, 3) * 10
            )
            for i in range(30)
        ]
        
        # Process in windows of 10 frames
        window_size = 10
        detector = AnomalyDetector()
        
        results = []
        for i in range(0, len(frames), window_size):
            window = frames[i:i + window_size]
            result = detector.detect(window)
            results.append(result)
        
        assert len(results) == 3
        assert all(isinstance(r, DetectionResult) for r in results)
    
    def test_pattern_batch_processing(self):
        """Test pattern: processing multiple batches of frames."""
        detector = AnomalyDetector()
        preprocessor = Preprocessor()
        
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


class TestErrorHandlingPatterns:
    """Test patterns related to error handling and robustness."""
    
    def test_pattern_graceful_degradation_missing_data(self):
        """Test pattern: handling missing or incomplete data gracefully."""
        frames = [
            PointCloudFrame(
                timestamp=1000.0 + i * 100,
                frame_id=f"frame_{i}",
                points=np.random.rand(100, 3) * 10 if i % 2 == 0 else np.array([]).reshape(0, 3)
            )
            for i in range(10)
        ]
        
        # Should handle frames with varying quality
        detector = AnomalyDetector()
        result = detector.detect(frames)
        
        assert isinstance(result, DetectionResult)
    
    def test_pattern_reusing_detector_instance(self):
        """Test pattern: reusing detector instance for multiple datasets."""
        detector = AnomalyDetector()
        
        # First dataset
        frames1 = [
            PointCloudFrame(
                timestamp=1000.0 + i * 100,
                frame_id=f"frame_{i}",
                points=np.random.rand(100, 3) * 10
            )
            for i in range(5)
        ]
        
        # Second dataset
        frames2 = [
            PointCloudFrame(
                timestamp=2000.0 + i * 100,
                frame_id=f"frame_{i}",
                points=np.random.rand(150, 3) * 10
            )
            for i in range(5)
        ]
        
        result1 = detector.detect(frames1)
        result2 = detector.detect(frames2)
        
        assert isinstance(result1, DetectionResult)
        assert isinstance(result2, DetectionResult)
        # Results should be independent
        assert result1 is not result2


class TestIntegrationPatterns:
    """Test patterns for component integration."""
    
    def test_pattern_full_pipeline(self):
        """Test pattern: complete pipeline from data to report."""
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
    
    def test_pattern_comparison_workflow(self):
        """Test pattern: comparing different configurations."""
        frames = [
            PointCloudFrame(
                timestamp=1000.0 + i * 100,
                frame_id=f"frame_{i}",
                points=np.random.rand(100, 3) * 10
            )
            for i in range(5)
        ]
        
        # Compare with and without preprocessing
        detector = AnomalyDetector()
        
        result_raw = detector.detect(frames)
        
        preprocessor = Preprocessor(voxel_size=0.5)
        processed = preprocessor.process_sequence(frames)
        result_processed = detector.detect(processed)
        
        # Both should produce valid results
        assert isinstance(result_raw, DetectionResult)
        assert isinstance(result_processed, DetectionResult)
