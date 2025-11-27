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


class TestEdgeCaseFrames:
    """Test edge cases for PointCloudFrame."""
    
    def test_frame_with_zero_points(self):
        """Test frame with empty points array."""
        points = np.array([]).reshape(0, 3)
        frame = PointCloudFrame(timestamp=1000.0, frame_id="empty", points=points)
        
        assert frame.num_points == 0
        assert frame.points.shape[0] == 0
    
    def test_frame_with_single_point(self):
        """Test frame with exactly one point."""
        points = np.array([[1.0, 2.0, 3.0]])
        frame = PointCloudFrame(timestamp=1000.0, frame_id="single", points=points)
        
        assert frame.num_points == 1
    
    def test_frame_with_nan_values(self):
        """Test frame containing NaN values."""
        points = np.array([[1.0, 2.0, np.nan], [4.0, 5.0, 6.0]])
        frame = PointCloudFrame(timestamp=1000.0, frame_id="nan", points=points)
        
        assert frame.num_points == 2
        assert np.isnan(frame.points[0, 2])
    
    def test_frame_with_inf_values(self):
        """Test frame containing infinity values."""
        points = np.array([[1.0, 2.0, np.inf], [4.0, -np.inf, 6.0]])
        frame = PointCloudFrame(timestamp=1000.0, frame_id="inf", points=points)
        
        assert frame.num_points == 2
        assert np.isinf(frame.points[0, 2])
    
    def test_frame_with_very_large_coordinates(self):
        """Test frame with extremely large coordinate values."""
        points = np.array([[1e10, 1e10, 1e10], [1e15, 1e15, 1e15]])
        frame = PointCloudFrame(timestamp=1000.0, frame_id="large", points=points)
        
        assert frame.num_points == 2
    
    def test_frame_with_negative_timestamp(self):
        """Test frame with negative timestamp."""
        points = np.array([[1.0, 2.0, 3.0]])
        frame = PointCloudFrame(timestamp=-1000.0, frame_id="neg_time", points=points)
        
        assert frame.timestamp == -1000.0
    
    def test_frame_with_zero_timestamp(self):
        """Test frame with zero timestamp."""
        points = np.array([[1.0, 2.0, 3.0]])
        frame = PointCloudFrame(timestamp=0.0, frame_id="zero_time", points=points)
        
        assert frame.timestamp == 0.0


class TestEdgeCaseDetector:
    """Test edge cases for AnomalyDetector."""
    
    def test_detector_with_none_frames(self):
        """Test detector behavior when None is in frame list."""
        detector = AnomalyDetector()
        points = np.random.rand(10, 3)
        frames = [
            PointCloudFrame(timestamp=1000.0, frame_id="f1", points=points),
            None,  # This shouldn't happen but test robustness
        ]
        
        # Should handle gracefully or raise appropriate error
        with pytest.raises((TypeError, AttributeError)):
            detector.detect(frames)
    
    def test_detector_with_all_detectors_disabled(self):
        """Test detector with all detection methods disabled."""
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
    
    def test_detector_with_extreme_thresholds(self):
        """Test detector with threshold values at extremes."""
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
        
        # All thresholds at 1.0
        detector = AnomalyDetector(
            density_threshold=1.0,
            spatial_threshold=1.0,
            ghost_threshold=1.0,
            temporal_threshold=1.0
        )
        result = detector.detect([frame])
        assert isinstance(result, DetectionResult)
    
    def test_detector_with_identical_frames(self):
        """Test detector with multiple identical frames."""
        detector = AnomalyDetector()
        
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        frames = [
            PointCloudFrame(timestamp=1000.0 + i, frame_id=f"f{i}", points=points.copy())
            for i in range(10)
        ]
        
        result = detector.detect(frames)
        assert isinstance(result, DetectionResult)
    
    def test_detector_with_very_sparse_frames(self):
        """Test detector with frames containing very few points."""
        detector = AnomalyDetector()
        
        frames = [
            PointCloudFrame(
                timestamp=1000.0 + i,
                frame_id=f"f{i}",
                points=np.array([[i, i, i]])  # Single point per frame
            )
            for i in range(5)
        ]
        
        result = detector.detect(frames)
        assert isinstance(result, DetectionResult)
    
    def test_detector_with_very_dense_frames(self):
        """Test detector with frames containing many points."""
        detector = AnomalyDetector()
        
        frames = [
            PointCloudFrame(
                timestamp=1000.0 + i,
                frame_id=f"f{i}",
                points=np.random.rand(10000, 3)  # 10k points per frame
            )
            for i in range(3)
        ]
        
        result = detector.detect(frames)
        assert isinstance(result, DetectionResult)


class TestEdgeCasePreprocessor:
    """Test edge cases for Preprocessor."""
    
    def test_preprocessor_downsample_with_zero_voxel_size(self):
        """Test downsampling with zero voxel size."""
        preprocessor = Preprocessor(voxel_size=0.0)
        
        points = np.random.rand(100, 3) * 10
        frame = PointCloudFrame(timestamp=1000.0, frame_id="test", points=points)
        
        # Should handle gracefully or raise appropriate error
        with pytest.raises((ValueError, ZeroDivisionError, RuntimeError)):
            preprocessor.process_sequence([frame])
    
    def test_preprocessor_downsample_with_negative_voxel_size(self):
        """Test downsampling with negative voxel size."""
        preprocessor = Preprocessor(voxel_size=-1.0)
        
        points = np.random.rand(100, 3) * 10
        frame = PointCloudFrame(timestamp=1000.0, frame_id="test", points=points)
        
        # Should handle gracefully or raise appropriate error
        with pytest.raises((ValueError, RuntimeError)):
            preprocessor.process_sequence([frame])
    
    def test_preprocessor_downsample_with_huge_voxel_size(self):
        """Test downsampling with very large voxel size."""
        preprocessor = Preprocessor(voxel_size=1000.0)
        
        points = np.random.rand(100, 3) * 10
        frame = PointCloudFrame(timestamp=1000.0, frame_id="test", points=points)
        
        # With huge voxel size, should collapse to very few points
        result = preprocessor.process_sequence([frame])
        assert len(result) == 1
        assert result[0].num_points <= frame.num_points
    
    def test_preprocessor_remove_outliers_with_single_point(self):
        """Test outlier removal with only one point."""
        preprocessor = Preprocessor(remove_outliers=True)
        
        points = np.array([[1.0, 2.0, 3.0]])
        frame = PointCloudFrame(timestamp=1000.0, frame_id="test", points=points)
        
        result = preprocessor.process_sequence([frame])
        assert len(result) == 1
    
    def test_preprocessor_with_frames_of_varying_sizes(self):
        """Test preprocessing frames with dramatically different point counts."""
        # Disable outlier removal to avoid issues with small point clouds
        preprocessor = Preprocessor(voxel_size=0.5, remove_outliers=False)
        
        frames = [
            PointCloudFrame(timestamp=1000.0, frame_id="f1", points=np.random.rand(5, 3)),
            PointCloudFrame(timestamp=2000.0, frame_id="f2", points=np.random.rand(5000, 3)),
            PointCloudFrame(timestamp=3000.0, frame_id="f3", points=np.random.rand(50, 3)),
        ]
        
        result = preprocessor.process_sequence(frames)
        assert len(result) == 3


class TestEdgeCaseBagLoader:
    """Test edge cases for BagLoader."""
    
    def test_bag_loader_with_nonexistent_path(self):
        """Test BagLoader with path that doesn't exist."""
        with pytest.raises(FileNotFoundError):
            BagLoader("/nonexistent/path/to/bag")
    
    def test_bag_loader_with_empty_string_path(self):
        """Test BagLoader with empty string path."""
        with pytest.raises((FileNotFoundError, RuntimeError)):
            BagLoader("")
    
    def test_bag_loader_with_none_path(self):
        """Test BagLoader with None as path."""
        with pytest.raises((TypeError, FileNotFoundError)):
            BagLoader(None)


class TestEdgeCaseAnomalyDataclass:
    """Test edge cases for Anomaly dataclass."""
    
    def test_anomaly_with_zero_severity(self):
        """Test anomaly with zero severity."""
        anomaly = Anomaly(
            frame_index=0,
            timestamp=1000.0,
            anomaly_type="test",
            severity=0.0,
            description="Zero severity"
        )
        
        assert anomaly.severity == 0.0
    
    def test_anomaly_with_max_severity(self):
        """Test anomaly with maximum severity."""
        anomaly = Anomaly(
            frame_index=0,
            timestamp=1000.0,
            anomaly_type="test",
            severity=1.0,
            description="Max severity"
        )
        
        assert anomaly.severity == 1.0
    
    def test_anomaly_with_negative_frame_index(self):
        """Test anomaly with negative frame index."""
        anomaly = Anomaly(
            frame_index=-1,
            timestamp=1000.0,
            anomaly_type="test",
            severity=0.5,
            description="Negative index"
        )
        
        assert anomaly.frame_index == -1
    
    def test_anomaly_with_empty_metadata(self):
        """Test anomaly with empty metadata dict."""
        anomaly = Anomaly(
            frame_index=0,
            timestamp=1000.0,
            anomaly_type="test",
            severity=0.5,
            description="Empty metadata",
            metadata={}
        )
        
        assert anomaly.metadata == {}
    
    def test_anomaly_with_empty_description(self):
        """Test anomaly with empty description string."""
        anomaly = Anomaly(
            frame_index=0,
            timestamp=1000.0,
            anomaly_type="test",
            severity=0.5,
            description=""
        )
        
        assert anomaly.description == ""
