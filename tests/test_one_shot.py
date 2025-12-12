import pytest
from roboqa_temporal.synchronization import TemporalSyncValidator

class MockStream:
    def __init__(self, name, freq, count):
        self.name = name
        self.frequency = freq
        self.timestamps = [i * (1.0/freq) for i in range(count)]

def test_temporal_sync_one_shot_perfect_match(tmp_path):
    """
    author: xinxintai
    reviewer: Snehul0
    category: one-shot test
    """
    validator = TemporalSyncValidator(output_dir=str(tmp_path))
    
    # Construct a perfect match scenario (One-Shot: Specific Input -> Specific Output)
    camera = MockStream("camera", 10.0, 3)
    camera.timestamps = [0.0, 0.1, 0.2]
    
    lidar = MockStream("lidar", 10.0, 3)
    lidar.timestamps = [0.0, 0.1, 0.2] # Exactly the same
    
    streams = {"camera": camera, "lidar": lidar}
    
    report = validator.analyze_streams(streams, bag_name="oneshot", include_visualizations=False)
    
    # In this perfect case, the offset should be exactly 0
    assert report.metrics["mean_offset_ms"] == 0.0
