import pytest
from roboqa_temporal.synchronization import TemporalSyncValidator

class MockStream:
    def __init__(self, name, freq, count):
        self.name = name
        self.frequency = freq
        self.timestamps = [i * (1.0/freq) for i in range(count)]

def test_temporal_sync_pattern_variable_frequencies(tmp_path):
    """
    author: xinxintai
    reviewer: Snehul0
    category: pattern test
    justification: Verifies that the validator can handle a pattern of varying frequencies across multiple sensors.
    """
    validator = TemporalSyncValidator(output_dir=str(tmp_path))
    
    # Pattern test: Mixing sensors with different frequencies
    camera = MockStream("camera", 30.0, 30) # 30Hz
    lidar = MockStream("lidar", 10.0, 10)   # 10Hz
    
    streams = {"camera": camera, "lidar": lidar}
    
    report = validator.analyze_streams(streams, bag_name="pattern", include_visualizations=False)
    
    assert report is not None
