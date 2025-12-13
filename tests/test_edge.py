import pytest
from roboqa_temporal.synchronization import TemporalSyncValidator

class MockStream:
    def __init__(self, name, freq, count):
        self.name = name
        self.frequency = freq
        self.timestamps = [i * (1.0/freq) for i in range(count)]

def test_temporal_sync_handles_no_matches_edge(tmp_path):
    """
    author: xinxintai
    reviewer: Snehul0
    category: edge test
    justification: Tests the validator's behavior when sensor streams have disjoint timestamps (0 overlaps). 
                   This ensures the system handles empty intersection sets gracefully without crashing.
    """
    # Setup
    validator = TemporalSyncValidator(output_dir=str(tmp_path))
    
    # Construct edge case: Completely disjoint timestamps
    # Camera: 0.0, 0.1
    camera = MockStream("camera", 10.0, 2)
    camera.timestamps = [0.0, 0.1]
    
    # Lidar: 1000.0, 1000.1 (Far away from camera timestamps)
    lidar = MockStream("lidar", 10.0, 2)
    lidar.timestamps = [1000.0, 1000.1]
    
    streams = {"camera": camera, "lidar": lidar}

    # Action
    report = validator.analyze_streams(streams, bag_name="edge_test", include_visualizations=False)

    # Assertion
    # As long as it doesn't crash and returns a report, it passes.
    assert report is not None
