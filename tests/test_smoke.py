import pytest
from roboqa_temporal.synchronization import TemporalSyncValidator

# Simple helper class for mock data
class MockStream:
    def __init__(self, name, freq, count):
        self.name = name
        self.frequency = freq
        # Generate simple timestamps: 0.0, 0.1, 0.2 ...
        self.timestamps = [i * (1.0/freq) for i in range(count)]

def test_temporal_sync_validator_smoke(tmp_path):
    """
    author: xinxintai
    reviewer: Snehul0
    category: smoke test
    """
    # Setup: Create minimal mock data
    validator = TemporalSyncValidator(output_dir=str(tmp_path))
    camera = MockStream("camera", 10.0, 5)
    lidar = MockStream("lidar", 10.0, 5)
    streams = {"camera": camera, "lidar": lidar}

    # Action: Run the core pipeline
    report = validator.analyze_streams(streams, bag_name="smoke_test", include_visualizations=False)

    # Assertion: Smoke tests only care that the program finished without crashing.
    # TA Feedback: Do not check specific metric scores here.
    assert report is not None
    assert report.compliance_flags is not None
