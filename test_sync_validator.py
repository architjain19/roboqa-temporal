import pytest
import os
import shutil
import pandas as pd
from sync_validator import SyncValidator

# --- Fixture for Setup and Teardown ---
@pytest.fixture
def temp_data_dir():
    """
    Creates a temporary directory for testing and removes it afterwards.
    This replaces setUp() and tearDown() in unittest.
    """
    test_dir = "./test_data_pytest_temp"
    
    # Setup: Create directories
    os.makedirs(os.path.join(test_dir, "sensor_A"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "sensor_B"), exist_ok=True)
    
    yield test_dir  # This passes the path to the test functions
    
    # Teardown: Remove directories after test finishes
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

def create_dummy_file(base_dir, sensor, timestamps):
    """Helper function to create dummy timestamp files."""
    path = os.path.join(base_dir, sensor, "timestamps.txt")
    with open(path, "w") as f:
        for t in timestamps:
            f.write(f"{t}\n")

# --- Test Cases ---

# 1. SMOKE TEST: Does it initialize and load without crashing?
def test_smoke_initialization(temp_data_dir):
    # Create a simple valid file
    create_dummy_file(temp_data_dir, "sensor_A", ["2011-09-26 13:00:00.000000000"])
    
    validator = SyncValidator(temp_data_dir)
    result = validator.load_kitti_timestamps("sensor_A")
    
    # In pytest, we use simple 'assert' statements
    assert result is True
    assert "sensor_A" in validator.timestamps

# 2. ONE-SHOT TEST: Does it verify a specific known output?
def test_one_shot_calculation(temp_data_dir):
    # Sensor A is at 0ms, Sensor B is at 100ms. Difference is 100ms.
    t1 = ["2011-09-26 13:00:00.000000000"]
    t2 = ["2011-09-26 13:00:00.100000000"]
    
    create_dummy_file(temp_data_dir, "sensor_A", t1)
    create_dummy_file(temp_data_dir, "sensor_B", t2)
    
    validator = SyncValidator(temp_data_dir)
    validator.load_kitti_timestamps("sensor_A")
    validator.load_kitti_timestamps("sensor_B")
    
    metrics = validator.calculate_sync_metrics("sensor_A", "sensor_B")
    
    # Check if mean offset is exactly 100.0 ms
    # pytest.approx handles floating point precision issues
    assert metrics['mean_offset_ms'] == pytest.approx(100.0, abs=0.01)
    
    # Check if the score calculation logic works (Score should NOT be 100)
    assert metrics['quality_score'] < 100.0

# 3. EDGE TEST: How does it handle empty files?
def test_edge_empty_file(temp_data_dir):
    create_dummy_file(temp_data_dir, "sensor_A", []) # Empty list
    
    validator = SyncValidator(temp_data_dir)
    validator.load_kitti_timestamps("sensor_A")
    
    # If using pandas read_csv on empty file, it might contain empty DF or not load
    if "sensor_A" in validator.timestamps:
        assert len(validator.timestamps["sensor_A"]) == 0

# 4. PATTERN TEST: Can it detect drifting synchronization?
def test_pattern_drift(temp_data_dir):
    # Sensor A: Constant 100ms interval
    t1 = ["2011-09-26 13:00:00.000", "2011-09-26 13:00:00.100", "2011-09-26 13:00:00.200"]
    # Sensor B: Increasing interval (110ms) -> Drifting away
    t2 = ["2011-09-26 13:00:00.000", "2011-09-26 13:00:00.110", "2011-09-26 13:00:00.220"]
    
    create_dummy_file(temp_data_dir, "sensor_A", t1)
    create_dummy_file(temp_data_dir, "sensor_B", t2)
    
    validator = SyncValidator(temp_data_dir)
    validator.load_kitti_timestamps("sensor_A")
    validator.load_kitti_timestamps("sensor_B")
    
    metrics = validator.calculate_sync_metrics("sensor_A", "sensor_B")
    
    # Frame 0 diff: 0ms
    # Frame 1 diff: 10ms
    # Frame 2 diff: 20ms -> Max drift should be 20ms
    assert metrics['max_drift_ms'] == pytest.approx(20.0, abs=0.01)
