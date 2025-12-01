import unittest
import shutil
import os
import pandas as pd
from sync_validator import SyncValidator

class TestSyncValidator(unittest.TestCase):

    def setUp(self):
        """Setup: Create temporary dummy data folder structure."""
        self.test_dir = "./test_data_temp"
        os.makedirs(os.path.join(self.test_dir, "sensor_A"), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, "sensor_B"), exist_ok=True)
        self.validator = SyncValidator(self.test_dir)

    def tearDown(self):
        """Teardown: Remove temporary data folder."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def create_dummy_file(self, sensor, timestamps):
        path = os.path.join(self.test_dir, sensor, "timestamps.txt")
        with open(path, "w") as f:
            for t in timestamps:
                f.write(f"{t}\n")

    # TEST 1: Smoke Test (Basic Functionality)
    # Checks if the class instantiates and runs without crashing on valid input.
    def test_smoke(self):
        self.create_dummy_file("sensor_A", ["2011-09-26 13:00:00.000000000"])
        result = self.validator.load_kitti_timestamps("sensor_A")
        self.assertTrue(result, "Smoke test failed: Should load valid file successfully")

    # TEST 2: One-Shot Test (Specific Known Output)
    # Checks if the math is exactly correct for a known input.
    def test_one_shot_offset(self):
        # Time A: 0ms, Time B: 100ms. Difference should be 100ms.
        t1 = ["2011-09-26 13:00:00.100000000"]
        t2 = ["2011-09-26 13:00:00.000000000"]
        self.create_dummy_file("sensor_A", t1)
        self.create_dummy_file("sensor_B", t2)
        
        self.validator.load_kitti_timestamps("sensor_A")
        self.validator.load_kitti_timestamps("sensor_B")
        
        metrics = self.validator.calculate_sync_metrics("sensor_A", "sensor_B")
        self.assertAlmostEqual(metrics['mean_offset_ms'], 100.0, places=2)

    # TEST 3: Edge Test (Empty or Invalid Data)
    # Checks behavior when files are empty.
    def test_edge_empty_file(self):
        self.create_dummy_file("sensor_A", []) # Empty file
        result = self.validator.load_kitti_timestamps("sensor_A")
        # Depending on logic, it might return True (loaded 0 lines) or False. 
        # Our code prints 0 timestamps but strictly pandas might return empty DF.
        if "sensor_A" in self.validator.timestamps:
            self.assertEqual(len(self.validator.timestamps["sensor_A"]), 0)

    # TEST 4: Pattern Test (Drifting Data)
    # Checks if we can detect increasing drift over time.
    def test_pattern_drift(self):
        # Sensor A stays constant interval (100ms)
        t1 = ["2011-09-26 13:00:00.000", "2011-09-26 13:00:00.100"] 
        # Sensor B is slower (110ms interval) -> Drift increases
        t2 = ["2011-09-26 13:00:00.000", "2011-09-26 13:00:00.110"]
        
        self.create_dummy_file("sensor_A", t1)
        self.create_dummy_file("sensor_B", t2)
        self.validator.load_kitti_timestamps("sensor_A")
        self.validator.load_kitti_timestamps("sensor_B")
        
        metrics = self.validator.calculate_sync_metrics("sensor_A", "sensor_B")
        # First frame diff 0, second frame diff 10. Max drift should be 10.
        self.assertAlmostEqual(metrics['max_drift_ms'], 10.0, places=1)

if __name__ == '__main__':
    unittest.main()
