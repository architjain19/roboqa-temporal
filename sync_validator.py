import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json  # [Added] Used to export machine-readable data
from datetime import datetime

class SyncValidator:
    """
    Validates temporal alignment between sensor streams.
    Reference: RoboQA-Temporal Feature 1
    """
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.timestamps = {}

    def load_kitti_timestamps(self, sensor_name):
        """
        Loads timestamps from KITTI raw data text files.
        """
        file_path = os.path.join(self.data_path, sensor_name, 'timestamps.txt')
        
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            return False

        try:
            # Read timestamps
            df = pd.read_csv(file_path, header=None, names=['datetime'])
            df['dt'] = pd.to_datetime(df['datetime'], errors='coerce')
            df = df.dropna()
            self.timestamps[sensor_name] = df['dt']
            print(f"Successfully loaded {len(df)} timestamps for {sensor_name}")
            return True
        except Exception as e:
            print(f"Error loading {sensor_name}: {str(e)}")
            return False

    def calculate_quality_score(self, mean_offset_ms, max_drift_ms):
        """
        [Added] Converts raw error metrics into a 0-100 quality score.
        Requirement: 'Temporal Alignment Quality Score' for Feature 4 integration.
        
        Logic:
        - Perfect synchronization (0ms) = 100 points
        - 30ms error (typical threshold) = 60 points (passing grade)
        - > 50ms error = 0 points
        """
        # Use a simple linear penalty model
        # Penalty: Higher weight for mean offset, lower weight for max drift
        penalty = (mean_offset_ms * 2.0) + (max_drift_ms * 0.5)
        score = max(0.0, 100.0 - penalty)
        return round(score, 2)

    def calculate_sync_metrics(self, sensor_1, sensor_2):
        """
        Calculates temporal drift metrics and Quality Score.
        """
        t1 = self.timestamps[sensor_1]
        t2 = self.timestamps[sensor_2]
        
        # Ensure we compare the same number of frames
        min_len = min(len(t1), len(t2))
        t1 = t1.iloc[:min_len].reset_index(drop=True)
        t2 = t2.iloc[:min_len].reset_index(drop=True)

        # Calculate difference in milliseconds
        deltas = (t1 - t2).dt.total_seconds() * 1000.0 
        
        mean_offset = np.mean(np.abs(deltas))
        max_drift = np.max(np.abs(deltas))

        # [Modified] Calculate score based on metrics
        quality_score = self.calculate_quality_score(mean_offset, max_drift)

        metrics = {
            "mean_offset_ms": mean_offset,
            "std_dev_ms": np.std(deltas),
            "max_drift_ms": max_drift,
            "quality_score": quality_score, # [Added] 0-100 Score
            "frame_count": int(len(deltas)),
            "raw_deltas": deltas # raw_deltas is a Series, handled separately for JSON
        }
        
        return metrics

    def generate_report(self, metrics, sensor_pair, output_dir):
        """
        Generates:
        1. Visualization Plot (.png)
        2. Human Report (.txt)
        3. Machine Metadata (.json) -> For Feature 4 integration
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 1. Plot
        plt.figure(figsize=(10, 6))
        plt.plot(metrics["raw_deltas"], label='Offset', color='blue', alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        plt.title(f"Sync Quality (Score: {metrics['quality_score']}): {sensor_pair[0]} vs {sensor_pair[1]}")
        plt.xlabel("Frame Index")
        plt.ylabel("Time Delta (ms)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "sync_plot.png"))
        plt.close()

        # 2. Text Report
        report_path = os.path.join(output_dir, "sync_report.txt")
        status = "PASS" if metrics["quality_score"] >= 60 else "FAIL"
        
        with open(report_path, "w") as f:
            f.write(f"--- Synchronization Quality Report ---\n")
            f.write(f"Sensors: {sensor_pair[0]} vs {sensor_pair[1]}\n")
            f.write(f"Quality Score: {metrics['quality_score']} / 100\n") # [Added] Display score
            f.write(f"Status: {status}\n\n")
            f.write(f"Metrics:\n")
            f.write(f"  - Mean Offset: {metrics['mean_offset_ms']:.4f} ms\n")
            f.write(f"  - Max Drift: {metrics['max_drift_ms']:.4f} ms\n")

        # 3. JSON Export (For Feature 4 Integration)
        # Remove raw_deltas (too large and not serializable), keep statistics
        json_metrics = metrics.copy()
        del json_metrics["raw_deltas"] 
        json_metrics["sensor_pair"] = sensor_pair
        json_metrics["status"] = status
        
        json_path = os.path.join(output_dir, "sync_metrics.json")
        with open(json_path, "w") as f:
            json.dump(json_metrics, f, indent=4)
        
        print(f"Success! Generated:\n - Plot: sync_plot.png\n - Report: sync_report.txt\n - Data: sync_metrics.json (For Feature 4)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RoboQA-Temporal Sync Validator")
    parser.add_argument("--path", type=str, required=True, help="Path to KITTI drive")
    args = parser.parse_args()

    validator = SyncValidator(args.path)

    # Demo execution
    if validator.load_kitti_timestamps("image_02") and validator.load_kitti_timestamps("image_03"):
        stats = validator.calculate_sync_metrics("image_02", "image_03")
        validator.generate_report(stats, ("image_02", "image_03"), "./output_report")
    # Fallback
    elif validator.load_kitti_timestamps("image_02") and validator.load_kitti_timestamps("image_00"):
        stats = validator.calculate_sync_metrics("image_02", "image_00")
        validator.generate_report(stats, ("image_02", "image_00"), "./output_report")
    else:
        print("Could not load sensors.")
