import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        Typically found in: /drive_path/sensor_name/timestamps.txt
        """
        file_path = os.path.join(self.data_path, sensor_name, 'timestamps.txt')
        
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            return False

        try:
            # Read timestamps. KITTI format is usually: '2011-09-26 13:02:25.669806480'
            df = pd.read_csv(file_path, header=None, names=['datetime'])
            # Convert to datetime objects
            df['dt'] = pd.to_datetime(df['datetime'], errors='coerce')
            # Drop any rows that failed to parse
            df = df.dropna()
            self.timestamps[sensor_name] = df['dt']
            print(f"Successfully loaded {len(df)} timestamps for {sensor_name}")
            return True
        except Exception as e:
            print(f"Error loading {sensor_name}: {str(e)}")
            return False

    def calculate_sync_metrics(self, sensor_1, sensor_2):
        """
        Calculates temporal drift metrics (Feature 1 requirement).
        Computes Mean Offset, Max Drift, and Standard Deviation.
        """
        t1 = self.timestamps[sensor_1]
        t2 = self.timestamps[sensor_2]
        
        # Ensure we compare the same number of frames
        min_len = min(len(t1), len(t2))
        t1 = t1.iloc[:min_len].reset_index(drop=True)
        t2 = t2.iloc[:min_len].reset_index(drop=True)

        # Calculate difference in milliseconds
        deltas = (t1 - t2).dt.total_seconds() * 1000.0 
        
        metrics = {
            "mean_offset_ms": np.mean(np.abs(deltas)),
            "std_dev_ms": np.std(deltas),
            "max_drift_ms": np.max(np.abs(deltas)),
            "raw_deltas": deltas
        }
        
        return metrics

    def generate_report(self, metrics, sensor_pair, output_dir):
        """
        Generates visualization plots and actionable recommendations text file.
        Output: sync_plot.png and sync_report.txt
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 1. Generate Visualization Plot (Matplotlib)
        plt.figure(figsize=(10, 6))
        plt.plot(metrics["raw_deltas"], label='Frame-to-Frame Offset', color='blue', alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        plt.title(f"Temporal Synchronization: {sensor_pair[0]} vs {sensor_pair[1]}")
        plt.xlabel("Frame Index")
        plt.ylabel("Time Delta (ms)")
        plt.legend()
        plt.grid(True)
        
        plot_path = os.path.join(output_dir, "sync_plot.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Visualization saved to {plot_path}")

        # 2. Generate Text Report
        report_path = os.path.join(output_dir, "sync_report.txt")
        # Threshold: 30ms is a typical strict threshold for AV sync
        status = "PASS" if metrics["mean_offset_ms"] < 30 else "FAIL" 
        
        with open(report_path, "w") as f:
            f.write(f"--- Synchronization Quality Report ---\n")
            f.write(f"Sensors: {sensor_pair[0]} vs {sensor_pair[1]}\n")
            f.write(f"Overall Status: {status}\n\n")
            f.write(f"Key Metrics:\n")
            f.write(f"  - Mean Offset: {metrics['mean_offset_ms']:.4f} ms\n")
            f.write(f"  - Max Drift: {metrics['max_drift_ms']:.4f} ms\n")
            f.write(f"  - Std Deviation: {metrics['std_dev_ms']:.4f} ms\n\n")
            f.write(f"Actionable Recommendations:\n")
            if status == "FAIL":
                f.write("  - [CRITICAL] Large temporal offset detected (>30ms).\n")
                f.write("  - Investigate PTP (Precision Time Protocol) settings on hardware.\n")
                f.write("  - Perform hardware recalibration for LiDAR-Camera trigger.\n")
            else:
                f.write("  - Synchronization is within acceptable limits.\n")
                f.write("  - Data is safe for sensor fusion algorithms.\n")
        print(f"Report saved to {report_path}")

if __name__ == "__main__":
    # Uses argparse to avoid hardcoded paths
    parser = argparse.ArgumentParser(description="RoboQA-Temporal Sync Validator")
    parser.add_argument("--path", type=str, required=True, help="Path to the KITTI drive folder (containing sensor folders)")
    args = parser.parse_args()

    validator = SyncValidator(args.path)

    # For KITTI, standard folders are 'image_02' (Camera) and 'image_00' or 'velodyne_points'
    # We will try to load Camera 02 and Camera 03 (or 00) for this demo test
    print("Loading sensor data...")
    cam_02_loaded = validator.load_kitti_timestamps("image_02")
    cam_03_loaded = validator.load_kitti_timestamps("image_03") # Using image_03 as second sensor if avail

    if cam_02_loaded and cam_03_loaded:
        print("Calculating metrics...")
        stats = validator.calculate_sync_metrics("image_02", "image_03")
        validator.generate_report(stats, ("image_02", "image_03"), "./output_report")
        print("\nSUCCESS: Report generated in './output_report' folder.")
    else:
        # Fallback for testing if image_03 isn't there, try image_00
        cam_00_loaded = validator.load_kitti_timestamps("image_00")
        if cam_02_loaded and cam_00_loaded:
             stats = validator.calculate_sync_metrics("image_02", "image_00")
             validator.generate_report(stats, ("image_02", "image_00"), "./output_report")
             print("\nSUCCESS: Report generated in './output_report' folder.")
        else:
             print("Error: Could not find two sensors to compare in the provided path.")
