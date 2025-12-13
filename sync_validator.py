import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import base64
from io import BytesIO
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
        file_path = os.path.join(self.data_path, sensor_name, "timestamps.txt")

        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            return False

        try:
            df = pd.read_csv(file_path, header=None, names=["datetime"])
            df["dt"] = pd.to_datetime(df["datetime"], errors="coerce")
            df = df.dropna()
            self.timestamps[sensor_name] = df["dt"]
            print(f"Successfully loaded {len(df)} timestamps for {sensor_name}")
            return True
        except Exception as e:
            print(f"Error loading {sensor_name}: {str(e)}")
            return False

    def calculate_quality_score(self, mean_offset_ms, max_drift_ms):
        """
        Converts raw error metrics into a 0-100 quality score.
        """
        penalty = (mean_offset_ms * 2.0) + (max_drift_ms * 0.5)
        score = max(0.0, 100.0 - penalty)
        return round(score, 2)

    def calculate_sync_metrics(self, sensor_1, sensor_2):
        """
        Calculates temporal drift metrics and Quality Score.
        Includes advanced metrics: Drift Rate and Sync Success Rate.
        """
        t1 = self.timestamps[sensor_1]
        t2 = self.timestamps[sensor_2]

        min_len = min(len(t1), len(t2))
        t1 = t1.iloc[:min_len].reset_index(drop=True)
        t2 = t2.iloc[:min_len].reset_index(drop=True)

        # Calculate difference in milliseconds
        deltas = (t1 - t2).dt.total_seconds() * 1000.0

        # 1. Basic Metrics
        mean_offset = np.mean(np.abs(deltas))
        max_drift = np.max(np.abs(deltas))
        std_dev = np.std(deltas)

        # 2. Advanced Metric: Drift Rate (ms per second)
        total_duration_sec = (t1.iloc[-1] - t1.iloc[0]).total_seconds()
        if total_duration_sec > 0:
            drift_change = np.abs(deltas.iloc[-1]) - np.abs(deltas.iloc[0])
            drift_rate = drift_change / total_duration_sec
        else:
            drift_rate = 0.0

        # 3. Advanced Metric: Sync Success Rate (< 30ms threshold)
        threshold_ms = 30.0
        success_count = np.sum(np.abs(deltas) < threshold_ms)
        success_rate = (success_count / len(deltas)) * 100.0

        quality_score = self.calculate_quality_score(mean_offset, max_drift)

        metrics = {
            "mean_offset_ms": mean_offset,
            "std_dev_ms": std_dev,
            "max_drift_ms": max_drift,
            "drift_rate_ms_per_sec": drift_rate,
            "sync_success_rate": success_rate,
            "quality_score": quality_score,
            "frame_count": int(len(deltas)),
            "raw_deltas": deltas,
        }

        return metrics

    def generate_report(self, metrics, sensor_pair, output_dir):
        """
        Generates:
        1. HTML Report (Visuals + Text) -> For Humans/PPT
        2. JSON Data -> For Feature 4 Teammate
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # --- 1. Create Plot in Memory ---
        plt.figure(figsize=(10, 6))
        plt.plot(metrics["raw_deltas"], label="Offset", color="dodgerblue", alpha=0.8)
        plt.axhline(y=0, color="red", linestyle="--", alpha=0.3)
        plt.axhline(y=30, color="orange", linestyle=":", label="Threshold (30ms)")
        plt.axhline(y=-30, color="orange", linestyle=":")
        plt.title(f"Temporal Sync: {sensor_pair[0]} vs {sensor_pair[1]}")
        plt.xlabel("Frame Index")
        plt.ylabel("Time Delta (ms)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save plot to a bytes buffer instead of a file
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        # Encode image to base64 string
        image_base64 = base64.b64encode(buf.read()).decode("utf-8")

        # --- 2. Generate HTML Content ---
        status = "PASS" if metrics["quality_score"] >= 60 else "FAIL"
        color = "#28a745" if status == "PASS" else "#dc3545"  # Green or Red

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>RoboQA Sync Report</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 40px; background-color: #f8f9fa; }}
                .container {{ max-width: 900px; margin: 0 auto; background: white; padding: 30px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); border-radius: 8px; }}
                h1 {{ color: #333; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
                .summary-box {{ display: flex; justify-content: space-between; background: #e9ecef; padding: 20px; border-radius: 8px; margin: 20px 0; }}
                .score-item {{ text-align: center; }}
                .score-val {{ font-size: 32px; font-weight: bold; color: #007bff; }}
                .status-val {{ font-size: 32px; font-weight: bold; color: {color}; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                th, td {{ border: 1px solid #dee2e6; padding: 12px; text-align: left; }}
                th {{ background-color: #f8f9fa; }}
                img {{ width: 100%; height: auto; margin-top: 30px; border: 1px solid #dee2e6; }}
                .recommendations {{ background-color: #fff3cd; padding: 15px; border-left: 6px solid #ffc107; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Synchronization Quality Report</h1>
                <p><strong>Sensors:</strong> {sensor_pair[0]} vs {sensor_pair[1]} | <strong>Date:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
                
                <div class="summary-box">
                    <div class="score-item">
                        <div>Quality Score</div>
                        <div class="score-val">{metrics["quality_score"]}</div>
                    </div>
                    <div class="score-item">
                        <div>Status</div>
                        <div class="status-val">{status}</div>
                    </div>
                    <div class="score-item">
                        <div>Success Rate</div>
                        <div class="score-val">{metrics["sync_success_rate"]:.1f}%</div>
                    </div>
                </div>

                <h3>Detailed Metrics</h3>
                <table>
                    <tr><th>Metric</th><th>Value</th><th>Description</th></tr>
                    <tr><td>Mean Offset</td><td>{metrics["mean_offset_ms"]:.4f} ms</td><td>Average time difference between sensors.</td></tr>
                    <tr><td>Standard Deviation</td><td>{metrics["std_dev_ms"]:.4f} ms</td><td>Stability of synchronization (Jitter).</td></tr>
                    <tr><td>Max Drift</td><td>{metrics["max_drift_ms"]:.4f} ms</td><td>Worst-case single frame offset.</td></tr>
                    <tr><td>Drift Rate</td><td>{metrics["drift_rate_ms_per_sec"]:.5f} ms/sec</td><td>Speed at which sensors are diverging.</td></tr>
                </table>

                <h3>Visualization</h3>
                <img src="data:image/png;base64,{image_base64}" alt="Sync Plot">

                <div class="recommendations">
                    <h3>Actionable Recommendations</h3>
                    <ul>
                        {"<li>Synchronization is stable and within ISO limits. Proceed to sensor fusion.</li>" if status == "PASS" else "<li><strong>CRITICAL:</strong> High offset detected. Check PTP hardware settings.</li><li>Recalibration required.</li>"}
                        <li>Monitor Drift Rate: {metrics["drift_rate_ms_per_sec"]:.5f} ms/s indicates {"stable" if abs(metrics["drift_rate_ms_per_sec"]) < 0.1 else "unstable"} clocks.</li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """

        # Write HTML File
        html_path = os.path.join(output_dir, "sync_report.html")
        with open(html_path, "w") as f:
            f.write(html_content)

        # --- 3. JSON Export (Essential for Feature 4) ---
        json_metrics = metrics.copy()
        del json_metrics["raw_deltas"]
        json_metrics["sensor_pair"] = sensor_pair
        json_metrics["status"] = status

        json_path = os.path.join(output_dir, "sync_metrics.json")
        with open(json_path, "w") as f:
            json.dump(json_metrics, f, indent=4)

        # --- 4. Terminal Output with Links ---
        # 获取绝对路径
        abs_html_path = os.path.abspath(html_path)

        print("\n" + "=" * 60)
        print(" ANALYSIS COMPLETE ")
        print("=" * 60)
        print("\n[REPORT]:\nTo view the report, Ctrl+Click the link below:")
        print(f"\n   file://{abs_html_path}\n")  # 这在 VS Code 终端里是可以点击的

        print(
            "[SHORTCUT]:\nOr copy and paste this command to open directly in Windows:"
        )
        # 尝试构建一个 explorer.exe 命令来直接在 Windows 中打开
        print(f'\n   explorer.exe "{abs_html_path}"\n')

        print(
            f"[DATA]:\nJSON metrics for Feature 4 saved to:\n   {os.path.abspath(json_path)}"
        )
        print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RoboQA-Temporal Sync Validator")
    parser.add_argument("--path", type=str, required=True, help="Path to KITTI drive")
    args = parser.parse_args()

    validator = SyncValidator(args.path)

    if validator.load_kitti_timestamps("image_02") and validator.load_kitti_timestamps(
        "image_03"
    ):
        stats = validator.calculate_sync_metrics("image_02", "image_03")
        validator.generate_report(stats, ("image_02", "image_03"), "./output_report")
    elif validator.load_kitti_timestamps(
        "image_02"
    ) and validator.load_kitti_timestamps("image_00"):
        stats = validator.calculate_sync_metrics("image_02", "image_00")
        validator.generate_report(stats, ("image_02", "image_00"), "./output_report")
    else:
        print("Could not load sensors.")
