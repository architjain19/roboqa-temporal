"""
################################################################
File: feature4_demo_visuals.py
Created: 2025-12-07
Created by: Sayali Nehul (snehul@uw.edu)
Last Modified: 2025-12-07
Last Modified by: Sayali Nehul (snehul@uw.edu)
################################################################
Dataset Quality Scoring & Cross-Benchmarking (Feature 4)
KITTI Visual Demo — Camera + LiDAR Quick Viewer.
This module generates simple visual previews for KITTI sequences,
including random sampled camera images from image_00 to image_03
and 3D visualization of a Velodyne LiDAR .bin scan.Outputs produced 
for each sequence is camera_samples_<seq>.png lidar_pointcloud_<seq>.png
################################################################
"""
import numpy as np
import os
import matplotlib.pyplot as plt

def load_bin(bin_path):                                                                                               #Load single KITTI .bin LiDAR frame.
    scan = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)                                    
    return scan[:, 0], scan[:, 1], scan[:, 2]                                                                         # x, y, z only (no intensity)


def get_first_frame(sequence_path):                                                                                   #Return the first .bin file path inside velodyne_points/data.
    velo_path = os.path.join(sequence_path, "velodyne_points", "data")
    if not os.path.exists(velo_path):
        raise FileNotFoundError(f"No velodyne_points/data folder in {sequence_path}")

    files = sorted([f for f in os.listdir(velo_path) if f.endswith(".bin")])
    if not files:
        raise FileNotFoundError(f"No .bin files found in {velo_path}")

    return os.path.join(velo_path, files[0])


def plot_three_sequences(seq_paths, seq_names):                                                                       #Plot FIRST LiDAR frame of each sequence with clear labeling.
    colors = ["blue", "green", "red"]

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    for seq_path, seq_name, color in zip(seq_paths, seq_names, colors):
        frame_path = get_first_frame(seq_path)

        print(f"[INFO] Loading frame: {frame_path}")
        x, y, z = load_bin(frame_path)

        ax.scatter(x, y, z, s=1, c=color, label=f"{seq_name} (frame 0)")

    ax.set_title("KITTI LiDAR Point Cloud — Sequences 0005, 0023, 0070", fontsize=14)
    ax.set_xlabel("X (m) — forward")
    ax.set_ylabel("Y (m) — left/right")
    ax.set_zlabel("Z (m) — height")                                                                                                                         
    ax.legend(loc="upper left")                                                                                         # legend explaining color encoding
    ax.view_init(elev=25, azim=-65)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    seqs = [
        "./data/sequences/2011_09_26_drive_0005_sync",
        "./data/sequences/2011_09_26_drive_0023_sync",
        "./data/sequences/2011_09_26_drive_0070_sync"
    ]

    names = ["0005", "0023", "0070"]

    plot_three_sequences(seqs, names)

