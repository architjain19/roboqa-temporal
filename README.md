# RoboQA-Temporal

**Automated Quality Assessment and Anomaly Detection for Multi-Sensor Robotics Datasets**

RoboQA-Temporal is an open-source, professional Python toolkit focused on automated quality assessment and anomaly detection for multi-sensor robotics datasets, with a special focus on ROS2 bag files. It provides automated, objective, and reproducible health checks for robotics datasets used in ML, SLAM, and perception workflows.

![RoboQA-Logo](.docs/roboqa_logo.png)

## Design/Components/Stories:

### User Stories:
> User Stories for this project can be found here - [USER_STORIES.md](.docs/USER_STORIES.md)

### User Design:
> User Desigs for this project can be found here - [USER_DESIGN.md](.docs/USER_DESIGN.md)

### User Components:
> User Components for this project can be found here - [USER_COMPONENTS.md](.docs/USER_COMPONENTS.md)

## Installation

### Prerequisites

- Python 3.10 or higher
- ROS2 (for ROS2 bag file support) (Distro - `Humble`)

The following ROS packages are provided by ROS 2 Humble installation (`APT`)

```bash
sudo apt update
sudo apt install ros-humble-rclpy ros-humble-rosbag2 ros-humble-sensor-msgs
```

Source ROS 2 environment before running code or tests:

```bash
source /opt/ros/humble/setup.bash
```

> **Note:** Use the system Python (or add /opt/ros/.../dist-packages to PYTHONPATH) if running inside a venv.

### Install from Source

```bash
# Clone the repository
git clone https://github.com/architjain19/roboqa-temporal.git
cd roboqa-temporal

# Create a virtual environment:
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate`

# 5. Source ROS
source /opt/ros/humble/setup.bash

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Dependencies

- `numpy` >= 1.21.0
- `scipy` >= 1.7.0
- `scikit-learn` >= 1.0.0
- `open3d` >= 0.15.0
- `pandas` >= 1.3.0
- `matplotlib` >= 3.5.0
- `seaborn` >= 0.11.0
- `pyyaml` >= 6.0
- `tqdm` >= 4.62.0
- `tabulate` >= 0.9.0
- `lark-parser` >= 0.12.0


## Testing

The integration tests exercise the ROS 2 loader and other anomaly and quality testing tools against the sample bag in `dataset/mybag`.

1. Ensure ROS 2 Humble is installed (see prerequisites above) and source it:
   ```bash
   source /opt/ros/humble/setup.bash
   ```
2. Install development dependencies:
   ```bash
   pip install -e .[dev]
   ```
3. Run the test suite:
   ```bash
   pytest -q
   ```

> **Note**: GitHub Actions setup for this package, uses the same ROS packages via `ros-tooling/setup-ros` and runs these tests on every push and pull request.

## Quick Start

### Usage

RoboQA-Temporal now supports two main operation modes:

#### 1. Anomaly Detection Mode (for ROS2 bags)

```bash
# Analyze a ROS2 bag file for anomalies
roboqa anomaly path/to/bag_file.db3

# Specify output format
roboqa anomaly path/to/bag_file.db3 --output html

# Limit number of frames
roboqa anomaly path/to/bag_file.db3 --max-frames 1000

# Use configuration file
roboqa anomaly path/to/bag_file.db3 --config examples/config_anomaly_kitti.yaml

# Combined usage
roboqa anomaly path/to/bag_file.db3 --output html --max-frames 1000 --config examples/config_anomaly_kitti.yaml 
```

#### 2. Synchronization Analysis Mode (for multi-sensor datasets)

```bash
# Analyze temporal synchronization in a KITTI-format dataset
roboqa sync dataset/2011_09_26_drive_0005_sync/

# Specify output format
roboqa sync dataset/2011_09_26_drive_0005_sync/ --output html

# Limit frames and customize output
roboqa sync dataset/2011_09_26_drive_0005_sync/ --max-frames 500 --output-dir reports/sync/

# Use configuration file for custom sensor mappings and thresholds
roboqa sync dataset/2011_09_26_drive_0005_sync/ --config examples/config_sync.yaml
```

### Troubleshooting (Handling Large Point Clouds (KITTI, Outdoor LiDAR))

For datasets with large point clouds (100k+ points per frame), use voxel downsampling to avoid memory issues:

```bash
# Recommended settings for KITTI or similar large datasets
roboqa anomaly path/to/bag_file.db3 --voxel-size 0.1 --config examples/config_anomaly_kitti.yaml

# Or use CLI arguments directly
roboqa anomaly path/to/bag_file.db3 --voxel-size 0.1 --max-points-for-outliers 50000
```

**Key parameters:**
- `--voxel-size 0.1`: Downsample point clouds to ~20-30k points (adjust based on your needs)
- `--max-points-for-outliers 50000`: Skip outlier removal for point clouds exceeding this limit
- Use the pre-configured `examples/config_anomaly_kitti.yaml` for optimal KITTI settings

## Project Structure

(For more details check [here](PROJECT_STRUCTURE.md))

```
.
└── roboqa-temporal
    ├── CONTRIBUTING.md                     # Contributing guidelines
    ├── dataset/                            # Sample datasets (KITTI format)
    ├── .docs/                              # Documentation and presentations
    ├── examples                            # Example scripts and configs
    │   ├── config_anomaly_example.yaml
    │   ├── config_anomaly_kitti.yaml
    │   ├── config_sync.yaml
    │   ├── example_anomaly.py
    │   ├── example_sync.py
    │   └── synthetic_data_generator.py
    ├── LICENSE                             # MIT License
    ├── PROJECT_STRUCTURE.md                # Detailed documentation on project structure
    ├── pyproject.toml
    ├── README.md
    ├── reports                             # Output reports directory
    ├── requirements.txt
    ├── src
    │   └── roboqa_temporal                 # Main package
    │       ├── __init__.py
    │       ├── cli                         # Command-line interface
    │       │   └── main.py
    │       ├── detection                   # Anomaly detection algorithms
    │       │   ├── detector.py
    │       │   ├── detectors.py
    │       ├── loader                      # ROS2 bag file loader
    │       │   ├── bag_loader.py
    │       ├── preprocessing               # Point cloud preprocessing
    │       │   └── preprocessor.py
    │       ├── reporting                   # Report generation
    │       │   └── report_generator.py
    │       └── synchronization             # Cross-modal synchronization analysis
    │           └── temporal_validator.py
    └── tests
        ├── conftest.py
        ├── test_bag_loader.py
        ├── test_detection_pipeline.py
        ├── test_edge.py
        ├── test_one_shot.py
        ├── test_pattern.py
        └── test_smoke.py
```

## Features

### Core Capabilities

#### 1. Anomaly Detection (for ROS2 bags)
- **Automated Quality Assessment**: Objective and reproducible health checks for robotics datasets
- **Point Cloud Anomaly Detection**: Detects various anomalies in point cloud sequences:
  - **Density Drops & Occlusions**: Identifies sudden drops in point density
  - **Spatial Discontinuities**: Analyzes geometric changes and frame-to-frame transformations
  - **Ghost Points**: Highlights points likely due to reflections, multi-path returns, or hardware artifacts
  - **Temporal Inconsistency**: Quantifies smoothness and consistency in spatio-temporal evolution

#### 2. Cross-Modal Synchronization Analysis (for multi-sensor datasets)
- **Timestamp Drift Detection**: Measure clock offset and drift between sensor streams (e.g., LiDAR, camera, IMU)
- **Data Loss & Duplication Flagging**: Identify missing, skipped, or duplicate messages across all sensor topics
- **Temporal Alignment Quality Score**: Quantify overall synchronization fidelity as a numeric metric per sequence
- **Multi-Sensor Support**: Works with KITTI and similar dataset formats with timestamp files
- **Comprehensive Drift Analysis**: Includes chi-square tests, Kalman filtering, and cross-correlation analysis

### Key Features

- **Dual Operation Modes**: Support for both ROS2 bag anomaly detection and multi-sensor synchronization analysis
- **ROS2 Bag Support**: Efficient reading and processing of ROS2 bag files
- **Multi-Sensor Dataset Support**: Works with KITTI-format datasets and other image sequence formats
- **Modular Architecture**: Extensible design with separate modules for loading, preprocessing, detection, synchronization, and reporting
- **Multiple Output Formats**: Generate reports in Markdown, HTML (with visualizations), and CSV
- **Comprehensive CLI**: Easy-to-use command-line interface with mode selection
- **Configuration Support**: YAML-based configuration files for flexible parameter tuning
- **Visualization**: Interactive plots, temporal heatmaps, and charts in HTML reports
- **Timestamp Corrections**: Exports recommended timestamp corrections as YAML parameter files

## Contributing

> Contributions are welcome! Please checkout instructions for contributing here - [CONTRIBUTING](CONTRIBUTING.md)

## License

[MIT License](LICENSE)

## Support

For issues, questions, or contributions, please open an issue on GitHub.

## Acknowledgments

Built for the robotics research community to improve dataset quality and reliability.

### Authors/Contributors
- Archit Jain ( [architj@uw.edu](mailto:architj@uw.edu) )
- Dharineesh Somisetty ( [dhar007@uw.edu](mailto:dhar007@uw.edu) )
- Xinxin Tai ( [xtaiuw@uw.edu](mailto:xtaiuw@uw.edu) )
- Sayali Nehul ( [snehul@uw.edu](mailto:snehul@uw.edu))
