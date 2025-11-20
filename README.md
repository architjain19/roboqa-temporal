# RoboQA-Temporal

**Automated Quality Assessment and Anomaly Detection for Multi-Sensor Robotics Datasets**

RoboQA-Temporal is an open-source, professional Python toolkit focused on automated quality assessment and anomaly detection for multi-sensor robotics datasets, with a special focus on ROS2 bag files. It provides automated, objective, and reproducible health checks for robotics datasets used in ML, SLAM, and perception workflows.

## Features

### Core Capabilities

- **Automated Quality Assessment**: Objective and reproducible health checks for robotics datasets
- **Point Cloud Anomaly Detection**: Detects various anomalies in point cloud sequences:
  - **Density Drops & Occlusions**: Identifies sudden drops in point density
  - **Spatial Discontinuities**: Analyzes geometric changes and frame-to-frame transformations
  - **Ghost Points**: Highlights points likely due to reflections, multi-path returns, or hardware artifacts
  - **Temporal Inconsistency**: Quantifies smoothness and consistency in spatio-temporal evolution

### Key Features

- **ROS2 Bag Support**: Efficient reading and processing of ROS2 bag files
- **Modular Architecture**: Extensible design with separate modules for loading, preprocessing, detection, and reporting
- **Multiple Output Formats**: Generate reports in Markdown, HTML (with visualizations), and CSV
- **Comprehensive CLI**: Easy-to-use command-line interface
- **Configuration Support**: YAML-based configuration files for flexible parameter tuning
- **Visualization**: Interactive plots and charts in HTML reports


## Anomaly Detection Methods

### 1. Density Drop Detection

Detects sudden drops in point density that may indicate:
- Sensor occlusions
- Hardware faults
- Environmental changes

**Algorithm**: Moving average with z-score analysis

### 2. Spatial Discontinuity Detection

Analyzes geometric changes and frame-to-frame transformations to identify:
- Irregular motion
- Environmental shifts
- Sensor misalignment

**Algorithm**: Centroid translation, bounding box changes, and nearest neighbor distances

### 3. Ghost Point Detection

Identifies points likely due to:
- Reflections
- Multi-path returns
- Hardware artifacts

**Algorithm**: Statistical outlier detection (elliptic envelope) and distance-based heuristics

### 4. Temporal Consistency Detection

Quantifies smoothness and consistency in spatio-temporal evolution:
- Point count changes
- Centroid velocity
- Bounding box changes
- Acceleration analysis

**Algorithm**: Frame-to-frame difference metrics and second-order derivatives

## Output Reports

### Markdown Report

Human-readable markdown report with:
- Executive summary
- Health metrics
- Detected anomalies
- Frame statistics

### HTML Report

Interactive HTML report with:
- All markdown content
- Visualizations (point count over time, severity distributions)
- Color-coded severity indicators
- Embedded plots

### CSV Export

Structured CSV file with:
- All detected anomalies
- Frame-by-frame statistics
- Metadata for each anomaly

## Development

### Contributing

Contributions are welcome! Please checkout instructions for contributing here - [CONTRIBUTING](CONTRIBUTING.md)

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
