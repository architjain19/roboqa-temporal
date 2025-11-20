# RoboQA-Temporal Project Structure

## Overview

This document describes the structure and organization of the RoboQA-Temporal project.

## Directory Structure

```
roboqa-temporal/
├── roboqa_temporal/          # Main package
│   ├── __init__.py           # Package initialization and exports
│   ├── loader/               # ROS2 bag file loading
│   │   ├── __init__.py
│   │   └── bag_loader.py     # BagLoader class for reading ROS2 bags
│   ├── preprocessing/        # Point cloud preprocessing
│   │   ├── __init__.py
│   │   └── preprocessor.py   # Preprocessor class for cleaning/normalizing
│   ├── detection/            # Anomaly detection algorithms
│   │   ├── __init__.py
│   │   ├── detector.py       # Main AnomalyDetector orchestrator
│   │   └── detectors.py      # Individual detector implementations
│   ├── reporting/            # Report generation
│   │   ├── __init__.py
│   │   └── report_generator.py  # ReportGenerator for multiple formats
│   ├── cli/                  # Command-line interface
│   │   ├── __init__.py
│   │   └── main.py           # CLI entry point
│   └── config/               # Configuration utilities
│       └── __init__.py
├── tests/                    # Unit tests
│   ├── __init__.py
│   ├── test_preprocessing.py
│   └── test_detection.py
├── examples/                 # Example scripts and configs
│   ├── basic_usage.py
│   ├── synthetic_data_generator.py
│   ├── advanced_usage.py
│   └── config_example.yaml
├── .docs/                     # Documentation
├── README.md                 # Main documentation (project overview)
├── CONTRIBUTING.md           # Contribution guidelines
├── LICENSE                   # MIT License
├── setup.py                  # Setup script
├── pyproject.toml            # Modern Python project configuration
├── requirements.txt          # Python dependencies
└── .gitignore               # Git ignore rules
```

## Module Descriptions

### loader/
- **BagLoader**: Reads ROS2 bag files and extracts point cloud data
- Supports topic filtering, frame ID filtering, and efficient iteration
- Handles PointCloud2 message deserialization

### preprocessing/
- **Preprocessor**: Cleans and normalizes point cloud data
- Supports voxel-based downsampling
- Multiple outlier removal methods (statistical, radius-based, LOF)
- Time alignment capabilities

### detection/
- **AnomalyDetector**: Main orchestrator that runs multiple detectors
- **DensityDropDetector**: Detects sudden drops in point density
- **SpatialDiscontinuityDetector**: Analyzes geometric changes
- **GhostPointDetector**: Identifies reflection/multi-path artifacts
- **TemporalConsistencyDetector**: Quantifies temporal smoothness

### reporting/
- **ReportGenerator**: Creates quality assessment reports
- Supports Markdown, HTML (with plots), and CSV formats
- Generates visualizations and statistics

### cli/
- **main**: Command-line interface entry point
- Supports configuration files, various output formats, and flexible options

## Data Flow

1. **Input**: ROS2 bag file
2. **Loading**: BagLoader extracts point cloud frames
3. **Preprocessing**: Preprocessor cleans and normalizes data
4. **Detection**: AnomalyDetector runs multiple detection algorithms
5. **Reporting**: ReportGenerator creates output reports

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

## Testing

Tests are organized by module:
- `test_preprocessing.py`: Tests for preprocessing functionality
- `test_detection.py`: Tests for anomaly detection algorithms

Run tests with: `pytest tests/`

## Configuration

Configuration is supported via YAML files. See `examples/config_example.yaml` for available options.
