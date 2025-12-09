# RoboQA-Temporal Project Structure

## Overview

This document describes the structure and organization of the RoboQA-Temporal project, including both anomaly detection and cross-modal synchronization analysis features.

## Directory Structure

```
roboqa-temporal/
├── src/roboqa_temporal/                # Main package (src layout)
│   ├── __init__.py                     # Package initialization and exports
│   ├── loader/                         # ROS2 bag file loading
│   │   ├── __init__.py         
│   │   └── bag_loader.py               # BagLoader class for reading ROS2 bags
│   ├── preprocessing/                  # Point cloud preprocessing
│   │   ├── __init__.py         
│   │   └── preprocessor.py             # Preprocessor class for cleaning/normalizing
│   ├── detection/                      # Anomaly detection algorithms
│   │   ├── __init__.py 
│   │   ├── detector.py                 # Main AnomalyDetector orchestrator
│   │   └── detectors.py                # Individual detector implementations
│   ├── synchronization/                # Cross-modal synchronization analysis
│   │   ├── __init__.py 
│   │   └── temporal_validator.py       # TemporalSyncValidator for multi-sensor sync
│   ├── fusion/                         # Camera-LiDAR fusion quality assessment
│   │   ├── __init__.py
│   │   └── fusion_quality_validator.py # CalibrationQualityValidator for fusion metrics
│   ├── health_reporting/               # Dataset health assessment & quality dashboards
│   │   ├── __init__.py
│   │   ├── pipeline.py                 # Core metrics computation and orchestration
│   │   ├── dashboard.py                # Interactive dashboards and visualizations
│   │   ├── exporters.py                # Multi-format export (CSV/JSON/YAML)
│   │   └── curation.py                 # Sequence curation recommendations
│   ├── reporting/                      # Report generation
│   │   ├── __init__.py 
│   │   └── report_generator.py         # ReportGenerator for multiple formats
│   └── cli/                            # Command-line interface
│       ├── __init__.py         
│       └── main.py                     # CLI entry point with mode selection
├── tests/                              # Unit tests
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_bag_loader.py
│   ├── test_detection_pipeline.py
│   ├── test_edge.py
│   ├── test_one_shot.py
│   ├── test_pattern.py
│   ├── test_smoke.py
├── examples/                           # Example scripts and configs
│   ├── example_anomaly.py              # Anomaly detection example
│   ├── example_sync.py                 # Synchronization analysis example
│   ├── synthetic_data_generator.py     # Synthetic data generator script
│   ├── config_anomaly_example.yaml     # Anomaly detector configuration sample
│   ├── config_anomaly_kitti.yaml       # Anomaly detector configuration for kitti
│   └── config_sync.yaml                # Synchronization configuration
├── dataset/                            # Sample datasets
├── reports/                            # Generated reports (output)
├── .docs/                              # Internal documentation
├── .github/                            # GitHub Actions CI/CD workflows
├── README.md                           # Main documentation (project overview)
├── CONTRIBUTING.md                     # Contribution guidelines
├── LICENSE                             # MIT License
├── PROJECT_STRUCTURE.md                # This file
├── pyproject.toml                      # Modern Python project configuration
├── requirements.txt                    # Python dependencies
└── .gitignore                          # Git ignore rules
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

### synchronization/
- **TemporalSyncValidator**: Validates temporal alignment across multi-sensor datasets
- **SensorStream**: Represents individual sensor data streams with timestamps
- **PairwiseDriftResult**: Stores analysis results for sensor pairs
- **TemporalSyncReport**: Aggregates all synchronization validation results

### fusion/
- **CalibrationQualityValidator**: Main validator for camera-LiDAR fusion quality assessment
- **CalibrationStream**: Container for calibration stream data
- **CalibrationPairResult**: Result of calibration quality assessment for a sensor pair
- **ProjectionErrorFrame**: Projection error metrics for a single frame
- **IlluminationFrame**: Illumination change metrics for a single frame
- **MovingObjectFrame**: Moving object detection quality for a single frame
- **CalibrationQualityReport**: Complete fusion quality assessment report

### health_reporting/
- **run_health_check**: Main orchestration function for dataset health assessment
- **pipeline.py**: Core metrics computation (temporal score, anomaly score, completeness metrics)
- **dashboard.py**: Interactive Plotly dashboards and PNG visualizations
- **exporters.py**: Multi-format export (CSV, JSON, YAML) and summary reports
- **curation.py**: Automated sequence curation recommendations with severity levels

### reporting/
- **ReportGenerator**: Creates quality assessment reports
- Supports Markdown, HTML (with plots), and CSV formats
- Generates visualizations and statistics
- Used by both anomaly detection and synchronization modules

### cli/
- **main**: Command-line interface entry point with dual operation modes
- **Mode: anomaly** - ROS2 bag anomaly detection
- **Mode: sync** - Multi-sensor synchronization analysis
- Supports configuration files, various output formats, and flexible options

## Data Flow

### Anomaly Detection Pipeline
1. **Input**: ROS2 bag file
2. **Loading**: BagLoader extracts point cloud frames
3. **Preprocessing**: Preprocessor cleans and normalizes data
4. **Detection**: AnomalyDetector runs multiple detection algorithms
5. **Reporting**: ReportGenerator creates output reports

### Synchronization Analysis Pipeline
1. **Input**: Multi-sensor dataset folder (e.g., KITTI format)
2. **Loading**: TemporalSyncValidator loads timestamps from sensor folders
3. **Stream Creation**: Creates SensorStream objects for each sensor
4. **Pairwise Analysis**: Analyzes timestamp alignment between sensor pairs
5. **Drift Detection**: Computes drift rates, chi-square tests, Kalman predictions
6. **Reporting**: Generates reports with heatmaps and correction parameters

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

## Synchronization Analysis Features

### Key Capabilities

1. **Timestamp Drift Detection**
   - Measures clock offset and drift between sensor streams
   - Computes drift rate in milliseconds per second
   - Uses Kalman filtering to predict future drift

2. **Data Loss & Duplication Flagging**
   - Identifies missing frames based on expected frequency
   - Detects duplicate or near-duplicate timestamps
   - Flags sequences with irregular sampling rates

3. **Temporal Alignment Quality Score**
   - Calculates numeric quality metric (0-1) for each sensor pair
   - Aggregates scores across all pairs for overall synchronization quality
   - Exports recommended timestamp corrections as YAML files

4. **Statistical Analysis**
   - Chi-square tests for timestamp consistency
   - Cross-correlation lag detection
   - Rolling statistics for temporal trends

### Supported Dataset Formats

- **KITTI Format**: Datasets with sensor subfolders containing timestamps.txt files
- **Timestamp Formats**:
  - KITTI datetime: `YYYY-MM-DD HH:MM:SS.nanoseconds`
  - Unix seconds: `1632825072.351950336`
  - Unix nanoseconds: `1632825072351950336`

### Default Sensor Pairs Analyzed

- camera_left ↔ lidar
- camera_left ↔ camera_right
- lidar ↔ imu
- camera_left ↔ imu

## Camera-LiDAR Fusion Quality Assessment Features

### Key Capabilities

1. **Calibration Drift Estimation**
   - Tests for changes in calibration matrices over time
   - Measures edge alignment between camera edges and LiDAR projections
   - Computes normalized mutual information between camera and LiDAR data
   - Suggests potential hardware re-calibration needs when quality drops

2. **Projection Error Quantification**
   - Measures reprojection error when projecting 3D points into camera images
   - Tracks error trends across the sequence (stable/increasing/decreasing)
   - Identifies frames with maximum projection errors
   - Highlights instances with significant calibration drift

3. **Illumination and Scene Change Detection**
   - Detects brightness changes and lighting variations
   - Measures image contrast (Michelson contrast)
   - Identifies light source changes that may affect fusion quality
   - Tracks edge density changes indicating scene modifications
   - Assesses impact on feature matching and tracking reliability

4. **Moving Object Detection Quality**
   - Evaluates consistency of dynamic object detection across frames
   - Quantifies detection confidence and fusion quality scores
   - Tracks temporal consistency in object detection
   - Identifies frames with detection quality issues
   - Assesses overall camera-LiDAR fusion capability

### Supported Dataset Formats

- **KITTI Format**: Datasets with camera and LiDAR subfolders
- **Camera Data**: PNG images in `camera_*/data/` directories
- **LiDAR Data**: Binary point cloud files in `lidar/data/` directories
- **Calibration Files**: `calib_velo_to_cam.txt` and `calib_cam_to_cam.txt`

### Output Metrics

**Calibration Quality:**
- Edge Alignment Score (0.0-1.0): Higher indicates better geometric alignment
- Mutual Information Score (0.0-1.0+): Higher indicates better sensor correlation
- Contrastive Score: Deep learning-based similarity metric (if enabled)

**Projection Error:**
- Mean Reprojection Error: Average error across all frames
- Max Reprojection Error: Peak error indicating worst-case calibration
- Increasing Error Count: Frames showing drift trend

**Illumination:**
- Mean Brightness: Average image brightness
- Brightness Standard Deviation: Variation in illumination
- Light Source Changes: Count of detected major lighting events

**Moving Objects:**
- Mean Detected Objects: Average number of detected objects per frame
- Detection Confidence: Average confidence in object detection
- Fusion Quality Score: Combined metric for fusion capability
- Consistency Score: Temporal consistency of detections

## Output Reports

### Anomaly Detection Reports

#### Markdown Report
- Executive summary with health metrics
- Detected anomalies table
- Frame-by-frame statistics

#### HTML Report
- Interactive visualizations (point count, severity distributions)
- Color-coded severity indicators
- Embedded plots

#### CSV Export
- All detected anomalies
- Frame-by-frame statistics
- Metadata for each anomaly

### Synchronization Reports

#### Markdown Report
- Executive summary with quality score
- Sensor stream statistics
- Pairwise synchronization results
- Recommendations and compliance flags

#### HTML Report
- Styled tables with quality indicators
- Embedded temporal heatmaps
- Visual drift patterns over time

#### CSV Export
- Pairwise synchronization results
- Detailed metrics per sensor pair

#### YAML Parameters
- Recommended timestamp corrections
- Mean/max offsets per sensor pair
- Drift rates for each pair

### Health Reporting & Dataset Quality Assessment

#### HTML Report (Dashboard)
- Plotly-based multi-panel visualization
- Overall quality per sequence (bar chart)
- Timeliness dimension analysis (line chart)
- Completeness dimension analysis (line chart)
- Temporal vs anomaly score comparison
- Quality dimension status table with implementation status

#### Structured Metrics Exports
- **CSV**: Per-sensor detailed metrics and per-sequence aggregated metrics
- **JSON**: Structured format for external tool integration and ML pipelines
- **YAML**: Human-readable configuration format with quality metadata

#### Summary Reports
- **TXT**: Text-based summary with statistics, tier distribution, best/worst sequences
- Detailed per-sequence breakdown with all quality dimensions

#### Visualizations (PNG)
- Quality scores bar chart with color-coded tiers
- Dimension comparison chart (timeliness vs completeness)
- Health tier distribution chart

#### Curation Recommendations
- **TXT**: Detailed human-readable curation guidance
- **JSON**: Machine-readable recommendations for pipeline integration
- Severity-based categorization (critical/high/medium/low)
- Actionable recommendations: exclude, review, or monitor

## Testing

Tests are organized by functionality:
- `test_bag_loader.py`: ROS2 bag loading tests
- `test_detection_pipeline.py`: Anomaly detection pipeline tests
- `test_edge.py`: Edge case handling
- `test_one_shot.py`: Single-frame analysis
- `test_pattern.py`: Pattern detection tests
- `test_smoke.py`: Basic smoke tests

Run tests with: `pytest tests/`
