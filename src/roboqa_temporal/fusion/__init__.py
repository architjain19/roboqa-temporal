"""

################################################################

File: roboqa_temporal/fusion/__init__.py
Created: 2025-12-08
Created by: RoboQA-Temporal Authors
Last Modified: 2025-12-08
Last Modified by: RoboQA-Temporal Authors

#################################################################

Copyright: RoboQA-Temporal Authors
License: MIT License

################################################################

Camera-LiDAR Fusion Quality Assessment Module

Provides comprehensive validation of camera-LiDAR sensor fusion quality
including calibration drift estimation, projection error quantification,
illumination change detection, and moving object detection quality.

################################################################

"""

from roboqa_temporal.fusion.fusion_quality_validator import (
    CalibrationQualityValidator,
    CalibrationQualityReport,
    CalibrationPairResult,
    CalibrationStream,
    ProjectionErrorFrame,
    IlluminationFrame,
    MovingObjectFrame,
)

__all__ = [
    "CalibrationQualityValidator",
    "CalibrationQualityReport",
    "CalibrationPairResult",
    "CalibrationStream",
    "ProjectionErrorFrame",
    "IlluminationFrame",
    "MovingObjectFrame",
]
