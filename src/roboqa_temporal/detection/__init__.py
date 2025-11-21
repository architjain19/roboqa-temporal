"""

################################################################

File: roboqa_temporal/detection/__init__.py
Created: 2025-11-20
Created by: Archit Jain (architj@uw.edu)
Last Modified: 2025-11-20
Last Modified by: Archit Jain (architj@uw.edu)

#################################################################

Copyright: RoboQA-Temporal Authors
License: MIT License

################################################################

Anomaly detection module for RoboQA-Temporal.

################################################################

"""

from roboqa_temporal.detection.detector import AnomalyDetector
from roboqa_temporal.detection.detectors import (
    DensityDropDetector,
    SpatialDiscontinuityDetector,
    GhostPointDetector,
    TemporalConsistencyDetector,
)

__all__ = [
    "AnomalyDetector",
    "DensityDropDetector",
    "SpatialDiscontinuityDetector",
    "GhostPointDetector",
    "TemporalConsistencyDetector",
]

