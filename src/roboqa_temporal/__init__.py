"""

################################################################

File: roboqa_temporal/__init__.py
Created: 2025-11-20
Created by: Archit Jain (architj@uw.edu)
Last Modified: 2025-11-20
Last Modified by: Archit Jain (architj@uw.edu)

#################################################################

Copyright: RoboQA-Temporal Authors
License: MIT License

################################################################

RoboQA-Temporal: Automated quality assessment and anomaly detection
for multi-sensor robotics datasets, with focus on ROS2 bag files.

################################################################

"""

__version__ = "0.1.0"
__author__ = "Archit Jain"

__all__ = []

try:
    from roboqa_temporal.loader import BagLoader

    __all__.append("BagLoader")
except ImportError:  # pragma: no cover - requires ROS
    BagLoader = None  # type: ignore

try:
    from roboqa_temporal.preprocessing import Preprocessor

    __all__.append("Preprocessor")
except ImportError:  # pragma: no cover
    Preprocessor = None  # type: ignore

try:
    from roboqa_temporal.detection import AnomalyDetector

    __all__.append("AnomalyDetector")
except ImportError:  # pragma: no cover
    AnomalyDetector = None  # type: ignore

try:
    from roboqa_temporal.reporting import ReportGenerator

    __all__.append("ReportGenerator")
except ImportError:  # pragma: no cover
    ReportGenerator = None  # type: ignore

try:
    from roboqa_temporal.synchronization import TemporalSyncValidator
    __all__.append("TemporalSyncValidator")
except ImportError:
    pass

__all__.append("TemporalSyncValidator")
