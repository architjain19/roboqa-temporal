"""

################################################################

File: roboqa_temporal/synchronization/__init__.py
Created: 2025-12-07
Created by: RoboQA-Temporal Authors
Last Modified: 2025-12-07
Last Modified by: RoboQA-Temporal Authors

#################################################################

Copyright: RoboQA-Temporal Authors
License: MIT License

################################################################

Cross-Modal Synchronization Analysis module.

Provides temporal synchronization validation for multi-sensor data,
including timestamp drift detection, data loss flagging, and temporal
alignment quality scoring.

################################################################

"""

from roboqa_temporal.synchronization.temporal_validator import (
    TemporalSyncValidator,
    SensorStream,
    PairwiseDriftResult,
    TemporalSyncReport,
)

__all__ = [
    "TemporalSyncValidator",
    "SensorStream",
    "PairwiseDriftResult",
    "TemporalSyncReport",
]
