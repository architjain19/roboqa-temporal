"""
Temporary stub for detection utilities.

This exists so that tests importing `roboqa_temporal.detection.AnomalyDetector`
do not fail with ModuleNotFoundError. It can be replaced by a full implementation
in a future feature.
"""


class AnomalyDetector:
    """Minimal placeholder anomaly detector."""

    def __init__(self, *args, **kwargs):
        # Accept any arguments so tests don't break.
        pass

    def detect(self, data):
        """Return an empty list of anomalies for now."""
        return []
