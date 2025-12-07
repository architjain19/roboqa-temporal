"""
Temporary stub for detection utilities.

This exists so that tests importing `roboqa_temporal.detection`
do not fail with ModuleNotFoundError or ImportError.
"""

class AnomalyDetector:
    """Minimal placeholder anomaly detector."""
    def __init__(self, *args, **kwargs):
        # Accept any arguments so tests don't break.
        pass

    def detect(self, data):
        """Return an empty list of anomalies for now."""
        return []

class Anomaly:
    """Minimal placeholder for Anomaly data class."""
    def __init__(self, *args, **kwargs):
        pass

class DetectionResult:
    """Minimal placeholder for DetectionResult data class."""
    def __init__(self, *args, **kwargs):
        pass
