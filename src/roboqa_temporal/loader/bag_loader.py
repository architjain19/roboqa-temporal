"""
Temporary stub for loader utilities.

This exists so that tests importing `roboqa_temporal.loader.bag_loader`
do not fail with ModuleNotFoundError.
"""

class PointCloudFrame:
    """Minimal placeholder for PointCloudFrame."""
    def __init__(self, *args, **kwargs):
        pass

class BagLoader:
    """Minimal placeholder for BagLoader."""
    def __init__(self, *args, **kwargs):
        pass

    def load_bag(self, *args, **kwargs):
        """Return empty list to prevent runtime errors."""
        return []
