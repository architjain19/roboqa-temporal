"""

################################################################

File: roboqa_temporal/preprocessing/preprocessor.py
Created: 2025-11-20
Created by: Archit Jain (architj@uw.edu)
Last Modified: 2025-11-20
Last Modified by: Archit Jain (architj@uw.edu)

#################################################################

Copyright: RoboQA-Temporal Authors
License: MIT License

################################################################

Preprocessing module for RoboQA-Temporal. This module provides
functionality to clean and normalize point cloud data before
anomaly detection. It includes methods for downsampling, outlier
removal, time alignment, and normalization.

################################################################

"""

from typing import List, Optional, Tuple
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import LocalOutlierFactor

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

from roboqa_temporal.loader.bag_loader import PointCloudFrame


class Preprocessor:
    """
    Preprocessor for point cloud data.

    Supports:
    - Downsampling (voxel-based, random)
    - Outlier removal (statistical, radius-based)
    - Time alignment and synchronization
    - Normalization
    """

    def __init__(
        self,
        voxel_size: Optional[float] = None,
        remove_outliers: bool = True,
        outlier_method: str = "statistical",
        outlier_params: Optional[dict] = None,
    ):
        """
        Initialize preprocessor.

        Args:
            voxel_size: Voxel size for downsampling (None = no downsampling)
            remove_outliers: Whether to remove outliers
            outlier_method: Method for outlier removal ('statistical', 'radius', 'lof')
            outlier_params: Parameters for outlier removal
        """
        self.voxel_size = voxel_size
        self.remove_outliers = remove_outliers
        self.outlier_method = outlier_method
        self.outlier_params = outlier_params or {}

    def process_frame(self, frame: PointCloudFrame) -> PointCloudFrame:
        """
        Process a single point cloud frame.

        Args:
            frame: Input point cloud frame

        Returns:
            Processed point cloud frame
        """
        points = frame.points.copy()

        # Downsampling
        if self.voxel_size is not None:
            points = self._voxel_downsample(points)

        # Outlier removal
        if self.remove_outliers and len(points) > 0:
            points, mask = self._remove_outliers(points)
            # Apply mask to intensities and colors if they exist
            intensities = None
            colors = None
            if frame.intensities is not None and len(frame.intensities) == len(frame.points):
                intensities = frame.intensities[mask] if mask is not None else frame.intensities
            if frame.colors is not None and len(frame.colors) == len(frame.points):
                colors = frame.colors[mask] if mask is not None else frame.colors
        else:
            intensities = frame.intensities
            colors = frame.colors

        return PointCloudFrame(
            timestamp=frame.timestamp,
            frame_id=frame.frame_id,
            points=points,
            intensities=intensities,
            colors=colors,
        )

    def process_sequence(
        self, frames: List[PointCloudFrame]
    ) -> List[PointCloudFrame]:
        """
        Process a sequence of point cloud frames.

        Args:
            frames: List of input frames

        Returns:
            List of processed frames
        """
        return [self.process_frame(frame) for frame in frames]

    def _voxel_downsample(self, points: np.ndarray) -> np.ndarray:
        """
        Downsample points using voxel grid.

        Args:
            points: Point cloud (N, 3)

        Returns:
            Downsampled points
        """
        if not OPEN3D_AVAILABLE:
            # Fallback to simple grid-based downsampling
            return self._simple_voxel_downsample(points)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        return np.asarray(pcd.points)

    def _simple_voxel_downsample(self, points: np.ndarray) -> np.ndarray:
        """
        Simple voxel-based downsampling without Open3D.

        Args:
            points: Point cloud (N, 3)

        Returns:
            Downsampled points
        """
        if len(points) == 0:
            return points

        # Create voxel grid
        voxel_indices = np.floor(points / self.voxel_size).astype(int)

        # Get unique voxels and keep first point in each voxel
        _, unique_indices = np.unique(
            voxel_indices, axis=0, return_index=True
        )
        return points[unique_indices]

    def _remove_outliers(
        self, points: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Remove outliers from point cloud.

        Args:
            points: Point cloud (N, 3)

        Returns:
            Tuple of (filtered_points, mask)
        """
        if len(points) < 4:  # Need at least 4 points for outlier detection
            return points, None

        if self.outlier_method == "statistical":
            return self._statistical_outlier_removal(points)
        elif self.outlier_method == "radius":
            return self._radius_outlier_removal(points)
        elif self.outlier_method == "lof":
            return self._lof_outlier_removal(points)
        else:
            raise ValueError(f"Unknown outlier method: {self.outlier_method}")

    def _statistical_outlier_removal(
        self, points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove outliers using statistical method (mean distance).

        Args:
            points: Point cloud (N, 3)

        Returns:
            Tuple of (filtered_points, mask)
        """
        nb_neighbors = self.outlier_params.get("nb_neighbors", 20)
        std_ratio = self.outlier_params.get("std_ratio", 2.0)

        if len(points) < nb_neighbors + 1:
            return points, np.ones(len(points), dtype=bool)

        # Compute distances to k nearest neighbors
        distances = cdist(points, points)
        # Sort and take k nearest (excluding self)
        k_distances = np.partition(distances, nb_neighbors + 1, axis=1)[
            :, 1 : nb_neighbors + 1
        ]
        mean_distances = np.mean(k_distances, axis=1)

        # Compute threshold
        mean = np.mean(mean_distances)
        std = np.std(mean_distances)
        threshold = mean + std_ratio * std

        # Filter points
        mask = mean_distances < threshold
        return points[mask], mask

    def _radius_outlier_removal(
        self, points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove outliers using radius-based method.

        Args:
            points: Point cloud (N, 3)

        Returns:
            Tuple of (filtered_points, mask)
        """
        radius = self.outlier_params.get("radius", 0.1)
        min_neighbors = self.outlier_params.get("min_neighbors", 5)

        if len(points) == 0:
            return points, np.array([], dtype=bool)

        # Compute distances
        distances = cdist(points, points)
        # Count neighbors within radius
        neighbor_counts = np.sum(distances < radius, axis=1) - 1  # Exclude self

        # Filter points
        mask = neighbor_counts >= min_neighbors
        return points[mask], mask

    def _lof_outlier_removal(
        self, points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove outliers using Local Outlier Factor.

        Args:
            points: Point cloud (N, 3)

        Returns:
            Tuple of (filtered_points, mask)
        """
        n_neighbors = self.outlier_params.get("n_neighbors", 20)
        contamination = self.outlier_params.get("contamination", 0.1)

        if len(points) < n_neighbors + 1:
            return points, np.ones(len(points), dtype=bool)

        lof = LocalOutlierFactor(
            n_neighbors=min(n_neighbors, len(points) - 1),
            contamination=contamination,
        )
        labels = lof.fit_predict(points)
        mask = labels == 1  # 1 = inlier, -1 = outlier
        return points[mask], mask

    def align_timestamps(
        self, frames: List[PointCloudFrame], target_fps: Optional[float] = None
    ) -> List[PointCloudFrame]:
        """
        Align timestamps to uniform intervals.

        Args:
            frames: List of frames with timestamps
            target_fps: Target frame rate (None = keep original)

        Returns:
            List of frames with aligned timestamps
        """
        if not frames or target_fps is None:
            return frames

        timestamps = np.array([f.timestamp for f in frames])
        start_time = timestamps[0]
        dt = 1.0 / target_fps

        aligned_frames = []
        current_target = start_time

        for frame in frames:
            if frame.timestamp >= current_target:
                aligned_frames.append(frame)
                current_target += dt

        return aligned_frames

