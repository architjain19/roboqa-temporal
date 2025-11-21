"""

################################################################

File: roboqa_temporal/loader/bag_loader.py
Created: 2025-11-20
Created by: Archit Jain (architj@uw.edu)
Last Modified: 2025-11-20
Last Modified by: Archit Jain (architj@uw.edu)

#################################################################

Copyright: RoboQA-Temporal Authors
License: MIT License

################################################################

ROS2 bag file loader for point cloud and sensor data. This module
provides the BagLoader class which efficiently reads point cloud
messages from ROS2 bag files, supporting selective topic extraction,
frame-by-frame iteration, and sensor synchronization. It handles
various point cloud field formats and provides metadata for each
frame.

################################################################

"""

import os
from typing import List, Optional, Dict, Any, Iterator, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    import rosbag2_py
except ImportError as e:
    raise ImportError(
        "ROS2 dependencies not installed. Install with: pip install rclpy rosbag2-py"
    ) from e

try:
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs_py import point_cloud2
except ImportError as e:
    raise ImportError(
        "Sensor messages not available. Install with: pip install sensor-msgs"
    ) from e


@dataclass
class PointCloudFrame:
    """Container for a single point cloud frame with metadata."""

    timestamp: float
    frame_id: str
    points: np.ndarray  # Shape: (N, 3) for x, y, z
    intensities: Optional[np.ndarray] = None  # Shape: (N,)
    colors: Optional[np.ndarray] = None  # Shape: (N, 3) for RGB
    num_points: int = 0

    def __post_init__(self):
        """Calculate number of points after initialization."""
        if self.points is not None and len(self.points) > 0:
            self.num_points = len(self.points)


class BagLoader:
    """
    Efficient loader for ROS2 bag files, extracting point cloud and sensor data.

    Supports:
    - Reading point cloud messages from specified topics
    - Sensor synchronization and time alignment
    - Selective topic extraction
    - Frame-by-frame iteration
    """

    def __init__(
        self,
        bag_path: str,
        topics: Optional[List[str]] = None,
        frame_id: Optional[str] = None,
    ):
        """
        Initialize bag loader.

        Args:
            bag_path: Path to ROS2 bag file or directory
            topics: List of topic names to read (None = all point cloud topics)
            frame_id: Optional frame ID filter
        """
        self.bag_path = Path(bag_path)
        if not self.bag_path.exists():
            raise FileNotFoundError(f"Bag path does not exist: {bag_path}")

        self.topics = topics
        self.frame_id = frame_id
        self.reader = None
        self.topic_types = {}
        self._initialize_reader()

    def _initialize_reader(self):
        """Initialize ROS2 bag reader."""
        storage_options = rosbag2_py.StorageOptions(
            uri=str(self.bag_path), storage_id="sqlite3"
        )
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        )
        self.reader = rosbag2_py.SequentialReader()
        self.reader.open(storage_options, converter_options)

        # Get topic metadata
        topic_metadata = self.reader.get_all_topics_and_types()
        self.topic_types = {topic.name: topic.type for topic in topic_metadata}

        # Filter topics if specified
        if self.topics is None:
            # Auto-detect point cloud topics
            self.topics = [
                topic.name
                for topic in topic_metadata
                if "PointCloud2" in topic.type or "point_cloud" in topic.name.lower()
            ]
        else:
            # Validate specified topics exist
            available_topics = set(self.topic_types.keys())
            specified_topics = set(self.topics)
            missing = specified_topics - available_topics
            if missing:
                raise ValueError(
                    f"Topics not found in bag: {missing}. "
                    f"Available topics: {available_topics}"
                )

    def get_topic_info(self) -> Dict[str, Any]:
        """
        Get information about topics in the bag.

        Returns:
            Dictionary with topic names, types, and message counts
        """
        info = {}
        for topic_name in self.topics:
            msg_type = self.topic_types.get(topic_name, "unknown")
            info[topic_name] = {"type": msg_type}
        return info

    def read_point_clouds(
        self, max_frames: Optional[int] = None, progress: bool = True
    ) -> Iterator[PointCloudFrame]:
        """
        Read point cloud messages from bag file.

        Args:
            max_frames: Maximum number of frames to read (None = all)
            progress: Show progress bar

        Yields:
            PointCloudFrame objects with point cloud data
        """
        frame_count = 0
        pbar = tqdm(desc="Reading point clouds", disable=not progress)

        try:
            while self.reader.has_next():
                if max_frames is not None and frame_count >= max_frames:
                    break

                (topic, data, timestamp) = self.reader.read_next()

                if topic not in self.topics:
                    continue

                try:
                    msg_type = get_message(self.topic_types[topic])
                    msg = deserialize_message(data, msg_type)

                    if isinstance(msg, PointCloud2):
                        frame = self._extract_point_cloud(msg, timestamp)
                        if frame is not None:
                            pbar.update(1)
                            yield frame
                            frame_count += 1

                except Exception as e:
                    if progress:
                        pbar.write(f"Warning: Failed to read message: {e}")
                    continue
        finally:
            pbar.close()

    def _extract_point_cloud(
        self, msg: PointCloud2, timestamp: int
    ) -> Optional[PointCloudFrame]:
        """
        Extract point cloud data from ROS2 message.

        Args:
            msg: PointCloud2 ROS2 message
            timestamp: Message timestamp (nanoseconds)

        Returns:
            PointCloudFrame or None if extraction fails
        """
        try:
            # Filter by frame_id if specified
            if self.frame_id is not None and msg.header.frame_id != self.frame_id:
                return None

            # Inspect available fields to avoid asking for missing ones
            available_fields = [f.name for f in msg.fields]
            # Require x,y,z
            if not all(n in available_fields for n in ("x", "y", "z")):
                return None

            # Read XYZ (use skip_nans to avoid malformed rows)
            points_list = list(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
            if not points_list:
                return None

            pts_arr = np.asarray(points_list)
            # If read_points returned a structured array with named fields
            if getattr(pts_arr, "dtype", None) is not None and getattr(pts_arr.dtype, "names", None):
                try:
                    pts = np.vstack([pts_arr[name] for name in ("x", "y", "z")]).T.astype(np.float32)
                except Exception:
                    return None
            else:
                # pts_arr may be shape (N,3) or (N,) of tuples
                if pts_arr.ndim == 1:
                    pts = np.vstack(pts_arr).astype(np.float32)
                else:
                    pts = pts_arr.astype(np.float32)

            points = pts

            # Intensities: try common field names
            intensities = None
            intensity_candidates = ("intensity", "intensities", "i")
            intensity_field = next((c for c in intensity_candidates if c in available_fields), None)
            if intensity_field:
                try:
                    ints = list(point_cloud2.read_points(msg, field_names=(intensity_field,), skip_nans=True))
                    if ints:
                        # ensure we always take the first element of each returned tuple/row
                        intensities = np.array([float(v[0]) for v in ints], dtype=np.float32)
                except Exception:
                    intensities = None

            # Colors: prefer separate r,g,b fields; fall back to packed rgb/rgba
            colors = None
            if all(n in available_fields for n in ("r", "g", "b")):
                try:
                    cols = list(point_cloud2.read_points(msg, field_names=("r", "g", "b"), skip_nans=True))
                    if cols:
                        cols_arr = np.asarray(cols)
                        if getattr(cols_arr, "dtype", None) is not None and getattr(cols_arr.dtype, "names", None):
                            cols_pts = np.vstack([cols_arr[name] for name in ("r", "g", "b")]).T
                        else:
                            if cols_arr.ndim == 1:
                                cols_pts = np.vstack(cols_arr)
                            else:
                                cols_pts = cols_arr
                        colors = (cols_pts.astype(np.uint8) / 255.0)
                except Exception:
                    colors = None
            elif "rgb" in available_fields or "rgba" in available_fields:
                packed_name = "rgb" if "rgb" in available_fields else "rgba"
                try:
                    packed_vals = list(point_cloud2.read_points(msg, field_names=(packed_name,)))
                    if packed_vals:
                        import struct

                        rgb_list = []
                        for item in packed_vals:
                            # item may be (v,) or structured row; normalize to single value v
                            try:
                                v = item[0]
                            except Exception:
                                v = item
                            # v may be float32 containing packed uint32 bits or already int
                            try:
                                as_uint = struct.unpack("I", struct.pack("f", float(v)))[0]
                            except Exception:
                                try:
                                    as_uint = int(v)
                                except Exception:
                                    continue
                            r = (as_uint >> 16) & 0xFF
                            g = (as_uint >> 8) & 0xFF
                            b = as_uint & 0xFF
                            rgb_list.append((r, g, b))
                        if rgb_list:
                            colors = np.array(rgb_list, dtype=np.uint8) / 255.0
                except Exception:
                    colors = None

            # Convert timestamp from nanoseconds to seconds
            timestamp_sec = timestamp / 1e9

            return PointCloudFrame(
                timestamp=timestamp_sec,
                frame_id=msg.header.frame_id,
                points=points,
                intensities=intensities,
                colors=colors,
            )
        except Exception as e:
             print(f"Error extracting point cloud: {e}")
             return None

    def read_all_frames(
        self, max_frames: Optional[int] = None
    ) -> List[PointCloudFrame]:
        """
        Read all point cloud frames into memory.

        Args:
            max_frames: Maximum number of frames to read

        Returns:
            List of PointCloudFrame objects
        """
        return list(self.read_point_clouds(max_frames=max_frames))

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get basic statistics about the bag file.

        Returns:
            Dictionary with statistics
        """
        frames = list(self.read_point_clouds(max_frames=1000, progress=False))
        if not frames:
            return {"num_frames": 0, "error": "No point cloud frames found"}

        num_points = [f.num_points for f in frames]
        timestamps = [f.timestamp for f in frames]

        return {
            "num_frames_sampled": len(frames),
            "avg_points_per_frame": np.mean(num_points),
            "min_points": int(np.min(num_points)),
            "max_points": int(np.max(num_points)),
            "std_points": float(np.std(num_points)),
            "duration_sec": float(np.max(timestamps) - np.min(timestamps)),
            "avg_fps": len(frames) / (np.max(timestamps) - np.min(timestamps))
            if len(frames) > 1
            else 0.0,
        }

    def close(self):
        """Close the bag reader."""
        if self.reader is None:
            return

        # rosbag2_py.SequentialReader in some ROS2 versions doesn't implement .close()
        try:
            if callable(getattr(self.reader, "close", None)):
                self.reader.close()
            else:
                # trying alternative cleanup names
                for fn_name in ("reset", "reset_reader", "reset_storage", "finalize"):
                    fn = getattr(self.reader, fn_name, None)
                    if callable(fn):
                        fn()
                        break
        except Exception as e:
            print(f"Warning: failed to close rosbag reader cleanly: {e}")
        finally:
            self.reader = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

