from __future__ import annotations

import itertools
from pathlib import Path
from typing import Iterator, List

import pytest

pytest.importorskip("rosbag2_py")
pytest.importorskip("rclpy")
pytest.importorskip("rosidl_runtime_py")
pytest.importorskip("sensor_msgs")
pytest.importorskip("sensor_msgs_py")

from roboqa_temporal.loader import BagLoader
from roboqa_temporal.loader.bag_loader import PointCloudFrame

_DATASET_DIR = Path(__file__).resolve().parents[1] / "dataset" / "mybag"


def _ensure_sample_bag_exists() -> Path:
    if not _DATASET_DIR.exists():
        pytest.skip("Sample ROS2 bag not available in dataset/mybag")
    if not (_DATASET_DIR / "metadata.yaml").exists():
        pytest.skip("metadata.yaml missing for sample ROS2 bag")
    return _DATASET_DIR


@pytest.fixture(scope="session")
def bag_directory() -> str:
    """Return path to the on-disk ROS2 bag used for integration tests."""
    return str(_ensure_sample_bag_exists())


@pytest.fixture()
def bag_loader(bag_directory: str) -> Iterator[BagLoader]:
    """Provide a fresh BagLoader instance for each test."""
    loader = BagLoader(bag_directory)
    try:
        yield loader
    finally:
        loader.close()


@pytest.fixture()
def sample_frames(bag_directory: str) -> List[PointCloudFrame]:
    """Read a small batch of frames for detector tests."""
    loader = BagLoader(bag_directory)
    try:
        frames = list(
            itertools.islice(loader.read_point_clouds(max_frames=20, progress=False), 20)
        )
    finally:
        loader.close()

    if not frames:
        pytest.skip("Sample bag did not yield any point cloud frames")
    return frames


