from __future__ import annotations

import itertools

from roboqa_temporal.loader.bag_loader import PointCloudFrame


def test_topic_info_includes_point_cloud_topic(bag_loader):
    """
    author: architjain
    reviewer: dharinesh
    category: integration test (for CI purposes)
    justification: Tests ROS2 bag integration functionality
    """
    topic_info = bag_loader.get_topic_info()
    assert "/synthetic_points" in topic_info
    assert topic_info["/synthetic_points"]["type"].endswith("PointCloud2")


def test_read_point_clouds_produces_frames(bag_loader):
    """
    author: architjain
    reviewer: dharinesh
    category: integration test (for CI purposes)
    justification: Tests ROS2 bag reading and frame production
    """
    frames = list(itertools.islice(bag_loader.read_point_clouds(progress=False), 5))
    assert frames, "Expected sample bag to yield at least one frame"

    first_frame: PointCloudFrame = frames[0]
    assert first_frame.points.shape[1] == 3
    assert first_frame.num_points == first_frame.points.shape[0]

    timestamps = [frame.timestamp for frame in frames]
    assert all(b >= a for a, b in zip(timestamps, timestamps[1:]))

