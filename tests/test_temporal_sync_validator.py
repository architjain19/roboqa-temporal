import math
from pathlib import Path

import numpy as np
import yaml

from roboqa_temporal.synchronization import SensorStream, TemporalSyncValidator


def _make_stream(name: str, freq_hz: float, count: int, offset_ms: float = 0.0, drift_ms: float = 0.0):
    base = np.arange(count, dtype=np.float64) / freq_hz
    delta = (offset_ms / 1000.0) + (drift_ms / 1000.0) * (base / base.max() if base.max() else 0.0)
    timestamps_ns = ((base + delta) * 1e9).astype(np.int64).tolist()
    return SensorStream(
        name=name,
        topic=f"/{name}",
        timestamps_ns=timestamps_ns,
        expected_frequency=freq_hz,
    )


def test_temporal_sync_perfect_alignment(tmp_path):
    validator = TemporalSyncValidator(output_dir=str(tmp_path))
    camera = _make_stream("camera", 30.0, 300, offset_ms=0.2)
    lidar = _make_stream("lidar", 10.0, 100)
    streams = {"camera": camera, "lidar": lidar}

    report = validator.analyze_streams(streams, bag_name="perfect", include_visualizations=False)

    assert math.isclose(report.metrics["temporal_offset_score"], 1.0, rel_tol=1e-2)
    assert not report.recommendations
    assert "camera_lidar" in report.pair_results
    pair = report.pair_results["camera_lidar"]
    assert pair.approx_time_pass
    assert pair.max_delta_ms < 5.0


def test_temporal_sync_detects_drift(tmp_path):
    validator = TemporalSyncValidator(output_dir=str(tmp_path))
    lidar = _make_stream("lidar", 10.0, 200)
    imu = _make_stream("imu", 200.0, 2000, drift_ms=5.0)
    streams = {"lidar": lidar, "imu": imu}

    report = validator.analyze_streams(streams, bag_name="drift", include_visualizations=False)

    assert report.metrics["avg_drift_rate_ms_per_s"] > 0.1
    assert any("drift" in rec for rec in report.recommendations)
    pair = report.pair_results["lidar_imu"]
    assert not pair.ptp_pass or not pair.approx_time_pass


def test_temporal_sync_parameter_file(tmp_path):
    validator = TemporalSyncValidator(output_dir=str(tmp_path))
    camera = _make_stream("camera", 30.0, 50)
    lidar = _make_stream("lidar", 10.0, 50, offset_ms=3.0)
    imu = _make_stream("imu", 200.0, 50)
    streams = {"camera": camera, "lidar": lidar, "imu": imu}

    report = validator.analyze_streams(streams, bag_name="params", include_visualizations=False)

    assert report.parameter_file is not None
    param_path = Path(report.parameter_file)
    assert param_path.exists()
    payload = yaml.safe_load(param_path.read_text())
    assert payload["metadata"]["iso_8000_61"] is True
    assert "temporal_corrections" in payload
    assert any(key.startswith("camera_lidar") for key in payload["temporal_corrections"])


def test_temporal_sync_report_structure_smoke(tmp_path):
    """
    author: xinxintai
    reviewer: buddy_placeholder
    category: smoke test
    """
    validator = TemporalSyncValidator(
        output_dir=str(tmp_path),
        expected_frequency_hz={"camera": 30.0, "lidar": 30.0, "imu": 200.0},
    )
    camera = _make_stream("camera", 30.0, 120)
    lidar = _make_stream("lidar", 30.0, 120)
    imu = _make_stream("imu", 200.0, 400)
    streams = {"camera": camera, "lidar": lidar, "imu": imu}

    report = validator.analyze_streams(streams, bag_name="structure", include_visualizations=False)

    expected_pairs = {"camera_lidar", "lidar_imu", "camera_imu"}
    assert expected_pairs.issubset(report.pair_results.keys())
    assert report.metrics["temporal_offset_score"] > 0.4
    for pair_name in expected_pairs:
        pair = report.pair_results[pair_name]
        assert pair.deltas_ms.size > 0
        assert pair.rolling_mean_ms.size == pair.deltas_ms.size


def test_temporal_sync_frequency_violation_one_shot(tmp_path):
    """
    author: xinxintai
    reviewer: buddy_placeholder
    category: one-shot test
    """
    validator = TemporalSyncValidator(output_dir=str(tmp_path))
    camera = _make_stream("camera", 10.0, 200)
    camera.expected_frequency = 30.0
    lidar = _make_stream("lidar", 10.0, 200)
    streams = {"camera": camera, "lidar": lidar}

    report = validator.analyze_streams(streams, bag_name="freq", include_visualizations=False)

    assert "camera_frequency_violation" in report.compliance_flags
    assert any("camera frequency deviation" in rec for rec in report.recommendations)


def test_temporal_sync_handles_no_matches_edge(tmp_path):
    """
    author: xinxintai
    reviewer: buddy_placeholder
    category: edge test
    """
    validator = TemporalSyncValidator(output_dir=str(tmp_path))
    camera = _make_stream("camera", 30.0, 60)
    lidar = _make_stream("lidar", 30.0, 60, offset_ms=6000.0)
    streams = {"camera": camera, "lidar": lidar}

    report = validator.analyze_streams(streams, bag_name="edge", include_visualizations=False)
    pair = report.pair_results["camera_lidar"]

    assert pair.deltas_ms.size == min(len(camera.timestamps_ns), len(lidar.timestamps_ns))
    assert not pair.approx_time_pass
    assert math.isclose(pair.max_delta_ms, 6000.0, rel_tol=0.01)


def test_auto_detect_topics_pattern_detection():
    """
    author: xinxintai
    reviewer: buddy_placeholder
    category: pattern test
    justification: verifies naming/type patterns trigger automatic topic detection
    """
    topic_types = {
        "/perception/front/image_color": "sensor_msgs/msg/Image",
        "/perception/rear/camera/compressed": "sensor_msgs/msg/CompressedImage",
        "/perception/front/pointcloud": "sensor_msgs/msg/PointCloud2",
        "/custom/front/image_stream": "custom_msgs/msg/ImageLike",
        "/imu/data": "sensor_msgs/msg/Imu",
    }

    matches = TemporalSyncValidator._auto_detect_topics("camera", topic_types)

    assert matches == sorted(
        [
            "/custom/front/image_stream",
            "/perception/front/image_color",
            "/perception/rear/camera/compressed",
        ]
    )


def test_temporal_sync_exports_report_files(tmp_path):
    validator = TemporalSyncValidator(
        output_dir=str(tmp_path),
        report_formats=["markdown", "csv"],
        auto_export_reports=True,
    )
    camera = _make_stream("camera", 30.0, 40)
    lidar = _make_stream("lidar", 10.0, 40, offset_ms=2.5)
    streams = {"camera": camera, "lidar": lidar}

    report = validator.analyze_streams(streams, bag_name="export", include_visualizations=False)

    assert {"markdown", "csv"}.issubset(report.report_files.keys())
    for path in report.report_files.values():
        assert Path(path).exists()
    md_path = Path(report.report_files["markdown"])
    assert "Temporal Synchronization Report" in md_path.read_text()
