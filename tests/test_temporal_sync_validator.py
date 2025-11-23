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
