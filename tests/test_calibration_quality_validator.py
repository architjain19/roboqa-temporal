import math
from pathlib import Path

import numpy as np
import yaml

from roboqa_temporal.calibration import (
    CalibrationQualityValidator,
    CalibrationStream,
)


def _make_synthetic_pair(
    name: str,
    n_frames: int = 50,
    miscalibration_pixels: float = 0.0,
) -> CalibrationStream:
    # In these synthetic tests we do not actually create images/pointclouds.
    # We just provide dummy paths and encode the miscalibration level into
    # the calibration_file name so the validator can infer it.
    image_paths = [f"/synthetic/{name}/image_{i:06d}.png" for i in range(n_frames)]
    pointcloud_paths = [
        f"/synthetic/{name}/cloud_{i:06d}.bin" for i in range(n_frames)
    ]
    calib_tag = f"miscalib_{miscalibration_pixels:.1f}px"
    calibration_file = f"/synthetic/calib/{name}_{calib_tag}.txt"

    return CalibrationStream(
        name=name,
        image_paths=image_paths,
        pointcloud_paths=pointcloud_paths,
        calibration_file=calibration_file,
        camera_id="image_02",
        lidar_id="velodyne",
    )


def test_calibration_quality_perfect_alignment(tmp_path):
    validator = CalibrationQualityValidator(output_dir=str(tmp_path))

    cam_lidar = _make_synthetic_pair(
        "cam_lidar", n_frames=80, miscalibration_pixels=0.5
    )
    pairs = {"cam_lidar": cam_lidar}

    report = validator.analyze_sequences(
        pairs,
        bag_name="perfect_calib",
        include_visualizations=False,
    )

    assert "edge_alignment_score" in report.metrics
    assert "mi_score" in report.metrics
    assert "contrastive_score" in report.metrics

    assert report.metrics["edge_alignment_score"] >= 0.9
    assert report.metrics["mi_score"] >= 0.9
    assert report.metrics["contrastive_score"] >= 0.9

    assert not report.recommendations

    assert "cam_lidar" in report.pair_results
    pair = report.pair_results["cam_lidar"]
    assert pair.overall_pass
    assert pair.pass_geom_edge
    assert pair.pass_mi
    assert pair.pass_contrastive


def test_calibration_quality_detects_miscalibration(tmp_path):
    validator = CalibrationQualityValidator(output_dir=str(tmp_path))

    good_pair = _make_synthetic_pair(
        "good_cam_lidar", n_frames=60, miscalibration_pixels=1.0
    )
    bad_pair = _make_synthetic_pair(
        "bad_cam_lidar", n_frames=60, miscalibration_pixels=15.0
    )

    pairs = {
        "good_cam_lidar": good_pair,
        "bad_cam_lidar": bad_pair,
    }

    report = validator.analyze_sequences(
        pairs,
        bag_name="drift_calib",
        include_visualizations=False,
    )

    assert "bad_cam_lidar" in report.pair_results
    bad_res = report.pair_results["bad_cam_lidar"]
    good_res = report.pair_results["good_cam_lidar"]

    assert not bad_res.overall_pass or not (
        bad_res.pass_geom_edge and bad_res.pass_mi and bad_res.pass_contrastive
    )

    assert good_res.geom_edge_score > bad_res.geom_edge_score
    assert good_res.mutual_information > bad_res.mutual_information
    assert good_res.contrastive_score > bad_res.contrastive_score

    assert any(
        "calibration" in rec.lower()
        or "misalignment" in rec.lower()
        or "drift" in rec.lower()
        for rec in report.recommendations
    )


def test_calibration_quality_parameter_file(tmp_path):
    validator = CalibrationQualityValidator(output_dir=str(tmp_path))

    good_pair = _make_synthetic_pair(
        "cam_lidar_good", n_frames=40, miscalibration_pixels=1.0
    )
    off_pair = _make_synthetic_pair(
        "cam_lidar_off", n_frames=40, miscalibration_pixels=6.0
    )

    pairs = {
        "cam_lidar_good": good_pair,
        "cam_lidar_off": off_pair,
    }

    report = validator.analyze_sequences(
        pairs,
        bag_name="calib_params",
        include_visualizations=False,
    )

    assert report.parameter_file is not None
    param_path = Path(report.parameter_file)
    assert param_path.exists()

    payload = yaml.safe_load(param_path.read_text())

    assert payload["metadata"]["iso_8000_61"] is True
    assert payload["metadata"]["type"] == "sensor_calibration_quality"

    assert "calibration_corrections" in payload

    keys = list(payload["calibration_corrections"].keys())
    assert any(key.startswith("cam_lidar_good") for key in keys)
    assert any(key.startswith("cam_lidar_off") for key in keys)

    good_entry = payload["calibration_corrections"]["cam_lidar_good"]
    off_entry = payload["calibration_corrections"]["cam_lidar_off"]

    assert good_entry["quality_score"] > off_entry["quality_score"]
    assert "recalibrate" in off_entry.get("recommendation", "").lower()
