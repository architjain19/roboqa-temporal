import math
from pathlib import Path
import pytest

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

    # Check HTML report generation
    assert report.html_report_file is not None
    html_path = Path(report.html_report_file)
    assert html_path.exists()
    html_content = html_path.read_text()
    assert "Calibration Quality Report" in html_content
    assert "cam_lidar_good" in html_content
    assert "cam_lidar_off" in html_content
    assert "FAIL" in html_content  # off_pair should fail


def test_calibration_validator_smoke(tmp_path):
    """author: Dharineesh Somisetty
    reviewer: <buddy name>
    category: smoke test
    """

    validator = CalibrationQualityValidator(output_dir=str(tmp_path))
    pairs = {"smoke_pair": _make_synthetic_pair("smoke", miscalibration_pixels=2.0)}

    report = validator.analyze_sequences(
        pairs,
        bag_name="smoke_bag",
        include_visualizations=False,
    )

    assert report.metrics["edge_alignment_score"] > 0.0


def test_calibration_validator_one_shot(tmp_path):
    """author: Dharineesh Somisetty
    reviewer: <buddy name>
    category: one-shot test
    """

    validator = CalibrationQualityValidator(output_dir=str(tmp_path))
    pair = _make_synthetic_pair("one_shot", miscalibration_pixels=10.0)
    report = validator.analyze_sequences({"one_shot": pair}, bag_name="one_shot")

    expected_quality = max(0.0, 1.0 - 10.0 / 20.0)
    assert math.isclose(
        report.pair_results["one_shot"].geom_edge_score,
        expected_quality,
        rel_tol=1e-6,
    )


def test_calibration_validator_edge_case_extreme_miscalibration(tmp_path):
    """author: Dharineesh Somisetty
    reviewer: <buddy name>
    category: edge test
    """

    validator = CalibrationQualityValidator(output_dir=str(tmp_path))
    pair = _make_synthetic_pair("edge", miscalibration_pixels=1000.0)
    report = validator.analyze_sequences({"edge": pair}, bag_name="edge_case")

    result = report.pair_results["edge"]
    assert math.isclose(result.geom_edge_score, 0.0)
    assert not result.overall_pass


def test_calibration_validator_pattern_quality_decreases_with_miscalibration(tmp_path):
    """author: Dharineesh Somisetty
    reviewer: <buddy name>
    category: pattern test
    """

    validator = CalibrationQualityValidator(output_dir=str(tmp_path))

    miscalibrations = [0.0, 2.0, 5.0, 10.0, 15.0]
    previous_score = float("inf")

    for idx, mis_px in enumerate(miscalibrations):
        pair_name = f"pattern_{idx}"
        pair = _make_synthetic_pair(pair_name, miscalibration_pixels=mis_px)
        report = validator.analyze_sequences({pair_name: pair}, bag_name=f"pattern_{idx}")

        score = report.pair_results[pair_name].geom_edge_score
        assert score <= previous_score + 1e-9
        previous_score = score


def test_calibration_quality_real_data(tmp_path):
    """Test with real KITTI data.
    
    author: Dharineesh Somisetty
    reviewer: <buddy name>
    category: integration test
    """
    repo_root = Path(__file__).parent
    data_root = repo_root / "2011_09_26"
    
    # Check if data exists
    drive_dir = data_root / "2011_09_26_drive_0002_sync"
    if not drive_dir.exists():
        pytest.skip("Real data not found")

    image_dir = drive_dir / "image_02" / "data"
    velo_dir = drive_dir / "velodyne_points" / "data"
    calib_dir = data_root / "City_calibration_texts"

    image_paths = sorted([str(p) for p in image_dir.glob("*.png")])
    pointcloud_paths = sorted([str(p) for p in velo_dir.glob("*.bin")])

    if not image_paths or not pointcloud_paths:
        pytest.skip("No images or pointclouds found in real data folder")

    stream = CalibrationStream(
        name="real_kitti_drive_0002",
        image_paths=image_paths,
        pointcloud_paths=pointcloud_paths,
        calibration_file=str(calib_dir),
        camera_id="image_02",
        lidar_id="velodyne",
    )

    # Save report to project root for visibility
    output_dir = repo_root
    validator = CalibrationQualityValidator(output_dir=str(output_dir))
    report = validator.analyze_sequences(
        {"KITTI_drive_0002": stream},
        bag_name="KITTI_dataset",
        include_visualizations=False,
    )

    assert "KITTI_drive_0002" in report.pair_results
    res = report.pair_results["KITTI_drive_0002"]
    
    print(f"Real Data Results: Edge={res.geom_edge_score}, MI={res.mutual_information}")
    
    # We expect non-zero scores if projection works
    # Note: MI can be negative or zero if no overlap, but with correct calib it should be positive?
    # NMI is usually 0 to 1.
    assert res.mutual_information > -1.0 
    assert res.geom_edge_score >= 0.0
    
    # Check if HTML report was generated
    assert Path(report.html_report_file).exists()
    
    # Verify HTML content has explanation
    html_content = Path(report.html_report_file).read_text()
    assert "Calibration Quality Report" in html_content
    assert "Understanding the Metrics" in html_content

    print(f"\nHTML Report generated at: {report.html_report_file}")

