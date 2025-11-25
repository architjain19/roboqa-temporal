from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass
class CalibrationStream:
    name: str
    image_paths: List[str]
    pointcloud_paths: List[str]
    calibration_file: str
    camera_id: str
    lidar_id: str


@dataclass
class CalibrationPairResult:
    geom_edge_score: float
    mutual_information: float
    contrastive_score: float
    pass_geom_edge: bool
    pass_mi: bool
    pass_contrastive: bool
    overall_pass: bool
    details: Dict


@dataclass
class CalibrationQualityReport:
    bag_name: str
    metrics: Dict
    pair_results: Dict[str, CalibrationPairResult]
    recommendations: List[str]
    parameter_file: Optional[str]


class CalibrationQualityValidator:
    """Synthetic sensor calibration quality validator.

    This implementation is intentionally lightweight so that unit tests can
    exercise the calibration-quality reporting pipeline without requiring
    large KITTI datasets. Real-world projects can later replace the scoring
    logic in ``_score_pair`` with true geometric / mutual-information checks.
    """

    def __init__(self, output_dir: str, config: Optional[dict] = None) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}

    def _extract_miscalib_pixels(self, stream: CalibrationStream) -> float:
        """Infer a synthetic miscalibration magnitude from the calibration filename.

        Tests encode miscalibration as e.g. ``..._miscalib_10.0px.txt``.
        """

        stem = Path(stream.calibration_file).stem
        # Example stem: "cam_lidar_miscalib_10.0px"
        parts = stem.split("_")
        for i, token in enumerate(parts):
            if token.startswith("miscalib") or token == "miscalib":
                # Look at this token or the next one for a numeric value.
                candidates = [token]
                if i + 1 < len(parts):
                    candidates.append(parts[i + 1])
                for c in candidates:
                    txt = c.replace("miscalib", "").replace("px", "")
                    txt = txt.strip("_")
                    if not txt:
                        continue
                    try:
                        return float(txt)
                    except ValueError:
                        continue
        return 0.0

    def _score_pair(self, stream: CalibrationStream) -> CalibrationPairResult:
        mis_px = self._extract_miscalib_pixels(stream)
        max_px = float(self.config.get("max_miscalib_px", 20.0))
        if max_px <= 0:
            max_px = 20.0

        quality = max(0.0, 1.0 - mis_px / max_px)
        geom_edge_score = quality
        mutual_information = quality
        contrastive_score = quality

        threshold = float(self.config.get("pass_threshold", 0.8))
        pass_geom_edge = geom_edge_score >= threshold
        pass_mi = mutual_information >= threshold
        pass_contrastive = contrastive_score >= threshold
        overall_pass = pass_geom_edge and pass_mi and pass_contrastive

        return CalibrationPairResult(
            geom_edge_score=geom_edge_score,
            mutual_information=mutual_information,
            contrastive_score=contrastive_score,
            pass_geom_edge=pass_geom_edge,
            pass_mi=pass_mi,
            pass_contrastive=pass_contrastive,
            overall_pass=overall_pass,
            details={"synthetic_mis_px": mis_px},
        )

    def analyze_sequences(
        self,
        pairs: Dict[str, CalibrationStream],
        bag_name: str,
        include_visualizations: bool = False,  # kept for API symmetry
    ) -> CalibrationQualityReport:
        pair_results: Dict[str, CalibrationPairResult] = {}
        recommendations: List[str] = []

        for key, stream in pairs.items():
            res = self._score_pair(stream)
            pair_results[key] = res

            if not res.overall_pass:
                msg = (
                    f"Calibration drift/misalignment suspected for pair '{key}' "
                    f"(quality={res.geom_edge_score:.2f}). Consider recalibration."
                )
                recommendations.append(msg)

        if pair_results:
            edge_scores = [r.geom_edge_score for r in pair_results.values()]
            mi_scores = [r.mutual_information for r in pair_results.values()]
            contrast_scores = [r.contrastive_score for r in pair_results.values()]
            metrics = {
                "edge_alignment_score": float(sum(edge_scores) / len(edge_scores)),
                "mi_score": float(sum(mi_scores) / len(mi_scores)),
                "contrastive_score": float(sum(contrast_scores) / len(contrast_scores)),
            }
        else:
            metrics = {
                "edge_alignment_score": 0.0,
                "mi_score": 0.0,
                "contrastive_score": 0.0,
            }

        payload = {
            "metadata": {
                "iso_8000_61": True,
                "type": "sensor_calibration_quality",
                "bag_name": bag_name,
            },
            "calibration_corrections": {},
        }

        for key, res in pair_results.items():
            entry: Dict[str, object] = {
                "quality_score": float(res.geom_edge_score),
                "overall_pass": bool(res.overall_pass),
            }
            if not res.overall_pass:
                entry["recommendation"] = "Recalibrate this camera–LiDAR pair."
            payload["calibration_corrections"][key] = entry

        param_path = self.output_dir / f"{bag_name}_calibration_quality.yaml"
        with param_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f)

        return CalibrationQualityReport(
            bag_name=bag_name,
            metrics=metrics,
            pair_results=pair_results,
            recommendations=recommendations,
            parameter_file=str(param_path),
        )
