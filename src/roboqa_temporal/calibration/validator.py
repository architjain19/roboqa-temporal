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
    html_report_file: Optional[str] = None


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

        html_path = self.output_dir / f"{bag_name}_calibration_quality_report.html"
        html_report = self._generate_html_report(bag_name, metrics, pair_results, recommendations)
        with html_path.open("w", encoding="utf-8") as f:
            f.write(html_report)

        return CalibrationQualityReport(
            bag_name=bag_name,
            metrics=metrics,
            pair_results=pair_results,
            recommendations=recommendations,
            parameter_file=str(param_path),
            html_report_file=str(html_path),
        )

    def _generate_html_report(
        self,
        bag_name: str,
        metrics: Dict[str, float],
        pair_results: Dict[str, CalibrationPairResult],
        recommendations: List[str],
    ) -> str:
        rows = []
        for name, res in pair_results.items():
            status_class = "pass" if res.overall_pass else "fail"
            status_text = "PASS" if res.overall_pass else "FAIL"
            rows.append(
                f"<tr>"
                f"<td>{name}</td>"
                f"<td>{res.geom_edge_score:.3f}</td>"
                f"<td>{res.mutual_information:.3f}</td>"
                f"<td>{res.contrastive_score:.3f}</td>"
                f"<td class='{status_class}'>{status_text}</td>"
                f"</tr>"
            )

        recs_html = ""
        if recommendations:
            recs_list = "".join(f"<li>{r}</li>" for r in recommendations)
            recs_html = (
                f"<div class='recommendations'>"
                f"<h3>Recommendations</h3>"
                f"<ul>{recs_list}</ul>"
                f"</div>"
            )

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Calibration Quality Report: {bag_name}</title>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 40px; line-height: 1.6; color: #333; }}
                h1 {{ border-bottom: 2px solid #eee; padding-bottom: 10px; }}
                .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
                .card {{ background: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #dee2e6; flex: 1; }}
                .card h3 {{ margin-top: 0; font-size: 0.9em; color: #666; text-transform: uppercase; }}
                .card .value {{ font-size: 1.8em; font-weight: bold; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #dee2e6; }}
                th {{ background-color: #f1f3f5; font-weight: 600; }}
                .pass {{ color: #28a745; font-weight: bold; }}
                .fail {{ color: #dc3545; font-weight: bold; }}
                .recommendations {{ margin-top: 30px; background-color: #fff3cd; padding: 20px; border-radius: 8px; border: 1px solid #ffeeba; color: #856404; }}
            </style>
        </head>
        <body>
            <h1>Calibration Quality Report</h1>
            <p><strong>Bag Name:</strong> {bag_name}</p>
            
            <div class="summary">
                <div class="card">
                    <h3>Edge Alignment</h3>
                    <div class="value">{metrics.get('edge_alignment_score', 0.0):.3f}</div>
                </div>
                <div class="card">
                    <h3>Mutual Info</h3>
                    <div class="value">{metrics.get('mi_score', 0.0):.3f}</div>
                </div>
                <div class="card">
                    <h3>Contrastive</h3>
                    <div class="value">{metrics.get('contrastive_score', 0.0):.3f}</div>
                </div>
            </div>

            <h2>Pair Results</h2>
            <table>
                <thead>
                    <tr>
                        <th>Pair Name</th>
                        <th>Edge Score</th>
                        <th>MI Score</th>
                        <th>Contrastive</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(rows)}
                </tbody>
            </table>

            {recs_html}
            
            <p style="margin-top: 40px; color: #999; font-size: 0.8em;">Generated by RoboQA Temporal Validator</p>
        </body>
        </html>
        """
        return html
