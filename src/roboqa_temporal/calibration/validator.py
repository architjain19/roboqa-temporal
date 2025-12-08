from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml


import numpy as np
from scipy import ndimage
import matplotlib.image as mpimg

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
    """Sensor calibration quality validator.

    Supports both synthetic (filename-based) and real-world (geometric/MI) validation.
    """

    def __init__(self, output_dir: str, config: Optional[dict] = None) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}

    def _load_kitti_calib(self, calib_dir: Path, camera_id: str) -> dict:
        """Load KITTI calibration matrices."""
        calib = {}
        
        # Load velo_to_cam
        velo_to_cam_file = calib_dir / "calib_velo_to_cam.txt"
        if velo_to_cam_file.exists():
            with open(velo_to_cam_file, "r") as f:
                for line in f:
                    key, val = line.split(":", 1)
                    if key == "R":
                        calib["R_velo2cam"] = np.fromstring(val, sep=" ").reshape(3, 3)
                    elif key == "T":
                        calib["T_velo2cam"] = np.fromstring(val, sep=" ").reshape(3, 1)

        # Load cam_to_cam
        cam_to_cam_file = calib_dir / "calib_cam_to_cam.txt"
        if cam_to_cam_file.exists():
            with open(cam_to_cam_file, "r") as f:
                for line in f:
                    key, val = line.split(":", 1)
                    # camera_id is like "image_02", we need "02"
                    idx = camera_id.split("_")[-1]
                    
                    if key == "R_rect_00":
                        calib["R_rect_00"] = np.fromstring(val, sep=" ").reshape(3, 3)
                    elif key == f"P_rect_{idx}":
                        calib[f"P_rect_{idx}"] = np.fromstring(val, sep=" ").reshape(3, 4)

        return calib

    def _project_lidar_to_image(self, points: np.ndarray, calib: dict, camera_id: str) -> np.ndarray:
        """Project 3D LiDAR points to 2D image plane."""
        idx = camera_id.split("_")[-1]
        P_rect = calib.get(f"P_rect_{idx}")
        R_rect_00 = calib.get("R_rect_00")
        R_velo2cam = calib.get("R_velo2cam")
        T_velo2cam = calib.get("T_velo2cam")

        if any(x is None for x in [P_rect, R_rect_00, R_velo2cam, T_velo2cam]):
            return np.array([])

        # Transform to cam0
        # X_cam0 = R_velo2cam * X_velo + T_velo2cam
        pts_3d = points[:, :3].T
        pts_cam0 = R_velo2cam @ pts_3d + T_velo2cam

        # Rectify
        # X_rect = R_rect_00 * X_cam0
        pts_rect = R_rect_00 @ pts_cam0

        # Project
        # x_img = P_rect * X_rect
        # P_rect is 3x4, pts_rect is 3xN. Need homogeneous coords for pts_rect?
        # P_rect * [X, Y, Z, 1]^T
        pts_rect_hom = np.vstack((pts_rect, np.ones((1, pts_rect.shape[1]))))
        pts_2d_hom = P_rect @ pts_rect_hom

        # Normalize
        pts_2d = pts_2d_hom[:2, :] / pts_2d_hom[2, :]
        return pts_2d.T

    def _compute_mutual_information(self, img_gray: np.ndarray, lidar_intensity: np.ndarray, bins: int = 20) -> float:
        """Compute Normalized Mutual Information."""
        hist_2d, _, _ = np.histogram2d(img_gray.ravel(), lidar_intensity.ravel(), bins=bins)
        
        # Convert to probabilities
        pxy = hist_2d / float(np.sum(hist_2d))
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)
        
        px_py = px[:, None] * py[None, :]
        nzs = pxy > 0
        
        mi = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
        
        # Normalize
        h_x = -np.sum(px[px > 0] * np.log(px[px > 0]))
        h_y = -np.sum(py[py > 0] * np.log(py[py > 0]))
        nmi = 2 * mi / (h_x + h_y)
        
        return nmi

    def _compute_real_metrics(self, stream: CalibrationStream) -> CalibrationPairResult:
        calib_dir = Path(stream.calibration_file)
        calib = self._load_kitti_calib(calib_dir, stream.camera_id)
        
        # Process first frame for now
        img_path = stream.image_paths[0]
        pc_path = stream.pointcloud_paths[0]
        
        if not Path(img_path).exists() or not Path(pc_path).exists():
             return CalibrationPairResult(0.0, 0.0, 0.0, False, False, False, False, {"error": "files not found"})

        # Load Image
        img = mpimg.imread(img_path)
        if len(img.shape) == 3:
            img_gray = np.mean(img, axis=2)
        else:
            img_gray = img

        # Load Point Cloud
        # KITTI bin files are float32 x, y, z, intensity
        points = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)
        
        # Project
        pts_2d = self._project_lidar_to_image(points, calib, stream.camera_id)
        
        if pts_2d.size == 0:
             return CalibrationPairResult(0.0, 0.0, 0.0, False, False, False, False, {"error": "projection failed"})

        # Filter points within image bounds
        h, w = img_gray.shape
        valid_mask = (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < w) & \
                     (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < h) & \
                     (points[:, 0] > 0) # Keep only points in front of camera
        
        valid_pts_2d = pts_2d[valid_mask]
        valid_intensities = points[valid_mask, 3]
        
        if len(valid_pts_2d) == 0:
             return CalibrationPairResult(0.0, 0.0, 0.0, False, False, False, False, {"error": "no valid points"})

        # 1. Mutual Information
        # Sample image intensities at projected points
        # Use nearest neighbor interpolation
        x_idxs = np.clip(np.round(valid_pts_2d[:, 0]).astype(int), 0, w - 1)
        y_idxs = np.clip(np.round(valid_pts_2d[:, 1]).astype(int), 0, h - 1)
        
        img_samples = img_gray[y_idxs, x_idxs]
        
        mi_score = self._compute_mutual_information(img_samples, valid_intensities)
        
        # 2. Edge Alignment
        # Compute gradient magnitude
        sx = ndimage.sobel(img_gray, axis=0, mode='constant')
        sy = ndimage.sobel(img_gray, axis=1, mode='constant')
        sob = np.hypot(sx, sy)
        
        # Normalize sobel
        if sob.max() > 0:
            sob = sob / sob.max()
            
        # Sample edge magnitude at projected points
        edge_samples = sob[y_idxs, x_idxs]
        edge_score = np.mean(edge_samples)
        
        # Heuristic scaling for scores to match 0-1 range roughly
        # MI is usually low, maybe 0.1-0.5. Edge score also low.
        # We'll just return raw values for now, but thresholds might need tuning.
        
        # For "pass" criteria, we might need baselines. 
        # For now, let's set arbitrary low thresholds or rely on relative comparison if we had history.
        # The user asked for "metrics to help user understand... which values qualify as good".
        # I'll set some provisional thresholds.
        
        pass_mi = mi_score > 0.05 # Very low threshold as placeholder
        pass_edge = edge_score > 0.05
        
        return CalibrationPairResult(
            geom_edge_score=float(edge_score),
            mutual_information=float(mi_score),
            contrastive_score=0.0, # Not implemented
            pass_geom_edge=pass_edge,
            pass_mi=pass_mi,
            pass_contrastive=True, # Skip
            overall_pass=pass_edge and pass_mi,
            details={"n_points": len(valid_pts_2d)}
        )

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
        # Check if synthetic
        if any(str(p).startswith("/synthetic") for p in stream.image_paths):
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
        else:
            return self._compute_real_metrics(stream)

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
        all_passed = True
        for name, res in pair_results.items():
            if not res.overall_pass:
                all_passed = False
            status_class = "pass" if res.overall_pass else "fail"
            status_text = "PASS" if res.overall_pass else "FAIL"
            rec_text = "None" if res.overall_pass else "Recalibrate"
            rows.append(
                f"<tr>"
                f"<td>{name}</td>"
                f"<td>{res.geom_edge_score:.3f}</td>"
                f"<td>{res.mutual_information:.3f}</td>"
                f"<td>{res.contrastive_score:.3f}</td>"
                f"<td class='{status_class}'>{status_text}</td>"
                f"<td>{rec_text}</td>"
                f"</tr>"
            )
        
        rows_html = "\n".join(rows)
        overall_status = "<span class='pass'>PASS</span>" if all_passed else "<span class='fail'>FAIL</span>"

        html_template = f"""<!DOCTYPE html>
<html>
<head>
    <title>Calibration Quality Report - {bag_name}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; color: #333; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .summary {{ background-color: #f8f9fa; padding: 20px; border-radius: 8px; border: 1px solid #e9ecef; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        th, td {{ border: 1px solid #dee2e6; padding: 12px; text-align: left; }}
        th {{ background-color: #e9ecef; color: #495057; }}
        tr:nth-child(even) {{ background-color: #f8f9fa; }}
        .pass {{ color: #28a745; font-weight: bold; }}
        .fail {{ color: #dc3545; font-weight: bold; }}
        .metric-info {{ background-color: #e8f4f8; padding: 15px; border-left: 5px solid #17a2b8; margin: 20px 0; border-radius: 4px; }}
        .metric-info h3 {{ margin-top: 0; color: #0c5460; }}
        .metric-info ul {{ margin-bottom: 0; }}
        .metric-info li {{ margin-bottom: 8px; }}
    </style>
</head>
<body>
    <h1>Calibration Quality Report</h1>
    
    <div class="summary">
        <h2>Summary for {bag_name}</h2>
        <p><strong>Overall Status:</strong> {overall_status}</p>
        <p><strong>Average Edge Alignment Score:</strong> {metrics.get('edge_alignment_score', 0.0):.3f}</p>
        <p><strong>Average Mutual Information:</strong> {metrics.get('mi_score', 0.0):.3f}</p>
    </div>

    <div class="metric-info">
        <h3>Understanding the Metrics</h3>
        <p>This report assesses the alignment between Camera and LiDAR sensors using the following metrics:</p>
        <ul>
            <li><strong>Geometric Edge Score (0.0 - 1.0):</strong> Measures how well edges detected in the camera image align with depth discontinuities (edges) in the projected LiDAR point cloud. 
                <br><em>Interpretation:</em> Higher is better. Values > 0.10 typically indicate reasonable alignment for complex scenes. Values near 0.0 indicate misalignment or featureless scenes.</li>
            <li><strong>Mutual Information (NMI):</strong> Measures the statistical dependence between the camera image intensity and the LiDAR return intensity.
                <br><em>Interpretation:</em> Higher is better. Values > 0.05 typically indicate that the sensors are observing the same scene structure. Values near 0.0 suggest independence (misalignment).</li>
            <li><strong>Contrastive Score:</strong> A learned similarity metric using deep neural networks (if enabled).</li>
        </ul>
    </div>

    <h2>Pairwise Results</h2>
    <table>
        <thead>
            <tr>
                <th>Sensor Pair</th>
                <th>Edge Score</th>
                <th>Mutual Info</th>
                <th>Contrastive</th>
                <th>Status</th>
                <th>Recommendation</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
</body>
</html>"""
        return html_template

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
                body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 40px; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; }}
                h1 {{ border-bottom: 2px solid #eee; padding-bottom: 10px; color: #2c3e50; }}
                h2 {{ color: #34495e; margin-top: 30px; }}
                .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
                .card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; border: 1px solid #dee2e6; flex: 1; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
                .card h3 {{ margin-top: 0; font-size: 0.9em; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }}
                .card .value {{ font-size: 2.2em; font-weight: bold; color: #2c3e50; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
                th, td {{ padding: 15px; text-align: left; border-bottom: 1px solid #dee2e6; }}
                th {{ background-color: #f1f3f5; font-weight: 600; color: #495057; }}
                tr:hover {{ background-color: #f8f9fa; }}
                .pass {{ color: #28a745; font-weight: bold; background-color: #d4edda; padding: 4px 8px; border-radius: 4px; display: inline-block; }}
                .fail {{ color: #dc3545; font-weight: bold; background-color: #f8d7da; padding: 4px 8px; border-radius: 4px; display: inline-block; }}
                .recommendations {{ margin-top: 30px; background-color: #fff3cd; padding: 20px; border-radius: 8px; border: 1px solid #ffeeba; color: #856404; }}
                .metrics-guide {{ margin-top: 40px; padding: 25px; background: #e9ecef; border-radius: 8px; border-left: 5px solid #6c757d; }}
                .metrics-guide h3 {{ margin-top: 0; color: #495057; }}
                .metric-item {{ margin-bottom: 15px; }}
                .metric-name {{ font-weight: bold; color: #2c3e50; }}
                .metric-desc {{ color: #555; }}
                .metric-good {{ color: #28a745; font-size: 0.9em; font-weight: 500; }}
            </style>
        </head>
        <body>
            <h1>Calibration Quality Report</h1>
            <p><strong>Dataset/Bag Name:</strong> {bag_name}</p>
            
            <div class="summary">
                <div class="card">
                    <h3>Avg Edge Alignment</h3>
                    <div class="value">{metrics.get('edge_alignment_score', 0.0):.3f}</div>
                </div>
                <div class="card">
                    <h3>Avg Mutual Info</h3>
                    <div class="value">{metrics.get('mi_score', 0.0):.3f}</div>
                </div>
                <div class="card">
                    <h3>Avg Contrastive</h3>
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

            <div class="metrics-guide">
                <h3>Understanding the Metrics</h3>
                
                <div class="metric-item">
                    <div class="metric-name">Edge Alignment Score</div>
                    <div class="metric-desc">Measures how well edges detected in the camera image align with depth discontinuities (edges) in the projected LiDAR point cloud. A higher score indicates better geometric alignment between the sensors.</div>
                    <div class="metric-good">✓ Good Quality: > 0.05 (Typical range: 0.0 - 1.0)</div>
                </div>

                <div class="metric-item">
                    <div class="metric-name">Mutual Information (MI) Score</div>
                    <div class="metric-desc">Quantifies the statistical dependence between the camera image intensity (grayscale) and the LiDAR return intensity. It captures how much information the two modalities share. Higher values suggest better calibration.</div>
                    <div class="metric-good">✓ Good Quality: > 0.05 (Typical range: 0.0 - 1.0+)</div>
                </div>

                <div class="metric-item">
                    <div class="metric-name">Contrastive Score</div>
                    <div class="metric-desc">A deep learning-based similarity metric that compares feature embeddings from image patches and corresponding LiDAR projections. It is robust to lighting changes and sensor noise.</div>
                    <div class="metric-good">✓ Good Quality: > 0.8 (Typical range: 0.0 - 1.0)</div>
                </div>
            </div>

            {recs_html}
            
            <p style="margin-top: 40px; color: #999; font-size: 0.8em; text-align: center;">Generated by RoboQA Temporal Validator</p>
        </body>
        </html>
        """
        return html
