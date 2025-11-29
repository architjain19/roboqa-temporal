from __future__ import annotations

from typing import List, Dict

import numpy as np


def compute_percentile_vs_reference(
    reference_scores: List[float],
    candidate_score: float,
) -> Dict[str, float]:
    """
    Feature 4: Cross-benchmarking helper.

    Given:
        - reference_scores: quality scores from a reference dataset
          (e.g., KITTI / nuScenes).
        - candidate_score: overall_quality_score for one sequence / dataset.

    Returns:
        {
            "percentile": float in [0, 100],
            "ref_mean": float,
            "ref_std": float,
        }
    """
    if len(reference_scores) == 0:
        raise ValueError("reference_scores must not be empty")

    ref = np.asarray(reference_scores, dtype=float)
    percentile = (ref < candidate_score).sum() / len(ref) * 100.0

    return {
        "percentile": float(percentile),
        "ref_mean": float(ref.mean()),
        "ref_std": float(ref.std(ddof=0)),
    }
