"""
Feature 4: Dataset quality scoring.

Given per-topic timestamps and expected frequencies, this module
computes:
    - completeness_score
    - consistency_score
    - timeliness_score
    - overall_quality_score
"""

from __future__ import annotations

from typing import Dict, List

# --------- IMPORTS WITH FALLBACK (PACKAGE + STANDALONE) ---------
try:
    # Case 1: imported as part of the package roboqa_temporal
    from .metrics import (
        compute_completeness,
        compute_temporal_consistency,
        compute_timeliness,
    )
except ImportError:
    # Case 2: imported as a standalone module when tests modify sys.path
    # to point directly at src/roboqa_temporal
    from metrics import (
        compute_completeness,
        compute_temporal_consistency,
        compute_timeliness,
    )

# Define type aliases *locally* so we do NOT rely on metrics.py for them
SequenceData = Dict[str, List[float]]
FreqDict = Dict[str, float]


class QualityAnalyzer:
    """
    Core of Feature 4: turn raw timestamps into a single quality score.
    """

    def __init__(
        self,
        expected_freqs: FreqDict,
        weights: Dict[str, float] | None = None,
    ) -> None:
        """
        Args:
            expected_freqs: mapping topic -> expected frequency (Hz)
            weights: optional weights for combining:
                completeness, consistency, timeliness.
        """
        self.expected_freqs = expected_freqs
        self.weights = weights or {
            "completeness": 0.4,
            "consistency": 0.4,
            "timeliness": 0.2,
        }

    def analyze_sequence(self, sequence_data: SequenceData) -> Dict[str, float]:
        """
        Compute per-dimension metrics and an overall quality score.

        Returns:
            dict with keys:
                - completeness_score
                - consistency_score
                - timeliness_score
                - overall_quality_score
                - plus topic-level metrics
        """
        metrics: Dict[str, float] = {}

        compl = compute_completeness(sequence_data, self.expected_freqs)
        cons = compute_temporal_consistency(sequence_data, self.expected_freqs)
        time = compute_timeliness(sequence_data)

        metrics.update(compl)
        metrics.update(cons)
        metrics.update(time)

        c = metrics.get("completeness_score", 0.0)
        s = metrics.get("consistency_score", 0.0)
        t = metrics.get("timeliness_score", 0.0)

        overall = (
            self.weights["completeness"] * c
            + self.weights["consistency"] * s
            + self.weights["timeliness"] * t
        )

        metrics["overall_quality_score"] = float(overall)
        return metrics

