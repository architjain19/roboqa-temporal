import math
import os
import sys

# Make src/roboqa_temporal importable without ROS2
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PKG_DIR = os.path.join(ROOT_DIR, "src", "roboqa_temporal")
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

from quality_analyzer import QualityAnalyzer
from benchmarking import compute_percentile_vs_reference


# ---------- 1. SMOKE TEST ----------

def test_feature4_smoke_quality_and_benchmarking():
    """
    author: sayali
    reviewer: Archit,Dharineesh,Xinxin
    category: smoke test

    goal:
        End-to-end check: compute an overall quality score and then
        compare it to a reference dataset using the benchmarking helper.
    """
    sequence = {
        "/lidar": [0.0, 0.1, 0.2, 0.3],
        "/camera": [0.0, 0.1, 0.2, 0.3],
    }
    expected_freqs = {
        "/lidar": 10.0,
        "/camera": 10.0,
    }

    qa = QualityAnalyzer(expected_freqs)
    metrics = qa.analyze_sequence(sequence)

    assert "overall_quality_score" in metrics
    overall = metrics["overall_quality_score"]
    assert 0.0 <= overall <= 100.0

    ref_scores = [60.0, 70.0, 80.0, 90.0]
    bench = compute_percentile_vs_reference(ref_scores, overall)

    assert "percentile" in bench
    assert 0.0 <= bench["percentile"] <= 100.0


# ---------- 2. ONE-SHOT TEST ----------

def test_feature4_one_shot_known_percentile():
    """
    author: sayali
    reviewer: Archit,Dharineesh,Xinxin
    category: one-shot test

    goal:
        For a simple reference distribution, verify the percentile when
        we know the exact ordering.
    """
    reference_scores = [10.0, 20.0, 30.0, 40.0, 50.0]
    candidate_score = 30.0  # 2 out of 5 scores are below → 40%

    out = compute_percentile_vs_reference(reference_scores, candidate_score)
    assert math.isclose(out["percentile"], 40.0, rel_tol=1e-12)


# ---------- 3. EDGE TEST ----------

def test_feature4_edge_empty_reference():
    """
    author: sayali
    reviewer: Archit,Dharineesh,Xinxin
    category: edge test

    goal:
        If the reference dataset library is empty, we should not silently
        compute a percentile. Instead, we raise a ValueError.
    """
    try:
        compute_percentile_vs_reference([], 75.0)
    except ValueError as e:
        assert "reference_scores must not be empty" in str(e)
    else:
        assert False, "Expected ValueError for empty reference_scores"


# ---------- 4. PATTERN TEST ----------

def test_feature4_pattern_monotonic_benchmarking():
    """
    author: sayali
    reviewer: Archit,Dharineesh,Xinxin
    category: pattern test

    pattern:
        For a fixed reference dataset, as the candidate quality score
        increases, the benchmarking percentile should not decrease.

    test idea:
        Fix reference_scores and evaluate percentiles for increasing
        candidate_scores; assert monotonic non-decreasing behavior.
    """
    reference_scores = [20.0, 40.0, 60.0, 80.0]
    candidates = [10.0, 30.0, 50.0, 70.0, 90.0]

    percentiles = [
        compute_percentile_vs_reference(reference_scores, s)
        ["percentile"]
        for s in candidates
    ]

    for i in range(1, len(percentiles)):
        assert percentiles[i] >= percentiles[i - 1]
