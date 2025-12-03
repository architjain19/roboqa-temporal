from typing import List, Optional

import numpy as np
import pandas as pd


def add_temporal_score(
    df: pd.DataFrame,
    error_cols: Optional[List[str]] = None,
    positive_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Add a 'temporal_score' column in [0, 1], where higher is better.

    Supports both:
      - TXT fallback metrics:
          global_mean_sync_error, matched_fraction
      - Feature 1 (TemporalSyncValidator) metrics:
          temporal_offset_score, avg_drift_rate_ms_per_s,
          max_predicted_drift_ms, iso_8000_61_pass
    """

    if error_cols is None:
        # larger = worse
        error_cols = [
            "global_mean_sync_error",    # TXT fallback
            "avg_drift_rate_ms_per_s",   # Feature 1
            "max_predicted_drift_ms",    # Feature 1
        ]

    if positive_cols is None:
        # larger = better
        positive_cols = [
            "matched_fraction",          # TXT fallback
            "temporal_offset_score",     # Feature 1
            "iso_8000_61_pass",          # Feature 1 (0 or 1)
        ]

    norm_cols: List[str] = []

    def _normalize_series(v: pd.Series, flip: bool) -> pd.Series:
        v = v.astype(float)
        v_min = v.min()
        v_max = v.max()
        if np.isclose(v_max, v_min):
            # no variation -> neutral 0.5
            return pd.Series(0.5, index=v.index)
        if flip:
            # larger = worse -> lower score
            return (v_max - v) / (v_max - v_min)
        else:
            # larger = better -> higher score
            return (v - v_min) / (v_max - v_min)

    # Error-type metrics
    for col in error_cols:
        if col not in df.columns:
            continue
        norm_name = f"norm_{col}"
        df[norm_name] = _normalize_series(df[col], flip=True)
        norm_cols.append(norm_name)

    # Good metrics
    for col in positive_cols:
        if col not in df.columns:
            continue
        norm_name = f"norm_{col}"
        df[norm_name] = _normalize_series(df[col], flip=False)
        norm_cols.append(norm_name)

    if not norm_cols:
        df["temporal_score"] = 1.0
        return df

    df["temporal_score"] = df[norm_cols].mean(axis=1)
    return df


def add_anomaly_score(
    df: pd.DataFrame,
    error_cols: Optional[List[str]] = None,
    positive_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Add an 'anomaly_score' column in [0, 1], where higher is better.

    Works with:
      - Simple Feature 4 fallback metric:
          mean_anomaly_ratio  (0..1, larger = worse)
      - Feature 2 metrics if exported, e.g.:
          overall_health_score (0..1, larger = better)
          density_avg_density, ghost_avg_ghost_ratio, etc.
    """

    if error_cols is None:
        error_cols = [
            "mean_anomaly_ratio",        # Fallback
            "avg_ghost_ratio",          # Example from GhostPointDetector
            "density_cv",               # Coefficient of variation (if present)
        ]

    if positive_cols is None:
        positive_cols = [
            "overall_health_score",      # From AnomalyDetector.health_metrics
        ]

    norm_cols: List[str] = []

    def _normalize_series(v: pd.Series, flip: bool) -> pd.Series:
        v = v.astype(float)
        v_min = v.min()
        v_max = v.max()
        if np.isclose(v_max, v_min):
            return pd.Series(0.5, index=v.index)
        if flip:
            return (v_max - v) / (v_max - v_min)
        else:
            return (v - v_min) / (v_max - v_min)

    for col in error_cols:
        if col not in df.columns:
            continue
        norm_name = f"norm_{col}"
        df[norm_name] = _normalize_series(df[col], flip=True)
        norm_cols.append(norm_name)

    for col in positive_cols:
        if col not in df.columns:
            continue
        norm_name = f"norm_{col}"
        df[norm_name] = _normalize_series(df[col], flip=False)
        norm_cols.append(norm_name)

    if not norm_cols:
        df["anomaly_score"] = 1.0
        return df

    df["anomaly_score"] = df[norm_cols].mean(axis=1)
    return df


def add_multimodal_health_score(
    df: pd.DataFrame,
    temporal_col: str = "temporal_score",
    anomaly_col: str = "anomaly_score",
    alpha_temporal: float = 0.5,
) -> pd.DataFrame:
    """
    Combine temporal + anomaly scores into a single multi-modal score.

    multi_modal_score = alpha * temporal + (1 - alpha) * anomaly
    """

    if temporal_col not in df.columns:
        df[temporal_col] = 1.0
    if anomaly_col not in df.columns:
        df[anomaly_col] = 1.0

    alpha = float(alpha_temporal)
    alpha = min(max(alpha, 0.0), 1.0)
    df["multimodal_health_score"] = (
        alpha * df[temporal_col].astype(float)
        + (1.0 - alpha) * df[anomaly_col].astype(float)
    )
    return df


def add_health_tiers(
    df: pd.DataFrame,
    score_col: str = "multimodal_health_score",
    tier_col: str = "health_tier",
) -> pd.DataFrame:
    """
    Map multi-modal health score into discrete tiers.

    Example:
      score >= 0.85 -> 'excellent'
      score >= 0.70 -> 'good'
      score >= 0.50 -> 'fair'
      else          -> 'poor'
    """
    if score_col not in df.columns:
        df[score_col] = 1.0

    def _tier(x: float) -> str:
        if x >= 0.85:
            return "excellent"
        if x >= 0.70:
            return "good"
        if x >= 0.50:
            return "fair"
        return "poor"

    df[tier_col] = df[score_col].astype(float).map(_tier)
    return df
