"""
################################################################

File: roboqa_temporal/health_reporting/curation.py
Created: 2025-12-08
Created by: RoboQA-Temporal Authors
Last Modified: 2025-12-08
Last Modified by: RoboQA-Temporal Authors

#################################################################

Copyright: RoboQA-Temporal Authors
License: MIT License

################################################################

Sequence curation recommendations module.

Generates actionable recommendations for dataset curation based on
quality metrics and threshold-based rules.

################################################################

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple

import pandas as pd


@dataclass
class CurationRecommendation:
    """A single curation recommendation."""
    
    sequence: str
    severity: str  # "critical", "high", "medium", "low"
    category: str  # "temporal", "completeness", "quality", "anomaly"
    message: str
    metric_value: float
    threshold: float
    action: str  # "exclude", "review", "monitor"


def generate_curation_recommendations(
    df_per_sensor: pd.DataFrame,
    df_per_sequence: pd.DataFrame,
    temporal_threshold: float = 0.6,
    completeness_threshold: float = 0.6,
    quality_threshold: float = 0.5,
) -> List[CurationRecommendation]:
    """
    Generate sequence curation recommendations based on metric thresholds.
    
    Args:
        df_per_sensor: DataFrame with per-sensor metrics
        df_per_sequence: DataFrame with per-sequence aggregated metrics
        temporal_threshold: Minimum acceptable timeliness score
        completeness_threshold: Minimum acceptable completeness score
        quality_threshold: Minimum acceptable overall quality score
        
    Returns:
        List of CurationRecommendation objects
    """
    recommendations: List[CurationRecommendation] = []
    
    for _, row in df_per_sequence.iterrows():
        sequence_name = row["sequence"]
        
        # Checking overall quality
        overall_score = row["overall_quality_score"]
        if overall_score < quality_threshold:
            severity = "critical" if overall_score < 0.3 else "high"
            action = "exclude" if overall_score < 0.3 else "review"
            rec = CurationRecommendation(
                sequence=sequence_name,
                severity=severity,
                category="quality",
                message=f"Overall quality score is low ({overall_score:.1%} < {quality_threshold:.1%})",
                metric_value=overall_score,
                threshold=quality_threshold,
                action=action,
            )
            recommendations.append(rec)
        
        # Checking timeliness
        timeliness_score = row["dim_timeliness"]
        if timeliness_score < temporal_threshold:
            severity = "high" if timeliness_score < 0.4 else "medium"
            rec = CurationRecommendation(
                sequence=sequence_name,
                severity=severity,
                category="temporal",
                message=f"Temporal irregularities detected (timeliness {timeliness_score:.1%} < {temporal_threshold:.1%})",
                metric_value=timeliness_score,
                threshold=temporal_threshold,
                action="review",
            )
            recommendations.append(rec)
        
        # Checking completeness
        completeness_score = row["dim_completeness"]
        if completeness_score < completeness_threshold:
            severity = "high" if completeness_score < 0.4 else "medium"
            rec = CurationRecommendation(
                sequence=sequence_name,
                severity=severity,
                category="completeness",
                message=f"Missing or dropped frames detected (completeness {completeness_score:.1%} < {completeness_threshold:.1%})",
                metric_value=completeness_score,
                threshold=completeness_threshold,
                action="review",
            )
            recommendations.append(rec)
        
        # Checking anomaly score
        anomaly_score = row["anomaly_score"]
        if anomaly_score < 0.5:
            severity = "high" if anomaly_score < 0.3 else "medium"
            rec = CurationRecommendation(
                sequence=sequence_name,
                severity=severity,
                category="anomaly",
                message=f"Significant timing anomalies detected (anomaly score {anomaly_score:.1%} is low)",
                metric_value=anomaly_score,
                threshold=0.5,
                action="review",
            )
            recommendations.append(rec)
    
    # Sorting by severity
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    recommendations.sort(key=lambda r: (severity_order.get(r.severity, 999), r.sequence))
    
    return recommendations


def generate_curation_report(
    recommendations: List[CurationRecommendation],
    output_path: str,
) -> str:
    """
    Generate a human-readable curation recommendations report.
    
    Args:
        recommendations: List of CurationRecommendation objects
        output_path: Path to save report
        
    Returns:
        Path to the generated report
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("SEQUENCE CURATION RECOMMENDATIONS\n")
        f.write("=" * 100 + "\n\n")
        
        if not recommendations:
            f.write("No curation issues found. All sequences meet quality thresholds.\n\n")
        else:
            # Summary statistics
            severity_counts = {}
            for rec in recommendations:
                severity_counts[rec.severity] = severity_counts.get(rec.severity, 0) + 1
            
            f.write("SUMMARY\n")
            f.write("-" * 100 + "\n")
            f.write(f"Total Issues: {len(recommendations)}\n")
            for severity in ["critical", "high", "medium", "low"]:
                if severity in severity_counts:
                    f.write(f"  {severity.upper():8s}: {severity_counts[severity]}\n")
            f.write("\n")
            
            # By action
            action_counts = {}
            for rec in recommendations:
                action_counts[rec.action] = action_counts.get(rec.action, 0) + 1
            
            f.write("RECOMMENDED ACTIONS\n")
            f.write("-" * 100 + "\n")
            for action in ["exclude", "review", "monitor"]:
                if action in action_counts:
                    f.write(f"  {action.upper():8s}: {action_counts[action]} sequences\n")
            f.write("\n")
            
            # Detailed recommendations
            f.write("DETAILED RECOMMENDATIONS\n")
            f.write("-" * 100 + "\n\n")
            
            current_sequence = None
            for rec in recommendations:
                if rec.sequence != current_sequence:
                    if current_sequence is not None:
                        f.write("\n")
                    current_sequence = rec.sequence
                    f.write(f"SEQUENCE: {rec.sequence}\n")
                    f.write("  " + "-" * 96 + "\n")
                
                severity_indicator = "!" * (4 - ["critical", "high", "medium", "low"].index(rec.severity))
                f.write(f"  [{severity_indicator}] {rec.severity.upper():8s} - {rec.category.upper():12s}\n")
                f.write(f"      Message:  {rec.message}\n")
                f.write(f"      Action:   {rec.action.upper()}\n")
                f.write(f"      Value:    {rec.metric_value:.3f} (threshold: {rec.threshold:.3f})\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 100 + "\n")
    
    print(f"[INFO] Saved curation recommendations: {output_path}")
    return output_path


def generate_curation_json(
    recommendations: List[CurationRecommendation],
    output_path: str,
) -> str:
    """
    Generate curation recommendations in JSON format.
    
    Args:
        recommendations: List of CurationRecommendation objects
        output_path: Path to save JSON
        
    Returns:
        Path to the generated JSON file
    """
    import json
    
    rec_dicts = []
    for rec in recommendations:
        rec_dicts.append({
            "sequence": rec.sequence,
            "severity": rec.severity,
            "category": rec.category,
            "message": rec.message,
            "metric_value": float(rec.metric_value),
            "threshold": float(rec.threshold),
            "action": rec.action,
        })
    
    output = {
        "feature": "sequence_curation",
        "version": "1.0.0",
        "total_recommendations": len(rec_dicts),
        "recommendations": rec_dicts,
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    
    print(f"[INFO] Saved curation recommendations JSON: {output_path}")
    return output_path


def get_sequences_to_exclude(
    recommendations: List[CurationRecommendation],
) -> List[str]:
    """
    Get list of sequences recommended for exclusion.
    
    Args:
        recommendations: List of CurationRecommendation objects
        
    Returns:
        List of sequence names to exclude
    """
    exclude_sequences = set()
    for rec in recommendations:
        if rec.action == "exclude":
            exclude_sequences.add(rec.sequence)
    
    return sorted(list(exclude_sequences))


def get_sequences_for_review(
    recommendations: List[CurationRecommendation],
) -> List[str]:
    """
    Get list of sequences recommended for review.
    
    Args:
        recommendations: List of CurationRecommendation objects
        
    Returns:
        List of sequence names to review
    """
    review_sequences = set()
    for rec in recommendations:
        if rec.action == "review":
            review_sequences.add(rec.sequence)
    
    return sorted(list(review_sequences))


def get_issues_by_severity(
    recommendations: List[CurationRecommendation],
) -> Dict[str, List[CurationRecommendation]]:
    """
    Group recommendations by severity level.
    
    Args:
        recommendations: List of CurationRecommendation objects
        
    Returns:
        Dictionary mapping severity levels to lists of recommendations
    """
    by_severity = {
        "critical": [],
        "high": [],
        "medium": [],
        "low": [],
    }
    
    for rec in recommendations:
        if rec.severity in by_severity:
            by_severity[rec.severity].append(rec)
    
    return by_severity


def get_issues_by_category(
    recommendations: List[CurationRecommendation],
) -> Dict[str, List[CurationRecommendation]]:
    """
    Group recommendations by category.
    
    Args:
        recommendations: List of CurationRecommendation objects
        
    Returns:
        Dictionary mapping categories to lists of recommendations
    """
    by_category = {}
    
    for rec in recommendations:
        if rec.category not in by_category:
            by_category[rec.category] = []
        by_category[rec.category].append(rec)
    
    return by_category
