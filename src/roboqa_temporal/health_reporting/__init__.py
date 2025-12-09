"""
################################################################

File: roboqa_temporal/health_reporting/__init__.py
Created: 2025-12-08
Created by: RoboQA-Temporal Authors
Last Modified: 2025-12-08
Last Modified by: RoboQA-Temporal Authors

#################################################################

Copyright: RoboQA-Temporal Authors
License: MIT License

################################################################

Health Reporting and Dataset Quality Assessment Module.

Provides automated quality dashboards, metrics export, and sequence
curation recommendations for multi-sensor robotics datasets.

################################################################

"""

from .pipeline import (
    run_health_check,
    compute_temporal_score,
    compute_anomaly_score,
    compute_completeness_metrics,
    health_tier_from_overall,
)

from .exporters import (
    export_json,
    export_csv,
    export_yaml,
)

from .curation import (
    generate_curation_recommendations,
)

__all__ = [
    "run_health_check",
    "compute_temporal_score",
    "compute_anomaly_score",
    "compute_completeness_metrics",
    "health_tier_from_overall",
    "export_json",
    "export_csv",
    "export_yaml",
    "generate_curation_recommendations",
]
