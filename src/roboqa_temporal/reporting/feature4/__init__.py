"""
################################################################
File: __init__.py
Created: 2025-12-07
Created by: Sayali Nehul
Last Modified: 2025-12-07
Last Modified by: Sayali Nehul
#################################################################
Reporting utilities for (Feature 4) Dataset Quality Scoring &
Cross-Benchmarking.
This package exposes the primary public functions used to convert
metric dictionaries into DataFrames, export results, build Plotly
dashboards, and save them as standalone HTML reports.
Modules exported:
- metrics_list_to_dataframe : convert metrics → DataFrame
- export_dataframe          : save metrics to CSV / JSON
- build_quality_dashboard   : construct 4-panel interactive dashboard
- save_dashboard_html       : export dashboard as HTML
These utilities form the final stage of the Feature 4 pipeline,
enabling reproducible reporting, visualization, and dataset curation.
################################################################
"""
from .metrics_to_df import metrics_list_to_dataframe
from .exporters import export_dataframe
from .dashboards import build_quality_dashboard
from .html_utils import save_dashboard_html

__all__ = [
    "metrics_list_to_dataframe",
    "export_dataframe",
    "build_quality_dashboard",
    "save_dashboard_html",
]
