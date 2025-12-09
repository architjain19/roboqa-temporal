"""
################################################################
File: html_utils.py
Created: 2025-12-07
Created by: Sayali Nehul
Last Modified: 2025-12-07
Last Modified by: Sayali Nehul
#################################################################
HTML Export Utility for Dataset Quality Scoring &Cross-Benchmarking
(Feature 4) Dashboards.This module provides`save_dashboard_html()`, 
which exports a PlotlyFigure as a standalone, browser-viewable HTML 
file. The output isportable, interactive, and suitable for dataset
reviews, academic reports, and reproducible analytics pipelines. 
Plotly JS is loaded via CDN to minimize file size.
################################################################
"""

from pathlib import Path
from plotly.io import write_html

def save_dashboard_html(fig, path: str):
    """
    Save dashboard as standalone HTML.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    write_html(
        fig,
        file=str(p),
        full_html=True,
        include_plotlyjs="cdn",
    )
