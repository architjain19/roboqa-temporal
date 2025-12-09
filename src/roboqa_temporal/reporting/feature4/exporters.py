"""
################################################################
File: exporters.py
Created: 2025-12-07
Created by: Sayali Nehul
Last Modified: 2025-12-07
Last Modified by: Sayali Nehul
#################################################################
Data export utilities for Feature 4.Exports a Pandas DataFrame 
containing dataset quality metrics into CSV or JSON format. 
The module ensures that output directories exist and produces 
clean, analysis-ready files for downstream visualizers,dashboards,
benchmarking tools, or external analytics workflows.
Supported formats:
- CSV  : comma-separated plain text table
- JSON : list-of-records encoding, readable by most data systems
################################################################
"""

from pathlib import Path
import pandas as pd

def export_dataframe(df: pd.DataFrame, path: str, fmt: str = "csv") -> None:
    """
    Export a DataFrame as CSV or JSON.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "csv":
        df.to_csv(p, index=False)
    elif fmt == "json":
        df.to_json(p, orient="records", indent=2)
    else:
        raise ValueError(f"Unsupported format: {fmt}")
