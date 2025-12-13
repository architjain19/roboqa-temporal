"""
################################################################
File: metrics_to_df.py
Created: 2025-12-07
Created by: Sayali Nehul (snehul@uw.edu)
Last Modified: 2025-12-07
Last Modified by: Sayali Nehul (snehul@uw.edu)
#################################################################
Convert aggregated per-sequence metrics into a structured Pandas
DataFrame. Dataset Quality Scoring &Cross-Benchmarking (Feature 4)
generates a list of metric dictionaries—one per dataset sequence. 
This module converts that list into a clean DataFrame where each row 
represents a sequence and each column represents a metric field.
The resulting DataFrame is used by exporters and dashboard builders 
to create visual summaries, reports, and cross-dataset comparisons.
################################################################
"""
from typing import List, Dict, Any
import pandas as pd

def metrics_list_to_dataframe(metrics_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert a list of per-sequence metrics dicts into a Pandas DataFrame.
    Each dict corresponds to one sequence.
    """
    return pd.DataFrame(metrics_list)
