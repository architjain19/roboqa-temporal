from typing import List, Dict, Any
import pandas as pd

def metrics_list_to_dataframe(metrics_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert a list of per-sequence metrics dicts into a Pandas DataFrame.
    Each dict corresponds to one sequence.
    """
    return pd.DataFrame(metrics_list)
