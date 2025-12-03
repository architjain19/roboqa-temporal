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
