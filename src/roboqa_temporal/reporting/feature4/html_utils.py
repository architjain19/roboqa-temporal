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
