"""
Test-only helper module for Feature 4 (dataset quality reporting).

This module dynamically loads the Feature 4 reporting helpers directly
from their source files under:

    src/roboqa_temporal/reporting/feature4/

We do this to:
- avoid importing the full roboqa_temporal package (which may pull in ROS2),
- keep Feature 4 tests independent of other features,
- ensure we always test the latest source files on disk.
"""

from __future__ import annotations

from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# Paths to Feature 4 modules
# --------------------------------------------------------------------------------------

# Repo root: .../roboqa-temporal
ROOT = Path(__file__).resolve().parents[1]

# Feature 4 directory: src/roboqa_temporal/reporting/feature4
FEATURE4_DIR = ROOT / "src" / "roboqa_temporal" / "reporting" / "feature4"

METRICS_PATH = FEATURE4_DIR / "metrics_to_df.py"
EXPORTERS_PATH = FEATURE4_DIR / "exporters.py"
DASHBOARDS_PATH = FEATURE4_DIR / "dashboards.py"
HTML_UTILS_PATH = FEATURE4_DIR / "html_utils.py"


def _load_module(name: str, path: Path):
    """
    Dynamically load a module from a given file path. This bypasses the
    normal package import system and avoids importing roboqa_temporal.__init__.
    """
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------------------
# Loaded modules
# --------------------------------------------------------------------------------------

metrics_mod = _load_module("metrics_to_df_mod", METRICS_PATH)
exporters_mod = _load_module("exporters_mod", EXPORTERS_PATH)
dashboards_mod = _load_module("dashboards_mod", DASHBOARDS_PATH)
html_utils_mod = _load_module("html_utils_mod", HTML_UTILS_PATH)


# --------------------------------------------------------------------------------------
# Public helpers re-exported for tests
# --------------------------------------------------------------------------------------

metrics_list_to_dataframe = metrics_mod.metrics_list_to_dataframe
export_dataframe = exporters_mod.export_dataframe
build_quality_dashboard = dashboards_mod.build_quality_dashboard
save_dashboard_html = html_utils_mod.save_dashboard_html


__all__ = [
    "np",
    "pd",
    "metrics_list_to_dataframe",
    "export_dataframe",
    "build_quality_dashboard",
    "save_dashboard_html",
]
