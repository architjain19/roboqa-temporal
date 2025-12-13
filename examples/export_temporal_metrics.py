"""
################################################################
File: export_temporal_metrics.py
Created: 2025-12-08
Created by: Sayali Nehul (snehul@uw.edu)
Last Modified: 2025-12-08
Last Modified by: Sayali Nehul (snehul@uw.edu)
################################################################

Exports temporal synchronization metrics (Feature 1) for ROS2 MCAP
bags in a JSON format compatible with Dataset Quality Scoring &
Cross-Benchmarking (Feature 4). This script runs
the TemporalSyncValidator, generates per-topic alignment metrics,
and saves optional visualizations in the specified output folder.

################################################################
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from roboqa_temporal.synchronization.temporal_validator import TemporalSyncValidator                     # Import temporal synchronization validator (Feature 1) (already implemented earlier)


def run_and_export(
    bag_path: str,
    output_dir: str,
    max_messages_per_topic: int | None,
) -> str:
    bag_path_obj = Path(bag_path)                                                                        # Convert bag into a Path object for safety & convenience
    if not bag_path_obj.exists():                                                                        # Safety check — ensures the bag actually exists
        raise FileNotFoundError(f"Bag path does not exist: {bag_path}")

    bag_name = bag_path_obj.stem
    os.makedirs(output_dir, exist_ok=True)                                                               # Ensure output folder exists

    validator = TemporalSyncValidator(output_dir=output_dir)                                             # Initialize the Feature 1 temporal sync validator
    report = validator.validate(                                                                         # Run the validator on the bag
        str(bag_path_obj),
        max_messages_per_topic=max_messages_per_topic,
        include_visualizations=True,
    )

    payload = {                                                                                           # Build a clean JSON payload that Dataset Quality Scoring and Cross- Benchmarking (Feature 4) will later read
        "sequence": bag_name,
        "metrics": report.metrics,
    }

    out_path = Path(output_dir) / f"{bag_name}_temporal_metrics.json"                                      # Final output file location
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print(f"[INFO] Temporal metrics written to: {out_path}")
    return str(out_path)


def _parse_args() -> argparse.Namespace:                                                                   # CLI parser for running this file as a script
    p = argparse.ArgumentParser(description="Export Feature 1 temporal metrics for Feature 4")
    p.add_argument("--bag", required=True, help="Path to ROS2 MCAP bag")                                   # Path to input ROS2 bag file (.mcap)
    p.add_argument(                                                                                        # Directory where output JSON will be saved
        "--output-dir",
        default="reports/temporal_sync",
        help="Directory where <bag>_temporal_metrics.json will be stored",
    )
    p.add_argument(
        "--max-messages-per-topic",
        type=int,
        default=None,
        help="Optional cap on messages per topic (useful for very large bags)",
    )
    return p.parse_args()


if __name__ == "__main__":                                                                               # Run parser and execute the export function
    args = _parse_args()
    run_and_export(args.bag, args.output_dir, args.max_messages_per_topic)rgs.bag, args.output_dir, args.max_messages_per_topic)
