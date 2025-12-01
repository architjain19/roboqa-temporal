from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from roboqa_temporal.synchronization.temporal_validator import TemporalSyncValidator


def run_and_export(
    bag_path: str,
    output_dir: str,
    max_messages_per_topic: int | None,
) -> str:
    bag_path_obj = Path(bag_path)
    if not bag_path_obj.exists():
        raise FileNotFoundError(f"Bag path does not exist: {bag_path}")

    bag_name = bag_path_obj.stem
    os.makedirs(output_dir, exist_ok=True)

    validator = TemporalSyncValidator(output_dir=output_dir)
    report = validator.validate(
        str(bag_path_obj),
        max_messages_per_topic=max_messages_per_topic,
        include_visualizations=True,
    )

    payload = {
        "sequence": bag_name,
        "metrics": report.metrics,
    }

    out_path = Path(output_dir) / f"{bag_name}_temporal_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print(f"[INFO] Temporal metrics written to: {out_path}")
    return str(out_path)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Feature 1 temporal metrics for Feature 4")
    p.add_argument("--bag", required=True, help="Path to ROS2 MCAP bag")
    p.add_argument(
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


if __name__ == "__main__":
    args = _parse_args()
    run_and_export(args.bag, args.output_dir, args.max_messages_per_topic)
