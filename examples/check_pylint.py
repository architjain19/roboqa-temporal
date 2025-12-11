#!/usr/bin/env python3
"""
Pylint checker script for RoboQA-Temporal

This script runs pylint on the specified directory to ensure code quality.

Usage:
    python scripts/check_pylint.py          # Check all source code
    python scripts/check_pylint.py src      # Check specific directory
    python scripts/check_pylint.py --help   # Show help
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_pylint(path: str, min_score: float = 7.0, strict: bool = False) -> int:
    """
    Run pylint on specified path.

    Args:
        path: Path to check
        min_score: Minimum score to pass
        strict: If True, fail on any issues (uses higher threshold)

    Returns:
        Exit code
    """
    cmd = [sys.executable, "-m", "pylint", path]
    
    threshold = 9.0 if strict else min_score
    cmd.extend(["--fail-under", str(threshold)])

    result = subprocess.run(cmd, text=True)
    return result.returncode


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run pylint checks on RoboQA-Temporal")
    parser.add_argument(
        "path",
        nargs="?",
        default="src/roboqa_temporal",
        help="Path to check (default: src/roboqa_temporal)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=7.0,
        help="Minimum pylint score to pass (default: 7.0)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use strict mode (fail-under 9.0)",
    )

    args = parser.parse_args()

    check_path = Path(args.path)
    if not check_path.is_absolute():
        check_path = Path.cwd() / check_path

    if not check_path.exists():
        print(f"Error: Path does not exist: {check_path}", file=sys.stderr)
        return 1

    print(f"Running pylint on {check_path}...")
    exit_code = run_pylint(str(check_path), args.min_score, args.strict)

    if exit_code == 0:
        print("\nPylint check passed!")
    else:
        print("\nPylint check failed!")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
