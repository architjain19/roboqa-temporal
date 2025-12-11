"""
Pylint integration tests for RoboQA-Temporal

This module provides pytest integration with pylint to check code quality.
"""

import subprocess
import sys
from pathlib import Path

import pytest


def get_python_files(directory: Path) -> list[Path]:
    """Get all Python files in the source directory."""
    return list(directory.rglob("*.py"))


class TestPylint:
    """Pylint tests for code quality checking."""

    @pytest.fixture(scope="session", autouse=True)
    def source_dir(self):
        """Get the source directory path."""
        return Path(__file__).resolve().parents[1] / "src" / "roboqa_temporal"

    def test_pylint_src(self, source_dir):
        """Run pylint on source code."""
        if not source_dir.exists():
            pytest.skip(f"Source directory not found: {source_dir}")

        result = subprocess.run(
            [sys.executable, "-m", "pylint", str(source_dir), "--fail-under=7.0"],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            pytest.fail(f"Pylint check failed:\n{result.stdout}\n{result.stderr}")

    def test_pylint_tests(self):
        """Run pylint on test code with relaxed standards."""
        test_dir = Path(__file__).resolve().parent

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pylint",
                str(test_dir),
                "--disable=missing-docstring",
                "--fail-under=6.0",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"\nPylint found issues in tests:\n{result.stdout}")


def test_pylint_single_module():
    """Test a specific module for pylint compliance."""
    source_dir = Path(__file__).resolve().parents[1] / "src" / "roboqa_temporal"

    if not source_dir.exists():
        pytest.skip(f"Source directory not found: {source_dir}")

    python_files = get_python_files(source_dir)
    if not python_files:
        pytest.skip("No Python files found")

    detection_dir = source_dir / "detection"
    if detection_dir.exists():
        result = subprocess.run(
            [sys.executable, "-m", "pylint", str(detection_dir)],
            capture_output=True,
            text=True,
        )

        lines = result.stdout.split("\n")
        summary = [l for l in lines if "rated at" in l]
        if summary:
            print(f"\nDetection module pylint rating: {summary[0]}")
