"""Builds a summary folder with catalogs and copied report spreadsheets for delivery.

This module is part of the ESRLab defense-ready codebase. It uses snake_case for functions and variables while keeping Python classes in PascalCase, and exposes documented helpers for reproducible experiments.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from experiments.reporting import build_defense_summary_folder


def build_argument_parser() -> argparse.ArgumentParser:
    """Run the build argument parser step and return its computed result."""
    parser = argparse.ArgumentParser(description="Build a compact summary folder for defense outputs without moving the main report directory.")
    parser.add_argument("all_config_output_root", type=Path, help="For example outputs/all_config_runs")
    parser.add_argument("--structural-output-root", type=Path, default=None, help="For example outputs/structural_sweep")
    parser.add_argument("--defense-output-root", type=Path, default=Path("outputs") / "defense_report", help="For example outputs/defense_report")
    return parser


def main() -> None:
    """Run the main step and return its computed result."""
    args = build_argument_parser().parse_args()
    artifacts = build_defense_summary_folder(
        all_config_output_root=args.all_config_output_root,
        structural_output_root=args.structural_output_root,
        output_root=args.defense_output_root,
    )
    for key, value in artifacts.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
