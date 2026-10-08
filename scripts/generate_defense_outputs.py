"""CLI wrapper that regenerates the defense report from existing aggregate results.

This module is part of the ESRLab defense-ready codebase. It uses snake_case for functions and variables while keeping Python classes in PascalCase, and exposes documented helpers for reproducible experiments.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from experiments.reporting import generate_defense_outputs


def build_argument_parser() -> argparse.ArgumentParser:
    """Builds the CLI parser for report generation."""

    parser = argparse.ArgumentParser(description="Generate clean defense-ready tables and charts from experiment outputs.")
    parser.add_argument("all_config_output_root", type=Path)
    parser.add_argument("--structural-output-root", type=Path, default=Path("outputs") / "structural_sweep")
    parser.add_argument("--output-root", type=Path, default=Path("outputs") / "defense_report")
    return parser


def main() -> None:
    """Runs the reporting pipeline."""

    args = build_argument_parser().parse_args()
    artifact_paths = generate_defense_outputs(
        all_config_output_root=args.all_config_output_root,
        structural_output_root=args.structural_output_root,
        output_root=args.output_root,
    )
    for key, value in artifact_paths.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
    # PyTorch/OpenMP backends may keep worker threads alive briefly after heavy
    # training/quantization CLIs.  Flush files and streams, then terminate the
    # command-line process deterministically so batch runs do not appear hung.
    import os
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
