"""Command-line entry points for dataset generation, experiments, reporting, repair, and sanity checks.

This module is part of the ESRLab defense-ready codebase. It uses snake_case for functions and variables while keeping Python classes in PascalCase, and exposes documented helpers for reproducible experiments.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from experiments.pipeline import run_full_pdf_pipeline


def build_argument_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser."""

    parser = argparse.ArgumentParser(description="Run the full project workflow aligned with the PDF.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs") / "full_pdf_pipeline")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--skip-dataset-generation", action="store_true")
    parser.add_argument("--smoke-mode", action="store_true")
    return parser


def main() -> None:
    """Runs the full PDF-aligned pipeline CLI."""

    args = build_argument_parser().parse_args()
    artifacts = run_full_pdf_pipeline(
        output_root=args.output_root,
        dataset_root=args.dataset_root,
        generate_dataset_first=not args.skip_dataset_generation,
        smoke_mode=args.smoke_mode,
    )
    for key, value in artifacts.items():
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
