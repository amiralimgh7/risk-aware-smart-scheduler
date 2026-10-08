"""Ensures plots use PDF as the canonical output and derives PNG copies from those PDFs.

This module is part of the ESRLab defense-ready codebase. It uses snake_case for functions and variables while keeping Python classes in PascalCase, and exposes documented helpers for reproducible experiments.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from experiments.reporting import ensure_pdf_for_png_outputs


def build_argument_parser() -> argparse.ArgumentParser:
    """Run the build argument parser step and return its computed result."""
    parser = argparse.ArgumentParser(
        description=(
            "Ensure plot PDF/PNG pairs under an output root. PDF is treated as the "
            "canonical chart output, and PNG files are rasterized from those PDFs."
        )
    )
    parser.add_argument("output_root", type=Path, help="Root to scan, for example outputs/defense_report")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild PNG counterparts from PDF even when PNG already exists.")
    return parser


def main() -> None:
    """Run the main step and return its computed result."""
    args = build_argument_parser().parse_args()
    audit = ensure_pdf_for_png_outputs(args.output_root, overwrite=args.overwrite)
    for key, value in audit.items():
        print(f"{key}={value}")
    if audit.get("pdf_failed", 0) or audit.get("png_failed_from_pdf", 0):
        sys.exit(2)


if __name__ == "__main__":
    main()
