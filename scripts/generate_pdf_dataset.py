"""Command-line entry points for dataset generation, experiments, reporting, repair, and sanity checks.

This module is part of the ESRLab defense-ready codebase. It uses snake_case for functions and variables while keeping Python classes in PascalCase, and exposes documented helpers for reproducible experiments.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from experiments.pipeline import (
    PDF_DENSITIES,
    PDF_GRAPH_COUNTS,
    PDF_REGULARITIES,
    PDF_WIDTHS,
    generate_pdf_phase_1_dataset,
)


def build_argument_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser."""

    parser = argparse.ArgumentParser(description="Generate the phase-1 dataset defined by the project PDF.")
    parser.add_argument("--output-root", type=Path, default=Path("datasets") / "pdf_phase_1")
    parser.add_argument("--graph-counts", type=int, nargs="*", default=PDF_GRAPH_COUNTS)
    parser.add_argument("--densities", type=float, nargs="*", default=PDF_DENSITIES)
    parser.add_argument("--regularities", type=float, nargs="*", default=PDF_REGULARITIES)
    parser.add_argument("--widths", type=float, nargs="*", default=PDF_WIDTHS)
    parser.add_argument("--num-nodes", type=int, default=50)
    parser.add_argument("--mean-graph-utilization", type=float, default=0.7)
    parser.add_argument("--max-graph-utilization", type=float, default=0.9)
    parser.add_argument("--period", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-regenerate", action="store_true")
    return parser


def main() -> None:
    """Runs the dataset generation CLI."""

    args = build_argument_parser().parse_args()
    generated_directories = generate_pdf_phase_1_dataset(
        output_root=args.output_root,
        graph_counts=args.graph_counts,
        densities=args.densities,
        regularities=args.regularities,
        widths=args.widths,
        num_nodes=args.num_nodes,
        mean_graph_utilization=args.mean_graph_utilization,
        max_graph_utilization=args.max_graph_utilization,
        period=args.period,
        seed=args.seed,
        skip_existing=not args.force_regenerate,
    )
    print(
        f"Prepared {len(generated_directories)} configuration directories under: {args.output_root} "
        f"(force_regenerate={args.force_regenerate})"
    )


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
