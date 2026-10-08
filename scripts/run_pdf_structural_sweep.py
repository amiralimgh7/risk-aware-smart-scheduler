"""Runs the structural sweep including proportional core/workload scaling and 64-core scenarios.

This module is part of the ESRLab defense-ready codebase. It uses snake_case for functions and variables while keeping Python classes in PascalCase, and exposes documented helpers for reproducible experiments.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from core.config import TrainingConfig
from experiments.pipeline import PDF_CORE_COUNTS, PDF_CORE_UTILIZATION_LEVELS, PDF_NODE_COUNTS, run_structural_sweep


def build_argument_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser."""

    parser = argparse.ArgumentParser(description="Run the structural sweeps requested by the project PDF.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "structural_sweep")
    parser.add_argument("--node-counts", type=int, nargs="*", default=PDF_NODE_COUNTS)
    parser.add_argument("--core-counts", type=int, nargs="*", default=PDF_CORE_COUNTS)
    parser.add_argument("--utilization-levels", type=float, nargs="*", default=PDF_CORE_UTILIZATION_LEVELS)
    parser.add_argument("--graphs-per-setting", type=int, default=10)
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--policy-episodes", type=int, default=6)
    parser.add_argument("--no-scale-nodes-with-cores", action="store_true", help="Keep graph node count fixed while core count changes. Default behavior scales nodes with cores.")
    parser.add_argument("--core-reference-count", type=int, default=4, help="Core count used as the baseline for proportional node scaling.")
    parser.add_argument("--max-scaled-node-count", type=int, default=None, help="Optional safety cap for the scaled node count.")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    """Runs the structural sweep CLI."""

    args = build_argument_parser().parse_args()
    rows = run_structural_sweep(
        output_dir=args.output_dir,
        node_counts=args.node_counts,
        core_counts=args.core_counts,
        utilization_levels=args.utilization_levels,
        graphs_per_setting=args.graphs_per_setting,
        training_config=TrainingConfig(learning_rate=5e-3, num_epochs=args.num_epochs, batch_size=1, seed=args.seed),
        policy_episodes=args.policy_episodes,
        seed=args.seed,
        scale_nodes_with_cores=not args.no_scale_nodes_with_cores,
        core_reference_count=args.core_reference_count,
        max_scaled_node_count=args.max_scaled_node_count,
    )
    print(f"Stored {len(rows)} structural sweep summary rows under: {args.output_dir}")


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
