"""Command-line entry points for dataset generation, experiments, reporting, repair, and sanity checks.

This module is part of the ESRLab defense-ready codebase. It uses snake_case for functions and variables while keeping Python classes in PascalCase, and exposes documented helpers for reproducible experiments.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from core.config import TrainingConfig
from experiments.pipeline import (
    load_applications_from_configuration,
    run_baseline_suite,
    run_quantization_experiment,
    run_uncertainty_ablation,
)
from hardware_model.hardware import build_big_little_platform


def build_argument_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser."""

    parser = argparse.ArgumentParser(description="Run all baseline and ablation experiments on one dataset configuration.")
    parser.add_argument("configuration_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "baseline_suite")
    parser.add_argument("--num-a7", type=int, default=4)
    parser.add_argument("--num-a15", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--policy-episodes", type=int, default=8)
    parser.set_defaults(skip_latency_benchmark=True)
    parser.add_argument("--skip-latency-benchmark", dest="skip_latency_benchmark", action="store_true")
    parser.add_argument("--enable-latency-benchmark", dest="skip_latency_benchmark", action="store_false")
    return parser


def main() -> None:
    """Runs the baseline, ablation, and INT8 experiments."""

    args = build_argument_parser().parse_args()
    applications = load_applications_from_configuration(args.configuration_dir)
    platform = build_big_little_platform(num_a7=args.num_a7, num_a15=args.num_a15)
    training_config = TrainingConfig(learning_rate=5e-3, num_epochs=args.num_epochs, batch_size=1, seed=0)

    baseline_artifacts = run_baseline_suite(
        applications=applications,
        platform=platform,
        output_dir=args.output_dir / "baseline_suite",
        training_config=training_config,
        policy_episodes=args.policy_episodes,
        benchmark_learned_model_latency=not args.skip_latency_benchmark,
    )
    uncertainty_artifacts = run_uncertainty_ablation(
        applications=applications,
        platform=platform,
        output_dir=args.output_dir / "uncertainty_ablation",
        training_config=training_config,
        policy_episodes=args.policy_episodes,
    )
    quantization_artifacts = run_quantization_experiment(
        applications=applications,
        platform=platform,
        output_dir=args.output_dir / "quantization",
        training_config=training_config,
        benchmark_latency=not args.skip_latency_benchmark,
    )

    print("Baseline algorithms:")
    for algorithm_name, summary in baseline_artifacts.summary_by_algorithm.items():
        print(f"- {algorithm_name}: DMR={summary['mean_deadline_miss_ratio']:.6f}, utilization={summary['mean_utilization']:.6f}")

    print("\nUncertainty ablation:")
    for variant_name, summary in uncertainty_artifacts.summary_by_variant.items():
        print(
            f"- {variant_name}: mode_switch_probability={summary['mean_mode_switch_probability']:.6f}, "
            f"DMR={summary['mean_deadline_miss_ratio']:.6f}"
        )

    print("\nQuantization methods:")
    for method_name, summary in (quantization_artifacts.summary_by_method or {}).items():
        print(
            f"- {method_name}: scheduler_size_bytes={float(summary['average_scheduler_size_bytes']):.6f}, "
            f"size_reduction={float(summary['average_relative_size_reduction_vs_fp32']):.6f}, "
            f"prediction_mae={float(summary['average_mean_absolute_prediction_error_vs_fp32']):.6f}"
        )


if __name__ == "__main__":
    # PyTorch/OpenMP backends may keep worker threads alive briefly after heavy
    # training/quantization CLIs.  Flush files and streams, then terminate the
    # command-line process deterministically so batch runs do not appear hung,
    # including the case where an exception is raised before normal shutdown.
    import os
    import sys
    import traceback

    exit_code = 0
    try:
        main()
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
