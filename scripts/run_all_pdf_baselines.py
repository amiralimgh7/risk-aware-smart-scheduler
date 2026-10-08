"""Command-line entry points for dataset generation, experiments, reporting, repair, and sanity checks.

This module is part of the ESRLab defense-ready codebase. It uses snake_case for functions and variables while keeping Python classes in PascalCase, and exposes documented helpers for reproducible experiments.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

from core.config import TrainingConfig
from dataset_tools.serialization import load_json, save_json
from experiments.pipeline import (
    collect_configuration_directories,
    load_applications_from_configuration,
    run_baseline_suite,
    run_quantization_experiment,
    run_uncertainty_ablation,
    save_rows_as_csv,
)
from hardware_model.hardware import build_big_little_platform


def _mirror_aggregate_outputs_to_legacy_root(output_root: Path, aggregate_dir: Path) -> None:
    """Copies aggregate outputs to the legacy root output directory for backward compatibility."""

    output_root.mkdir(parents=True, exist_ok=True)
    for file_name in [
        "run_index.csv",
        "baseline_summary_aggregate.csv",
        "uncertainty_summary_aggregate.csv",
        "quantization_summary_aggregate.csv",
        "learned_models_summary_aggregate.csv",
        "task_level_results_aggregate.csv",
        "core_level_results_aggregate.csv",
        "schedule_summary_extended_aggregate.csv",
        "energy_power_summary_aggregate.csv",
        "prediction_metrics_aggregate.csv",
        "prediction_detail_results_aggregate.csv",
        "power_trace_aggregate.csv",
        "training_loss_curves_aggregate.csv",
        "run_manifest.json",
    ]:
        source_path = aggregate_dir / file_name
        target_path = output_root / file_name
        if source_path.exists():
            shutil.copy2(source_path, target_path)


def build_argument_parser() -> argparse.ArgumentParser:
    """Builds the CLI parser for dataset-wide baseline execution."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the baseline suite, uncertainty ablation, and quantization experiment "
            "over every dataset configuration under one dataset root."
        )
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("outputs") / "all_config_runs")
    parser.add_argument("--num-a7", type=int, default=4)
    parser.add_argument("--num-a15", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--policy-episodes", type=int, default=8)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-configurations", type=int, default=None)
    parser.add_argument("--force-rerun", action="store_true")
    parser.set_defaults(skip_latency_benchmark=True)
    parser.add_argument(
        "--skip-latency-benchmark",
        dest="skip_latency_benchmark",
        action="store_true",
        help="Skip learned-model and quantization latency microbenchmarks. Skip learned-model and quantization latency microbenchmarks. This is the default for smoke-safe runs.",
    )
    parser.add_argument(
        "--enable-latency-benchmark",
        dest="skip_latency_benchmark",
        action="store_false",
        help="Enable learned-model and quantization latency microbenchmarks explicitly. Latency is disabled by default for smoke-safe runs.",
    )
    return parser


def _flatten_summary(
    summary: Dict[str, Dict[str, float]],
    configuration_relative_path: str,
    summary_kind: str,
) -> List[Dict[str, float | str]]:
    """Converts nested summary dictionaries to flat rows."""

    rows: List[Dict[str, float | str]] = []
    for algorithm_name, metrics in summary.items():
        row: Dict[str, float | str] = {
            "configuration": configuration_relative_path,
            "summary_kind": summary_kind,
            "algorithm": algorithm_name,
        }
        row.update(metrics)
        rows.append(row)
    return rows


def _flatten_quantization_methods(summary: Dict[str, Dict[str, float | str]], configuration_relative_path: str) -> List[Dict[str, float | str]]:
    """Converts quantization method summaries to flat aggregate rows."""

    return [
        {
            "configuration": configuration_relative_path,
            "summary_kind": "quantization_method",
            "method_name": method_name,
            **dict(metrics),
        }
        for method_name, metrics in summary.items()
    ]






DETAILED_BASELINE_FILENAMES = {
    "task_rows": "task_level_results.csv",
    "core_rows": "core_level_results.csv",
    "schedule_rows": "schedule_summary_extended.csv",
    "energy_rows": "energy_power_summary.csv",
    "prediction_rows": "prediction_metrics.csv",
    "prediction_detail_rows": "prediction_detail_results.csv",
    "power_trace_rows": "power_trace.csv",
    "training_loss_rows": "training_loss_curves.csv",
}


def _load_csv_rows(path: Path) -> List[Dict[str, float | str]]:
    """Loads CSV rows without pandas."""

    import csv

    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def _with_configuration(rows: List[Dict[str, float | str]], configuration_relative_path: str) -> List[Dict[str, float | str]]:
    """Adds or overwrites the configuration column for aggregate outputs."""

    return [{"configuration": configuration_relative_path, **dict(row)} for row in rows]


def _load_existing_detailed_rows(configuration_output_dir: Path, configuration_relative_path: str) -> Dict[str, List[Dict[str, float | str]]]:
    """Loads per-configuration detailed CSV outputs from an existing baseline-suite run."""

    baseline_dir = configuration_output_dir / "baseline_suite"
    return {
        key: _with_configuration(_load_csv_rows(baseline_dir / filename), configuration_relative_path)
        for key, filename in DETAILED_BASELINE_FILENAMES.items()
    }


def _all_detailed_files_exist(configuration_output_dir: Path) -> bool:
    """Returns True only when all detailed CSV outputs exist for this configuration."""

    baseline_dir = configuration_output_dir / "baseline_suite"
    return all((baseline_dir / filename).exists() for filename in DETAILED_BASELINE_FILENAMES.values())

def _has_positive_latency_value(value: object) -> bool:
    """Returns True when a loaded latency field contains a measured positive value."""

    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False


def _existing_results_have_latency(
    learned_models_summary_path: Path,
    quantization_summary_path: Path,
) -> bool:
    """Checks whether reusable result files already contain non-zero latency metrics.

    Older runs in this project often stored all latency fields as zero because the
    command-line wrappers skipped benchmarking by default.  When latency is enabled
    we should not silently reuse those stale files; otherwise the final report keeps
    showing zero latency even though the code is capable of measuring it.
    """

    latency_keys = (
        "predictor_latency_ms",
        "policy_latency_ms",
        "scheduler_latency_ms",
        "average_predictor_latency_ms",
        "average_policy_latency_ms",
        "average_scheduler_latency_ms",
        "average_float_predictor_latency_ms",
        "average_int8_predictor_latency_ms",
        "average_float_policy_latency_ms",
        "average_int8_policy_latency_ms",
    )

    try:
        with learned_models_summary_path.open("r", encoding="utf-8") as file:
            learned_rows = json.load(file)
    except Exception:
        learned_rows = []

    learned_has_latency = any(
        any(_has_positive_latency_value(row.get(key)) for key in latency_keys)
        for row in (learned_rows if isinstance(learned_rows, list) else [])
    )

    try:
        quantization_summary = load_json(quantization_summary_path)
    except Exception:
        quantization_summary = {}

    quantization_has_latency = any(
        _has_positive_latency_value(quantization_summary.get(key))
        for key in latency_keys
    )
    methods = quantization_summary.get("methods", {}) if isinstance(quantization_summary, dict) else {}
    if isinstance(methods, dict):
        quantization_has_latency = quantization_has_latency or any(
            any(_has_positive_latency_value(method_row.get(key)) for key in latency_keys)
            for method_row in methods.values()
            if isinstance(method_row, dict)
        )

    return learned_has_latency and quantization_has_latency


def _load_existing_rows(configuration_output_dir: Path, configuration_relative_path: str) -> Dict[str, List[Dict[str, float | str]]]:
    """Loads previously computed summary rows for one configuration."""

    baseline_summary = load_json(configuration_output_dir / "baseline_suite" / "baseline_summary.json")
    uncertainty_summary = load_json(configuration_output_dir / "uncertainty_ablation" / "uncertainty_ablation_summary.json")
    quantization_summary = load_json(configuration_output_dir / "quantization" / "quantization_summary.json")
    quantization_methods = dict(quantization_summary.get("methods", {}))
    if not quantization_methods:
        quantization_methods = {"legacy_int8": quantization_summary}
    with (configuration_output_dir / "baseline_suite" / "learned_models_summary.json").open("r", encoding="utf-8") as file:
        learned_models_summary = json.load(file)

    loaded = {
        "baseline_rows": _flatten_summary(baseline_summary, configuration_relative_path, "baseline"),
        "uncertainty_rows": _flatten_summary(uncertainty_summary, configuration_relative_path, "uncertainty_ablation"),
        "quantization_rows": _flatten_quantization_methods(quantization_methods, configuration_relative_path),
        "learned_models_rows": [{"configuration": configuration_relative_path, **dict(row)} for row in learned_models_summary],
    }
    loaded.update(_load_existing_detailed_rows(configuration_output_dir, configuration_relative_path))
    return loaded


def main() -> None:
    """Runs the dataset-wide baseline workflow."""

    args = build_argument_parser().parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    aggregate_dir = output_root / "_aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    all_configurations = collect_configuration_directories(dataset_root)
    selected_configurations = all_configurations[args.start_index :]
    if args.max_configurations is not None:
        selected_configurations = selected_configurations[: args.max_configurations]

    if not selected_configurations:
        raise RuntimeError("No dataset configurations were selected for execution.")

    platform = build_big_little_platform(num_a7=args.num_a7, num_a15=args.num_a15)
    training_config = TrainingConfig(learning_rate=5e-3, num_epochs=args.num_epochs, batch_size=1, seed=0)

    run_index_rows: List[Dict[str, float | str]] = []
    aggregate_baseline_rows: List[Dict[str, float | str]] = []
    aggregate_uncertainty_rows: List[Dict[str, float | str]] = []
    aggregate_quantization_rows: List[Dict[str, float | str]] = []
    aggregate_learned_models_rows: List[Dict[str, float | str]] = []
    aggregate_task_rows: List[Dict[str, float | str]] = []
    aggregate_core_rows: List[Dict[str, float | str]] = []
    aggregate_schedule_rows: List[Dict[str, float | str]] = []
    aggregate_energy_rows: List[Dict[str, float | str]] = []
    aggregate_prediction_rows: List[Dict[str, float | str]] = []
    aggregate_prediction_detail_rows: List[Dict[str, float | str]] = []
    aggregate_power_trace_rows: List[Dict[str, float | str]] = []
    aggregate_training_loss_rows: List[Dict[str, float | str]] = []

    total_configurations = len(selected_configurations)
    for local_index, configuration_dir in enumerate(selected_configurations):
        configuration_index = args.start_index + local_index
        configuration_relative_path = str(configuration_dir.relative_to(dataset_root))
        configuration_output_dir = output_root / configuration_relative_path
        baseline_summary_path = configuration_output_dir / "baseline_suite" / "baseline_summary.json"
        uncertainty_summary_path = configuration_output_dir / "uncertainty_ablation" / "uncertainty_ablation_summary.json"
        quantization_summary_path = configuration_output_dir / "quantization" / "quantization_summary.json"
        learned_models_summary_path = configuration_output_dir / "baseline_suite" / "learned_models_summary.json"

        progress_percent = ((local_index + 1) / total_configurations) * 100.0
        print(
            f"[Configurations] {local_index + 1}/{total_configurations} ({progress_percent:6.2f}%) -> {configuration_relative_path}"
        )

        existing_result_files_are_complete = (
            baseline_summary_path.exists()
            and uncertainty_summary_path.exists()
            and quantization_summary_path.exists()
            and learned_models_summary_path.exists()
            and _all_detailed_files_exist(configuration_output_dir)
        )
        existing_latency_is_usable = (
            args.skip_latency_benchmark
            or not existing_result_files_are_complete
            or _existing_results_have_latency(
                learned_models_summary_path=learned_models_summary_path,
                quantization_summary_path=quantization_summary_path,
            )
        )

        if not args.force_rerun and existing_result_files_are_complete and existing_latency_is_usable:
            loaded_rows = _load_existing_rows(configuration_output_dir, configuration_relative_path)
            aggregate_baseline_rows.extend(loaded_rows["baseline_rows"])
            aggregate_uncertainty_rows.extend(loaded_rows["uncertainty_rows"])
            aggregate_quantization_rows.extend(loaded_rows["quantization_rows"])
            aggregate_learned_models_rows.extend(loaded_rows["learned_models_rows"])
            aggregate_task_rows.extend(loaded_rows.get("task_rows", []))
            aggregate_core_rows.extend(loaded_rows.get("core_rows", []))
            aggregate_schedule_rows.extend(loaded_rows.get("schedule_rows", []))
            aggregate_energy_rows.extend(loaded_rows.get("energy_rows", []))
            aggregate_prediction_rows.extend(loaded_rows.get("prediction_rows", []))
            aggregate_prediction_detail_rows.extend(loaded_rows.get("prediction_detail_rows", []))
            aggregate_power_trace_rows.extend(loaded_rows.get("power_trace_rows", []))
            aggregate_training_loss_rows.extend(loaded_rows.get("training_loss_rows", []))
            status = "skipped_existing_results"
        else:
            applications = load_applications_from_configuration(configuration_dir)
            baseline_artifacts = run_baseline_suite(
                applications=applications,
                platform=platform,
                output_dir=configuration_output_dir / "baseline_suite",
                training_config=training_config,
                policy_episodes=args.policy_episodes,
                benchmark_learned_model_latency=not args.skip_latency_benchmark,
                configuration_label=configuration_relative_path,
            )
            uncertainty_artifacts = run_uncertainty_ablation(
                applications=applications,
                platform=platform,
                output_dir=configuration_output_dir / "uncertainty_ablation",
                training_config=training_config,
                policy_episodes=args.policy_episodes,
            )
            quantization_artifacts = run_quantization_experiment(
                applications=applications,
                platform=platform,
                output_dir=configuration_output_dir / "quantization",
                training_config=training_config,
                benchmark_latency=not args.skip_latency_benchmark,
            )

            aggregate_baseline_rows.extend(_flatten_summary(baseline_artifacts.summary_by_algorithm, configuration_relative_path, "baseline"))
            aggregate_uncertainty_rows.extend(_flatten_summary(uncertainty_artifacts.summary_by_variant, configuration_relative_path, "uncertainty_ablation"))
            aggregate_quantization_rows.extend(
                _flatten_quantization_methods(quantization_artifacts.summary_by_method or {}, configuration_relative_path)
            )
            aggregate_learned_models_rows.extend(
                {"configuration": configuration_relative_path, **row} for row in baseline_artifacts.learned_models_summary
            )
            detailed_rows = baseline_artifacts.detailed_rows or {}
            aggregate_task_rows.extend(_with_configuration(detailed_rows.get("task_rows", []), configuration_relative_path))
            aggregate_core_rows.extend(_with_configuration(detailed_rows.get("core_rows", []), configuration_relative_path))
            aggregate_schedule_rows.extend(_with_configuration(detailed_rows.get("schedule_rows", []), configuration_relative_path))
            aggregate_energy_rows.extend(_with_configuration(detailed_rows.get("energy_rows", []), configuration_relative_path))
            aggregate_prediction_rows.extend(_with_configuration(detailed_rows.get("prediction_rows", []), configuration_relative_path))
            aggregate_prediction_detail_rows.extend(_with_configuration(detailed_rows.get("prediction_detail_rows", []), configuration_relative_path))
            aggregate_power_trace_rows.extend(_with_configuration(detailed_rows.get("power_trace_rows", []), configuration_relative_path))
            training_loss_rows = _load_csv_rows(configuration_output_dir / "baseline_suite" / "training_loss_curves.csv")
            aggregate_training_loss_rows.extend(_with_configuration(training_loss_rows, configuration_relative_path))
            status = "completed"

        run_index_rows.append(
            {
                "configuration_index": float(configuration_index),
                "configuration": configuration_relative_path,
                "status": status,
            }
        )

        save_rows_as_csv(run_index_rows, aggregate_dir / "run_index.csv")
        save_rows_as_csv(aggregate_baseline_rows, aggregate_dir / "baseline_summary_aggregate.csv")
        save_rows_as_csv(aggregate_uncertainty_rows, aggregate_dir / "uncertainty_summary_aggregate.csv")
        save_rows_as_csv(aggregate_quantization_rows, aggregate_dir / "quantization_summary_aggregate.csv")
        save_rows_as_csv(aggregate_learned_models_rows, aggregate_dir / "learned_models_summary_aggregate.csv")
        save_rows_as_csv(aggregate_task_rows, aggregate_dir / "task_level_results_aggregate.csv")
        save_rows_as_csv(aggregate_core_rows, aggregate_dir / "core_level_results_aggregate.csv")
        save_rows_as_csv(aggregate_schedule_rows, aggregate_dir / "schedule_summary_extended_aggregate.csv")
        save_rows_as_csv(aggregate_energy_rows, aggregate_dir / "energy_power_summary_aggregate.csv")
        save_rows_as_csv(aggregate_prediction_rows, aggregate_dir / "prediction_metrics_aggregate.csv")
        save_rows_as_csv(aggregate_prediction_detail_rows, aggregate_dir / "prediction_detail_results_aggregate.csv")
        save_rows_as_csv(aggregate_power_trace_rows, aggregate_dir / "power_trace_aggregate.csv")
        save_rows_as_csv(aggregate_training_loss_rows, aggregate_dir / "training_loss_curves_aggregate.csv")

    run_manifest = {
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "aggregate_dir": str(aggregate_dir),
        "num_selected_configurations": len(selected_configurations),
        "num_completed_or_loaded": len(run_index_rows),
        "force_rerun": bool(args.force_rerun),
        "task_level_rows": len(aggregate_task_rows),
        "core_level_rows": len(aggregate_core_rows),
        "schedule_summary_extended_rows": len(aggregate_schedule_rows),
        "energy_power_rows": len(aggregate_energy_rows),
        "prediction_metrics_rows": len(aggregate_prediction_rows),
        "prediction_detail_rows": len(aggregate_prediction_detail_rows),
        "power_trace_rows": len(aggregate_power_trace_rows),
        "training_loss_rows": len(aggregate_training_loss_rows),
    }
    save_json(run_manifest, aggregate_dir / "run_manifest.json")
    _mirror_aggregate_outputs_to_legacy_root(output_root=output_root, aggregate_dir=aggregate_dir)
    print(f"dataset_root={dataset_root}")
    print(f"output_root={output_root}")
    print(f"aggregate_dir={aggregate_dir}")
    print(f"num_selected_configurations={len(selected_configurations)}")


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
