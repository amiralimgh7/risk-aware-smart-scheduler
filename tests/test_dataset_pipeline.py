"""Regression tests and smoke tests for the ESRLab pipeline.

This module is part of the ESRLab defense-ready codebase. It uses snake_case for functions and variables while keeping Python classes in PascalCase, and exposes documented helpers for reproducible experiments.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from core.config import TrainingConfig
from dataset_tools.serialization import load_application_json, load_json, save_application_json
from experiments.pipeline import (
    build_pdf_configuration_path,
    generate_pdf_phase_1_dataset,
    load_applications_from_configuration,
    run_baseline_suite,
    run_quantization_experiment,
    run_uncertainty_ablation,
    run_structural_sweep,
)
from hardware_model.hardware import build_big_little_platform


def test_application_serialization_round_trip(tmp_path: Path) -> None:
    """Run the test application serialization round trip step and return its computed result."""
    generated_directories = generate_pdf_phase_1_dataset(
        output_root=tmp_path / "dataset",
        graph_counts=[2],
        densities=[0.4],
        regularities=[0.5],
        widths=[0.6],
        num_nodes=12,
        mean_graph_utilization=0.6,
        max_graph_utilization=0.8,
        period=50.0,
        seed=10,
    )
    configuration_dir = generated_directories[0]
    applications = load_applications_from_configuration(configuration_dir)
    assert len(applications) == 2

    output_path = tmp_path / "round_trip.json"
    save_application_json(applications[0], output_path)
    loaded = load_application_json(output_path)
    assert loaded.application_id == applications[0].application_id
    assert loaded.graph.edge_count() == applications[0].graph.edge_count()
    assert loaded.graph.validate_acyclic()


def test_pdf_dataset_generation_creates_expected_structure_and_diverse_graphs(tmp_path: Path) -> None:
    """Run the test pdf dataset generation creates expected structure and diverse graphs step and return its computed result."""
    generated_directories = generate_pdf_phase_1_dataset(
        output_root=tmp_path / "dataset",
        graph_counts=[4],
        densities=[0.2],
        regularities=[0.5],
        widths=[0.6],
        num_nodes=14,
        mean_graph_utilization=0.65,
        max_graph_utilization=0.85,
        period=60.0,
        seed=3,
    )
    assert len(generated_directories) == 1

    configuration_dir = build_pdf_configuration_path(
        output_root=tmp_path / "dataset",
        graph_count=4,
        density=0.2,
        regularity=0.5,
        width=0.6,
    )
    assert configuration_dir.exists()
    assert (configuration_dir / "manifest.json").exists()
    application_files = sorted((configuration_dir / "applications").glob("application_*.json"))
    assert len(application_files) == 4

    applications = load_applications_from_configuration(configuration_dir)
    edge_counts = {application.graph.edge_count() for application in applications}
    level_signatures = {tuple(application.graph.level_widths()) for application in applications}
    assert len(edge_counts) > 1 or len(level_signatures) > 1


def test_smoke_experiment_runners_execute_store_outputs_and_save_models(tmp_path: Path) -> None:
    """Run the test smoke experiment runners execute store outputs and save models step and return its computed result."""
    generated_directories = generate_pdf_phase_1_dataset(
        output_root=tmp_path / "dataset",
        graph_counts=[4],
        densities=[0.4],
        regularities=[0.5],
        widths=[0.6],
        num_nodes=16,
        mean_graph_utilization=0.65,
        max_graph_utilization=0.85,
        period=60.0,
        seed=8,
    )
    applications = load_applications_from_configuration(generated_directories[0])
    platform = build_big_little_platform(num_a7=2, num_a15=2)
    training_config = TrainingConfig(learning_rate=5e-3, num_epochs=1, batch_size=1, seed=0)

    baseline_artifacts = run_baseline_suite(
        applications=applications,
        platform=platform,
        output_dir=tmp_path / "outputs" / "baseline_suite",
        training_config=training_config,
        policy_episodes=3,
    )
    assert "heft" in baseline_artifacts.summary_by_algorithm
    assert "gnn" in baseline_artifacts.summary_by_algorithm
    assert "proposed_gat_float32" in baseline_artifacts.summary_by_algorithm
    assert "proposed_gat_quantization_int8" in baseline_artifacts.summary_by_algorithm
    assert (tmp_path / "outputs" / "baseline_suite" / "baseline_summary.json").exists()
    assert (tmp_path / "outputs" / "trained_models" / "gnn" / "metadata.json").exists()
    assert (tmp_path / "outputs" / "trained_models" / "proposed_gat_float32" / "metadata.json").exists()

    uncertainty_artifacts = run_uncertainty_ablation(
        applications=applications,
        platform=platform,
        output_dir=tmp_path / "outputs" / "uncertainty_ablation",
        training_config=training_config,
        policy_episodes=3,
    )
    assert "with_uncertainty" in uncertainty_artifacts.summary_by_variant
    assert "without_uncertainty" in uncertainty_artifacts.summary_by_variant

    quantization_artifacts = run_quantization_experiment(
        applications=applications,
        platform=platform,
        output_dir=tmp_path / "outputs" / "quantization",
        training_config=training_config,
    )
    assert quantization_artifacts.average_float_scheduler_size_bytes > quantization_artifacts.average_int8_scheduler_size_bytes
    assert quantization_artifacts.average_relative_size_reduction > 0.0


def test_dataset_generation_skips_existing_configurations_by_default(tmp_path: Path) -> None:
    """Run the test dataset generation skips existing configurations by default step and return its computed result."""
    dataset_root = tmp_path / "dataset"
    generate_pdf_phase_1_dataset(
        output_root=dataset_root,
        graph_counts=[2],
        densities=[0.4],
        regularities=[0.5],
        widths=[0.6],
        num_nodes=12,
        mean_graph_utilization=0.6,
        max_graph_utilization=0.8,
        period=50.0,
        seed=10,
    )
    configuration_dir = build_pdf_configuration_path(
        output_root=dataset_root,
        graph_count=2,
        density=0.4,
        regularity=0.5,
        width=0.6,
    )
    original_manifest = load_json(configuration_dir / "manifest.json")

    generate_pdf_phase_1_dataset(
        output_root=dataset_root,
        graph_counts=[2],
        densities=[0.4],
        regularities=[0.5],
        widths=[0.6],
        num_nodes=12,
        mean_graph_utilization=0.6,
        max_graph_utilization=0.8,
        period=50.0,
        seed=999,
    )
    reused_manifest = load_json(configuration_dir / "manifest.json")
    assert reused_manifest["seed"] == original_manifest["seed"]


def test_run_all_pdf_baselines_cli_smoke(tmp_path: Path) -> None:
    """Run the test run all pdf baselines cli smoke step and return its computed result."""
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "all_runs"
    generate_pdf_phase_1_dataset(
        output_root=dataset_root,
        graph_counts=[4],
        densities=[0.4],
        regularities=[0.5],
        widths=[0.6],
        num_nodes=14,
        mean_graph_utilization=0.65,
        max_graph_utilization=0.85,
        period=60.0,
        seed=12,
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_all_pdf_baselines",
            str(dataset_root),
            "--output-root",
            str(output_root),
            "--num-epochs",
            "1",
            "--policy-episodes",
            "3",
            "--max-configurations",
            "1",
        ],
        check=True,
    )

    assert (output_root / "run_index.csv").exists()
    assert (output_root / "baseline_summary_aggregate.csv").exists()
    assert (output_root / "uncertainty_summary_aggregate.csv").exists()
    assert (output_root / "quantization_summary_aggregate.csv").exists()
    assert (output_root / "learned_models_summary_aggregate.csv").exists()


def test_run_structural_sweep_smoke(tmp_path: Path) -> None:
    """Run the test run structural sweep smoke step and return its computed result."""
    output_dir = tmp_path / "structural"
    rows = run_structural_sweep(
        output_dir=output_dir,
        node_counts=[20],
        core_counts=[4],
        utilization_levels=[0.5],
        graphs_per_setting=2,
        training_config=TrainingConfig(learning_rate=5e-3, num_epochs=1, batch_size=1, seed=2),
        policy_episodes=2,
        seed=2,
    )
    assert len(rows) == 6
    assert (output_dir / "structural_sweep_summary.csv").exists()
