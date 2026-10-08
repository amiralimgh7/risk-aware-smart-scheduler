"""Regression tests and smoke tests for the ESRLab pipeline.

This module is part of the ESRLab defense-ready codebase. It uses snake_case for functions and variables while keeping Python classes in PascalCase, and exposes documented helpers for reproducible experiments.
"""
from __future__ import annotations

from baselines.classic_gcn import ClassicGcnBaseline
from baselines.heft import HeftScheduler
from baselines.nsga_iii import NsgaThreeConfig, NsgaThreeScheduler
from baselines.vd_edf import VdEdfScheduler
from core.config import ApplicationTimingConfig, DaggenConfig, MixedCriticalityConfig, TrainingConfig
from generators.workload import DagWorkloadBuilder
from hardware_model.hardware import build_big_little_platform
from models.quantization_eval import compare_float_and_int8_schedulers
from simulator.reward import RewardWeights
from training.rl_policy import RiskAwareGatScheduler


def _build_small_application_set():
    """Internal helper for build small application set."""
    builder = DagWorkloadBuilder(
        dag_config=DaggenConfig(
            n=16,
            width=0.6,
            regularity=0.5,
            density=0.4,
            seed=17,
        ),
        timing_config=ApplicationTimingConfig(period=60.0, relative_deadline=60.0),
        mc_config=MixedCriticalityConfig(
            hi_task_ratio=0.3,
            hi_wcet_scale_min=1.2,
            hi_wcet_scale_max=1.6,
            seed=17,
        ),
    )
    applications = [
        builder.build_single_application(application_id=f"app_{index}", total_utilization=0.55 + 0.05 * index)
        for index in range(3)
    ]
    return applications


def test_classic_baselines_produce_metrics() -> None:
    """Run the test classic baselines produce metrics step and return its computed result."""
    applications = _build_small_application_set()
    application = applications[0]
    platform = build_big_little_platform(num_a7=2, num_a15=2)

    heft_metrics = HeftScheduler().schedule(application=application, platform=platform, rng_seed=3)
    vd_edf_metrics = VdEdfScheduler().schedule(application=application, platform=platform, rng_seed=3)
    nsga_metrics = NsgaThreeScheduler(
        config=NsgaThreeConfig(population_size=8, num_generations=3, reference_divisions=4, seed=3)
    ).schedule(application=application, platform=platform)

    for metrics in (heft_metrics, vd_edf_metrics, nsga_metrics):
        assert metrics.makespan >= 0.0
        assert 0.0 <= metrics.utilization <= 1.0 + 1e-6
        assert 0.0 <= metrics.deadline_miss_ratio <= 1.0


def test_gnn_baseline_trains_saves_loads_and_schedules(tmp_path) -> None:
    """Run the test gnn baseline trains saves loads and schedules step and return its computed result."""
    applications = _build_small_application_set()
    platform = build_big_little_platform(num_a7=2, num_a15=2)
    baseline = ClassicGcnBaseline(device="cpu")
    baseline.fit(
        applications=applications,
        training_config=TrainingConfig(learning_rate=1e-2, num_epochs=2, batch_size=1, seed=1),
        verbose=False,
    )
    checkpoint_dir = tmp_path / "gnn_checkpoint"
    baseline.save_checkpoint(checkpoint_dir)
    loaded_baseline = ClassicGcnBaseline.load_checkpoint(checkpoint_dir, device="cpu")
    metrics = loaded_baseline.schedule(application=applications[0], platform=platform, rng_seed=1)
    assert metrics.makespan >= 0.0
    assert 0.0 <= metrics.utilization <= 1.0 + 1e-6


def test_proposed_gat_scheduler_trains_predictor_policy_saves_loads_and_runs(tmp_path) -> None:
    """Run the test proposed gat scheduler trains predictor policy saves loads and runs step and return its computed result."""
    applications = _build_small_application_set()
    platform = build_big_little_platform(num_a7=2, num_a15=2)
    scheduler = RiskAwareGatScheduler(device="cpu")
    predictor_history = scheduler.fit_predictor(
        applications=applications,
        platform=platform,
        training_config=TrainingConfig(learning_rate=5e-3, num_epochs=2, batch_size=1, seed=2),
        cvar_alpha=0.1,
        verbose=False,
    )
    assert len(predictor_history.epoch_losses) == 2

    policy_history = scheduler.fit_policy(
        applications=applications,
        platform=platform,
        num_episodes=4,
        learning_rate=1e-3,
        reward_weights=RewardWeights(w_u=2.0, w_s=1.0, w_r=0.3, w_d=2.0, w_a=0.2),
        verbose=False,
    )
    assert len(policy_history.rewards) == 4

    checkpoint_dir = tmp_path / "proposed_checkpoint"
    scheduler.save_checkpoint(checkpoint_dir)
    loaded_scheduler = RiskAwareGatScheduler.load_checkpoint(checkpoint_dir, device="cpu")
    metrics = loaded_scheduler.schedule(application=applications[0], platform=platform)
    assert metrics.makespan >= 0.0
    assert 0.0 <= metrics.mode_switch_probability <= 1.0


def test_real_int8_scheduler_comparison_runs() -> None:
    """Run the test real int8 scheduler comparison runs step and return its computed result."""
    applications = _build_small_application_set()
    platform = build_big_little_platform(num_a7=2, num_a15=2)
    float_scheduler = RiskAwareGatScheduler(device="cpu")

    float_scheduler.fit_predictor(
        applications=applications,
        platform=platform,
        training_config=TrainingConfig(learning_rate=5e-3, num_epochs=1, batch_size=1, seed=3),
        verbose=False,
    )
    float_scheduler.fit_policy(
        applications=applications,
        platform=platform,
        num_episodes=2,
        verbose=False,
    )
    int8_scheduler = float_scheduler.build_int8_inference_variant()

    comparison = compare_float_and_int8_schedulers(
        float_scheduler=float_scheduler,
        int8_scheduler=int8_scheduler,
        application=applications[0],
        platform=platform,
        device="cpu",
    )
    assert comparison.float_scheduler_size_bytes > comparison.int8_scheduler_size_bytes
    assert comparison.relative_size_reduction > 0.0
    assert comparison.mean_absolute_prediction_error >= 0.0
