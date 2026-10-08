"""Small example scripts for manual inspection.

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


def build_demo_dataset():
    """Run the build demo dataset step and return its computed result."""
    builder = DagWorkloadBuilder(
        dag_config=DaggenConfig(
            n=20,
            width=0.6,
            regularity=0.5,
            density=0.4,
            seed=42,
        ),
        timing_config=ApplicationTimingConfig(period=80.0, relative_deadline=80.0),
        mc_config=MixedCriticalityConfig(
            hi_task_ratio=0.3,
            hi_wcet_scale_min=1.2,
            hi_wcet_scale_max=1.7,
            seed=42,
        ),
    )
    applications = [
        builder.build_single_application(application_id=f"demo_app_{index}", total_utilization=0.6 + 0.05 * index)
        for index in range(4)
    ]
    return applications


def print_metrics(title: str, metrics) -> None:
    """Run the print metrics step and return its computed result."""
    print(f"\n[{title}]")
    for key, value in metrics.reward_ready_dict.items():
        print(f"{key}: {value:.6f}")


def main() -> None:
    """Run the main step and return its computed result."""
    applications = build_demo_dataset()
    platform = build_big_little_platform(num_a7=2, num_a15=2)
    target_application = applications[0]

    heft_metrics = HeftScheduler().schedule(target_application, platform)
    vd_edf_metrics = VdEdfScheduler().schedule(target_application, platform)
    nsga_metrics = NsgaThreeScheduler(
        NsgaThreeConfig(population_size=10, num_generations=4, reference_divisions=4, seed=0)
    ).schedule(target_application, platform)

    gcn_baseline = ClassicGcnBaseline(device="cpu")
    gcn_baseline.fit(applications, TrainingConfig(learning_rate=1e-2, num_epochs=3, batch_size=1))
    gcn_metrics = gcn_baseline.schedule(target_application, platform)

    proposed_scheduler = RiskAwareGatScheduler(device="cpu", use_qat=False)
    proposed_scheduler.fit_predictor(
        applications=applications,
        platform=platform,
        training_config=TrainingConfig(learning_rate=5e-3, num_epochs=3, batch_size=1),
    )
    proposed_scheduler.fit_policy(
        applications=applications,
        platform=platform,
        num_episodes=6,
        learning_rate=1e-3,
        reward_weights=RewardWeights(w_u=2.0, w_s=1.0, w_r=0.3, w_d=2.0, w_a=0.2),
    )
    proposed_metrics = proposed_scheduler.schedule(target_application, platform)

    qat_scheduler = RiskAwareGatScheduler(device="cpu", use_qat=True)
    qat_scheduler.fit_predictor(
        applications=applications,
        platform=platform,
        training_config=TrainingConfig(learning_rate=5e-3, num_epochs=2, batch_size=1),
    )
    quantization_comparison = compare_float_and_int8_schedulers(
        float_scheduler=proposed_scheduler,
        int8_scheduler=proposed_scheduler.build_int8_inference_variant(),
        application=target_application,
        platform=platform,
    )

    print_metrics("HEFT", heft_metrics)
    print_metrics("VD-EDF", vd_edf_metrics)
    print_metrics("NSGA-III", nsga_metrics)
    print_metrics("Classic GCN", gcn_metrics)
    print_metrics("Proposed GAT + RL", proposed_metrics)

    print("\n[Quantization comparison]")
    print(f"float_size_bytes: {quantization_comparison.float_scheduler_size_bytes}")
    print(f"int8_size_bytes: {quantization_comparison.int8_scheduler_size_bytes}")
    print(f"relative_size_reduction: {quantization_comparison.relative_size_reduction:.6f}")
    print(f"mean_absolute_prediction_error: {quantization_comparison.mean_absolute_prediction_error:.6f}")


if __name__ == "__main__":
    main()
