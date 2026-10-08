"""Regression tests and smoke tests for the ESRLab pipeline.

This module is part of the ESRLab defense-ready codebase. It uses snake_case for functions and variables while keeping Python classes in PascalCase, and exposes documented helpers for reproducible experiments.
"""
from __future__ import annotations

from core.config import ApplicationTimingConfig, DaggenConfig, MixedCriticalityConfig, UUniFastConfig
from generators.daggen import DaggenGenerator
from generators.uunifast import uunifast, uunifast_discard
from generators.workload import DagWorkloadBuilder
from hardware_model.hardware import build_big_little_platform
from risk_modeling.risk_models import empirical_cvar_upper, predict_mode_switch_probability
from simulator.reward import RewardWeights, compute_reward


def test_uunifast_preserves_total_utilization() -> None:
    """Run the test uunifast preserves total utilization step and return its computed result."""
    values = uunifast(num_items=8, total_utilization=3.2, seed=7)
    assert len(values) == 8
    assert abs(sum(values) - 3.2) < 1e-9
    assert all(value > 0.0 for value in values)


def test_uunifast_discard_respects_cap() -> None:
    """Run the test uunifast discard respects cap step and return its computed result."""
    values = uunifast_discard(
        UUniFastConfig(
            num_items=6,
            total_utilization=2.4,
            utilization_cap_per_item=0.8,
            seed=11,
        )
    )
    assert len(values) == 6
    assert max(values) <= 0.8 + 1e-12


def test_daggen_produces_acyclic_graph() -> None:
    """Run the test daggen produces acyclic graph step and return its computed result."""
    generator = DaggenGenerator(
        DaggenConfig(
            n=50,
            width=0.6,
            regularity=0.5,
            density=0.4,
            jump=1,
            seed=5,
        )
    )
    graph = generator.generate("g0")
    assert graph.validate_acyclic()


def test_workload_builder_assigns_mixed_criticality_wcets() -> None:
    """Run the test workload builder assigns mixed criticality wcets step and return its computed result."""
    builder = DagWorkloadBuilder(
        dag_config=DaggenConfig(
            n=30,
            width=0.6,
            regularity=0.5,
            density=0.4,
            seed=21,
        ),
        timing_config=ApplicationTimingConfig(period=100.0, relative_deadline=100.0),
        mc_config=MixedCriticalityConfig(
            hi_task_ratio=0.4,
            hi_wcet_scale_min=1.2,
            hi_wcet_scale_max=1.8,
            seed=21,
        ),
    )
    application = builder.build_single_application(application_id="app0", total_utilization=0.7)
    hi_seen = False
    lo_seen = False
    total_lo_budget = 0.0
    for task_id in application.graph.real_task_ids:
        node = application.graph.nodes[task_id]
        total_lo_budget += node.lo_wcet
        assert node.lo_wcet > 0.0
        assert "robust_lo_wcet" in node.metadata
        if node.criticality == "HI":
            hi_seen = True
            assert node.hi_wcet > node.lo_wcet
            assert "hi_wcet_scale" in node.metadata
        else:
            lo_seen = True
            assert 0.0 <= node.hi_wcet <= node.lo_wcet
            assert "lo_hi_wcet_scale" in node.metadata
    assert hi_seen
    assert lo_seen
    assert abs(total_lo_budget - 70.0) < 1e-6


def test_platform_and_risk_helpers_run() -> None:
    """Run the test platform and risk helpers run step and return its computed result."""
    platform = build_big_little_platform(num_a7=4, num_a15=4)
    assert len(platform.cores) == 8
    assert all(len(core.dvfs_levels) == 5 for core in platform.cores)

    samples = [4.0, 5.0, 6.0, 12.0, 20.0]
    cvar = empirical_cvar_upper(samples, alpha=0.2)
    probability = predict_mode_switch_probability(lo_budget=10.0, execution_samples=samples)
    reward = compute_reward(
        utilization=0.8,
        mode_switch_probability=0.1,
        cvar_upper_value=0.2,
        deadline_miss=False,
        aging_index=0.05,
        weights=RewardWeights(w_u=2.0, w_s=1.0, w_r=3.0, w_d=4.0, w_a=5.0),
    )
    assert cvar == 20.0
    assert abs(probability - 0.4) < 1e-12
    assert isinstance(reward, float)
