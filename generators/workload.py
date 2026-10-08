"""DAG, UUniFast utilization, and workload generation utilities.

This module is part of the ESRLab defense-ready codebase. It uses snake_case for functions and variables while keeping Python classes in PascalCase, and exposes documented helpers for reproducible experiments.
"""
from __future__ import annotations

from typing import List, Optional

from core.config import ApplicationTimingConfig, DaggenConfig, MixedCriticalityConfig, SimulationConfig, UUniFastConfig
from core.random_utils import build_rng
from core.task_graph import DagApplication, RuntimeDistribution, TaskGraph
from generators.daggen import DaggenGenerator
from generators.uunifast import uunifast_discard
from risk_modeling.risk_models import robust_lo_wcet_from_samples


class DagWorkloadBuilder:
    """Builds periodic DAG applications from DAGGEN topology and UUniFast load."""

    def __init__(
        self,
        dag_config: DaggenConfig,
        timing_config: ApplicationTimingConfig,
        mc_config: Optional[MixedCriticalityConfig] = None,
        simulation_config: Optional[SimulationConfig] = None,
    ) -> None:
        """Initialize the object with validated configuration and runtime state."""
        self.dag_config = dag_config
        self.timing_config = timing_config
        self.mc_config = mc_config or MixedCriticalityConfig(seed=dag_config.seed)
        self.simulation_config = simulation_config or SimulationConfig(random_seed=dag_config.seed)
        self.rng = build_rng(self.mc_config.seed)
        self.generation_rng = build_rng(dag_config.seed)

    def build_single_application(
        self,
        application_id: str,
        total_utilization: float,
    ) -> DagApplication:
        """Builds a single periodic DAG application."""

        graph_seed = self.generation_rng.randint(0, 2**31 - 1)
        dag_config = DaggenConfig(
            n=self.dag_config.n,
            width=self.dag_config.width,
            regularity=self.dag_config.regularity,
            density=self.dag_config.density,
            jump=self.dag_config.jump,
            min_data=self.dag_config.min_data,
            max_data=self.dag_config.max_data,
            min_alpha=self.dag_config.min_alpha,
            max_alpha=self.dag_config.max_alpha,
            ccr_mode=self.dag_config.ccr_mode,
            add_dummy_source_sink=self.dag_config.add_dummy_source_sink,
            seed=graph_seed,
        )
        generator = DaggenGenerator(dag_config)
        graph = generator.generate(graph_id=application_id)
        self._assign_execution_budgets(graph, total_utilization, self.timing_config.period)

        deadline = (
            self.timing_config.relative_deadline
            if self.timing_config.relative_deadline is not None
            else self.timing_config.period
        )
        graph.deadline = deadline
        graph.period = self.timing_config.period

        return DagApplication(
            application_id=application_id,
            graph=graph,
            period=self.timing_config.period,
            deadline=deadline,
            total_utilization=total_utilization,
        )

    def build_application_set(
        self,
        num_applications: int,
        total_system_utilization: float,
        per_application_cap: Optional[float] = None,
    ) -> List[DagApplication]:
        """Builds a set of periodic DAG applications using UUniFast-Discard."""

        utilization_vector = uunifast_discard(
            UUniFastConfig(
                num_items=num_applications,
                total_utilization=total_system_utilization,
                utilization_cap_per_item=per_application_cap,
                seed=self.dag_config.seed,
            )
        )

        applications: List[DagApplication] = []
        for application_index, utilization in enumerate(utilization_vector):
            applications.append(
                self.build_single_application(
                    application_id=f"application_{application_index}",
                    total_utilization=utilization,
                )
            )
        return applications

    def _assign_execution_budgets(
        self,
        graph: TaskGraph,
        total_utilization: float,
        period: float,
    ) -> None:
        """Assigns LO-WCET, HI-WCET, and runtime distributions to graph nodes."""

        total_budget = total_utilization * period
        real_task_ids = graph.real_task_ids
        total_raw_cost = sum(max(graph.nodes[task_id].raw_cost, 1.0) for task_id in real_task_ids)
        hi_task_count = int(round(self.mc_config.hi_task_ratio * len(real_task_ids)))
        hi_task_ids = set(self.rng.sample(real_task_ids, hi_task_count)) if hi_task_count > 0 else set()
        runtime_rng = build_rng(self.simulation_config.random_seed)

        for task_id in real_task_ids:
            node = graph.nodes[task_id]
            normalized_weight = max(node.raw_cost, 1.0) / total_raw_cost
            lo_wcet = max(1e-9, normalized_weight * total_budget)

            if task_id in hi_task_ids:
                scale = self.rng.uniform(
                    self.mc_config.hi_wcet_scale_min,
                    self.mc_config.hi_wcet_scale_max,
                )
                node.criticality = "HI"
                node.lo_wcet = lo_wcet
                node.hi_wcet = lo_wcet * scale
                node.metadata["hi_wcet_scale"] = scale
                sigma = self.mc_config.hi_runtime_sigma
            else:
                degraded_scale = self.rng.uniform(
                    self.mc_config.lo_hi_wcet_scale_min,
                    self.mc_config.lo_hi_wcet_scale_max,
                )
                node.criticality = "LO"
                node.lo_wcet = lo_wcet
                node.hi_wcet = lo_wcet * degraded_scale
                node.metadata["lo_hi_wcet_scale"] = degraded_scale
                sigma = self.mc_config.lo_runtime_sigma

            node.predicted_lo_wcet = node.lo_wcet
            node.predicted_uncertainty = sigma
            node.runtime_distribution = RuntimeDistribution(mean_reference=node.lo_wcet, sigma=sigma)
            runtime_samples = node.runtime_samples(
                rng=runtime_rng,
                num_samples=self.simulation_config.num_runtime_samples,
            )
            node.metadata["robust_lo_wcet"] = robust_lo_wcet_from_samples(
                samples=runtime_samples,
                alpha=self.simulation_config.cvar_alpha,
            )
