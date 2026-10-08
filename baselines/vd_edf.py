"""Classical and learned baseline schedulers used for comparison.

This module is part of the ESRLab defense-ready codebase. It uses snake_case for
functions and variables while keeping Python classes in PascalCase, and exposes
documented helpers for reproducible experiments.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from core.metrics import ScheduleEntry, ScheduleMetrics
from core.random_utils import build_rng
from core.task_graph import DagApplication, TaskGraph
from hardware_model.hardware import CoreSpec, HeterogeneousPlatform
from simulator.list_scheduling import (
    DEFAULT_MODE_SWITCH_POWER_OVERHEAD,
    DEFAULT_MODE_SWITCH_TIME_OVERHEAD,
    DecodedSchedule,
    _estimated_budget_for_mode,
    _mixed_criticality_runtime_work,
    compute_sub_deadlines,
    evaluate_decoded_schedule,
)


@dataclass(slots=True)
class VdEdfConfig:
    """Configuration for the operational EDF-VD baseline.

    If ``hi_virtual_deadline_factor`` is ``None``, the scheduler computes the
    EDF-VD scaling factor ``x`` from the RTSS'16 IMC utilization test.  Providing
    a numeric value keeps a manual override for ablation runs.
    """

    hi_virtual_deadline_factor: float | None = None
    fallback_hi_virtual_deadline_factor: float = 0.5
    mode_switch_time_overhead: float = DEFAULT_MODE_SWITCH_TIME_OVERHEAD
    mode_switch_power_overhead: float = DEFAULT_MODE_SWITCH_POWER_OVERHEAD


class VdEdfScheduler:
    """EDF-VD baseline with IMC degraded-quality semantics.

    The uploaded RTSS'16 paper studies EDF-VD for imprecise mixed-criticality
    systems: HI tasks use tightened virtual deadlines in LO mode, and LO tasks
    are not automatically discarded after a mode switch.  Instead, LO tasks keep
    running in HI mode with reduced execution budgets.  This implementation keeps
    that budget semantics and adapts it to the project's dependency-aware DAG and
    heterogeneous multicore simulator.
    """

    def __init__(self, config: VdEdfConfig | None = None) -> None:
        """Initialize the object with validated configuration and runtime state."""
        self.config = config or VdEdfConfig()

    def schedule(
        self,
        application: DagApplication,
        platform: HeterogeneousPlatform,
        rng_seed: int = 0,
    ) -> ScheduleMetrics:
        """Schedules one application using dependency-aware EDF-VD dispatch."""

        graph = application.clone_graph()
        decoded = self._simulate(graph=graph, platform=platform, rng_seed=rng_seed)
        return evaluate_decoded_schedule(graph=graph, platform=platform, decoded=decoded)

    def _simulate(self, graph: TaskGraph, platform: HeterogeneousPlatform, rng_seed: int) -> DecodedSchedule:
        """Runs the EDF-VD-inspired event simulation.

        A single LO->HI transition is triggered by the first HI task whose sampled
        demand exceeds its LO budget.  After that transition, HI tasks use their
        HI budgets and LO tasks use their reduced HI-mode budgets.  LO tasks are
        only marked as service loss when their reduced HI-mode budget is exactly
        zero, which is the classical-MC special case.
        """

        rng = build_rng(rng_seed)
        decoded = DecodedSchedule()
        sub_deadlines = compute_sub_deadlines(graph)
        deadline_scaling_factor = self._deadline_scaling_factor(graph)
        completed: Set[int] = set()
        excluded: Set[int] = set()
        core_available = {core.core_id: 0.0 for core in platform.cores}
        current_time = 0.0
        hi_mode = False

        while len(completed) + len(excluded) < len(graph.nodes):
            if hi_mode:
                self._drop_zero_budget_lo_tasks(
                    graph=graph,
                    decoded=decoded,
                    completed=completed,
                    excluded=excluded,
                    current_time=current_time,
                )
                if len(completed) + len(excluded) >= len(graph.nodes):
                    break

            ready = self._ready_tasks(graph=graph, completed=completed, excluded=excluded, hi_mode=hi_mode)
            if not ready:
                raise RuntimeError("EDF-VD found no ready task under the current mixed-criticality mode.")

            ready.sort(
                key=lambda task_id: self._effective_deadline(
                    task_id=task_id,
                    graph=graph,
                    sub_deadlines=sub_deadlines,
                    hi_mode=hi_mode,
                    deadline_scaling_factor=deadline_scaling_factor,
                )
            )
            selected_task = ready[0]
            node = graph.nodes[selected_task]

            (
                selected_core,
                selected_dvfs,
                estimated_duration,
                actual_duration,
                runtime_samples,
                triggered_mode_switch,
            ) = self._select_core_and_dvfs(
                graph=graph,
                platform=platform,
                task_id=selected_task,
                current_time=current_time,
                core_available=core_available,
                completed_finish_times=decoded.finish_times,
                assigned_cores=decoded.assigned_cores,
                hi_mode=hi_mode,
                rng=rng,
            )

            ready_time = graph.predecessor_finish_constraints(
                task_id=selected_task,
                finish_times=decoded.finish_times,
                assigned_cores=decoded.assigned_cores,
                target_core_id=selected_core.core_id,
            ) if node.predecessors else 0.0
            start_time = max(current_time, core_available[selected_core.core_id], ready_time)

            switch_time_overhead = 0.0
            switch_energy_overhead = 0.0
            if triggered_mode_switch and not hi_mode:
                hi_mode = True
                decoded.mode_switch_count += 1
                switch_time_overhead = max(0.0, self.config.mode_switch_time_overhead)
                switch_energy_overhead = switch_time_overhead * max(0.0, self.config.mode_switch_power_overhead)

            finish_time = start_time + actual_duration + switch_time_overhead

            entry = ScheduleEntry(
                task_id=selected_task,
                core_id=selected_core.core_id,
                dvfs_level_id=selected_dvfs,
                start_time=start_time,
                finish_time=finish_time,
                estimated_runtime=estimated_duration,
                actual_runtime=actual_duration,
                criticality=node.criticality,
                mode_switch_time_overhead=switch_time_overhead,
                mode_switch_energy_overhead=switch_energy_overhead,
            )
            decoded.entries.append(entry)
            decoded.start_times[selected_task] = start_time
            decoded.finish_times[selected_task] = finish_time
            decoded.assigned_cores[selected_task] = selected_core.core_id
            decoded.assigned_dvfs[selected_task] = selected_dvfs
            decoded.durations[selected_task] = actual_duration
            decoded.runtime_samples_by_task[selected_task] = runtime_samples
            core_available[selected_core.core_id] = finish_time
            completed.add(selected_task)
            current_time = min(core_available.values())

        decoded.entries.sort(key=lambda item: item.start_time)
        return decoded

    def _deadline_scaling_factor(self, graph: TaskGraph) -> float:
        """Computes the EDF-VD virtual-deadline factor from the RTSS'16 test."""

        if self.config.hi_virtual_deadline_factor is not None:
            return min(1.0, max(1e-6, float(self.config.hi_virtual_deadline_factor)))

        period = graph.period or graph.deadline or graph.critical_path_length(use_hi_wcet=False)
        period = max(float(period), 1e-9)
        u_lo_lo = 0.0
        u_hi_lo = 0.0
        u_lo_hi = 0.0
        u_hi_hi = 0.0
        for task_id in graph.real_task_ids:
            node = graph.nodes[task_id]
            if node.criticality == "HI":
                u_lo_hi += node.lo_wcet / period
                u_hi_hi += max(node.hi_wcet, node.lo_wcet) / period
            else:
                u_lo_lo += node.lo_wcet / period
                u_hi_lo += min(node.lo_wcet, max(0.0, node.hi_wcet)) / period

        # Theorem 3, Eq. (8): ordinary EDF is enough; no tightening is needed.
        if u_hi_hi + u_lo_lo <= 1.0:
            return 1.0

        denominator_lower = 1.0 - u_lo_lo
        denominator_upper = u_lo_lo - u_hi_lo
        if denominator_lower <= 1e-12 or denominator_upper <= 1e-12:
            return min(1.0, max(1e-6, self.config.fallback_hi_virtual_deadline_factor))

        lower = u_lo_hi / denominator_lower
        upper = (1.0 - (u_hi_hi + u_hi_lo)) / denominator_upper
        if not (math.isfinite(lower) and math.isfinite(upper)):
            return min(1.0, max(1e-6, self.config.fallback_hi_virtual_deadline_factor))
        if u_hi_hi + u_hi_lo >= 1.0 or u_lo_lo >= 1.0 or u_lo_lo <= u_hi_lo or lower > upper:
            return min(1.0, max(1e-6, self.config.fallback_hi_virtual_deadline_factor))
        return min(1.0, max(1e-6, 0.5 * (lower + upper)))

    @staticmethod
    def _drop_zero_budget_lo_tasks(
        graph: TaskGraph,
        decoded: DecodedSchedule,
        completed: Set[int],
        excluded: Set[int],
        current_time: float,
    ) -> None:
        """Drops LO tasks only when their HI-mode budget is zero."""

        changed = True
        while changed:
            changed = False
            for task_id in graph.topological_order():
                if task_id in completed or task_id in excluded:
                    continue
                task_node = graph.nodes[task_id]
                if task_node.criticality != "LO" or task_node.is_dummy or task_node.hi_wcet > 0.0:
                    continue
                exclusion_time = max(
                    (
                        decoded.finish_times.get(predecessor_id, current_time)
                        for predecessor_id in task_node.predecessors
                        if predecessor_id in completed or predecessor_id in excluded
                    ),
                    default=current_time,
                )
                excluded.add(task_id)
                decoded.dropped_lo_tasks.add(task_id)
                decoded.start_times[task_id] = exclusion_time
                decoded.finish_times[task_id] = exclusion_time
                decoded.durations[task_id] = 0.0
                decoded.runtime_samples_by_task[task_id] = [0.0]
                changed = True

    def _effective_deadline(
        self,
        task_id: int,
        graph: TaskGraph,
        sub_deadlines: Dict[int, float],
        hi_mode: bool,
        deadline_scaling_factor: float,
    ) -> float:
        """Returns effective EDF-VD deadline for one task."""

        node = graph.nodes[task_id]
        deadline = sub_deadlines[task_id]
        if node.criticality == "HI" and not hi_mode and not node.is_dummy:
            return deadline_scaling_factor * deadline
        return deadline

    @staticmethod
    def _ready_tasks(graph: TaskGraph, completed: Set[int], excluded: Set[int], hi_mode: bool) -> List[int]:
        """Returns ready tasks under the current system mode."""

        ready: List[int] = []
        for task_id in graph.topological_order():
            if task_id in completed or task_id in excluded:
                continue
            node = graph.nodes[task_id]
            if hi_mode and node.criticality == "LO" and not node.is_dummy and node.hi_wcet <= 0.0:
                continue
            if all(predecessor_id in completed or predecessor_id in excluded for predecessor_id in node.predecessors):
                ready.append(task_id)
        return ready

    def _select_core_and_dvfs(
        self,
        graph: TaskGraph,
        platform: HeterogeneousPlatform,
        task_id: int,
        current_time: float,
        core_available: Dict[int, float],
        completed_finish_times: Dict[int, float],
        assigned_cores: Dict[int, int],
        hi_mode: bool,
        rng,
    ) -> Tuple[CoreSpec, int, float, float, List[float], bool]:
        """Selects a core and DVFS level for the next EDF-VD dispatch."""

        node = graph.nodes[task_id]
        if node.is_dummy:
            return platform.cores[0], 0, 0.0, 0.0, [0.0], False

        sampled_runtime_value = node.runtime_distribution.sample(rng) if node.runtime_distribution is not None else node.lo_wcet
        bounded_work, _, triggers_mode_switch = _mixed_criticality_runtime_work(
            node=node,
            sampled_work=sampled_runtime_value,
            hi_mode=hi_mode,
        )
        estimated_budget = _estimated_budget_for_mode(node=node, use_predictions=False, hi_mode=hi_mode)
        raw_runtime_samples = node.runtime_samples(rng=rng, num_samples=64)
        best_option = None

        for core in platform.cores:
            ready_time = graph.predecessor_finish_constraints(
                task_id=task_id,
                finish_times=completed_finish_times,
                assigned_cores=assigned_cores,
                target_core_id=core.core_id,
            ) if node.predecessors else 0.0
            start_time = max(current_time, core_available[core.core_id], ready_time)
            for dvfs_level in core.dvfs_levels:
                estimated_duration = core.execution_time(estimated_budget, dvfs_level.level_id)
                actual_duration = core.execution_time(bounded_work, dvfs_level.level_id)
                switch_overhead = self.config.mode_switch_time_overhead if triggers_mode_switch and not hi_mode else 0.0
                finish_time = start_time + actual_duration + switch_overhead
                energy = core.energy_for_runtime(runtime=actual_duration, dvfs_level_id=dvfs_level.level_id)
                if switch_overhead > 0.0:
                    energy += switch_overhead * max(0.0, self.config.mode_switch_power_overhead)
                candidate = (finish_time, energy, start_time, core.core_id, dvfs_level.level_id, estimated_duration, actual_duration)
                if best_option is None or candidate < best_option:
                    best_option = candidate

        if best_option is None:
            raise RuntimeError(f"EDF-VD could not assign task {task_id}.")
        _, _, _, core_id, dvfs_level_id, estimated_duration, actual_duration = best_option
        selected_core = platform.core_by_id(core_id)
        runtime_samples = [
            selected_core.execution_time(max(0.0, sample), dvfs_level_id)
            for sample in raw_runtime_samples
        ]
        return selected_core, dvfs_level_id, estimated_duration, actual_duration, runtime_samples, triggers_mode_switch
