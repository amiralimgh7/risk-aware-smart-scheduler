"""Classical and learned baseline schedulers used for comparison.

This module is part of the ESRLab defense-ready codebase. It uses snake_case for functions and variables while keeping Python classes in PascalCase, and exposes documented helpers for reproducible experiments.
"""
from __future__ import annotations

from typing import Dict, Tuple

from core.metrics import ScheduleMetrics
from core.task_graph import DagApplication, TaskGraph
from hardware_model.hardware import HeterogeneousPlatform
from simulator.list_scheduling import decode_priority_schedule, evaluate_decoded_schedule


class HeftScheduler:
    """Canonical HEFT-style baseline for heterogeneous DAG scheduling."""

    def schedule(
        self,
        application: DagApplication,
        platform: HeterogeneousPlatform,
        rng_seed: int = 0,
    ) -> ScheduleMetrics:
        """Schedules one application using HEFT ranks and EFT selection."""

        graph = application.clone_graph()
        priority_scores = self._compute_upward_ranks(graph=graph, platform=platform)
        core_assignments, dvfs_assignments = self._assign_cores(
            graph=graph,
            platform=platform,
            priority_scores=priority_scores,
        )
        decoded = decode_priority_schedule(
            graph=graph,
            platform=platform,
            priority_scores=priority_scores,
            core_assignments=core_assignments,
            dvfs_assignments=dvfs_assignments,
            rng_seed=rng_seed,
        )
        return evaluate_decoded_schedule(graph=graph, platform=platform, decoded=decoded)

    @staticmethod
    def _compute_upward_ranks(graph: TaskGraph, platform: HeterogeneousPlatform) -> Dict[int, float]:
        """Computes HEFT upward ranks using average computation cost."""

        def average_compute(task_id: int) -> float:
            """Run the average compute step and return its computed result."""
            node = graph.nodes[task_id]
            values = []
            for core in platform.cores:
                dvfs_level = max(core.dvfs_levels, key=lambda item: item.frequency_ghz)
                values.append(core.execution_time(node.lo_wcet, dvfs_level.level_id))
            return sum(values) / len(values)

        rank_u: Dict[int, float] = {}
        for task_id in reversed(graph.topological_order()):
            node = graph.nodes[task_id]
            successors = list(node.successors)
            if not successors:
                rank_u[task_id] = average_compute(task_id)
            else:
                rank_u[task_id] = average_compute(task_id) + max(
                    node.communication_costs.get(successor_id, 0.0) + rank_u[successor_id]
                    for successor_id in successors
                )
        return rank_u

    @staticmethod
    def _assign_cores(
        graph: TaskGraph,
        platform: HeterogeneousPlatform,
        priority_scores: Dict[int, float],
    ) -> Tuple[Dict[int, int], Dict[int, int]]:
        """Assigns each task to the core yielding earliest finish time."""

        task_order = sorted(graph.nodes.keys(), key=lambda task_id: priority_scores[task_id], reverse=True)
        core_assignments: Dict[int, int] = {}
        dvfs_assignments: Dict[int, int] = {}
        finish_times: Dict[int, float] = {}
        core_available = {core.core_id: 0.0 for core in platform.cores}

        pending = list(task_order)
        while pending:
            progress_made = False
            for task_id in list(pending):
                node = graph.nodes[task_id]
                if any(predecessor_id not in finish_times for predecessor_id in node.predecessors):
                    continue
                best_option = None
                for core in platform.cores:
                    dvfs_level_id = max(core.dvfs_levels, key=lambda item: item.frequency_ghz).level_id
                    ready_time = graph.predecessor_finish_constraints(
                        task_id=task_id,
                        finish_times=finish_times,
                        assigned_cores=core_assignments,
                        target_core_id=core.core_id,
                    ) if node.predecessors else 0.0
                    start_time = max(core_available[core.core_id], ready_time)
                    finish_time = start_time + core.execution_time(node.lo_wcet, dvfs_level_id)
                    candidate = (finish_time, core.core_id, dvfs_level_id)
                    if best_option is None or candidate < best_option:
                        best_option = candidate
                if best_option is None:
                    raise RuntimeError(f"HEFT could not find a slot for task {task_id}.")
                finish_time, core_id, dvfs_level_id = best_option
                core_assignments[task_id] = core_id
                dvfs_assignments[task_id] = dvfs_level_id
                finish_times[task_id] = finish_time
                core_available[core_id] = finish_time
                pending.remove(task_id)
                progress_made = True
            if not progress_made:
                raise RuntimeError("HEFT assignment loop made no progress; graph may be inconsistent.")
        return core_assignments, dvfs_assignments
