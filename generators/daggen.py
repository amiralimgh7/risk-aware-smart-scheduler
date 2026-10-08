"""DAG, UUniFast utilization, and workload generation utilities.

This module is part of the ESRLab defense-ready codebase. It uses snake_case for functions and variables while keeping Python classes in PascalCase, and exposes documented helpers for reproducible experiments.
"""
from __future__ import annotations

import math
from typing import List, Optional, Set

from core.config import DaggenConfig
from core.random_utils import build_rng, get_int_random_number_around, get_random_number_between
from core.task_graph import TaskGraph, TaskNode


N_2 = 1
N_LOG_N = 2
N_3 = 3


class DaggenGenerator:
    """Generates a DAG following the standard DAGGEN design pattern.

    The generation pipeline is intentionally close to the usual DAGGEN family:
    1. Decide a target width and derive ideal tasks per level.
    2. Sample actual level sizes using the regularity parameter.
    3. Connect each node to one or more parents based on density.
    4. Assign computation and communication costs.
    5. Optionally add a single dummy source and sink.
    """

    def __init__(self, config: DaggenConfig) -> None:
        """Initialize the object with validated configuration and runtime state."""
        self.config = config
        self.rng = build_rng(config.seed)

    def generate(self, graph_id: str = "dag_0") -> TaskGraph:
        """Generates one task graph."""

        graph = TaskGraph(graph_id=graph_id)
        level_sizes = self._generate_level_sizes()
        level_task_ids: List[List[int]] = []
        next_task_id = 0

        for level_index, level_size in enumerate(level_sizes):
            current_level_ids: List[int] = []
            for _ in range(level_size):
                node = self._create_task_node(task_id=next_task_id, level=level_index)
                graph.add_node(node)
                current_level_ids.append(next_task_id)
                next_task_id += 1
            level_task_ids.append(current_level_ids)

        self._generate_dependencies(graph=graph, level_task_ids=level_task_ids)

        if self.config.add_dummy_source_sink:
            graph = self._add_dummy_source_sink(graph)

        if not graph.validate_acyclic():
            raise RuntimeError("Generated graph is cyclic, which should not happen.")

        return graph

    def _generate_level_sizes(self) -> List[int]:
        """Generates the number of tasks in each level."""

        perfect_tasks_per_level = max(1, int(self.config.n ** self.config.width))
        latitude_percent = 100.0 * (1.0 - self.config.regularity)

        level_sizes: List[int] = []
        total_tasks = 0
        while total_tasks < self.config.n:
            sampled = get_int_random_number_around(
                rng=self.rng,
                center=perfect_tasks_per_level,
                latitude_percent=latitude_percent,
            )
            sampled = min(sampled, self.config.n - total_tasks)
            level_sizes.append(sampled)
            total_tasks += sampled
        return level_sizes

    def _create_task_node(self, task_id: int, level: int) -> TaskNode:
        """Creates a single DAGGEN-style task node."""

        data_size = int(get_random_number_between(self.rng, self.config.min_data, self.config.max_data))
        data_size = max(1024, (data_size // 1024) * 1024)
        alpha = get_random_number_between(self.rng, self.config.min_alpha, self.config.max_alpha)
        operation_factor = get_random_number_between(self.rng, 64.0, 512.0)

        if self.config.ccr_mode == 0:
            complexity_mode = self.rng.randint(N_2, N_3)
        else:
            complexity_mode = self.config.ccr_mode

        raw_cost = self._compute_raw_cost(
            data_size=data_size,
            operation_factor=operation_factor,
            complexity_mode=complexity_mode,
        )

        return TaskNode(
            task_id=task_id,
            level=level,
            data_size=data_size,
            alpha=alpha,
            complexity_mode=complexity_mode,
            raw_cost=raw_cost,
        )

    @staticmethod
    def _compute_raw_cost(data_size: int, operation_factor: float, complexity_mode: int) -> float:
        """Computes the node raw cost from its size and complexity mode."""

        if complexity_mode == N_2:
            return operation_factor * (data_size ** 2)
        if complexity_mode == N_LOG_N:
            return 2.0 * operation_factor * (data_size ** 2) * math.log2(max(data_size, 2))
        if complexity_mode == N_3:
            return operation_factor * (data_size ** 3)
        raise ValueError(f"Unsupported complexity mode: {complexity_mode}")

    def _generate_dependencies(self, graph: TaskGraph, level_task_ids: List[List[int]]) -> None:
        """Generates parent-child relations based on density and jump."""

        for level_index in range(1, len(level_task_ids)):
            previous_level_size = len(level_task_ids[level_index - 1])
            max_density_parents = max(1.0, self.config.density * previous_level_size)

            for task_id in level_task_ids[level_index]:
                requested_parent_count = min(
                    previous_level_size,
                    max(1, int(round(get_random_number_between(self.rng, 1.0, max_density_parents + 1.0)))),
                )
                selected_parents: Set[int] = set()

                for _ in range(requested_parent_count):
                    parent_level_offset = int(
                        get_random_number_between(self.rng, 1.0, float(self.config.jump) + 1.0)
                    )
                    parent_level = max(0, level_index - parent_level_offset)
                    parent_level_task_ids = level_task_ids[parent_level]
                    parent_index = int(
                        get_random_number_between(self.rng, 0.0, float(len(parent_level_task_ids)))
                    )
                    parent_index = min(parent_index, len(parent_level_task_ids) - 1)

                    parent_id = self._pick_unique_parent(
                        parent_level_task_ids=parent_level_task_ids,
                        current_parent_index=parent_index,
                        selected_parents=selected_parents,
                    )
                    if parent_id is None:
                        continue

                    selected_parents.add(parent_id)
                    graph.add_edge(src_id=parent_id, dst_id=task_id, communication_cost=0.0)

        self._assign_communication_costs(graph)

    @staticmethod
    def _pick_unique_parent(
        parent_level_task_ids: List[int],
        current_parent_index: int,
        selected_parents: Set[int],
    ) -> Optional[int]:
        """Selects a unique parent by circular probing."""

        for _ in range(len(parent_level_task_ids)):
            candidate_id = parent_level_task_ids[current_parent_index]
            if candidate_id not in selected_parents:
                return candidate_id
            current_parent_index = (current_parent_index + 1) % len(parent_level_task_ids)
        return None

    @staticmethod
    def _assign_communication_costs(graph: TaskGraph) -> None:
        """Assigns moderate communication costs derived from DAGGEN alpha and data size."""

        for node in graph.nodes.values():
            if node.is_dummy:
                continue
            for child_id in list(node.successors):
                node.communication_costs[child_id] = float(max(0.0, node.alpha * (node.data_size / 1024.0)))

    def _add_dummy_source_sink(self, graph: TaskGraph) -> TaskGraph:
        """Adds one dummy source and one dummy sink node."""

        shifted_graph = TaskGraph(graph_id=graph.graph_id)
        for node in graph.nodes.values():
            shifted_graph.add_node(
                TaskNode(
                    task_id=node.task_id,
                    level=node.level + 1,
                    data_size=node.data_size,
                    alpha=node.alpha,
                    complexity_mode=node.complexity_mode,
                    raw_cost=node.raw_cost,
                    criticality=node.criticality,
                    lo_wcet=node.lo_wcet,
                    hi_wcet=node.hi_wcet,
                    predicted_lo_wcet=node.predicted_lo_wcet,
                    predicted_uncertainty=node.predicted_uncertainty,
                    runtime_distribution=node.runtime_distribution,
                    is_dummy=node.is_dummy,
                    metadata=dict(node.metadata),
                )
            )

        for src_id, src_node in graph.nodes.items():
            for dst_id in src_node.successors:
                shifted_graph.add_edge(
                    src_id=src_id,
                    dst_id=dst_id,
                    communication_cost=src_node.communication_costs[dst_id],
                )

        source_id = max(shifted_graph.nodes.keys()) + 1
        sink_id = source_id + 1
        source_node = TaskNode(
            task_id=source_id,
            level=0,
            data_size=0,
            alpha=0.0,
            complexity_mode=N_2,
            raw_cost=0.0,
            is_dummy=True,
        )
        shifted_graph.add_node(source_node)

        sink_level = max(node.level for node in shifted_graph.nodes.values()) + 1
        sink_node = TaskNode(
            task_id=sink_id,
            level=sink_level,
            data_size=0,
            alpha=0.0,
            complexity_mode=N_2,
            raw_cost=0.0,
            is_dummy=True,
        )
        shifted_graph.add_node(sink_node)

        for task_id, node in list(shifted_graph.nodes.items()):
            if node.is_dummy:
                continue
            if len(node.predecessors) == 0:
                shifted_graph.add_edge(src_id=source_id, dst_id=task_id, communication_cost=0.0)
            if len(node.successors) == 0:
                shifted_graph.add_edge(src_id=task_id, dst_id=sink_id, communication_cost=0.0)

        shifted_graph.source_id = source_id
        shifted_graph.sink_id = sink_id
        return shifted_graph
