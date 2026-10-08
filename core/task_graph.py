"""Shared configuration, task graph data structures, metrics, and random utilities.

This module is part of the ESRLab defense-ready codebase. It uses snake_case for functions and variables while keeping Python classes in PascalCase, and exposes documented helpers for reproducible experiments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import math
import random


@dataclass(slots=True)
class RuntimeDistribution:
    """Log-normal runtime distribution for one task.

    ``mean_reference`` is the nominal LO estimate used to build the log-normal
    distribution. ``sigma`` controls uncertainty and tail heaviness.
    """

    mean_reference: float
    sigma: float

    def sample(self, rng: random.Random) -> float:
        """Samples one runtime value."""

        mean_reference = max(self.mean_reference, 1e-12)
        sigma = max(self.sigma, 1e-9)
        mu = math.log(mean_reference) - 0.5 * (sigma ** 2)
        return max(1e-12, rng.lognormvariate(mu, sigma))

    def sample_many(self, rng: random.Random, num_samples: int) -> List[float]:
        """Samples multiple runtime values."""

        return [self.sample(rng) for _ in range(num_samples)]


@dataclass(slots=True)
class TaskNode:
    """Represents one node in a DAG task graph."""

    task_id: int
    level: int
    data_size: int
    alpha: float
    complexity_mode: int
    raw_cost: float
    criticality: str = "LO"
    lo_wcet: float = 0.0
    hi_wcet: float = 0.0
    predicted_lo_wcet: float = 0.0
    predicted_uncertainty: float = 0.0
    runtime_distribution: Optional[RuntimeDistribution] = None
    predecessors: Set[int] = field(default_factory=set)
    successors: Set[int] = field(default_factory=set)
    communication_costs: Dict[int, float] = field(default_factory=dict)
    is_dummy: bool = False
    metadata: Dict[str, float] = field(default_factory=dict)

    def add_successor(self, successor_id: int, communication_cost: float) -> None:
        """Adds a successor with communication cost."""

        self.successors.add(successor_id)
        self.communication_costs[successor_id] = communication_cost

    def add_predecessor(self, predecessor_id: int) -> None:
        """Adds a predecessor relation."""

        self.predecessors.add(predecessor_id)

    def runtime_samples(self, rng: random.Random, num_samples: int) -> List[float]:
        """Returns runtime samples from the node distribution.

        Dummy nodes return zero-valued samples.
        """

        if self.is_dummy:
            return [0.0 for _ in range(num_samples)]
        if self.runtime_distribution is None:
            return [self.lo_wcet for _ in range(num_samples)]
        return self.runtime_distribution.sample_many(rng=rng, num_samples=num_samples)


@dataclass(slots=True)
class TaskGraph:
    """Represents a directed acyclic task graph."""

    graph_id: str
    nodes: Dict[int, TaskNode] = field(default_factory=dict)
    levels: List[List[int]] = field(default_factory=list)
    source_id: Optional[int] = None
    sink_id: Optional[int] = None
    deadline: Optional[float] = None
    period: Optional[float] = None

    def add_node(self, node: TaskNode) -> None:
        """Adds a node to the graph."""

        if node.task_id in self.nodes:
            raise ValueError(f"Duplicate node id: {node.task_id}")
        self.nodes[node.task_id] = node
        while len(self.levels) <= node.level:
            self.levels.append([])
        self.levels[node.level].append(node.task_id)

    def add_edge(self, src_id: int, dst_id: int, communication_cost: float) -> None:
        """Adds a directed edge between two existing nodes."""

        if src_id == dst_id:
            raise ValueError("Self loops are not allowed.")
        if src_id not in self.nodes or dst_id not in self.nodes:
            raise KeyError("Both nodes must exist before an edge can be added.")
        self.nodes[src_id].add_successor(dst_id, communication_cost)
        self.nodes[dst_id].add_predecessor(src_id)

    @property
    def real_task_ids(self) -> List[int]:
        """Returns ids of non-dummy nodes."""

        return [task_id for task_id, node in self.nodes.items() if not node.is_dummy]

    def topological_order(self) -> List[int]:
        """Returns a level-based topological order."""

        ordered: List[int] = []
        for level in self.levels:
            ordered.extend(level)
        return ordered

    def validate_acyclic(self) -> bool:
        """Checks whether the graph is acyclic using a Kahn traversal."""

        indegrees = {task_id: len(node.predecessors) for task_id, node in self.nodes.items()}
        ready = [task_id for task_id, degree in indegrees.items() if degree == 0]
        visited = 0

        while ready:
            current = ready.pop()
            visited += 1
            for successor_id in self.nodes[current].successors:
                indegrees[successor_id] -= 1
                if indegrees[successor_id] == 0:
                    ready.append(successor_id)

        return visited == len(self.nodes)

    def edge_count(self) -> int:
        """Returns the total number of directed edges."""

        return sum(len(node.successors) for node in self.nodes.values())

    def indegree(self, task_id: int) -> int:
        """Returns the indegree of a node."""

        return len(self.nodes[task_id].predecessors)

    def outdegree(self, task_id: int) -> int:
        """Returns the outdegree of a node."""

        return len(self.nodes[task_id].successors)

    def level_widths(self) -> List[int]:
        """Returns widths of all levels."""

        return [len(level) for level in self.levels]

    def normalized_level(self, task_id: int) -> float:
        """Returns the node level normalized to [0, 1]."""

        if len(self.levels) <= 1:
            return 0.0
        return self.nodes[task_id].level / (len(self.levels) - 1)

    def critical_path_lengths(self, use_hi_wcet: bool = False) -> Dict[int, float]:
        """Returns longest path lengths from each node to sink."""

        order = list(reversed(self.topological_order()))
        longest: Dict[int, float] = {}

        for task_id in order:
            node = self.nodes[task_id]
            node_cost = node.hi_wcet if use_hi_wcet else node.lo_wcet
            if not node.successors:
                longest[task_id] = node_cost
                continue
            longest[task_id] = node_cost + max(
                node.communication_costs.get(successor_id, 0.0) + longest[successor_id]
                for successor_id in node.successors
            )
        return longest

    def critical_path_length(self, use_hi_wcet: bool = False) -> float:
        """Returns the graph critical path length."""

        longest = self.critical_path_lengths(use_hi_wcet=use_hi_wcet)
        if self.source_id is not None:
            return longest[self.source_id]
        roots = [task_id for task_id, node in self.nodes.items() if len(node.predecessors) == 0]
        return max(longest[task_id] for task_id in roots)

    def communication_cost(self, src_id: int, dst_id: int) -> float:
        """Returns communication cost of one edge."""

        return self.nodes[src_id].communication_costs.get(dst_id, 0.0)

    def predecessor_finish_constraints(
        self,
        task_id: int,
        finish_times: Dict[int, float],
        assigned_cores: Dict[int, int],
        target_core_id: int,
    ) -> float:
        """Returns the predecessor-based earliest release time on a target core."""

        release_time = 0.0
        for predecessor_id in self.nodes[task_id].predecessors:
            finish_time = finish_times.get(predecessor_id, 0.0)
            communication_cost = 0.0
            if predecessor_id in assigned_cores and assigned_cores[predecessor_id] != target_core_id:
                communication_cost = self.communication_cost(predecessor_id, task_id)
            release_time = max(release_time, finish_time + communication_cost)
        return release_time


@dataclass(slots=True)
class DagApplication:
    """Represents one periodic DAG application."""

    application_id: str
    graph: TaskGraph
    period: float
    deadline: float
    total_utilization: float

    def clone_graph(self) -> TaskGraph:
        """Returns a deep-copy-like graph clone suitable for simulation."""

        cloned = TaskGraph(
            graph_id=self.graph.graph_id,
            source_id=self.graph.source_id,
            sink_id=self.graph.sink_id,
            deadline=self.graph.deadline,
            period=self.graph.period,
        )
        for node in self.graph.nodes.values():
            runtime_distribution = None
            if node.runtime_distribution is not None:
                runtime_distribution = RuntimeDistribution(
                    mean_reference=node.runtime_distribution.mean_reference,
                    sigma=node.runtime_distribution.sigma,
                )
            cloned.add_node(
                TaskNode(
                    task_id=node.task_id,
                    level=node.level,
                    data_size=node.data_size,
                    alpha=node.alpha,
                    complexity_mode=node.complexity_mode,
                    raw_cost=node.raw_cost,
                    criticality=node.criticality,
                    lo_wcet=node.lo_wcet,
                    hi_wcet=node.hi_wcet,
                    predicted_lo_wcet=node.predicted_lo_wcet,
                    predicted_uncertainty=node.predicted_uncertainty,
                    runtime_distribution=runtime_distribution,
                    is_dummy=node.is_dummy,
                    metadata=dict(node.metadata),
                )
            )
        for src_id, src_node in self.graph.nodes.items():
            for dst_id in src_node.successors:
                cloned.add_edge(src_id, dst_id, src_node.communication_costs[dst_id])
        return cloned


ScheduleTuple = Tuple[int, int, int]
