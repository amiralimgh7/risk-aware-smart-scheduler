"""GAT, GCN, graph tensor utilities, quantization, and quantization evaluation.

This module is part of the ESRLab defense-ready codebase. It uses snake_case for functions and variables while keeping Python classes in PascalCase, and exposes documented helpers for reproducible experiments.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from core.task_graph import TaskGraph


@dataclass(slots=True)
class GraphTensors:
    """Tensor package for one task graph."""

    task_ids: List[int]
    node_features: torch.Tensor
    adjacency: torch.Tensor
    target_lo_wcet: torch.Tensor
    target_sigma: torch.Tensor


def build_graph_tensors(graph: TaskGraph) -> GraphTensors:
    """Builds tensors from a task graph for GNN training and inference."""

    task_ids = graph.topological_order()
    num_nodes = len(task_ids)
    node_features: List[List[float]] = []
    target_lo: List[List[float]] = []
    target_sigma: List[List[float]] = []

    max_lo = max((graph.nodes[task_id].lo_wcet for task_id in task_ids), default=1.0)
    max_hi = max((graph.nodes[task_id].hi_wcet for task_id in task_ids), default=1.0)
    max_data = max((graph.nodes[task_id].data_size for task_id in task_ids), default=1)
    max_raw = max((graph.nodes[task_id].raw_cost for task_id in task_ids), default=1.0)

    index_of = {task_id: index for index, task_id in enumerate(task_ids)}
    adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

    for task_id in task_ids:
        node = graph.nodes[task_id]
        hi_flag = 1.0 if node.criticality == "HI" else 0.0
        dummy_flag = 1.0 if node.is_dummy else 0.0
        node_features.append(
            [
                float(node.level),
                graph.normalized_level(task_id),
                len(node.predecessors),
                len(node.successors),
                node.data_size / max(max_data, 1),
                node.alpha,
                float(node.complexity_mode),
                node.raw_cost / max(max_raw, 1e-9),
                node.hi_wcet / max(max_hi, 1e-9),
                hi_flag,
                dummy_flag,
            ]
        )
        target_lo.append([node.lo_wcet])
        sigma = 0.0 if node.runtime_distribution is None else node.runtime_distribution.sigma
        target_sigma.append([sigma])

        adjacency[index_of[task_id], index_of[task_id]] = 1.0
        for successor_id in node.successors:
            adjacency[index_of[task_id], index_of[successor_id]] = 1.0
            adjacency[index_of[successor_id], index_of[task_id]] = 1.0

    degree = torch.sum(adjacency, dim=1)
    inv_sqrt_degree = torch.diag(torch.pow(degree.clamp_min(1.0), -0.5))
    normalized_adjacency = inv_sqrt_degree @ adjacency @ inv_sqrt_degree

    return GraphTensors(
        task_ids=task_ids,
        node_features=torch.tensor(node_features, dtype=torch.float32),
        adjacency=normalized_adjacency,
        target_lo_wcet=torch.tensor(target_lo, dtype=torch.float32),
        target_sigma=torch.tensor(target_sigma, dtype=torch.float32),
    )
