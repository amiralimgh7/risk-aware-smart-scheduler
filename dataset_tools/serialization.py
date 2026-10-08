"""Dataset serialization and memory-safe spreadsheet helpers.

This module is part of the ESRLab defense-ready codebase. It uses snake_case for functions and variables while keeping Python classes in PascalCase, and exposes documented helpers for reproducible experiments.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from core.task_graph import DagApplication, RuntimeDistribution, TaskGraph, TaskNode


def runtime_distribution_to_dict(distribution: RuntimeDistribution | None) -> Dict[str, Any] | None:
    """Serializes a runtime distribution to a JSON-compatible dictionary."""

    if distribution is None:
        return None
    return {
        "mean_reference": distribution.mean_reference,
        "sigma": distribution.sigma,
    }


def runtime_distribution_from_dict(payload: Dict[str, Any] | None) -> RuntimeDistribution | None:
    """Deserializes a runtime distribution from a dictionary."""

    if payload is None:
        return None
    return RuntimeDistribution(
        mean_reference=float(payload["mean_reference"]),
        sigma=float(payload["sigma"]),
    )


def task_graph_to_dict(graph: TaskGraph) -> Dict[str, Any]:
    """Serializes a task graph to a JSON-compatible dictionary."""

    nodes: List[Dict[str, Any]] = []
    for task_id in graph.topological_order():
        node = graph.nodes[task_id]
        nodes.append(
            {
                "task_id": node.task_id,
                "level": node.level,
                "data_size": node.data_size,
                "alpha": node.alpha,
                "complexity_mode": node.complexity_mode,
                "raw_cost": node.raw_cost,
                "criticality": node.criticality,
                "lo_wcet": node.lo_wcet,
                "hi_wcet": node.hi_wcet,
                "predicted_lo_wcet": node.predicted_lo_wcet,
                "predicted_uncertainty": node.predicted_uncertainty,
                "runtime_distribution": runtime_distribution_to_dict(node.runtime_distribution),
                "predecessors": sorted(node.predecessors),
                "successors": sorted(node.successors),
                "communication_costs": {
                    str(child_id): node.communication_costs[child_id]
                    for child_id in sorted(node.communication_costs.keys())
                },
                "is_dummy": node.is_dummy,
                "metadata": node.metadata,
            }
        )

    return {
        "graph_id": graph.graph_id,
        "source_id": graph.source_id,
        "sink_id": graph.sink_id,
        "deadline": graph.deadline,
        "period": graph.period,
        "levels": graph.levels,
        "nodes": nodes,
    }


def task_graph_from_dict(payload: Dict[str, Any]) -> TaskGraph:
    """Deserializes a task graph from a dictionary."""

    graph = TaskGraph(
        graph_id=str(payload["graph_id"]),
        source_id=payload.get("source_id"),
        sink_id=payload.get("sink_id"),
        deadline=payload.get("deadline"),
        period=payload.get("period"),
    )

    nodes_payload = payload["nodes"]
    for node_payload in nodes_payload:
        graph.add_node(
            TaskNode(
                task_id=int(node_payload["task_id"]),
                level=int(node_payload["level"]),
                data_size=int(node_payload["data_size"]),
                alpha=float(node_payload["alpha"]),
                complexity_mode=int(node_payload["complexity_mode"]),
                raw_cost=float(node_payload["raw_cost"]),
                criticality=str(node_payload["criticality"]),
                lo_wcet=float(node_payload["lo_wcet"]),
                hi_wcet=float(node_payload["hi_wcet"]),
                predicted_lo_wcet=float(node_payload.get("predicted_lo_wcet", 0.0)),
                predicted_uncertainty=float(node_payload.get("predicted_uncertainty", 0.0)),
                runtime_distribution=runtime_distribution_from_dict(node_payload.get("runtime_distribution")),
                is_dummy=bool(node_payload.get("is_dummy", False)),
                metadata={
                    str(key): float(value) if isinstance(value, (int, float)) else value
                    for key, value in dict(node_payload.get("metadata", {})).items()
                },
            )
        )

    for node_payload in nodes_payload:
        src_id = int(node_payload["task_id"])
        for dst_id_str, communication_cost in dict(node_payload.get("communication_costs", {})).items():
            graph.add_edge(src_id=src_id, dst_id=int(dst_id_str), communication_cost=float(communication_cost))

    return graph


def dag_application_to_dict(application: DagApplication) -> Dict[str, Any]:
    """Serializes an application to a JSON-compatible dictionary."""

    return {
        "application_id": application.application_id,
        "period": application.period,
        "deadline": application.deadline,
        "total_utilization": application.total_utilization,
        "graph": task_graph_to_dict(application.graph),
    }


def dag_application_from_dict(payload: Dict[str, Any]) -> DagApplication:
    """Deserializes an application from a dictionary."""

    return DagApplication(
        application_id=str(payload["application_id"]),
        graph=task_graph_from_dict(payload["graph"]),
        period=float(payload["period"]),
        deadline=float(payload["deadline"]),
        total_utilization=float(payload["total_utilization"]),
    )


def save_application_json(application: DagApplication, output_path: str | Path) -> None:
    """Writes one application JSON file."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(dag_application_to_dict(application), file, ensure_ascii=False, indent=2)


def load_application_json(input_path: str | Path) -> DagApplication:
    """Loads one application JSON file."""

    path = Path(input_path)
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return dag_application_from_dict(payload)


def save_json(payload: Dict[str, Any], output_path: str | Path) -> None:
    """Writes an arbitrary JSON payload."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def load_json(input_path: str | Path) -> Dict[str, Any]:
    """Loads an arbitrary JSON payload."""

    path = Path(input_path)
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return dict(payload)
