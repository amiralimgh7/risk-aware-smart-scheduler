"""Classical and learned baseline schedulers used for comparison.

This module is part of the ESRLab defense-ready codebase. It uses snake_case for functions and variables while keeping Python classes in PascalCase, and exposes documented helpers for reproducible experiments.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import torch

from core.config import TrainingConfig
from core.metrics import ScheduleMetrics
from core.task_graph import DagApplication, TaskGraph
from dataset_tools.serialization import load_json, save_json
from hardware_model.hardware import HeterogeneousPlatform
from models.gcn_model import ClassicGcnRegressor
from models.graph_utils import build_graph_tensors
from models.quantization import benchmark_callable_latency_ms, count_parameters, estimate_model_size_bytes
from simulator.list_scheduling import compute_sub_deadlines, decode_priority_schedule, evaluate_decoded_schedule
from training.supervised import SupervisedTrainer, apply_gcn_predictions


class ClassicGcnBaseline:
    """Classic GCN baseline used for comparison with the proposed GAT model."""

    def __init__(
        self,
        in_features: int = 11,
        hidden_features: int = 32,
        embedding_features: int = 32,
        device: str = "cpu",
    ) -> None:
        """Initialize the object with validated configuration and runtime state."""
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.embedding_features = embedding_features
        self.model = ClassicGcnRegressor(
            in_features=in_features,
            hidden_features=hidden_features,
            embedding_features=embedding_features,
        )
        self.trainer = SupervisedTrainer(device=device)
        self.device = device

    def fit(self, applications: List[DagApplication], training_config: TrainingConfig, verbose: bool = True):
        """Trains the GCN model on the provided dataset and returns its loss history."""

        return self.trainer.train_gcn(
            model=self.model,
            applications=applications,
            config=training_config,
            verbose=verbose,
            progress_prefix="GNN",
        )

    def save_checkpoint(self, checkpoint_dir: str | Path) -> None:
        """Saves the trained GNN baseline."""

        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), checkpoint_dir / "model_state.pt")
        save_json(
            {
                "in_features": self.in_features,
                "hidden_features": self.hidden_features,
                "embedding_features": self.embedding_features,
                "device": self.device,
                "parameter_count": count_parameters(self.model),
                "model_size_bytes": estimate_model_size_bytes(self.model),
                "algorithm": "gnn",
            },
            checkpoint_dir / "metadata.json",
        )

    @classmethod
    def load_checkpoint(cls, checkpoint_dir: str | Path, device: str = "cpu") -> "ClassicGcnBaseline":
        """Loads a trained GNN baseline from disk."""

        checkpoint_dir = Path(checkpoint_dir)
        metadata = load_json(checkpoint_dir / "metadata.json")
        baseline = cls(
            in_features=int(metadata["in_features"]),
            hidden_features=int(metadata["hidden_features"]),
            embedding_features=int(metadata["embedding_features"]),
            device=device,
        )
        state_dict = torch.load(checkpoint_dir / "model_state.pt", map_location=device)
        baseline.model.load_state_dict(state_dict)
        baseline.model.to(device)
        baseline.model.eval()
        return baseline

    def parameter_count(self) -> int:
        """Returns the number of model parameters."""

        return count_parameters(self.model)

    def model_size_bytes(self) -> int:
        """Returns the serialized model size in bytes."""

        return estimate_model_size_bytes(self.model)

    @torch.no_grad()
    def benchmark_predictor_latency_ms(
        self,
        application: DagApplication,
        warmup_runs: int = 1,
        timed_runs: int = 5,
    ) -> float:
        """Benchmarks predictor inference latency in milliseconds."""

        graph_tensors = build_graph_tensors(application.graph)

        def predictor_forward() -> object:
            """Run the predictor forward step and return its computed result."""
            return self.model(
                node_features=graph_tensors.node_features.to(self.device),
                adjacency=graph_tensors.adjacency.to(self.device),
            )

        return benchmark_callable_latency_ms(
            predictor_forward,
            warmup_runs=warmup_runs,
            timed_runs=timed_runs,
        )

    @torch.no_grad()
    def benchmark_schedule_latency_ms(
        self,
        application: DagApplication,
        platform: HeterogeneousPlatform,
        warmup_runs: int = 1,
        timed_runs: int = 3,
    ) -> float:
        """Benchmarks end-to-end deterministic scheduling latency."""

        def schedule_forward() -> object:
            """Run the schedule forward step and return its computed result."""
            return self.schedule(application=application, platform=platform)

        return benchmark_callable_latency_ms(
            schedule_forward,
            warmup_runs=warmup_runs,
            timed_runs=timed_runs,
        )

    def schedule(
        self,
        application: DagApplication,
        platform: HeterogeneousPlatform,
        rng_seed: int = 0,
    ) -> ScheduleMetrics:
        """Schedules one graph using the trained GCN predictor and a greedy decoder."""

        graph = application.clone_graph()
        apply_gcn_predictions(model=self.model, graph=graph, device=self.device)
        priority_scores = self._compute_predicted_upward_ranks(graph=graph, platform=platform)
        core_assignments, dvfs_assignments = self._assign_greedily(
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
            use_predictions=True,
        )
        return evaluate_decoded_schedule(graph=graph, platform=platform, decoded=decoded)

    @staticmethod
    def _compute_predicted_upward_ranks(graph: TaskGraph, platform: HeterogeneousPlatform) -> Dict[int, float]:
        """Computes upward ranks using predicted LO-WCET values."""

        def average_compute(task_id: int) -> float:
            """Run the average compute step and return its computed result."""
            node = graph.nodes[task_id]
            values: List[float] = []
            for core in platform.cores:
                dvfs_level = max(core.dvfs_levels, key=lambda item: item.frequency_ghz)
                values.append(core.execution_time(node.predicted_lo_wcet, dvfs_level.level_id))
            return sum(values) / len(values)

        rank_u: Dict[int, float] = {}
        for task_id in reversed(graph.topological_order()):
            successors = list(graph.nodes[task_id].successors)
            if not successors:
                rank_u[task_id] = average_compute(task_id)
            else:
                rank_u[task_id] = average_compute(task_id) + max(
                    graph.communication_cost(task_id, successor_id) + rank_u[successor_id]
                    for successor_id in successors
                )
        return rank_u

    @staticmethod
    def _assign_greedily(
        graph: TaskGraph,
        platform: HeterogeneousPlatform,
        priority_scores: Dict[int, float],
    ) -> Tuple[Dict[int, int], Dict[int, int]]:
        """Assigns cores and DVFS levels greedily with predicted sub-deadlines."""

        sub_deadlines = compute_sub_deadlines(graph)
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
                    ready_time = graph.predecessor_finish_constraints(
                        task_id=task_id,
                        finish_times=finish_times,
                        assigned_cores=core_assignments,
                        target_core_id=core.core_id,
                    ) if node.predecessors else 0.0
                    for level in core.dvfs_levels:
                        start_time = max(core_available[core.core_id], ready_time)
                        predicted_duration = core.execution_time(node.predicted_lo_wcet, level.level_id)
                        finish_time = start_time + predicted_duration
                        energy = core.energy_for_runtime(runtime=predicted_duration, dvfs_level_id=level.level_id)
                        deadline_violation = max(0.0, finish_time - sub_deadlines[task_id])
                        candidate = (deadline_violation, finish_time, energy, core.core_id, level.level_id)
                        if best_option is None or candidate < best_option:
                            best_option = candidate
                if best_option is None:
                    raise RuntimeError(f"GCN baseline could not assign task {task_id}.")
                _, finish_time, _, core_id, level_id = best_option
                core_assignments[task_id] = core_id
                dvfs_assignments[task_id] = level_id
                finish_times[task_id] = finish_time
                core_available[core_id] = finish_time
                pending.remove(task_id)
                progress_made = True
            if not progress_made:
                raise RuntimeError("GCN greedy assignment loop made no progress.")
        return core_assignments, dvfs_assignments
