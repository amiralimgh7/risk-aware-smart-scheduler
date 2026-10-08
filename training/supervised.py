"""Training loops for supervised predictors and reward-driven policies.

This module is part of the ESRLab defense-ready codebase. It uses snake_case for functions and variables while keeping Python classes in PascalCase, and exposes documented helpers for reproducible experiments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import torch
from torch import nn

from core.config import TrainingConfig
from core.task_graph import DagApplication, TaskGraph
from hardware_model.hardware import HeterogeneousPlatform
from models.gat_risk import RiskAwareGatRegressor
from models.gcn_model import ClassicGcnRegressor
from models.graph_utils import GraphTensors, build_graph_tensors
from risk_modeling.risk_models import gaussian_upper_cvar


@dataclass(slots=True)
class TrainingHistory:
    """Stores loss curves of a training run."""

    epoch_losses: List[float] = field(default_factory=list)


@dataclass(slots=True)
class GatTrainingHistory(TrainingHistory):
    """Stores detailed loss terms of GAT training.

    ``epoch_losses`` keeps the actual optimized objective, which can be negative
    for a Gaussian NLL term when the learned variance is small.  The defense
    chart uses ``prediction_mse_losses`` instead, because it is non-negative and
    comparable with the classic GNN MSE curve.
    """

    nll_losses: List[float] = field(default_factory=list)
    sigma_losses: List[float] = field(default_factory=list)
    cvar_losses: List[float] = field(default_factory=list)
    prediction_mse_losses: List[float] = field(default_factory=list)


class SupervisedTrainer:
    """Provides supervised training utilities for GCN and GAT models."""

    def __init__(self, device: str = "cpu") -> None:
        """Initialize the object with validated configuration and runtime state."""
        self.device = torch.device(device)

    def train_gcn(
        self,
        model: ClassicGcnRegressor,
        applications: Sequence[DagApplication],
        config: TrainingConfig,
        verbose: bool = True,
        progress_prefix: str = "GNN",
    ) -> TrainingHistory:
        """Trains the classic GCN regressor."""

        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        loss_fn = nn.MSELoss()
        history = TrainingHistory()

        for epoch_index in range(config.num_epochs):
            epoch_loss = 0.0
            for application in applications:
                graph_tensors = build_graph_tensors(application.graph)
                optimizer.zero_grad()
                output = model(
                    node_features=graph_tensors.node_features.to(self.device),
                    adjacency=graph_tensors.adjacency.to(self.device),
                )
                loss = loss_fn(output.lo_wcet_prediction, graph_tensors.target_lo_wcet.to(self.device))
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
            average_epoch_loss = epoch_loss / max(1, len(applications))
            history.epoch_losses.append(average_epoch_loss)
            if verbose:
                progress_percent = ((epoch_index + 1) / config.num_epochs) * 100.0
                print(
                    f"[{progress_prefix}] epoch {epoch_index + 1}/{config.num_epochs} "
                    f"({progress_percent:6.2f}%) loss={average_epoch_loss:.6f}"
                )
        return history

    def train_gat(
        self,
        model: RiskAwareGatRegressor,
        applications: Sequence[DagApplication],
        platform: HeterogeneousPlatform,
        config: TrainingConfig,
        cvar_alpha: float = 0.1,
        verbose: bool = True,
        progress_prefix: str = "ProposedGAT",
        enable_uncertainty_losses: bool = True,
    ) -> GatTrainingHistory:
        """Trains the proposed GAT model with uncertainty and CVaR losses."""

        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        history = GatTrainingHistory()

        for epoch_index in range(config.num_epochs):
            epoch_total = 0.0
            epoch_nll = 0.0
            epoch_sigma = 0.0
            epoch_cvar = 0.0
            epoch_prediction_mse = 0.0
            for application in applications:
                graph = application.graph
                graph_tensors = build_graph_tensors(graph)
                destination_reliability = build_training_reliability_vector(graph=graph, platform=platform).to(self.device)
                node_features = graph_tensors.node_features.to(self.device)
                adjacency = graph_tensors.adjacency.to(self.device)
                target_lo = graph_tensors.target_lo_wcet.to(self.device)
                target_sigma = graph_tensors.target_sigma.to(self.device)
                target_cvar = build_target_cvar_tensor(graph=graph, graph_tensors=graph_tensors, alpha=cvar_alpha).to(self.device)

                optimizer.zero_grad()
                output = model(
                    node_features=node_features,
                    adjacency=adjacency,
                    destination_reliability=destination_reliability,
                )
                variance = torch.exp(output.log_variance).clamp_min(1e-8)
                predicted_sigma = torch.sqrt(variance)
                prediction_mse_loss = torch.mean((target_lo - output.lo_mean) ** 2)
                if enable_uncertainty_losses:
                    nll_loss = 0.5 * (output.log_variance + ((target_lo - output.lo_mean) ** 2) / variance).mean()
                    sigma_loss = torch.mean((predicted_sigma - target_sigma) ** 2)
                    predicted_cvar = output.lo_mean + 1.5 * predicted_sigma
                    cvar_loss = torch.mean((predicted_cvar - target_cvar) ** 2)
                    total_loss = nll_loss + 0.5 * sigma_loss + 0.5 * cvar_loss
                else:
                    nll_loss = torch.mean((target_lo - output.lo_mean) ** 2)
                    sigma_loss = torch.zeros((), device=self.device)
                    cvar_loss = torch.zeros((), device=self.device)
                    total_loss = nll_loss
                total_loss.backward()
                optimizer.step()

                epoch_total += float(total_loss.item())
                epoch_nll += float(nll_loss.item())
                epoch_sigma += float(sigma_loss.item())
                epoch_cvar += float(cvar_loss.item())
                epoch_prediction_mse += float(prediction_mse_loss.item())

            denominator = max(1, len(applications))
            average_total = epoch_total / denominator
            average_nll = epoch_nll / denominator
            average_sigma = epoch_sigma / denominator
            average_cvar = epoch_cvar / denominator
            average_prediction_mse = epoch_prediction_mse / denominator
            history.epoch_losses.append(average_total)
            history.nll_losses.append(average_nll)
            history.sigma_losses.append(average_sigma)
            history.cvar_losses.append(average_cvar)
            history.prediction_mse_losses.append(average_prediction_mse)
            if verbose:
                progress_percent = ((epoch_index + 1) / config.num_epochs) * 100.0
                print(
                    f"[{progress_prefix}] epoch {epoch_index + 1}/{config.num_epochs} ({progress_percent:6.2f}%) "
                    f"total={average_total:.6f} mse={average_prediction_mse:.6f} "
                    f"nll={average_nll:.6f} sigma={average_sigma:.6f} cvar={average_cvar:.6f}"
                )
        return history


def build_training_reliability_vector(graph: TaskGraph, platform: HeterogeneousPlatform) -> torch.Tensor:
    """Builds a node-wise reliability vector for GAT training.

    HI-critical tasks are biased toward the more powerful A15 island and LO-critical
    tasks toward the A7 island. This gives the GAT layer a meaningful reliability
    context during supervised learning.
    """

    a7_cores = [core for core in platform.cores if core.core_type == "A7"]
    a15_cores = [core for core in platform.cores if core.core_type == "A15"]
    reliabilities: List[float] = []
    for task_id in graph.topological_order():
        node = graph.nodes[task_id]
        if node.is_dummy:
            reliabilities.append(1.0)
            continue
        candidate_cores = a15_cores if node.criticality == "HI" and a15_cores else a7_cores or platform.cores
        values = [core.reliability_over_interval(0.0, max(node.lo_wcet, 1e-9)) for core in candidate_cores]
        reliabilities.append(sum(values) / len(values))
    return torch.tensor(reliabilities, dtype=torch.float32)


def build_target_cvar_tensor(graph: TaskGraph, graph_tensors: GraphTensors, alpha: float) -> torch.Tensor:
    """Builds the supervised target tensor for robust LO-WCET."""

    values: List[List[float]] = []
    for task_id in graph_tensors.task_ids:
        node = graph.nodes[task_id]
        if "robust_lo_wcet" in node.metadata:
            robust_value = float(node.metadata["robust_lo_wcet"])
        else:
            sigma = 0.0 if node.runtime_distribution is None else node.runtime_distribution.sigma
            robust_value = gaussian_upper_cvar(mean_value=node.lo_wcet, std_value=sigma, alpha=alpha)
        values.append([robust_value])
    return torch.tensor(values, dtype=torch.float32)


@torch.no_grad()
def apply_gcn_predictions(model: ClassicGcnRegressor, graph: TaskGraph, device: str = "cpu") -> None:
    """Writes GCN predictions back into the graph object."""

    model.eval()
    graph_tensors = build_graph_tensors(graph)
    output = model(
        node_features=graph_tensors.node_features.to(device),
        adjacency=graph_tensors.adjacency.to(device),
    )
    predictions = output.lo_wcet_prediction.cpu().squeeze(-1)
    for index, task_id in enumerate(graph_tensors.task_ids):
        graph.nodes[task_id].predicted_lo_wcet = float(predictions[index].item())
        graph.nodes[task_id].predicted_uncertainty = 0.0


@torch.no_grad()
def apply_gat_predictions(
    model: RiskAwareGatRegressor,
    graph: TaskGraph,
    platform: HeterogeneousPlatform,
    device: str = "cpu",
) -> torch.Tensor:
    """Writes GAT predictions back into the graph and returns node embeddings."""

    model.eval()
    graph_tensors = build_graph_tensors(graph)
    destination_reliability = build_training_reliability_vector(graph=graph, platform=platform).to(device)
    output = model(
        node_features=graph_tensors.node_features.to(device),
        adjacency=graph_tensors.adjacency.to(device),
        destination_reliability=destination_reliability,
    )
    means = output.lo_mean.cpu().squeeze(-1)
    stds = torch.sqrt(torch.exp(output.log_variance).clamp_min(1e-8)).cpu().squeeze(-1)
    for index, task_id in enumerate(graph_tensors.task_ids):
        graph.nodes[task_id].predicted_lo_wcet = float(means[index].item())
        graph.nodes[task_id].predicted_uncertainty = float(stds[index].item())
    return output.embeddings.detach().cpu()
