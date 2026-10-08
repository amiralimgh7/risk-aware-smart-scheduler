"""GAT, GCN, graph tensor utilities, quantization, and quantization evaluation.

This module is part of the ESRLab defense-ready codebase. It uses snake_case for functions and variables while keeping Python classes in PascalCase, and exposes documented helpers for reproducible experiments.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class GraphConvolution(nn.Module):
    """Simple graph convolution layer using pre-normalized adjacency."""

    def __init__(self, in_features: int, out_features: int) -> None:
        """Initialize the object with validated configuration and runtime state."""
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """Applies one graph convolution."""

        return adjacency @ self.linear(node_features)


@dataclass(slots=True)
class GcnOutput:
    """Output package of the classic GCN baseline."""

    lo_wcet_prediction: torch.Tensor
    embeddings: torch.Tensor


class ClassicGcnRegressor(nn.Module):
    """Classic GCN baseline for LO-WCET regression."""

    def __init__(self, in_features: int, hidden_features: int, embedding_features: int) -> None:
        """Initialize the object with validated configuration and runtime state."""
        super().__init__()
        self.conv_1 = GraphConvolution(in_features=in_features, out_features=hidden_features)
        self.conv_2 = GraphConvolution(in_features=hidden_features, out_features=embedding_features)
        self.regressor = nn.Linear(embedding_features, 1)
        self.activation = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> GcnOutput:
        """Runs the classic GCN regressor."""

        hidden = self.activation(self.conv_1(node_features=node_features, adjacency=adjacency))
        embeddings = self.activation(self.conv_2(node_features=hidden, adjacency=adjacency))
        lo_prediction = self.softplus(self.regressor(embeddings)) + 1e-8
        return GcnOutput(lo_wcet_prediction=lo_prediction, embeddings=embeddings)
