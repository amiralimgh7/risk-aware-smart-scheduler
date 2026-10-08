"""GAT, GCN, graph tensor utilities, quantization, and quantization evaluation.

This module is part of the ESRLab defense-ready codebase. It uses snake_case for functions and variables while keeping Python classes in PascalCase, and exposes documented helpers for reproducible experiments.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from models.quantization import FakeQuantizeTensor


class ReliabilityAwareGraphAttentionLayer(nn.Module):
    """Graph attention layer modulated by destination reliability."""

    def __init__(self, in_features: int, out_features: int, num_heads: int = 2, use_qat: bool = False) -> None:
        """Initialize the object with validated configuration and runtime state."""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.linear = nn.Linear(in_features, out_features * num_heads, bias=False)
        self.use_qat = use_qat
        self.weight_fake_quant = FakeQuantizeTensor(num_bits=8, symmetric=True, per_channel=True, channel_axis=0)
        self.activation_fake_quant = FakeQuantizeTensor(num_bits=8, symmetric=False)
        self.attention_source = nn.Parameter(torch.empty(num_heads, out_features))
        self.attention_destination = nn.Parameter(torch.empty(num_heads, out_features))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.output_projection = nn.Linear(out_features * num_heads, out_features)
        self.activation = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initializes parameters."""

        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.attention_source)
        nn.init.xavier_uniform_(self.attention_destination)
        nn.init.xavier_uniform_(self.output_projection.weight)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        destination_reliability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Applies reliability-aware graph attention."""

        num_nodes = node_features.size(0)
        if self.use_qat:
            node_features = self.activation_fake_quant(node_features)
            projected = F.linear(node_features, self.weight_fake_quant(self.linear.weight), self.linear.bias)
        else:
            projected = self.linear(node_features)
        projected = projected.view(num_nodes, self.num_heads, self.out_features)

        source_scores = torch.einsum("nhf,hf->nh", projected, self.attention_source)
        destination_scores = torch.einsum("nhf,hf->nh", projected, self.attention_destination)
        attention_logits = source_scores.unsqueeze(1) + destination_scores.unsqueeze(0)
        attention_logits = self.leaky_relu(attention_logits)
        reliability_scale = destination_reliability.clamp(min=0.0, max=1.0).view(1, num_nodes, 1)
        attention_logits = attention_logits * reliability_scale

        adjacency_mask = adjacency > 0.0
        mask = adjacency_mask.unsqueeze(-1).expand_as(attention_logits)
        attention_logits = attention_logits.masked_fill(~mask, float("-inf"))
        attention_weights = torch.softmax(attention_logits, dim=1)
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0, posinf=0.0, neginf=0.0)

        aggregated = torch.einsum("ijh,jhf->ihf", attention_weights, projected)
        aggregated = aggregated.reshape(num_nodes, self.num_heads * self.out_features)
        if self.use_qat:
            projected_output = F.linear(
                aggregated,
                self.weight_fake_quant(self.output_projection.weight),
                self.output_projection.bias,
            )
        else:
            projected_output = self.output_projection(aggregated)
        output = self.activation(projected_output)
        if self.use_qat:
            output = self.activation_fake_quant(output)
        return output, attention_weights


@dataclass(slots=True)
class GatOutput:
    """Output package of the proposed risk-aware GAT model."""

    lo_mean: torch.Tensor
    log_variance: torch.Tensor
    embeddings: torch.Tensor


class RiskAwareGatRegressor(nn.Module):
    """Proposed GAT model with uncertainty heads.

    When ``use_qat=True``, selected linear weights and intermediate activations inside
    the GAT message-passing layers are fake-quantized during training. Output heads,
    attention-score parameters, and Softmax behavior stay floating point to follow the
    stability rule described in the project PDF.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        embedding_features: int,
        num_heads: int = 2,
        use_qat: bool = False,
    ) -> None:
        """Initialize the object with validated configuration and runtime state."""
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.embedding_features = embedding_features
        self.num_heads = num_heads
        self.use_qat = use_qat

        self.attention_1 = ReliabilityAwareGraphAttentionLayer(
            in_features=in_features,
            out_features=hidden_features,
            num_heads=num_heads,
            use_qat=use_qat,
        )
        self.attention_2 = ReliabilityAwareGraphAttentionLayer(
            in_features=hidden_features,
            out_features=embedding_features,
            num_heads=num_heads,
            use_qat=use_qat,
        )
        self.mean_head = nn.Linear(embedding_features, 1)
        self.log_variance_head = nn.Linear(embedding_features, 1)
        self.softplus = nn.Softplus()

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        destination_reliability: torch.Tensor,
    ) -> GatOutput:
        """Runs the GAT predictor."""

        hidden, _ = self.attention_1(
            node_features=node_features,
            adjacency=adjacency,
            destination_reliability=destination_reliability,
        )
        embeddings, _ = self.attention_2(
            node_features=hidden,
            adjacency=adjacency,
            destination_reliability=destination_reliability,
        )
        lo_mean = self.softplus(self.mean_head(embeddings)) + 1e-8
        log_variance = self.log_variance_head(embeddings).clamp(min=-8.0, max=6.0)
        return GatOutput(lo_mean=lo_mean, log_variance=log_variance, embeddings=embeddings)
