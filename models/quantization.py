"""GAT, GCN, graph tensor utilities, quantization, and quantization evaluation.

This module is part of the ESRLab defense-ready codebase. It uses snake_case for functions and variables while keeping Python classes in PascalCase, and exposes documented helpers for reproducible experiments.
"""
from __future__ import annotations

import copy
import time
from typing import Callable

import torch
from torch import nn


class FakeQuantizeTensor(nn.Module):
    """Simple fake-quantization layer used for QAT-compatible experiments.

    It applies an affine quantize-dequantize operation during the forward pass, so
    training can learn under the numerical noise that will appear after INT8 export.
    Weight tensors can be fake-quantized per output channel, matching the PDF
    quantization policy for linear-layer weights.
    """

    def __init__(
        self,
        num_bits: int = 8,
        symmetric: bool = False,
        per_channel: bool = False,
        channel_axis: int = 0,
    ) -> None:
        """Initialize the object with validated configuration and runtime state."""
        super().__init__()
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.channel_axis = channel_axis

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Returns a fake-quantized tensor using STE-compatible PyTorch ops."""

        if not tensor.is_floating_point() or tensor.numel() == 0:
            return tensor
        if self.symmetric:
            quant_min = -((2 ** (self.num_bits - 1)) - 1)
            quant_max = (2 ** (self.num_bits - 1)) - 1
        else:
            quant_min = 0
            quant_max = (2 ** self.num_bits) - 1

        if self.symmetric and self.per_channel and tensor.ndim >= 2:
            axis = self.channel_axis if self.channel_axis >= 0 else tensor.ndim + self.channel_axis
            reduce_dims = [dim for dim in range(tensor.ndim) if dim != axis]
            max_abs = tensor.detach().abs().amax(dim=reduce_dims).clamp_min(1e-12)
            scales = max_abs / float(quant_max)
            zero_points = torch.zeros_like(scales, dtype=torch.int32)
            return torch.fake_quantize_per_channel_affine(
                tensor,
                scales.to(dtype=torch.float32),
                zero_points,
                axis,
                quant_min,
                quant_max,
            )

        if self.symmetric:
            max_abs = tensor.detach().abs().max().clamp_min(1e-12)
            scale = max_abs / float(quant_max)
            zero_point = torch.tensor(0, dtype=torch.int32, device=tensor.device)
        else:
            min_value = tensor.detach().min()
            max_value = tensor.detach().max()
            scale = ((max_value - min_value) / float(quant_max - quant_min)).clamp_min(1e-12)
            zero_point = torch.round(quant_min - min_value / scale).clamp(quant_min, quant_max).to(torch.int32)
        return torch.fake_quantize_per_tensor_affine(tensor, float(scale), int(zero_point.item()), quant_min, quant_max)


def quantize_dynamic_int8_model(model: nn.Module) -> nn.Module:
    """Returns a real dynamic-INT8 quantized copy of ``model``.

    The quantization is applied to all ``nn.Linear`` modules. This is real PyTorch
    dynamic quantization for CPU inference, not fake quantization.
    """

    model_copy = copy.deepcopy(model).cpu().eval()
    return torch.ao.quantization.quantize_dynamic(
        model_copy,
        {nn.Linear},
        dtype=torch.qint8,
    )


def clone_fp16_model(model: nn.Module) -> nn.Module:
    """Returns a float16 inference copy of the model for mixed-precision comparison."""

    return copy.deepcopy(model).cpu().eval().half()


def _quant_dequant_tensor_per_tensor_symmetric(tensor: torch.Tensor) -> torch.Tensor:
    """Quantizes and dequantizes a tensor using symmetric INT8 parameters."""

    if not tensor.is_floating_point() or tensor.numel() == 0:
        return tensor
    max_abs = tensor.detach().abs().max().clamp_min(1e-12)
    scale = max_abs / 127.0
    quantized = torch.round(tensor / scale).clamp(-127, 127)
    return quantized * scale


def _quant_dequant_tensor_per_channel_symmetric(tensor: torch.Tensor) -> torch.Tensor:
    """Quantizes and dequantizes a weight tensor per output channel."""

    if not tensor.is_floating_point() or tensor.numel() == 0 or tensor.ndim < 2:
        return tensor
    flat = tensor.detach().reshape(tensor.shape[0], -1)
    scales = flat.abs().max(dim=1).values.clamp_min(1e-12) / 127.0
    view_shape = [tensor.shape[0]] + [1] * (tensor.ndim - 1)
    scaled = scales.reshape(view_shape)
    quantized = torch.round(tensor / scaled).clamp(-127, 127)
    return quantized * scaled


def quantize_dequantize_int8_weights_model(
    model: nn.Module,
    per_channel: bool = True,
    include_parameter: Callable[[str, torch.Tensor], bool] | None = None,
) -> nn.Module:
    """Returns a float-executable model whose selected weights were INT8-rounded.

    This emulates the numerical effect of static/weight-only INT8 export while still
    using ordinary PyTorch float kernels, which keeps the experiment portable on
    Python installations that do not have a backend-specific static quantization
    runtime configured.
    """

    model_copy = copy.deepcopy(model).cpu().eval()
    with torch.no_grad():
        for name, parameter in model_copy.named_parameters():
            if include_parameter is not None and not include_parameter(name, parameter):
                continue
            if parameter.ndim >= 2 and parameter.is_floating_point():
                if per_channel:
                    parameter.copy_(_quant_dequant_tensor_per_channel_symmetric(parameter))
                else:
                    parameter.copy_(_quant_dequant_tensor_per_tensor_symmetric(parameter))
    return model_copy


def pdf_qat_parameter_selector(name: str, tensor: torch.Tensor) -> bool:
    """Selects parameters matching the quantization policy described in the PDF.

    The PDF-oriented policy quantizes computational linear weights but leaves
    attention-score parameters, Softmax behavior, uncertainty/output heads, and
    final decision heads in floating point for numerical stability.
    """

    if tensor.ndim < 2:
        return False
    excluded_tokens = (
        "attention_source",
        "attention_destination",
        "reliability_projection",
        "mean_head",
        "log_variance_head",
        "priority_head",
        "core_head",
        "dvfs_head",
    )
    if any(token in name for token in excluded_tokens):
        return False
    included_tokens = (
        "attention_1.linear",
        "attention_1.output_projection",
        "attention_2.linear",
        "attention_2.output_projection",
        "shared.0",
        "shared.2",
    )
    return any(token in name for token in included_tokens)


def estimate_int8_weight_storage_bytes(
    model: nn.Module,
    per_channel: bool = True,
    include_parameter: Callable[[str, torch.Tensor], bool] | None = None,
) -> int:
    """Estimates deployment storage when selected weight tensors are stored as INT8."""

    total = 0
    for name, value in model.state_dict().items():
        if not isinstance(value, torch.Tensor):
            continue
        should_quantize = value.ndim >= 2 and value.is_floating_point()
        if include_parameter is not None:
            should_quantize = should_quantize and include_parameter(name, value)
        if should_quantize:
            total += value.numel()  # int8 payload
            if per_channel:
                total += int(value.shape[0]) * 4  # one fp32 scale per output channel
            else:
                total += 4
            total += 4  # zero point / metadata placeholder
        else:
            total += _tensor_storage_bytes(value)
    return total


def _tensor_storage_bytes(tensor: torch.Tensor) -> int:
    """Estimates the storage footprint of one tensor in bytes.

    For quantized tensors, this function counts the underlying integer payload plus
    lightweight scale/zero-point metadata. This is a more meaningful deployment
    estimate than the raw ``torch.save`` archive size, whose container overhead can
    dominate for small models.
    """

    if tensor.is_quantized:
        payload = tensor.int_repr()
        total = payload.numel() * payload.element_size()
        qscheme = tensor.qscheme()
        if qscheme in {torch.per_tensor_affine, torch.per_tensor_symmetric}:
            total += 4  # scale
            total += 4  # zero point
        elif qscheme in {torch.per_channel_affine, torch.per_channel_symmetric, torch.per_channel_affine_float_qparams}:
            scales = tensor.q_per_channel_scales()
            zero_points = tensor.q_per_channel_zero_points()
            total += scales.numel() * scales.element_size()
            total += zero_points.numel() * zero_points.element_size()
        return total
    return tensor.numel() * tensor.element_size()


def _state_value_storage_bytes(value: object) -> int:
    """Recursively estimates the storage footprint of a state-dict value."""

    if isinstance(value, nn.Parameter):
        return _tensor_storage_bytes(value.detach())
    if isinstance(value, torch.Tensor):
        return _tensor_storage_bytes(value)
    if isinstance(value, dict):
        return sum(_state_value_storage_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_state_value_storage_bytes(item) for item in value)
    return 0


def estimate_model_size_bytes(model: nn.Module) -> int:
    """Returns an estimated model storage footprint in bytes.

    The estimate is based on the tensor storage required by the model state rather
    than the serialized archive size. This avoids misleading cases where a small INT8
    model appears larger than its FP32 counterpart purely because of zip/pickle
    overhead inside ``torch.save``.
    """

    state_dict = model.state_dict()
    return sum(_state_value_storage_bytes(value) for value in state_dict.values())


def count_parameters(model: nn.Module) -> int:
    """Counts total parameters of a model."""

    return sum(parameter.numel() for parameter in model.parameters())


def benchmark_callable_latency_ms(
    callable_fn: Callable[[], object],
    warmup_runs: int = 1,
    timed_runs: int = 5,
) -> float:
    """Benchmarks average callable latency in milliseconds.

    ``time.perf_counter_ns`` is used to avoid rounding very fast calls down to
    zero.  The default repeat count is intentionally small because CPU dynamic
    INT8 kernels can be slow on some development machines, and the value is used
    for comparative reporting rather than micro-benchmark publication.
    """

    for _ in range(max(0, warmup_runs)):
        callable_fn()

    timed_runs = int(timed_runs)
    if timed_runs <= 0:
        return 0.0

    start_time_ns = time.perf_counter_ns()
    for _ in range(timed_runs):
        callable_fn()
    elapsed_ns = time.perf_counter_ns() - start_time_ns
    if elapsed_ns <= 0:
        return 1e-6
    return max((elapsed_ns / timed_runs) / 1_000_000.0, 1e-6)
