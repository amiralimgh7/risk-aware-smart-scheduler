"""GAT, GCN, graph tensor utilities, quantization, and quantization evaluation.

This module is part of the ESRLab defense-ready codebase. It uses snake_case for functions and variables while keeping Python classes in PascalCase, and exposes documented helpers for reproducible experiments.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List

import torch

from core.metrics import ScheduleMetrics
from core.task_graph import DagApplication
from hardware_model.hardware import HeterogeneousPlatform
from models.graph_utils import build_graph_tensors
from models.quantization import (
    benchmark_callable_latency_ms,
    clone_fp16_model,
    estimate_int8_weight_storage_bytes,
    estimate_model_size_bytes,
    pdf_qat_parameter_selector,
    quantize_dequantize_int8_weights_model,
)
from simulator.list_scheduling import decode_priority_schedule, evaluate_decoded_schedule
from training.rl_policy import RiskAwareGatScheduler
from training.supervised import build_training_reliability_vector


QUANTIZATION_METHOD_RATIONALES: Dict[str, str] = {
    "original_fp32_reference": "مدل مرجع بدون چندی‌سازی؛ برای سنجش افت دقت، کاهش اندازه و تغییر latency کنار همه روش‌ها قرار می‌گیرد.",
    "qat_int8_pdf": "روش اصلی PDF؛ fake-quantization در آموزش و سیاست INT8 برای وزن‌های خطی منتخب، همراه با حفظ attention score، Softmax و سرهای خروجی به‌صورت شناور برای پایداری عددی.",
    "weight_only_int8_per_channel": "روش کم‌ریسک برای کاهش حافظه؛ وزن‌ها به‌صورت per-channel INT8 ذخیره می‌شوند ولی محاسبه می‌تواند با dequantized kernels انجام شود، بنابراین افت دقت آن معمولاً قابل کنترل است.",
    "fp16_mixed_precision": "روش رایج mixed precision؛ حجم وزن‌ها را تقریباً نصف می‌کند و برای مقایسه با INT8 نشان می‌دهد آیا کاهش دقت شدید لازم است یا FP16 کافی است.",
}


@dataclass(slots=True)
class QuantizationComparison:
    """Stores float-vs-real-INT8 comparison results kept for backward compatibility."""

    float_predictor_size_bytes: int
    int8_predictor_size_bytes: int
    float_policy_size_bytes: int
    int8_policy_size_bytes: int
    float_scheduler_size_bytes: int
    int8_scheduler_size_bytes: int
    relative_size_reduction: float
    mean_absolute_prediction_error: float
    float_predictor_latency_ms: float
    int8_predictor_latency_ms: float
    float_policy_latency_ms: float
    int8_policy_latency_ms: float


@dataclass(slots=True)
class QuantizationMethodComparison:
    """Stores comparison metrics for one quantization method beside FP32."""

    method_name: str
    display_name: str
    reason_for_inclusion: str
    is_pdf_method: bool
    predictor_size_bytes: int
    policy_size_bytes: int
    scheduler_size_bytes: int
    relative_size_reduction_vs_fp32: float
    mean_absolute_prediction_error_vs_fp32: float | None
    predictor_latency_ms: float
    policy_latency_ms: float
    scheduler_latency_ms: float
    status: str
    error_message: str
    makespan: float | None
    energy: float | None
    utilization: float | None
    deadline_miss_ratio: float | None
    application_deadline_miss_ratio: float | None
    task_deadline_miss_ratio: float | None
    mode_switch_probability: float | None
    average_cvar: float | None
    aging_index: float | None
    average_reliability: float | None
    completed_hi_ratio: float | None
    completed_lo_ratio: float | None
    dropped_lo_ratio: float | None
    completed_task_ratio: float | None
    service_loss_ratio: float | None


def _scheduler_sizes(scheduler: RiskAwareGatScheduler) -> tuple[int, int, int]:
    """Returns predictor, policy, and total scheduler sizes."""

    predictor_size = estimate_model_size_bytes(scheduler.predictor)
    policy_size = 0 if scheduler.policy_network is None else estimate_model_size_bytes(scheduler.policy_network)
    return predictor_size, policy_size, predictor_size + policy_size


def _safe_latency(callable_fn, benchmark_latency: bool) -> float:
    """Benchmarks latency and returns zero when benchmarking is disabled or unsupported."""

    if not benchmark_latency:
        return 0.0
    try:
        return benchmark_callable_latency_ms(callable_fn)
    except Exception:
        return 0.0


def _predictor_forward(model, graph_tensors, destination_reliability, dtype: torch.dtype = torch.float32):
    """Runs a predictor with consistent dtype handling."""

    node_features = graph_tensors.node_features.to(dtype=dtype)
    adjacency = graph_tensors.adjacency.to(dtype=dtype)
    reliability = destination_reliability.to(dtype=dtype)
    return model(node_features=node_features, adjacency=adjacency, destination_reliability=reliability)


def _policy_forward(policy_network, task_features: torch.Tensor, dtype: torch.dtype = torch.float32):
    """Runs a policy network with consistent dtype handling."""

    return policy_network(task_features.to(dtype=dtype))


def _truncate_error_message(exc: Exception, limit: int = 320) -> str:
    """Builds a concise error message for report tables."""

    text = f"{exc.__class__.__name__}: {exc}"
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _empty_schedule_metrics() -> Dict[str, float | None]:
    """Returns blank scheduling fields for failed quantization methods."""

    return {
        "makespan": None,
        "energy": None,
        "utilization": None,
        "deadline_miss_ratio": None,
        "application_deadline_miss_ratio": None,
        "task_deadline_miss_ratio": None,
        "mode_switch_probability": None,
        "average_cvar": None,
        "aging_index": None,
        "average_reliability": None,
        "completed_hi_ratio": None,
        "completed_lo_ratio": None,
        "dropped_lo_ratio": None,
        "completed_task_ratio": None,
        "service_loss_ratio": None,
    }


def _schedule_metrics_to_fields(metrics: ScheduleMetrics) -> Dict[str, float | None]:
    """Converts ScheduleMetrics to quantization-row fields."""

    return {
        "makespan": metrics.makespan,
        "energy": metrics.energy,
        "utilization": metrics.utilization,
        "deadline_miss_ratio": metrics.deadline_miss_ratio,
        "application_deadline_miss_ratio": metrics.application_deadline_miss_ratio,
        "task_deadline_miss_ratio": metrics.task_deadline_miss_ratio,
        "mode_switch_probability": metrics.mode_switch_probability,
        "average_cvar": metrics.average_cvar,
        "aging_index": metrics.aging_index,
        "average_reliability": metrics.average_reliability,
        "completed_hi_ratio": metrics.completed_hi_ratio,
        "completed_lo_ratio": metrics.completed_lo_ratio,
        "dropped_lo_ratio": metrics.dropped_lo_ratio,
        "completed_task_ratio": metrics.completed_task_ratio,
        "service_loss_ratio": metrics.service_loss_ratio,
    }


@torch.no_grad()
def _schedule_with_variant(
    reference_scheduler: RiskAwareGatScheduler,
    predictor,
    policy,
    application: DagApplication,
    platform: HeterogeneousPlatform,
    dtype: torch.dtype,
) -> ScheduleMetrics:
    """Runs deterministic scheduling with one quantized predictor/policy pair.

    This mirrors RiskAwareGatScheduler.schedule but explicitly controls dtype, so
    FP16 and dynamic-INT8 variants can be evaluated without falling back to FP32.
    """

    graph = application.clone_graph()
    graph_tensors = build_graph_tensors(graph)
    destination_reliability = build_training_reliability_vector(graph, platform)
    predictor_output = _predictor_forward(predictor, graph_tensors, destination_reliability, dtype=dtype)

    predicted_mean = predictor_output.lo_mean.detach().float().cpu()
    predicted_std = torch.sqrt(torch.exp(predictor_output.log_variance.float()).clamp_min(1e-8)).detach().cpu()
    embeddings = predictor_output.embeddings.detach().float().cpu()
    task_features = reference_scheduler._build_policy_task_features(
        graph=graph,
        embeddings=embeddings,
        predicted_mean=predicted_mean,
        predicted_std=predicted_std,
        use_uncertainty_features=reference_scheduler.use_uncertainty_features,
    )
    priority_mean, core_logits, dvfs_logits = _policy_forward(policy, task_features, dtype=dtype)

    task_ids = graph_tensors.task_ids
    priority_scores: Dict[int, float] = {}
    core_assignments: Dict[int, int] = {}
    dvfs_assignments: Dict[int, int] = {}
    for index, task_id in enumerate(task_ids):
        graph.nodes[task_id].predicted_lo_wcet = float(predicted_mean[index].item())
        graph.nodes[task_id].predicted_uncertainty = float(predicted_std[index].item())
        priority_scores[task_id] = float(priority_mean.detach().float().cpu().squeeze(-1)[index].item())
        core_id = int(torch.argmax(core_logits.detach().float().cpu()[index]).item())
        core_id = max(0, min(platform.num_cores - 1, core_id))
        dvfs_id = int(torch.argmax(dvfs_logits.detach().float().cpu()[index]).item())
        max_dvfs_id = len(platform.cores[core_id].dvfs_levels) - 1
        dvfs_id = max(0, min(max_dvfs_id, dvfs_id))
        core_assignments[task_id] = core_id
        dvfs_assignments[task_id] = dvfs_id

    decoded = decode_priority_schedule(
        graph=graph,
        platform=platform,
        priority_scores=priority_scores,
        core_assignments=core_assignments,
        dvfs_assignments=dvfs_assignments,
        use_predictions=True,
        drop_lo_after_mode_switch=False,
    )
    return evaluate_decoded_schedule(graph=graph, platform=platform, decoded=decoded)


@torch.no_grad()
def compare_float_and_int8_schedulers(
    float_scheduler: RiskAwareGatScheduler,
    int8_scheduler: RiskAwareGatScheduler,
    application: DagApplication,
    platform: HeterogeneousPlatform,
    device: str = "cpu",
    benchmark_latency: bool = False,
) -> QuantizationComparison:
    """Compares the trained float scheduler to its real dynamic-INT8 inference variant."""

    graph = application.graph
    graph_tensors = build_graph_tensors(graph)
    destination_reliability = build_training_reliability_vector(graph, platform).to(device)

    def float_predictor_forward():
        """Run the float predictor forward step and return its computed result."""
        return float_scheduler.predictor(
            node_features=graph_tensors.node_features.to(device),
            adjacency=graph_tensors.adjacency.to(device),
            destination_reliability=destination_reliability,
        )

    def int8_predictor_forward():
        """Run the int8 predictor forward step and return its computed result."""
        return int8_scheduler.predictor(
            node_features=graph_tensors.node_features.to(device),
            adjacency=graph_tensors.adjacency.to(device),
            destination_reliability=destination_reliability,
        )

    float_output = float_predictor_forward()
    int8_output = int8_predictor_forward()
    mae = torch.mean(torch.abs(float_output.lo_mean - int8_output.lo_mean)).item()

    float_predictor_size, float_policy_size, float_scheduler_size = _scheduler_sizes(float_scheduler)
    # The INT8 inference variant is kept float-executable for portability, so
    # storage must be estimated from the intended PDF/QAT INT8 representation,
    # not from the in-memory dequantized PyTorch tensors.
    int8_predictor_size = estimate_int8_weight_storage_bytes(
        float_scheduler.predictor,
        per_channel=True,
        include_parameter=pdf_qat_parameter_selector,
    )
    int8_policy_size = 0 if float_scheduler.policy_network is None else estimate_int8_weight_storage_bytes(
        float_scheduler.policy_network,
        per_channel=True,
        include_parameter=pdf_qat_parameter_selector,
    )
    int8_scheduler_size = int8_predictor_size + int8_policy_size
    relative_size_reduction = 0.0 if float_scheduler_size == 0 else 1.0 - (int8_scheduler_size / float_scheduler_size)

    if float_scheduler.policy_network is None or int8_scheduler.policy_network is None:
        raise RuntimeError("Both float and int8 schedulers must have a trained policy network.")

    task_features = float_scheduler._build_policy_task_features(
        graph=graph,
        embeddings=float_output.embeddings.detach().cpu(),
        predicted_mean=float_output.lo_mean.detach().cpu(),
        predicted_std=torch.sqrt(torch.exp(float_output.log_variance).clamp_min(1e-8)).detach().cpu(),
        use_uncertainty_features=float_scheduler.use_uncertainty_features,
    ).to(device)

    def float_policy_forward():
        """Run the float policy forward step and return its computed result."""
        return float_scheduler.policy_network(task_features)

    def int8_policy_forward():
        """Run the int8 policy forward step and return its computed result."""
        return int8_scheduler.policy_network(task_features)

    float_predictor_latency_ms = benchmark_callable_latency_ms(float_predictor_forward) if benchmark_latency else 0.0
    int8_predictor_latency_ms = benchmark_callable_latency_ms(int8_predictor_forward) if benchmark_latency else 0.0
    float_policy_latency_ms = benchmark_callable_latency_ms(float_policy_forward) if benchmark_latency else 0.0
    int8_policy_latency_ms = benchmark_callable_latency_ms(int8_policy_forward) if benchmark_latency else 0.0

    return QuantizationComparison(
        float_predictor_size_bytes=float_predictor_size,
        int8_predictor_size_bytes=int8_predictor_size,
        float_policy_size_bytes=float_policy_size,
        int8_policy_size_bytes=int8_policy_size,
        float_scheduler_size_bytes=float_scheduler_size,
        int8_scheduler_size_bytes=int8_scheduler_size,
        relative_size_reduction=relative_size_reduction,
        mean_absolute_prediction_error=mae,
        float_predictor_latency_ms=float_predictor_latency_ms,
        int8_predictor_latency_ms=int8_predictor_latency_ms,
        float_policy_latency_ms=float_policy_latency_ms,
        int8_policy_latency_ms=int8_policy_latency_ms,
    )


@torch.no_grad()
def compare_float_and_quantization_methods(
    float_scheduler: RiskAwareGatScheduler,
    qat_scheduler: RiskAwareGatScheduler,
    application: DagApplication,
    platform: HeterogeneousPlatform,
    benchmark_latency: bool = True,
) -> List[QuantizationMethodComparison]:
    """Compares FP32, PDF-QAT, weight-only INT8, and FP16 variants.

    Dynamic INT8 Linear was removed from defense tables because the repeated
    custom-GAT scheduling loop does not execute it reliably on all CPU/PyTorch
    builds. Keeping its skipped rows produced misleading zeros in latency, DMR,
    utilization, energy, and success-rate columns.
    """

    if float_scheduler.policy_network is None or qat_scheduler.policy_network is None:
        raise RuntimeError("Both float and QAT schedulers must have trained policy networks.")

    graph = application.graph
    graph_tensors = build_graph_tensors(graph)
    destination_reliability = build_training_reliability_vector(graph, platform)

    float_output = _predictor_forward(float_scheduler.predictor, graph_tensors, destination_reliability)
    float_predictor_size, float_policy_size, float_scheduler_size = _scheduler_sizes(float_scheduler)

    weight_only_predictor = quantize_dequantize_int8_weights_model(float_scheduler.predictor, per_channel=True)
    weight_only_policy = quantize_dequantize_int8_weights_model(float_scheduler.policy_network, per_channel=True)
    fp16_predictor = clone_fp16_model(float_scheduler.predictor)
    fp16_policy = clone_fp16_model(float_scheduler.policy_network)

    variant_specs = [
        {
            "method_name": "original_fp32_reference",
            "display_name": "Original FP32 Reference",
            "is_pdf_method": False,
            "predictor": float_scheduler.predictor,
            "policy": float_scheduler.policy_network,
            "dtype": torch.float32,
            "predictor_size": float_predictor_size,
            "policy_size": float_policy_size,
        },
        {
            "method_name": "qat_int8_pdf",
            "display_name": "QAT INT8 - PDF Method",
            "is_pdf_method": True,
            "predictor": qat_scheduler.predictor,
            "policy": qat_scheduler.policy_network,
            "dtype": torch.float32,
            "predictor_size": estimate_int8_weight_storage_bytes(qat_scheduler.predictor, per_channel=True, include_parameter=pdf_qat_parameter_selector),
            "policy_size": estimate_int8_weight_storage_bytes(qat_scheduler.policy_network, per_channel=True, include_parameter=pdf_qat_parameter_selector),
        },
        {
            "method_name": "weight_only_int8_per_channel",
            "display_name": "Weight-Only INT8 Per-Channel",
            "is_pdf_method": False,
            "predictor": weight_only_predictor,
            "policy": weight_only_policy,
            "dtype": torch.float32,
            "predictor_size": estimate_int8_weight_storage_bytes(float_scheduler.predictor, per_channel=True),
            "policy_size": estimate_int8_weight_storage_bytes(float_scheduler.policy_network, per_channel=True),
        },
        {
            "method_name": "fp16_mixed_precision",
            "display_name": "FP16 Mixed Precision",
            "is_pdf_method": False,
            "predictor": fp16_predictor,
            "policy": fp16_policy,
            "dtype": torch.float16,
            "predictor_size": estimate_model_size_bytes(fp16_predictor),
            "policy_size": estimate_model_size_bytes(fp16_policy),
        },
    ]

    comparisons: List[QuantizationMethodComparison] = []
    for spec in variant_specs:
        method_name = str(spec["method_name"])
        print(f"[QuantizationMethod] start {method_name}", flush=True)
        dtype = spec["dtype"]
        predictor = spec["predictor"]
        policy = spec["policy"]
        error_message = ""
        status = "success"
        mae: float | None = None
        schedule_fields = _empty_schedule_metrics()

        def predictor_call():
            """Run the predictor call step and return its computed result."""
            return _predictor_forward(predictor, graph_tensors, destination_reliability, dtype=dtype)

        def schedule_call():
            """Run the schedule call step and return its computed result."""
            return _schedule_with_variant(
                reference_scheduler=float_scheduler,
                predictor=predictor,
                policy=policy,
                application=application,
                platform=platform,
                dtype=dtype,
            )

        try:
            output = predictor_call()
            mae = torch.mean(torch.abs(float_output.lo_mean.float() - output.lo_mean.float())).item()
            metrics = schedule_call()
            schedule_fields = _schedule_metrics_to_fields(metrics)
        except Exception as exc:
            status = "failed"
            error_message = _truncate_error_message(exc)

        def policy_latency_call():
            """Run the policy latency call step and return its computed result."""
            output = predictor_call()
            task_features = float_scheduler._build_policy_task_features(
                graph=graph,
                embeddings=output.embeddings.detach().float().cpu(),
                predicted_mean=output.lo_mean.detach().float().cpu(),
                predicted_std=torch.sqrt(torch.exp(output.log_variance.float()).clamp_min(1e-8)).detach().cpu(),
                use_uncertainty_features=float_scheduler.use_uncertainty_features,
            )
            return _policy_forward(policy, task_features, dtype=dtype)

        predictor_latency = _safe_latency(predictor_call, benchmark_latency) if status == "success" else 0.0
        policy_latency = _safe_latency(policy_latency_call, benchmark_latency) if status == "success" else 0.0
        predictor_size = int(spec["predictor_size"])
        policy_size = int(spec["policy_size"])
        scheduler_size = predictor_size + policy_size
        relative_reduction = 0.0 if float_scheduler_size == 0 else 1.0 - (scheduler_size / float_scheduler_size)
        print(f"[QuantizationMethod] done {method_name} status={status}", flush=True)
        comparisons.append(
            QuantizationMethodComparison(
                method_name=method_name,
                display_name=str(spec["display_name"]),
                reason_for_inclusion=QUANTIZATION_METHOD_RATIONALES[method_name],
                is_pdf_method=bool(spec["is_pdf_method"]),
                predictor_size_bytes=predictor_size,
                policy_size_bytes=policy_size,
                scheduler_size_bytes=scheduler_size,
                relative_size_reduction_vs_fp32=relative_reduction,
                mean_absolute_prediction_error_vs_fp32=mae,
                predictor_latency_ms=predictor_latency,
                policy_latency_ms=policy_latency,
                scheduler_latency_ms=predictor_latency + policy_latency,
                status=status,
                error_message=error_message,
                makespan=schedule_fields["makespan"],
                energy=schedule_fields["energy"],
                utilization=schedule_fields["utilization"],
                deadline_miss_ratio=schedule_fields["deadline_miss_ratio"],
                application_deadline_miss_ratio=schedule_fields["application_deadline_miss_ratio"],
                task_deadline_miss_ratio=schedule_fields["task_deadline_miss_ratio"],
                mode_switch_probability=schedule_fields["mode_switch_probability"],
                average_cvar=schedule_fields["average_cvar"],
                aging_index=schedule_fields["aging_index"],
                average_reliability=schedule_fields["average_reliability"],
                completed_hi_ratio=schedule_fields["completed_hi_ratio"],
                completed_lo_ratio=schedule_fields["completed_lo_ratio"],
                dropped_lo_ratio=schedule_fields["dropped_lo_ratio"],
                completed_task_ratio=schedule_fields["completed_task_ratio"],
                service_loss_ratio=schedule_fields["service_loss_ratio"],
            )
        )
    return comparisons
