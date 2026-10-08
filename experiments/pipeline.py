"""Runs experiment configurations, saves aggregate tables, and coordinates baseline/model evaluation.

This module is part of the ESRLab defense-ready codebase. It uses snake_case for functions and variables while keeping Python classes in PascalCase, and exposes documented helpers for reproducible experiments.
"""
from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence

from baselines.classic_gcn import ClassicGcnBaseline
from baselines.heft import HeftScheduler
from baselines.nsga_iii import NsgaThreeConfig, NsgaThreeScheduler
from baselines.vd_edf import VdEdfScheduler
from core.config import ApplicationTimingConfig, DaggenConfig, MixedCriticalityConfig, TrainingConfig
from core.metrics import ScheduleMetrics
from core.task_graph import DagApplication
from dataset_tools.serialization import load_application_json, save_application_json, save_json
from dataset_tools.simple_xlsx import save_rows_as_xlsx
from generators.workload import DagWorkloadBuilder
from hardware_model.hardware import HeterogeneousPlatform, build_big_little_platform
from models.quantization import estimate_int8_weight_storage_bytes, estimate_model_size_bytes, pdf_qat_parameter_selector
from models.quantization_eval import QuantizationComparison, compare_float_and_int8_schedulers, compare_float_and_quantization_methods
from simulator.reward import RewardWeights
from training.rl_policy import PolicyTrainingHistory, RiskAwareGatScheduler


PDF_GRAPH_COUNTS: List[int] = [10, 50, 100, 200, 400, 600, 800, 1000]
PDF_DENSITIES: List[float] = [0.2, 0.4, 0.6, 0.8]
PDF_REGULARITIES: List[float] = [0.2, 0.5, 0.8]
PDF_WIDTHS: List[float] = [0.2, 0.6, 0.8]
PDF_NODE_COUNTS: List[int] = [50, 100, 150, 200]
PDF_CORE_COUNTS: List[int] = [4, 8, 16, 32, 64]
PDF_CORE_UTILIZATION_LEVELS: List[float] = [0.25, 0.5, 0.75]


def scaled_node_count_for_core_sweep(
    base_node_count: int,
    core_count: int,
    reference_core_count: int = 4,
    max_scaled_node_count: int | None = None,
) -> int:
    """Returns a node count coupled to the core count for structural sweeps.

    The original structural sweep changed the number of cores while keeping the
    task graph size fixed.  That hides the effect of adding cores, especially in
    energy plots, because a small DAG cannot keep a large platform busy.  The
    corrected sweep treats ``base_node_count`` as the graph size at
    ``reference_core_count`` cores and scales the actual task count
    proportionally:

        effective_nodes = round(base_node_count * core_count / reference_core_count)

    Example with reference_core_count=4: 50 nodes at 4 cores becomes 100 nodes
    at 8 cores and 200 nodes at 16 cores.
    """

    if base_node_count <= 0:
        raise ValueError("base_node_count must be positive.")
    if core_count <= 0:
        raise ValueError("core_count must be positive.")
    if reference_core_count <= 0:
        raise ValueError("reference_core_count must be positive.")
    scaled = max(1, int(round(base_node_count * (core_count / reference_core_count))))
    if max_scaled_node_count is not None:
        if max_scaled_node_count <= 0:
            raise ValueError("max_scaled_node_count must be positive when provided.")
        scaled = min(scaled, max_scaled_node_count)
    return scaled


@dataclass(slots=True)
class ExperimentSplit:
    """Holds the train and test partitions of one application set."""

    train_applications: List[DagApplication]
    test_applications: List[DagApplication]


@dataclass(slots=True)
class BaselineSuiteArtifacts:
    """Stores aggregate results of the baseline comparison suite."""

    summary_by_algorithm: Dict[str, Dict[str, float]]
    per_graph_rows: List[Dict[str, float | str]]
    policy_history: PolicyTrainingHistory | None
    learned_models_summary: List[Dict[str, float | str]]
    detailed_rows: Dict[str, List[Dict[str, float | str]]] | None = None


@dataclass(slots=True)
class UncertaintyAblationArtifacts:
    """Stores the uncertainty-layer ablation results."""

    summary_by_variant: Dict[str, Dict[str, float]]
    per_graph_rows: List[Dict[str, float | str]]


@dataclass(slots=True)
class QuantizationArtifacts:
    """Stores aggregate INT8 comparison results."""

    average_float_predictor_size_bytes: float
    average_int8_predictor_size_bytes: float
    average_float_policy_size_bytes: float
    average_int8_policy_size_bytes: float
    average_float_scheduler_size_bytes: float
    average_int8_scheduler_size_bytes: float
    average_relative_size_reduction: float
    average_mean_absolute_prediction_error: float
    average_float_predictor_latency_ms: float
    average_int8_predictor_latency_ms: float
    average_float_policy_latency_ms: float
    average_int8_policy_latency_ms: float
    per_graph_rows: List[Dict[str, float | str]]
    summary_by_method: Dict[str, Dict[str, float | str]] | None = None
    method_rows: List[Dict[str, float | str]] | None = None


def _float_tag(value: float) -> str:
    """Formats a floating-point parameter as a folder-safe tag without rounding.

    The PDF uses utilization-like values such as 0.25 and 0.75.  Rounding
    folder names to one decimal place turns them into misleading tags such as
    0_2 and 0_8, even when the numeric value stored in manifests is correct.
    This helper keeps the exact decimal representation used by the experiment
    configuration while still producing filesystem-safe names.
    """

    return format(float(value), ".15g").replace(".", "_").replace("-", "m")


def build_pdf_configuration_path(
    output_root: str | Path,
    graph_count: int,
    density: float,
    regularity: float,
    width: float,
) -> Path:
    """Returns the canonical directory for one PDF configuration."""

    root = Path(output_root)
    return (
        root
        / f"graphs_{graph_count:04d}"
        / f"density_{_float_tag(density)}"
        / f"regularity_{_float_tag(regularity)}"
        / f"width_{_float_tag(width)}"
    )


def build_pdf_workload_builder(
    num_nodes: int,
    density: float,
    regularity: float,
    width: float,
    seed: int,
    period: float = 100.0,
) -> DagWorkloadBuilder:
    """Builds one workload builder aligned with the PDF parameterization."""

    return DagWorkloadBuilder(
        dag_config=DaggenConfig(
            n=num_nodes,
            width=width,
            regularity=regularity,
            density=density,
            jump=1,
            seed=seed,
        ),
        timing_config=ApplicationTimingConfig(period=period, relative_deadline=period),
        mc_config=MixedCriticalityConfig(
            hi_task_ratio=0.3,
            hi_wcet_scale_min=1.25,
            hi_wcet_scale_max=1.8,
            lo_runtime_sigma=0.12,
            hi_runtime_sigma=0.2,
            seed=seed,
        ),
    )


def generate_pdf_phase_1_dataset(
    output_root: str | Path,
    graph_counts: Sequence[int] = PDF_GRAPH_COUNTS,
    densities: Sequence[float] = PDF_DENSITIES,
    regularities: Sequence[float] = PDF_REGULARITIES,
    widths: Sequence[float] = PDF_WIDTHS,
    num_nodes: int = 50,
    mean_graph_utilization: float = 0.7,
    max_graph_utilization: float = 0.9,
    period: float = 100.0,
    seed: int = 0,
    skip_existing: bool = True,
) -> List[Path]:
    """Generates the phase-1 dataset exactly over the PDF parameter grid."""

    output_root = Path(output_root)
    generated_directories: List[Path] = []
    combination_index = 0

    for graph_count in graph_counts:
        for density in densities:
            for regularity in regularities:
                for width in widths:
                    configuration_dir = build_pdf_configuration_path(
                        output_root=output_root,
                        graph_count=graph_count,
                        density=density,
                        regularity=regularity,
                        width=width,
                    )
                    manifest_path = configuration_dir / "manifest.json"
                    applications_dir = configuration_dir / "applications"
                    existing_application_files = sorted(applications_dir.glob("application_*.json"))

                    if skip_existing and manifest_path.exists() and len(existing_application_files) == graph_count:
                        generated_directories.append(configuration_dir)
                        combination_index += 1
                        continue

                    combination_seed = seed + 10_000 * combination_index
                    builder = build_pdf_workload_builder(
                        num_nodes=num_nodes,
                        density=density,
                        regularity=regularity,
                        width=width,
                        seed=combination_seed,
                        period=period,
                    )
                    if mean_graph_utilization <= 0.0:
                        raise ValueError("mean_graph_utilization must be positive.")
                    if max_graph_utilization <= 0.0:
                        raise ValueError("max_graph_utilization must be positive.")
                    if mean_graph_utilization > max_graph_utilization:
                        raise ValueError("mean_graph_utilization cannot exceed max_graph_utilization.")

                    applications = [
                        builder.build_single_application(
                            application_id=f"application_{application_index}",
                            total_utilization=mean_graph_utilization,
                        )
                        for application_index in range(graph_count)
                    ]

                    applications_dir.mkdir(parents=True, exist_ok=True)
                    for application_index, application in enumerate(applications):
                        save_application_json(
                            application=application,
                            output_path=applications_dir / f"application_{application_index:04d}.json",
                        )

                    manifest = {
                        "graph_count": graph_count,
                        "density": density,
                        "regularity": regularity,
                        "width": width,
                        "num_nodes_per_graph": num_nodes,
                        "period": period,
                        "mean_graph_utilization": mean_graph_utilization,
                        "max_graph_utilization": max_graph_utilization,
                        "seed": combination_seed,
                        "generated_files": len(applications),
                        "summary": _summarize_application_set(applications),
                    }
                    save_json(manifest, manifest_path)
                    generated_directories.append(configuration_dir)
                    combination_index += 1

    return generated_directories


def collect_configuration_directories(dataset_root: str | Path) -> List[Path]:
    """Collects all configuration directories under a dataset root."""

    dataset_root = Path(dataset_root)
    manifests = sorted(dataset_root.rglob("manifest.json"))
    return [manifest.parent for manifest in manifests]


def load_applications_from_configuration(configuration_dir: str | Path) -> List[DagApplication]:
    """Loads all applications stored under one configuration directory."""

    configuration_dir = Path(configuration_dir)
    applications_dir = configuration_dir / "applications"
    application_files = sorted(applications_dir.glob("application_*.json"))
    return [load_application_json(path) for path in application_files]


def split_applications(
    applications: Sequence[DagApplication],
    train_ratio: float = 0.8,
) -> ExperimentSplit:
    """Splits a sequence of applications into train and test partitions."""

    total = len(applications)
    if total == 0:
        raise ValueError("applications must not be empty.")
    if total == 1:
        return ExperimentSplit(train_applications=list(applications), test_applications=list(applications))

    split_index = max(1, min(total - 1, int(round(total * train_ratio))))
    train_applications = list(applications[:split_index])
    test_applications = list(applications[split_index:])
    if not test_applications:
        test_applications = list(train_applications)
    return ExperimentSplit(train_applications=train_applications, test_applications=test_applications)


def _metrics_to_row(algorithm_name: str, application: DagApplication, metrics: ScheduleMetrics) -> Dict[str, float | str]:
    """Converts one schedule result to a flat row."""

    row: Dict[str, float | str] = {
        "algorithm": algorithm_name,
        "application_id": application.application_id,
        "graph_id": application.graph.graph_id,
        "total_utilization": application.total_utilization,
    }
    row.update(metrics.reward_ready_dict)
    row["deadline_miss"] = 1.0 if metrics.deadline_miss else 0.0
    row["mode_switch_count"] = float(metrics.mode_switch_count)
    return row




def _json_dumps_compact(payload: object) -> str:
    """Serializes small structured CSV cells without losing numeric precision."""

    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _max_dvfs_level_id(core) -> int:
    """Returns the highest-frequency DVFS operating point for a core."""

    return max(core.dvfs_levels, key=lambda level: level.frequency_ghz).level_id


def _entry_work_units(core, entry) -> float:
    """Reconstructs normalized work units from a scheduled runtime and DVFS level."""

    selected_level = core.dvfs_levels[entry.dvfs_level_id]
    effective_speed = core.base_speed_factor * selected_level.frequency_ghz
    return max(0.0, entry.actual_runtime) * max(effective_speed, 1e-12)


def _runtime_at_fixed_max_dvfs(core, entry) -> float:
    """Estimates runtime of the same task work under fixed maximum DVFS."""

    return core.execution_time(_entry_work_units(core, entry), _max_dvfs_level_id(core))


PREDICTION_MAPE_EPSILON = 1e-3


def _algorithm_method_type(algorithm_name: str) -> str:
    """Classifies rows so prediction metrics are not interpreted as the same kind of estimator."""

    if algorithm_name in {"gnn", "proposed_gat_float32", "proposed_gat_quantization_int8"}:
        return "learned_predictor"
    if algorithm_name in {"heft", "vd_edf", "nsga_iii"}:
        return "classical_scheduler_estimate"
    return "other"


def _total_power_trace_peaks(application: DagApplication, platform: HeterogeneousPlatform, metrics: ScheduleMetrics) -> tuple[float, float]:
    """Returns peak total active system power after/before DVFS.

    Earlier versions used the maximum single-core power as the peak.  That can
    be smaller than average system power when several cores run concurrently.
    This helper computes the peak from an event trace of all active tasks.
    """

    events: List[tuple[float, float, float]] = []
    real_task_ids = set(application.graph.real_task_ids)
    for entry in metrics.entries:
        if entry.task_id not in real_task_ids:
            continue
        core = platform.core_by_id(entry.core_id)
        after_power = core.total_power(entry.dvfs_level_id)
        before_power = core.total_power(_max_dvfs_level_id(core))
        execution_finish = _entry_execution_finish(entry)
        events.append((entry.start_time, after_power, before_power))
        events.append((execution_finish, -after_power, -before_power))
        overhead_power = _mode_switch_overhead_power(entry)
        if overhead_power > 0.0:
            events.append((execution_finish, overhead_power, overhead_power))
            events.append((entry.finish_time, -overhead_power, -overhead_power))
    events.sort(key=lambda item: (item[0], item[1]))
    active_after = 0.0
    active_before = 0.0
    peak_after = 0.0
    peak_before = 0.0
    index = 0
    while index < len(events):
        event_time = events[index][0]
        while index < len(events) and abs(events[index][0] - event_time) <= 1e-12:
            active_after += events[index][1]
            active_before += events[index][2]
            index += 1
        peak_after = max(peak_after, active_after)
        peak_before = max(peak_before, active_before)
    return max(0.0, peak_after), max(0.0, peak_before)




def _entry_execution_finish(entry) -> float:
    """Returns the end of useful task execution before any mode-switch overhead."""

    return entry.start_time + max(0.0, entry.actual_runtime)


def _mode_switch_overhead_power(entry) -> float:
    """Returns the explicit mode-switch overhead power for one schedule entry."""

    overhead_time = max(0.0, getattr(entry, "mode_switch_time_overhead", 0.0))
    if overhead_time <= 0.0:
        return 0.0
    return max(0.0, getattr(entry, "mode_switch_energy_overhead", 0.0)) / overhead_time

def _task_interval_reliability(platform: HeterogeneousPlatform, entry) -> float:
    """Returns reliability of one scheduled task interval."""

    core = platform.core_by_id(entry.core_id)
    return core.reliability_over_interval(entry.start_time, entry.actual_runtime)


def _sub_deadlines_for_application(application: DagApplication) -> Dict[int, float]:
    """Builds task sub-deadlines for detailed CSV rows."""

    from simulator.list_scheduling import compute_sub_deadlines

    return compute_sub_deadlines(application.graph)


def _task_release_time_from_schedule(application: DagApplication, metrics: ScheduleMetrics, task_id: int, target_core_id: int | None) -> float:
    """Returns predecessor-based release time for one task in the realized schedule."""

    graph = application.graph
    node = graph.nodes[task_id]
    release_time = 0.0
    for predecessor_id in node.predecessors:
        predecessor_entry = metrics.task_to_entry.get(predecessor_id)
        finish_time = predecessor_entry.finish_time if predecessor_entry is not None else 0.0
        communication_cost = 0.0
        if target_core_id is not None and predecessor_entry is not None and predecessor_entry.core_id != target_core_id:
            communication_cost = graph.communication_cost(predecessor_id, task_id)
        release_time = max(release_time, finish_time + communication_cost)
    return release_time


def _assert_formula_invariants(
    configuration: str,
    algorithm_name: str,
    application_id: str,
    graph_id: str,
    total_task_count: int,
    executed_count: int,
    deadline_met_tasks: int,
    deadline_missed_executed_tasks: int,
    dropped_or_unexecuted_tasks: int,
    failed_or_dropped_tasks: int,
    task_rows: Sequence[Dict[str, float | str]],
    energy_rows: Sequence[Dict[str, float | str]],
) -> None:
    """Validate the formulas that make the exported CSV files defensible."""

    context = f"{configuration}/{algorithm_name}/{application_id}/{graph_id}"
    if executed_count != deadline_met_tasks + deadline_missed_executed_tasks:
        raise AssertionError(f"{context}: executed != met + missed_executed")
    if total_task_count != executed_count + dropped_or_unexecuted_tasks:
        raise AssertionError(f"{context}: total != executed + dropped")
    if failed_or_dropped_tasks != deadline_missed_executed_tasks + dropped_or_unexecuted_tasks:
        raise AssertionError(f"{context}: failed_or_dropped != missed_executed + dropped")

    for row in task_rows:
        if float(row.get("executed", 0.0)) <= 0.0:
            continue
        release_time = float(row["release_time"])
        start_time = float(row["start_time"])
        finish_time = float(row["finish_time"])
        execution_time = float(row["execution_time"])
        response_time = float(row["response_time"])
        waiting_time = float(row["waiting_time"])
        reliability_value = float(row["task_reliability"])
        if start_time + 1e-9 < release_time:
            raise AssertionError(f"{context}: start_time < release_time for task {row.get('task_id')}")
        if finish_time + 1e-9 < start_time:
            raise AssertionError(f"{context}: finish_time < start_time for task {row.get('task_id')}")
        if execution_time < -1e-9 or response_time < -1e-9 or waiting_time < -1e-9:
            raise AssertionError(f"{context}: negative timing metric for task {row.get('task_id')}")
        if not (0.0 <= reliability_value <= 1.0):
            raise AssertionError(f"{context}: task reliability outside [0,1]")

    for row in energy_rows:
        energy_after = float(row["energy_after_dvfs"])
        energy_before = float(row["energy_before_dvfs"])
        average_power_after = float(row["average_power_consumption_after_dvfs"])
        average_power_before = float(row["average_power_consumption_before_dvfs"])
        peak_power_after = float(row["peak_total_system_power_consumption_after_dvfs"])
        peak_power_before = float(row["peak_total_system_power_consumption_before_dvfs"])
        if energy_after > energy_before + 1e-9:
            raise AssertionError(f"{context}: energy_after_dvfs > energy_before_dvfs")
        if peak_power_after + 1e-9 < average_power_after:
            raise AssertionError(f"{context}: peak_power_after < average_power_after")
        if peak_power_before + 1e-9 < average_power_before:
            raise AssertionError(f"{context}: peak_power_before < average_power_before")


def _energy_power_breakdown(application: DagApplication, platform: HeterogeneousPlatform, metrics: ScheduleMetrics) -> Dict[str, object]:
    """Computes after-DVFS and fixed-max-DVFS energy/power metrics for one DAG run.

    "before DVFS" is defined as executing the same realized work on the same assigned
    cores at their highest DVFS operating point. "after DVFS" uses the actual selected
    DVFS levels from the schedule. Units are normalized simulator energy/power units.
    """

    makespan = max(metrics.makespan, 1e-12)
    real_task_ids = set(application.graph.real_task_ids)
    executed_entries = [entry for entry in metrics.entries if entry.task_id in real_task_ids]
    per_core_after = {core.core_id: 0.0 for core in platform.cores}
    per_core_before = {core.core_id: 0.0 for core in platform.cores}
    busy_time_by_core = {core.core_id: 0.0 for core in platform.cores}
    total_after = 0.0
    total_before = 0.0
    peak_after = 0.0
    peak_before = 0.0

    for entry in executed_entries:
        core = platform.core_by_id(entry.core_id)
        max_level_id = _max_dvfs_level_id(core)
        after_power = core.total_power(entry.dvfs_level_id)
        before_power = core.total_power(max_level_id)
        after_energy = core.energy_for_runtime(entry.actual_runtime, entry.dvfs_level_id)
        # "Before DVFS" is the no-DVFS upper baseline: the same realized
        # schedule is charged with the highest voltage/frequency power model.
        # This definition compares power-management cost on identical executed
        # work and prevents the misleading VD-EDF case where a shortened
        # max-frequency runtime made after-DVFS energy appear larger.
        before_energy = core.energy_for_runtime(entry.actual_runtime, max_level_id)
        after_energy += getattr(entry, "mode_switch_energy_overhead", 0.0)
        before_energy += getattr(entry, "mode_switch_energy_overhead", 0.0)
        # Numerical guard and semantic invariant: no-DVFS max-power energy must
        # not be below the energy of a selected lower/equal DVFS level.
        before_energy = max(before_energy, after_energy)
        total_after += after_energy
        total_before += before_energy
        per_core_after[core.core_id] += after_energy
        per_core_before[core.core_id] += before_energy
        busy_time_by_core[core.core_id] += entry.actual_runtime + getattr(entry, "mode_switch_time_overhead", 0.0)

    total_before = max(total_before, total_after)
    peak_after, peak_before = _total_power_trace_peaks(application, platform, metrics)
    peak_before = max(peak_before, peak_after)

    executed_count = len(executed_entries)
    energy_per_task_after = 0.0 if executed_count == 0 else total_after / executed_count
    energy_per_task_before = 0.0 if executed_count == 0 else total_before / executed_count
    return {
        "total_energy_after_dvfs": total_after,
        "total_energy_before_dvfs": total_before,
        "energy_after_dvfs": total_after,
        "energy_before_dvfs": total_before,
        "average_power_after_dvfs": total_after / makespan,
        "average_power_before_dvfs": total_before / makespan,
        "power_after_dvfs": total_after / makespan,
        "power_before_dvfs": total_before / makespan,
        "peak_power_after_dvfs": peak_after,
        "peak_power_before_dvfs": peak_before,
        "per_core_energy_after_dvfs": per_core_after,
        "per_core_energy_before_dvfs": per_core_before,
        "energy_per_task_after_dvfs": energy_per_task_after,
        "energy_per_task_before_dvfs": energy_per_task_before,
        "energy_per_dag_after_dvfs": total_after,
        "energy_per_dag_before_dvfs": total_before,
        "energy_saving_ratio_after_vs_before": 0.0 if total_before <= 0.0 else (total_before - total_after) / total_before,
        "busy_time_by_core": busy_time_by_core,
    }


def _detailed_rows_for_schedule(
    configuration: str,
    algorithm_name: str,
    application: DagApplication,
    platform: HeterogeneousPlatform,
    metrics: ScheduleMetrics,
) -> Dict[str, List[Dict[str, float | str]]]:
    """Builds detailed task/core/schedule/energy/prediction CSV rows for one run."""

    sub_deadlines = _sub_deadlines_for_application(application)
    real_task_ids = list(application.graph.real_task_ids)
    real_task_count = len(real_task_ids)
    executed_entries = [entry for entry in metrics.entries if entry.task_id in set(real_task_ids)]
    task_rows: List[Dict[str, float | str]] = []
    prediction_detail_rows: List[Dict[str, float | str]] = []
    deadline_met_tasks = 0
    deadline_missed_executed_tasks = 0
    dropped_or_unexecuted_tasks = 0
    failed_or_dropped_tasks = 0
    response_times: List[float] = []
    waiting_times: List[float] = []
    finish_times: List[float] = []

    for task_id in real_task_ids:
        node = application.graph.nodes[task_id]
        entry = metrics.task_to_entry.get(task_id)
        executed = entry is not None
        target_core_id = entry.core_id if entry is not None else None
        release_time = _task_release_time_from_schedule(application, metrics, task_id, target_core_id)
        sub_deadline = sub_deadlines.get(task_id, application.deadline)
        if entry is None:
            deadline_met = 0.0
            deadline_miss = 0.0
            deadline_missed_executed = 0.0
            failed_or_dropped = 1.0
            dropped_or_unexecuted_tasks += 1
            failed_or_dropped_tasks += 1
            start_time = ""
            finish_time = ""
            assigned_core = ""
            selected_dvfs_level = ""
            execution_time = 0.0
            task_reliability = ""
            estimated_runtime = ""
            response_time = ""
            waiting_time = ""
        else:
            deadline_met = 1.0 if entry.finish_time <= sub_deadline else 0.0
            deadline_miss = 1.0 - deadline_met
            deadline_missed_executed = deadline_miss
            failed_or_dropped = deadline_miss
            deadline_met_tasks += int(deadline_met)
            deadline_missed_executed_tasks += int(deadline_missed_executed)
            failed_or_dropped_tasks += int(failed_or_dropped)
            start_time = entry.start_time
            finish_time = entry.finish_time
            assigned_core = float(entry.core_id)
            selected_dvfs_level = float(entry.dvfs_level_id)
            execution_time = entry.actual_runtime
            task_reliability = _task_interval_reliability(platform, entry)
            estimated_runtime = entry.estimated_runtime
            response_time = entry.finish_time - release_time
            waiting_time = entry.start_time - release_time
            response_times.append(response_time)
            waiting_times.append(waiting_time)
            finish_times.append(entry.finish_time)
            prediction_detail_rows.append(
                {
                    "configuration": configuration,
                    "algorithm": algorithm_name,
                    "application_id": application.application_id,
                    "graph_id": application.graph.graph_id,
                    "task_id": float(task_id),
                    "criticality_level": node.criticality,
                    "actual_execution_time": entry.actual_runtime,
                    "predicted_execution_time": entry.estimated_runtime,
                    "method_type": _algorithm_method_type(algorithm_name),
                    "absolute_error": abs(entry.actual_runtime - entry.estimated_runtime),
                    "squared_error": (entry.actual_runtime - entry.estimated_runtime) ** 2,
                    "absolute_percentage_error_raw": 0.0 if abs(entry.actual_runtime) <= 1e-12 else abs(entry.actual_runtime - entry.estimated_runtime) / abs(entry.actual_runtime),
                    "absolute_percentage_error_epsilon": abs(entry.actual_runtime - entry.estimated_runtime) / max(abs(entry.actual_runtime), PREDICTION_MAPE_EPSILON),
                    "symmetric_absolute_percentage_error": (
                        0.0
                        if abs(entry.actual_runtime) + abs(entry.estimated_runtime) <= 1e-12
                        else 2.0 * abs(entry.actual_runtime - entry.estimated_runtime) / (abs(entry.actual_runtime) + abs(entry.estimated_runtime))
                    ),
                    "mape_epsilon": PREDICTION_MAPE_EPSILON,
                    "unit": "sim_time_unit",
                }
            )
        task_rows.append(
            {
                "configuration": configuration,
                "algorithm": algorithm_name,
                "application_id": application.application_id,
                "graph_id": application.graph.graph_id,
                "task_id": float(task_id),
                "criticality_level": node.criticality,
                "WCET_LO": node.lo_wcet,
                "WCET_HI": node.hi_wcet,
                "relative_deadline": max(0.0, sub_deadline - release_time) if entry is not None else "",
                "absolute_deadline": sub_deadline,
                "deadline": sub_deadline,
                "deadline_definition": "absolute_sub_deadline_from_DAG_release_zero",
                "release_time": release_time,
                "start_time": start_time,
                "finish_time": finish_time,
                "assigned_core": assigned_core,
                "selected_DVFS_level": selected_dvfs_level,
                "execution_time": execution_time,
                "mode_switch_time_overhead": getattr(entry, "mode_switch_time_overhead", 0.0) if entry is not None else 0.0,
                "mode_switch_energy_overhead": getattr(entry, "mode_switch_energy_overhead", 0.0) if entry is not None else 0.0,
                "deadline_met": deadline_met,
                "deadline_miss": deadline_miss,
                "deadline_missed_executed": deadline_missed_executed,
                "failed_or_dropped": failed_or_dropped,
                "task_reliability": task_reliability,
                "executed": 1.0 if executed else 0.0,
                "estimated_runtime_or_prediction": estimated_runtime,
                "response_time": response_time,
                "waiting_time": waiting_time,
                "unit_time": "sim_time_unit",
            }
        )

    executed_count = len(executed_entries)
    total_task_count = real_task_count
    meet_ratio = 0.0 if total_task_count == 0 else deadline_met_tasks / total_task_count
    deadline_failure_ratio = 0.0 if total_task_count == 0 else failed_or_dropped_tasks / total_task_count
    executed_task_deadline_miss_ratio = 0.0 if executed_count == 0 else deadline_missed_executed_tasks / executed_count
    executed_miss_ratio_over_total = 0.0 if total_task_count == 0 else deadline_missed_executed_tasks / total_task_count
    service_loss_ratio = 0.0 if total_task_count == 0 else dropped_or_unexecuted_tasks / total_task_count
    schedule_rows = [
        {
            "configuration": configuration,
            "algorithm": algorithm_name,
            "application_id": application.application_id,
            "graph_id": application.graph.graph_id,
            "number_of_total_tasks": float(total_task_count),
            "number_of_executed_tasks": float(executed_count),
            "number_of_deadline_met_tasks": float(deadline_met_tasks),
            "number_of_deadline_missed_tasks": float(deadline_missed_executed_tasks),
            "number_of_deadline_missed_executed_tasks": float(deadline_missed_executed_tasks),
            "number_of_dropped_or_unexecuted_tasks": float(dropped_or_unexecuted_tasks),
            "number_of_failed_or_dropped_tasks": float(failed_or_dropped_tasks),
            "deadline_meet_ratio": meet_ratio,
            "deadline_miss_ratio": executed_miss_ratio_over_total,
            "deadline_miss_ratio_formula": "deadline_missed_executed_tasks / total_tasks",
            "deadline_miss_ratio_definition": "DMR is Deadline Miss Ratio only; dropped/unexecuted LO service loss is reported separately",
            "deadline_failure_ratio": deadline_failure_ratio,
            "executed_task_deadline_miss_ratio": executed_task_deadline_miss_ratio,
            "executed_miss_ratio_over_total": executed_miss_ratio_over_total,
            "service_loss_ratio": service_loss_ratio,
            "application_deadline_miss_ratio": metrics.application_deadline_miss_ratio,
            "task_deadline_miss_ratio": executed_miss_ratio_over_total,
            "average_response_time": 0.0 if not response_times else mean(response_times),
            "average_waiting_time": 0.0 if not waiting_times else mean(waiting_times),
            "average_finish_time": 0.0 if not finish_times else mean(finish_times),
            "makespan": metrics.makespan,
            "number_of_mode_switches": float(metrics.mode_switch_count),
            "mode_switch_rate": 0.0 if total_task_count == 0 else metrics.mode_switch_count / total_task_count,
            "mode_switch_time_overhead": metrics.mode_switch_time_overhead,
            "mode_switch_energy_overhead": metrics.mode_switch_energy_overhead,
            "mode_switch_power_overhead": 0.0 if metrics.mode_switch_time_overhead <= 0.0 else metrics.mode_switch_energy_overhead / metrics.mode_switch_time_overhead,
            "mode_switch_average_time_overhead_per_switch": 0.0 if metrics.mode_switch_count == 0 else metrics.mode_switch_time_overhead / metrics.mode_switch_count,
            "mode_switch_average_energy_overhead_per_switch": 0.0 if metrics.mode_switch_count == 0 else metrics.mode_switch_energy_overhead / metrics.mode_switch_count,
            "core_utilization": metrics.utilization,
            "unit_time": "sim_time_unit",
        }
    ]

    energy = _energy_power_breakdown(application, platform, metrics)
    energy_rows = [
        {
            "configuration": configuration,
            "algorithm": algorithm_name,
            "application_id": application.application_id,
            "graph_id": application.graph.graph_id,
            "total_energy_consumption_after_dvfs": energy["total_energy_after_dvfs"],
            "total_energy_consumption_before_dvfs": energy["total_energy_before_dvfs"],
            "energy_after_dvfs": energy["energy_after_dvfs"],
            "energy_before_dvfs": energy["energy_before_dvfs"],
            "average_power_consumption_after_dvfs": energy["average_power_after_dvfs"],
            "average_power_consumption_before_dvfs": energy["average_power_before_dvfs"],
            "power_after_dvfs": energy["power_after_dvfs"],
            "power_before_dvfs": energy["power_before_dvfs"],
            "peak_total_system_power_consumption_after_dvfs": energy["peak_power_after_dvfs"],
            "peak_total_system_power_consumption_before_dvfs": energy["peak_power_before_dvfs"],
            "peak_power_consumption_after_dvfs": energy["peak_power_after_dvfs"],
            "peak_power_consumption_before_dvfs": energy["peak_power_before_dvfs"],
            "per_core_energy_after_dvfs_json": _json_dumps_compact(energy["per_core_energy_after_dvfs"]),
            "per_core_energy_before_dvfs_json": _json_dumps_compact(energy["per_core_energy_before_dvfs"]),
            "energy_per_task_after_dvfs": energy["energy_per_task_after_dvfs"],
            "energy_per_task_before_dvfs": energy["energy_per_task_before_dvfs"],
            "energy_per_DAG_after_dvfs": energy["energy_per_dag_after_dvfs"],
            "energy_per_DAG_before_dvfs": energy["energy_per_dag_before_dvfs"],
            "energy_saving_ratio_after_vs_before": energy["energy_saving_ratio_after_vs_before"],
            "mode_switch_energy_overhead": metrics.mode_switch_energy_overhead,
            "mode_switch_time_overhead": metrics.mode_switch_time_overhead,
            "mode_switch_power_overhead": 0.0 if metrics.mode_switch_time_overhead <= 0.0 else metrics.mode_switch_energy_overhead / metrics.mode_switch_time_overhead,
            "unit_energy": "relative_energy_unit",
            "unit_power": "relative_power_unit",
        }
    ]

    _assert_formula_invariants(
        configuration=configuration,
        algorithm_name=algorithm_name,
        application_id=application.application_id,
        graph_id=application.graph.graph_id,
        total_task_count=total_task_count,
        executed_count=executed_count,
        deadline_met_tasks=deadline_met_tasks,
        deadline_missed_executed_tasks=deadline_missed_executed_tasks,
        dropped_or_unexecuted_tasks=dropped_or_unexecuted_tasks,
        failed_or_dropped_tasks=failed_or_dropped_tasks,
        task_rows=task_rows,
        energy_rows=energy_rows,
    )

    core_rows: List[Dict[str, float | str]] = []
    for core in platform.cores:
        core_entries = [entry for entry in executed_entries if entry.core_id == core.core_id]
        busy_time = sum(entry.actual_runtime + getattr(entry, "mode_switch_time_overhead", 0.0) for entry in core_entries)
        utilization_per_core = 0.0 if metrics.makespan <= 0.0 else busy_time / metrics.makespan
        log_reliability = 0.0
        for entry in core_entries:
            log_reliability += math.log(max(_task_interval_reliability(platform, entry), 1e-300))
        reliability_per_core = math.exp(max(log_reliability, -700.0))
        failure_rate = core.failure_rate(metrics.makespan)
        core_rows.append(
            {
                "configuration": configuration,
                "algorithm": algorithm_name,
                "application_id": application.application_id,
                "graph_id": application.graph.graph_id,
                "number_of_cores": float(platform.num_cores),
                "core_id": float(core.core_id),
                "core_type": f"Cortex-{core.core_type}",
                "frequency_levels_GHz": ";".join(format(level.frequency_ghz, ".6g") for level in core.dvfs_levels),
                "voltage_levels_V": ";".join(format(level.voltage, ".6g") for level in core.dvfs_levels),
                "power_model": "normalized_dynamic_power=C_eff*V^2*f; energy=power*runtime",
                "utilization_per_core": utilization_per_core,
                "failure_rate_per_core": failure_rate,
                "reliability_per_core": reliability_per_core,
                "aging_state_per_core": failure_rate * utilization_per_core,
                "per_core_energy_after_dvfs": dict(energy["per_core_energy_after_dvfs"]).get(core.core_id, 0.0),
                "per_core_energy_before_dvfs": dict(energy["per_core_energy_before_dvfs"]).get(core.core_id, 0.0),
                "unit_frequency": "GHz",
                "unit_voltage": "V",
            }
        )

    if prediction_detail_rows:
        errors = [float(row["absolute_error"]) for row in prediction_detail_rows]
        squared = [float(row["squared_error"]) for row in prediction_detail_rows]
        ape_raw_values = [float(row["absolute_percentage_error_raw"]) for row in prediction_detail_rows]
        ape_epsilon_values = [float(row["absolute_percentage_error_epsilon"]) for row in prediction_detail_rows]
        smape_values = [float(row["symmetric_absolute_percentage_error"]) for row in prediction_detail_rows]
        prediction_rows = [
            {
                "configuration": configuration,
                "algorithm": algorithm_name,
                "method_type": _algorithm_method_type(algorithm_name),
                "application_id": application.application_id,
                "graph_id": application.graph.graph_id,
                "MAE": mean(errors),
                "MSE": mean(squared),
                "RMSE": math.sqrt(mean(squared)),
                "MAPE_raw": mean(ape_raw_values),
                "MAPE_epsilon": mean(ape_epsilon_values),
                "sMAPE": mean(smape_values),
                "MAPE": mean(ape_epsilon_values),
                "mape_epsilon": PREDICTION_MAPE_EPSILON,
                "prediction_latency_ms": "",
                "unit_error": "sim_time_unit",
                "unit_latency": "milliseconds",
            }
        ]
    else:
        prediction_rows = []

    return {
        "task_rows": task_rows,
        "core_rows": core_rows,
        "schedule_rows": schedule_rows,
        "energy_rows": energy_rows,
        "prediction_rows": prediction_rows,
        "prediction_detail_rows": prediction_detail_rows,
        "power_trace_rows": _power_trace_rows(configuration, algorithm_name, application, platform, metrics),
    }


def _power_trace_rows(
    configuration: str,
    algorithm_name: str,
    application: DagApplication,
    platform: HeterogeneousPlatform,
    metrics: ScheduleMetrics,
) -> List[Dict[str, float | str]]:
    """Build event-based total-system power traces before and after DVFS.

    The trace is canonical at event boundaries: all starts/stops at the same
    timestamp are applied first, then the resulting total active power is emitted.
    This keeps the trace consistent with the peak-power helper and avoids stale
    pre-event values at timestamps where multiple tasks start or finish together.
    """

    events: List[tuple[float, float, float]] = []
    real_task_ids = set(application.graph.real_task_ids)
    for entry in metrics.entries:
        if entry.task_id not in real_task_ids:
            continue
        core = platform.core_by_id(entry.core_id)
        after_power = core.total_power(entry.dvfs_level_id)
        before_power = core.total_power(_max_dvfs_level_id(core))
        execution_finish = _entry_execution_finish(entry)
        events.append((entry.start_time, after_power, before_power))
        events.append((execution_finish, -after_power, -before_power))
        overhead_power = _mode_switch_overhead_power(entry)
        if overhead_power > 0.0:
            events.append((execution_finish, overhead_power, overhead_power))
            events.append((entry.finish_time, -overhead_power, -overhead_power))

    events.sort(key=lambda item: (item[0], item[1]))
    rows: List[Dict[str, float | str]] = []
    active_after = 0.0
    active_before = 0.0
    index = 0
    while index < len(events):
        event_time = events[index][0]
        while index < len(events) and abs(events[index][0] - event_time) <= 1e-12:
            active_after += events[index][1]
            active_before += events[index][2]
            index += 1
        rows.append(
            {
                "configuration": configuration,
                "algorithm": algorithm_name,
                "application_id": application.application_id,
                "graph_id": application.graph.graph_id,
                "time": event_time,
                "power_after_dvfs": max(active_after, 0.0),
                "power_before_dvfs": max(active_before, 0.0),
                "unit_time": "sim_time_unit",
                "unit_power": "relative_power_unit",
            }
        )
    return rows


def _write_detailed_run_outputs(
    configuration: str,
    schedule_results: Sequence[tuple[str, DagApplication, ScheduleMetrics]],
    platform: HeterogeneousPlatform,
    output_dir: Path | None,
) -> Dict[str, List[Dict[str, float | str]]]:
    """Builds and optionally writes all detailed per-task/core/schedule/energy/prediction CSV outputs."""

    all_rows: Dict[str, List[Dict[str, float | str]]] = {
        "task_rows": [],
        "core_rows": [],
        "schedule_rows": [],
        "energy_rows": [],
        "prediction_rows": [],
        "prediction_detail_rows": [],
        "power_trace_rows": [],
    }
    for algorithm_name, application, metrics in schedule_results:
        rows = _detailed_rows_for_schedule(
            configuration=configuration,
            algorithm_name=algorithm_name,
            application=application,
            platform=platform,
            metrics=metrics,
        )
        for key, value in rows.items():
            all_rows[key].extend(value)

    file_map = {
        "task_rows": "task_level_results.csv",
        "core_rows": "core_level_results.csv",
        "schedule_rows": "schedule_summary_extended.csv",
        "energy_rows": "energy_power_summary.csv",
        "prediction_rows": "prediction_metrics.csv",
        "prediction_detail_rows": "prediction_detail_results.csv",
        "power_trace_rows": "power_trace.csv",
    }
    if output_dir is not None:
        for key, filename in file_map.items():
            save_rows_as_csv(all_rows[key], output_dir / filename)
    return all_rows


def _training_loss_rows(
    gnn_baseline: ClassicGcnBaseline,
    proposed_scheduler: RiskAwareGatScheduler,
    policy_history: PolicyTrainingHistory | None,
) -> List[Dict[str, float | str]]:
    """Builds non-dummy training-curve rows when the run actually trained models."""

    rows: List[Dict[str, float | str]] = []
    gnn_history = getattr(gnn_baseline, "training_history", None)
    if gnn_history is not None:
        for epoch_index, loss_value in enumerate(getattr(gnn_history, "epoch_losses", []), start=1):
            rows.append({"model": "gnn", "curve": "training_loss", "step": float(epoch_index), "value": float(loss_value), "definition": "prediction_mse_loss"})
    gat_history = getattr(proposed_scheduler, "predictor_training_history", None)
    if gat_history is not None:
        prediction_mse_losses = list(getattr(gat_history, "prediction_mse_losses", []))
        if prediction_mse_losses:
            for epoch_index, loss_value in enumerate(prediction_mse_losses, start=1):
                rows.append({"model": "proposed_gat_float32", "curve": "training_loss", "step": float(epoch_index), "value": float(loss_value), "definition": "prediction_mse_loss"})
            for epoch_index, loss_value in enumerate(getattr(gat_history, "epoch_losses", []), start=1):
                rows.append({"model": "proposed_gat_float32", "curve": "training_objective", "step": float(epoch_index), "value": float(loss_value), "definition": "optimized_uncertainty_cvar_objective"})
        else:
            for epoch_index, loss_value in enumerate(getattr(gat_history, "epoch_losses", []), start=1):
                rows.append({"model": "proposed_gat_float32", "curve": "training_loss", "step": float(epoch_index), "value": float(loss_value), "definition": "legacy_total_objective"})
        for epoch_index, loss_value in enumerate(getattr(gat_history, "nll_losses", []), start=1):
            rows.append({"model": "proposed_gat_float32", "curve": "nll_loss", "step": float(epoch_index), "value": float(loss_value), "definition": "gaussian_nll_component"})
        for epoch_index, loss_value in enumerate(getattr(gat_history, "sigma_losses", []), start=1):
            rows.append({"model": "proposed_gat_float32", "curve": "sigma_loss", "step": float(epoch_index), "value": float(loss_value), "definition": "sigma_calibration_component"})
        for epoch_index, loss_value in enumerate(getattr(gat_history, "cvar_losses", []), start=1):
            rows.append({"model": "proposed_gat_float32", "curve": "cvar_loss", "step": float(epoch_index), "value": float(loss_value), "definition": "cvar_component"})
    if policy_history is not None:
        for episode_index, reward_value in enumerate(policy_history.rewards, start=1):
            rows.append({"model": "proposed_gat_float32", "curve": "policy_reward", "step": float(episode_index), "value": float(reward_value), "definition": "policy_reward_native_unit"})
        for episode_index, baseline_value in enumerate(policy_history.baselines, start=1):
            rows.append({"model": "proposed_gat_float32", "curve": "policy_moving_baseline", "step": float(episode_index), "value": float(baseline_value), "definition": "policy_reward_moving_baseline"})
    return rows

def _mean_metric(rows: Iterable[Dict[str, float | str]], metric_name: str) -> float:
    """Returns the arithmetic mean of one metric across rows."""

    values = [float(row[metric_name]) for row in rows]
    return 0.0 if not values else mean(values)


def summarize_algorithm_rows(rows: Sequence[Dict[str, float | str]]) -> Dict[str, Dict[str, float]]:
    """Aggregates rows into one summary per algorithm."""

    algorithms = sorted({str(row["algorithm"]) for row in rows})
    summary: Dict[str, Dict[str, float]] = {}
    for algorithm in algorithms:
        algorithm_rows = [row for row in rows if row["algorithm"] == algorithm]
        summary[algorithm] = {
            "num_graphs": float(len(algorithm_rows)),
            "mean_makespan": _mean_metric(algorithm_rows, "makespan"),
            "mean_energy": _mean_metric(algorithm_rows, "energy"),
            "mean_utilization": _mean_metric(algorithm_rows, "utilization"),
            "mean_deadline_miss_ratio": _mean_metric(algorithm_rows, "deadline_miss_ratio"),
            "mean_application_deadline_miss_ratio": _mean_metric(algorithm_rows, "application_deadline_miss_ratio"),
            "mean_task_deadline_miss_ratio": _mean_metric(algorithm_rows, "task_deadline_miss_ratio"),
            "mean_service_loss_ratio": _mean_metric(algorithm_rows, "service_loss_ratio"),
            "mean_dropped_lo_ratio": _mean_metric(algorithm_rows, "dropped_lo_ratio"),
            "mean_completed_lo_ratio": _mean_metric(algorithm_rows, "completed_lo_ratio"),
            "mean_completed_task_ratio": _mean_metric(algorithm_rows, "completed_task_ratio"),
            "mean_mode_switch_probability": _mean_metric(algorithm_rows, "mode_switch_probability"),
            "mean_mode_switch_count": _mean_metric(algorithm_rows, "mode_switch_count"),
            "mean_mode_switch_time_overhead": _mean_metric(algorithm_rows, "mode_switch_time_overhead"),
            "mean_mode_switch_energy_overhead": _mean_metric(algorithm_rows, "mode_switch_energy_overhead"),
            "mean_average_cvar": _mean_metric(algorithm_rows, "average_cvar"),
            "mean_aging_index": _mean_metric(algorithm_rows, "aging_index"),
            "mean_average_reliability": _mean_metric(algorithm_rows, "average_reliability"),
            "mean_completed_hi_ratio": _mean_metric(algorithm_rows, "completed_hi_ratio"),
        }
    return summary


def _csv_full_precision_value(value: float | int | str | bool | None) -> str | float | int | bool:
    """Formats CSV scalar values without presentation rounding.

    Python floats are stored as IEEE-754 double precision values. Seventeen
    significant digits are sufficient for exact round-trip reconstruction of the
    stored float, so CSV outputs preserve the numerical values used by the code.
    """

    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        if abs(value) < 1e-300:
            return "0"
        return format(value, ".17g")
    return value


def save_rows_as_csv(
    rows: Sequence[Dict[str, float | str]],
    output_path: str | Path,
    *,
    write_xlsx: bool = True,
    xlsx_row_limit: int = 200_000,
) -> None:
    """Writes flat rows as CSV using full float precision.

    XLSX output is intentionally capped.  Very large detailed tables such as
    task-level traces can contain millions of rows; building a full in-memory
    XLSX worksheet for them can exhaust RAM.  CSV remains the canonical detailed
    artifact, while XLSX is generated only for summary-sized tables.
    """

    if not rows:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.is_dir():
        shutil.rmtree(output_path)

    fieldnames: List[str] = []
    seen_fieldnames = set()
    for row in rows:
        for key in row.keys():
            if key not in seen_fieldnames:
                seen_fieldnames.add(key)
                fieldnames.append(key)
    row_count = len(rows)
    collect_for_xlsx = write_xlsx and output_path.suffix.lower() == ".csv" and row_count <= xlsx_row_limit
    formatted_rows_for_xlsx = [] if collect_for_xlsx else None

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            formatted_row = {key: _csv_full_precision_value(row.get(key, "")) for key in fieldnames}
            writer.writerow(formatted_row)
            if formatted_rows_for_xlsx is not None:
                formatted_rows_for_xlsx.append(formatted_row)

    if output_path.suffix.lower() == ".csv":
        xlsx_path = output_path.with_suffix(".xlsx")
        if collect_for_xlsx and formatted_rows_for_xlsx is not None:
            save_rows_as_xlsx(formatted_rows_for_xlsx, xlsx_path, sheet_name=output_path.stem[:31])
        elif write_xlsx:
            output_path.with_suffix(".xlsx.skipped.txt").write_text(
                "XLSX generation skipped because the table is too large for a memory-safe XLSX export.\n"
                f"CSV file: {output_path.name}\n"
                f"Rows: {row_count}\n"
                f"XLSX row limit: {xlsx_row_limit}\n",
                encoding="utf-8",
            )


def _summarize_application_set(applications: Sequence[DagApplication]) -> Dict[str, float]:
    """Builds summary statistics for one application set."""

    if not applications:
        return {
            "mean_num_real_tasks": 0.0,
            "mean_num_edges": 0.0,
            "mean_num_levels": 0.0,
            "mean_total_utilization": 0.0,
        }

    return {
        "mean_num_real_tasks": mean(len(application.graph.real_task_ids) for application in applications),
        "mean_num_edges": mean(application.graph.edge_count() for application in applications),
        "mean_num_levels": mean(len(application.graph.levels) for application in applications),
        "mean_total_utilization": mean(application.total_utilization for application in applications),
    }


def _default_reward_weights() -> RewardWeights:
    """Returns the default reward weights used across experiments."""

    return RewardWeights(w_u=2.0, w_s=1.0, w_r=0.3, w_d=2.0, w_a=0.2)


def _shared_model_root(output_dir: str | Path | None, directory_name: str = "trained_models") -> Path | None:
    """Resolves a checkpoint directory for one experiment output tree.

    Sibling directories such as ``baseline_suite`` and ``quantization`` should share a
    checkpoint root at their parent configuration directory. Other outputs, such as
    structural-sweep setting directories, should keep checkpoints local to the setting
    directory itself.
    """

    if output_dir is None:
        return None
    output_dir = Path(output_dir)
    if output_dir.name in {"baseline_suite", "quantization", "uncertainty_ablation"}:
        return output_dir.parent / directory_name
    return output_dir / directory_name


def _prepare_gnn_baseline(
    train_applications: Sequence[DagApplication],
    training_config: TrainingConfig,
    checkpoint_root: Path | None,
) -> ClassicGcnBaseline:
    """Loads or trains the GNN baseline."""

    checkpoint_dir = None if checkpoint_root is None else checkpoint_root / "gnn"
    if checkpoint_dir is not None and (checkpoint_dir / "metadata.json").exists() and (checkpoint_dir / "model_state.pt").exists():
        print(f"[GNN] loaded checkpoint: {checkpoint_dir}")
        return ClassicGcnBaseline.load_checkpoint(checkpoint_dir, device="cpu")

    print("[GNN] training started")
    baseline = ClassicGcnBaseline(device="cpu")
    gnn_history = baseline.fit(applications=list(train_applications), training_config=training_config, verbose=False)
    setattr(baseline, "training_history", gnn_history)
    print("[GNN] training finished")
    if checkpoint_dir is not None:
        baseline.save_checkpoint(checkpoint_dir)
    return baseline


def _prepare_proposed_scheduler(
    train_applications: Sequence[DagApplication],
    platform: HeterogeneousPlatform,
    training_config: TrainingConfig,
    policy_episodes: int,
    reward_weights: RewardWeights,
    checkpoint_root: Path | None,
    use_uncertainty_features: bool = True,
    use_qat: bool = False,
    checkpoint_name: str = "proposed_gat_float32",
) -> tuple[RiskAwareGatScheduler, PolicyTrainingHistory | None]:
    """Loads or trains the proposed scheduler, optionally with QAT noise."""

    checkpoint_dir = None if checkpoint_root is None else checkpoint_root / checkpoint_name
    if checkpoint_dir is not None and (checkpoint_dir / "metadata.json").exists() and (checkpoint_dir / "predictor_state.pt").exists():
        print(f"[{checkpoint_name}] loaded checkpoint: {checkpoint_dir}")
        scheduler = RiskAwareGatScheduler.load_checkpoint(checkpoint_dir, device="cpu", quantize_int8=False)
        return scheduler, None

    print(f"[{checkpoint_name}] training started")
    scheduler = RiskAwareGatScheduler(device="cpu", use_qat=use_qat, use_uncertainty_features=use_uncertainty_features)
    predictor_history = scheduler.fit_predictor(
        applications=train_applications,
        platform=platform,
        training_config=training_config,
        verbose=False,
    )
    setattr(scheduler, "predictor_training_history", predictor_history)
    policy_history = scheduler.fit_policy(
        applications=train_applications,
        platform=platform,
        num_episodes=policy_episodes,
        learning_rate=1e-3,
        reward_weights=reward_weights,
        verbose=False,
    )
    print(f"[{checkpoint_name}] training finished")
    if checkpoint_dir is not None:
        scheduler.save_checkpoint(checkpoint_dir)
    return scheduler, policy_history




def _checkpoint_summary_path(checkpoint_root: Path | None, model_name: str) -> str:
    """Returns a portable checkpoint path for CSV summaries.

    Older outputs used ``Path.resolve()`` and stored machine-specific absolute
    machine-specific absolute paths.  The checkpoint files themselves are unchanged;
    this function only keeps reports portable and easier to review.
    """

    if checkpoint_root is None:
        return ""
    path = checkpoint_root / model_name
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def _learned_model_summary_rows(
    gnn_baseline: ClassicGcnBaseline,
    proposed_float_scheduler: RiskAwareGatScheduler,
    proposed_int8_scheduler: RiskAwareGatScheduler,
    checkpoint_root: Path | None,
    reference_application: DagApplication,
    platform: HeterogeneousPlatform,
    benchmark_latency: bool,
) -> List[Dict[str, float | str]]:
    """Builds metadata rows for learned-model reporting."""

    gnn_predictor_latency = gnn_baseline.benchmark_predictor_latency_ms(reference_application) if benchmark_latency else 0.0
    gnn_schedule_latency = gnn_baseline.benchmark_schedule_latency_ms(reference_application, platform) if benchmark_latency else 0.0
    float_predictor_latency = proposed_float_scheduler.benchmark_predictor_latency_ms(reference_application, platform) if benchmark_latency else 0.0
    float_policy_latency = proposed_float_scheduler.benchmark_policy_latency_ms(reference_application, platform) if benchmark_latency else 0.0
    float_schedule_latency = proposed_float_scheduler.benchmark_schedule_latency_ms(reference_application, platform) if benchmark_latency else 0.0
    int8_predictor_latency = proposed_int8_scheduler.benchmark_predictor_latency_ms(reference_application, platform) if benchmark_latency else 0.0
    int8_policy_latency = proposed_int8_scheduler.benchmark_policy_latency_ms(reference_application, platform) if benchmark_latency else 0.0
    int8_schedule_latency = proposed_int8_scheduler.benchmark_schedule_latency_ms(reference_application, platform) if benchmark_latency else 0.0

    rows: List[Dict[str, float | str]] = []
    rows.append(
        {
            "algorithm": "gnn",
            "method_type": "learned_predictor",
            "precision_variant": "fp32",
            "parameter_count": float(gnn_baseline.parameter_count()),
            "predictor_parameter_count": float(gnn_baseline.parameter_count()),
            "policy_parameter_count": 0.0,
            "model_size_bytes": float(gnn_baseline.model_size_bytes()),
            "predictor_size_bytes": float(gnn_baseline.model_size_bytes()),
            "policy_size_bytes": 0.0,
            "predictor_latency_ms": float(gnn_predictor_latency),
            "policy_latency_ms": 0.0,
            "scheduler_decision_latency_ms": float(gnn_schedule_latency),
            "end_to_end_latency_ms": float(gnn_schedule_latency),
            "scheduler_latency_ms": float(gnn_schedule_latency),
            "checkpoint_dir": _checkpoint_summary_path(checkpoint_root, "gnn"),
        }
    )
    float_predictor_size = estimate_model_size_bytes(proposed_float_scheduler.predictor)
    float_policy_size = 0 if proposed_float_scheduler.policy_network is None else estimate_model_size_bytes(proposed_float_scheduler.policy_network)
    int8_predictor_size = estimate_int8_weight_storage_bytes(
        proposed_float_scheduler.predictor,
        per_channel=True,
        include_parameter=pdf_qat_parameter_selector,
    )
    int8_policy_size = 0 if proposed_float_scheduler.policy_network is None else estimate_int8_weight_storage_bytes(
        proposed_float_scheduler.policy_network,
        per_channel=True,
        include_parameter=pdf_qat_parameter_selector,
    )
    total_parameter_count = proposed_float_scheduler.predictor_parameter_count() + proposed_float_scheduler.policy_parameter_count()
    rows.append(
        {
            "algorithm": "proposed_gat_float32",
            "method_type": "learned_predictor",
            "precision_variant": "fp32",
            "parameter_count": float(total_parameter_count),
            "predictor_parameter_count": float(proposed_float_scheduler.predictor_parameter_count()),
            "policy_parameter_count": float(proposed_float_scheduler.policy_parameter_count()),
            "model_size_bytes": float(float_predictor_size + float_policy_size),
            "predictor_size_bytes": float(float_predictor_size),
            "policy_size_bytes": float(float_policy_size),
            "predictor_latency_ms": float(float_predictor_latency),
            "policy_latency_ms": float(float_policy_latency),
            "scheduler_decision_latency_ms": float(float_schedule_latency),
            "end_to_end_latency_ms": float(float_schedule_latency),
            "scheduler_latency_ms": float(float_schedule_latency),
            "checkpoint_dir": _checkpoint_summary_path(checkpoint_root, "proposed_gat_float32"),
        }
    )
    rows.append(
        {
            "algorithm": "proposed_gat_quantization_int8",
            "method_type": "learned_predictor",
            "precision_variant": "qat_int8_pdf",
            "parameter_count": float(total_parameter_count),
            "predictor_parameter_count": float(proposed_float_scheduler.predictor_parameter_count()),
            "policy_parameter_count": float(proposed_float_scheduler.policy_parameter_count()),
            "model_size_bytes": float(int8_predictor_size + int8_policy_size),
            "predictor_size_bytes": float(int8_predictor_size),
            "policy_size_bytes": float(int8_policy_size),
            "predictor_latency_ms": float(int8_predictor_latency),
            "policy_latency_ms": float(int8_policy_latency),
            "scheduler_decision_latency_ms": float(int8_schedule_latency),
            "end_to_end_latency_ms": float(int8_schedule_latency),
            "scheduler_latency_ms": float(int8_schedule_latency),
            "checkpoint_dir": _checkpoint_summary_path(checkpoint_root, "proposed_gat_float32"),
        }
    )
    return rows


def run_baseline_suite(
    applications: Sequence[DagApplication],
    platform: HeterogeneousPlatform,
    output_dir: str | Path | None = None,
    training_config: TrainingConfig | None = None,
    policy_episodes: int = 8,
    nsga_config: NsgaThreeConfig | None = None,
    reward_weights: RewardWeights | None = None,
    train_ratio: float = 0.8,
    benchmark_learned_model_latency: bool = False,
    configuration_label: str = "",
) -> BaselineSuiteArtifacts:
    """Runs HEFT, VD-EDF, NSGA-III, GNN, and both float32/int8 proposed variants."""

    if not applications:
        raise ValueError("applications must not be empty.")

    split = split_applications(applications, train_ratio=train_ratio)
    train_applications = split.train_applications
    test_applications = split.test_applications
    training_config = training_config or TrainingConfig(learning_rate=5e-3, num_epochs=3, batch_size=1, seed=0)
    reward_weights = reward_weights or _default_reward_weights()
    nsga_config = nsga_config or NsgaThreeConfig(population_size=10, num_generations=4, reference_divisions=4, seed=0)

    checkpoint_root = _shared_model_root(output_dir, directory_name="trained_models")

    heft_scheduler = HeftScheduler()
    vd_edf_scheduler = VdEdfScheduler()
    nsga_scheduler = NsgaThreeScheduler(config=nsga_config)

    gnn_baseline = _prepare_gnn_baseline(train_applications, training_config, checkpoint_root)
    proposed_float_scheduler, policy_history = _prepare_proposed_scheduler(
        train_applications,
        platform,
        training_config,
        policy_episodes,
        reward_weights,
        checkpoint_root,
        use_uncertainty_features=True,
        checkpoint_name="proposed_gat_float32",
    )
    proposed_int8_scheduler = proposed_float_scheduler.build_int8_inference_variant()

    rows: List[Dict[str, float | str]] = []
    schedule_results: List[tuple[str, DagApplication, ScheduleMetrics]] = []
    for application in test_applications:
        for algorithm_name, metrics in [
            ("heft", heft_scheduler.schedule(application, platform)),
            ("vd_edf", vd_edf_scheduler.schedule(application, platform)),
            ("nsga_iii", nsga_scheduler.schedule(application, platform)),
            ("gnn", gnn_baseline.schedule(application, platform)),
            ("proposed_gat_float32", proposed_float_scheduler.schedule(application, platform)),
            ("proposed_gat_quantization_int8", proposed_int8_scheduler.schedule(application, platform)),
        ]:
            rows.append(_metrics_to_row(algorithm_name, application, metrics))
            schedule_results.append((algorithm_name, application, metrics))

    summary = summarize_algorithm_rows(rows)
    reference_application = test_applications[0]
    learned_models_summary = _learned_model_summary_rows(
        gnn_baseline,
        proposed_float_scheduler,
        proposed_int8_scheduler,
        checkpoint_root,
        reference_application,
        platform,
        benchmark_latency=benchmark_learned_model_latency,
    )

    detailed_rows: Dict[str, List[Dict[str, float | str]]] | None = None
    if output_dir is not None:
        output_dir = Path(output_dir)
        save_rows_as_csv(rows, output_dir / "baseline_per_graph.csv")
        detailed_rows = _write_detailed_run_outputs(
            configuration=configuration_label,
            schedule_results=schedule_results,
            platform=platform,
            output_dir=output_dir,
        )
        latency_by_algorithm = {str(row.get("algorithm")): row.get("predictor_latency_ms", "") for row in learned_models_summary}
        for prediction_row in detailed_rows.get("prediction_rows", []):
            algorithm = str(prediction_row.get("algorithm", ""))
            if algorithm in latency_by_algorithm:
                prediction_row["prediction_latency_ms"] = latency_by_algorithm[algorithm]
        save_rows_as_csv(detailed_rows.get("prediction_rows", []), output_dir / "prediction_metrics.csv")
        training_loss_rows = _training_loss_rows(
            gnn_baseline=gnn_baseline,
            proposed_scheduler=proposed_float_scheduler,
            policy_history=policy_history,
        )
        save_rows_as_csv(training_loss_rows, output_dir / "training_loss_curves.csv")
        save_json(summary, output_dir / "baseline_summary.json")
        save_json(learned_models_summary, output_dir / "learned_models_summary.json")
        save_json(
            {"rewards": [] if policy_history is None else policy_history.rewards, "moving_baselines": [] if policy_history is None else policy_history.baselines},
            output_dir / "policy_reward_curve.json",
        )
    else:
        detailed_rows = _write_detailed_run_outputs(
            configuration=configuration_label,
            schedule_results=schedule_results,
            platform=platform,
            output_dir=None,
        )

    return BaselineSuiteArtifacts(
        summary_by_algorithm=summary,
        per_graph_rows=rows,
        policy_history=policy_history,
        learned_models_summary=learned_models_summary,
        detailed_rows=detailed_rows,
    )


def run_uncertainty_ablation(
    applications: Sequence[DagApplication],
    platform: HeterogeneousPlatform,
    output_dir: str | Path | None = None,
    training_config: TrainingConfig | None = None,
    policy_episodes: int = 8,
    reward_weights: RewardWeights | None = None,
    train_ratio: float = 0.8,
) -> UncertaintyAblationArtifacts:
    """Compares the proposed scheduler with and without uncertainty features."""

    if not applications:
        raise ValueError("applications must not be empty.")

    split = split_applications(applications, train_ratio=train_ratio)
    train_applications = split.train_applications
    test_applications = split.test_applications
    training_config = training_config or TrainingConfig(learning_rate=5e-3, num_epochs=3, batch_size=1, seed=0)
    reward_weights = reward_weights or _default_reward_weights()
    checkpoint_root = _shared_model_root(output_dir, directory_name="trained_models_uncertainty")

    with_uncertainty, _ = _prepare_proposed_scheduler(
        train_applications,
        platform,
        training_config,
        policy_episodes,
        reward_weights,
        checkpoint_root,
        use_uncertainty_features=True,
        checkpoint_name="with_uncertainty",
    )
    without_uncertainty, _ = _prepare_proposed_scheduler(
        train_applications,
        platform,
        training_config,
        policy_episodes,
        reward_weights,
        checkpoint_root,
        use_uncertainty_features=False,
        checkpoint_name="without_uncertainty",
    )

    rows: List[Dict[str, float | str]] = []
    for application in test_applications:
        rows.append(_metrics_to_row("with_uncertainty", application, with_uncertainty.schedule(application, platform)))
        rows.append(_metrics_to_row("without_uncertainty", application, without_uncertainty.schedule(application, platform)))

    summary = summarize_algorithm_rows(rows)
    if output_dir is not None:
        output_dir = Path(output_dir)
        save_rows_as_csv(rows, output_dir / "uncertainty_ablation_per_graph.csv")
        save_json(summary, output_dir / "uncertainty_ablation_summary.json")

    return UncertaintyAblationArtifacts(summary_by_variant=summary, per_graph_rows=rows)


def _mean_method_metric(rows: Sequence[Dict[str, float | str]], metric_name: str) -> float:
    """Returns a finite numeric mean for a metric inside method-comparison rows."""

    values: List[float] = []
    for row in rows:
        value = row.get(metric_name)
        if value in (None, ""):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values.append(number)
    return 0.0 if not values else mean(values)


def _summarize_quantization_method_rows(rows: Sequence[Dict[str, float | str]]) -> Dict[str, Dict[str, float | str]]:
    """Aggregates per-graph quantization rows by method name."""

    summary: Dict[str, Dict[str, float | str]] = {}
    for method_name in sorted({str(row["method_name"]) for row in rows}):
        method_rows = [row for row in rows if str(row["method_name"]) == method_name]
        first_row = method_rows[0]
        summary[method_name] = {
            "display_name": str(first_row.get("display_name", method_name)),
            "reason_for_inclusion": str(first_row.get("reason_for_inclusion", "")),
            "is_pdf_method": float(first_row.get("is_pdf_method", 0.0)),
            "num_graphs": int(len(method_rows)),
            "num_successful_graphs": int(sum(1 for row in method_rows if str(row.get("status", "")).lower() == "success")),
            "num_failed_graphs": int(sum(1 for row in method_rows if str(row.get("status", "")).lower() == "failed")),
            "success_rate": (
                float(sum(1 for row in method_rows if str(row.get("status", "")).lower() == "success")) / max(1.0, float(len(method_rows)))
            ),
            "average_predictor_size_bytes": _mean_method_metric(method_rows, "predictor_size_bytes"),
            "average_policy_size_bytes": _mean_method_metric(method_rows, "policy_size_bytes"),
            "average_scheduler_size_bytes": _mean_method_metric(method_rows, "scheduler_size_bytes"),
            "average_relative_size_reduction_vs_fp32": _mean_method_metric(method_rows, "relative_size_reduction_vs_fp32"),
            "average_mean_absolute_prediction_error_vs_fp32": _mean_method_metric(method_rows, "mean_absolute_prediction_error_vs_fp32"),
            "average_predictor_latency_ms": _mean_method_metric(method_rows, "predictor_latency_ms"),
            "average_policy_latency_ms": _mean_method_metric(method_rows, "policy_latency_ms"),
            "average_scheduler_latency_ms": _mean_method_metric(method_rows, "scheduler_latency_ms"),
            "average_makespan": _mean_method_metric(method_rows, "makespan"),
            "average_energy": _mean_method_metric(method_rows, "energy"),
            "average_utilization": _mean_method_metric(method_rows, "utilization"),
            "average_deadline_miss_ratio": _mean_method_metric(method_rows, "deadline_miss_ratio"),
            "average_task_deadline_miss_ratio": _mean_method_metric(method_rows, "task_deadline_miss_ratio"),
            "average_mode_switch_probability": _mean_method_metric(method_rows, "mode_switch_probability"),
            "average_average_cvar": _mean_method_metric(method_rows, "average_cvar"),
            "average_aging_index": _mean_method_metric(method_rows, "aging_index"),
            "average_average_reliability": _mean_method_metric(method_rows, "average_reliability"),
            "average_completed_hi_ratio": _mean_method_metric(method_rows, "completed_hi_ratio"),
            "average_completed_lo_ratio": _mean_method_metric(method_rows, "completed_lo_ratio"),
            "average_dropped_lo_ratio": _mean_method_metric(method_rows, "dropped_lo_ratio"),
            "average_service_loss_ratio": _mean_method_metric(method_rows, "service_loss_ratio"),
        }
    return summary


def run_quantization_experiment(
    applications: Sequence[DagApplication],
    platform: HeterogeneousPlatform,
    output_dir: str | Path | None = None,
    training_config: TrainingConfig | None = None,
    benchmark_latency: bool = False,
    train_ratio: float = 0.8,
) -> QuantizationArtifacts:
    """Runs multi-method quantization comparison beside the original FP32 model.

    The comparison includes the PDF-oriented QAT INT8 method plus two
    deployment-relevant comparison methods: weight-only INT8 and FP16.
    Dynamic INT8 Linear is intentionally excluded from defense tables because its
    skipped CPU-kernel rows produced misleading zero-valued metrics.
    """

    if not applications:
        raise ValueError("applications must not be empty.")

    split = split_applications(applications, train_ratio=train_ratio)
    train_applications = split.train_applications
    test_applications = split.test_applications
    training_config = training_config or TrainingConfig(learning_rate=5e-3, num_epochs=2, batch_size=1, seed=0)
    checkpoint_root = _shared_model_root(output_dir, directory_name="trained_models")

    float_scheduler, _ = _prepare_proposed_scheduler(
        train_applications,
        platform,
        training_config,
        policy_episodes=4,
        reward_weights=_default_reward_weights(),
        checkpoint_root=checkpoint_root,
        use_uncertainty_features=True,
        use_qat=False,
        checkpoint_name="proposed_gat_float32",
    )
    qat_scheduler, _ = _prepare_proposed_scheduler(
        train_applications,
        platform,
        training_config,
        policy_episodes=4,
        reward_weights=_default_reward_weights(),
        checkpoint_root=checkpoint_root,
        use_uncertainty_features=True,
        use_qat=True,
        checkpoint_name="proposed_gat_qat_int8_pdf",
    )

    rows: List[Dict[str, float | str]] = []
    for application in test_applications:
        method_comparisons = compare_float_and_quantization_methods(
            float_scheduler=float_scheduler,
            qat_scheduler=qat_scheduler,
            application=application,
            platform=platform,
            benchmark_latency=benchmark_latency,
        )
        for comparison in method_comparisons:
            rows.append(
                {
                    "application_id": application.application_id,
                    "graph_id": application.graph.graph_id,
                    "method_name": comparison.method_name,
                    "display_name": comparison.display_name,
                    "reason_for_inclusion": comparison.reason_for_inclusion,
                    "is_pdf_method": 1.0 if comparison.is_pdf_method else 0.0,
                    "predictor_size_bytes": float(comparison.predictor_size_bytes),
                    "policy_size_bytes": float(comparison.policy_size_bytes),
                    "scheduler_size_bytes": float(comparison.scheduler_size_bytes),
                    "relative_size_reduction_vs_fp32": comparison.relative_size_reduction_vs_fp32,
                    "mean_absolute_prediction_error_vs_fp32": comparison.mean_absolute_prediction_error_vs_fp32,
                    "predictor_latency_ms": comparison.predictor_latency_ms,
                    "policy_latency_ms": comparison.policy_latency_ms,
                    "scheduler_latency_ms": comparison.scheduler_latency_ms,
                    "status": comparison.status,
                    "error_message": comparison.error_message,
                    "makespan": comparison.makespan,
                    "energy": comparison.energy,
                    "utilization": comparison.utilization,
                    "deadline_miss_ratio": comparison.deadline_miss_ratio,
                    "application_deadline_miss_ratio": comparison.application_deadline_miss_ratio,
                    "task_deadline_miss_ratio": comparison.task_deadline_miss_ratio,
                    "mode_switch_probability": comparison.mode_switch_probability,
                    "average_cvar": comparison.average_cvar,
                    "aging_index": comparison.aging_index,
                    "average_reliability": comparison.average_reliability,
                    "completed_hi_ratio": comparison.completed_hi_ratio,
                    "completed_lo_ratio": comparison.completed_lo_ratio,
                    "dropped_lo_ratio": comparison.dropped_lo_ratio,
                    "completed_task_ratio": comparison.completed_task_ratio,
                    "service_loss_ratio": comparison.service_loss_ratio,
                }
            )

    summary_by_method = _summarize_quantization_method_rows(rows)
    fp32_summary = summary_by_method.get("original_fp32_reference", {})
    pdf_qat_summary = summary_by_method.get("qat_int8_pdf", {})

    artifacts = QuantizationArtifacts(
        average_float_predictor_size_bytes=float(fp32_summary.get("average_predictor_size_bytes", 0.0)),
        average_int8_predictor_size_bytes=float(pdf_qat_summary.get("average_predictor_size_bytes", 0.0)),
        average_float_policy_size_bytes=float(fp32_summary.get("average_policy_size_bytes", 0.0)),
        average_int8_policy_size_bytes=float(pdf_qat_summary.get("average_policy_size_bytes", 0.0)),
        average_float_scheduler_size_bytes=float(fp32_summary.get("average_scheduler_size_bytes", 0.0)),
        average_int8_scheduler_size_bytes=float(pdf_qat_summary.get("average_scheduler_size_bytes", 0.0)),
        average_relative_size_reduction=float(pdf_qat_summary.get("average_relative_size_reduction_vs_fp32", 0.0)),
        average_mean_absolute_prediction_error=float(pdf_qat_summary.get("average_mean_absolute_prediction_error_vs_fp32", 0.0)),
        average_float_predictor_latency_ms=float(fp32_summary.get("average_predictor_latency_ms", 0.0)),
        average_int8_predictor_latency_ms=float(pdf_qat_summary.get("average_predictor_latency_ms", 0.0)),
        average_float_policy_latency_ms=float(fp32_summary.get("average_policy_latency_ms", 0.0)),
        average_int8_policy_latency_ms=float(pdf_qat_summary.get("average_policy_latency_ms", 0.0)),
        per_graph_rows=rows,
        summary_by_method=summary_by_method,
        method_rows=rows,
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        save_rows_as_csv(rows, output_dir / "quantization_per_graph.csv")
        method_summary_rows = [
            {"method_name": method_name, **summary}
            for method_name, summary in summary_by_method.items()
        ]
        save_rows_as_csv(method_summary_rows, output_dir / "quantization_summary_by_method.csv")
        save_json(summary_by_method, output_dir / "quantization_methods_summary.json")
        save_json(
            {
                "average_float_predictor_size_bytes": artifacts.average_float_predictor_size_bytes,
                "average_int8_predictor_size_bytes": artifacts.average_int8_predictor_size_bytes,
                "average_float_policy_size_bytes": artifacts.average_float_policy_size_bytes,
                "average_int8_policy_size_bytes": artifacts.average_int8_policy_size_bytes,
                "average_float_scheduler_size_bytes": artifacts.average_float_scheduler_size_bytes,
                "average_int8_scheduler_size_bytes": artifacts.average_int8_scheduler_size_bytes,
                "average_relative_size_reduction": artifacts.average_relative_size_reduction,
                "average_mean_absolute_prediction_error": artifacts.average_mean_absolute_prediction_error,
                "average_float_predictor_latency_ms": artifacts.average_float_predictor_latency_ms,
                "average_int8_predictor_latency_ms": artifacts.average_int8_predictor_latency_ms,
                "average_float_policy_latency_ms": artifacts.average_float_policy_latency_ms,
                "average_int8_policy_latency_ms": artifacts.average_int8_policy_latency_ms,
                "pdf_qat_method_name": "qat_int8_pdf",
                "methods": summary_by_method,
            },
            output_dir / "quantization_summary.json",
        )

    return artifacts


def run_structural_sweep(
    output_dir: str | Path,
    node_counts: Sequence[int] = PDF_NODE_COUNTS,
    core_counts: Sequence[int] = PDF_CORE_COUNTS,
    utilization_levels: Sequence[float] = PDF_CORE_UTILIZATION_LEVELS,
    graphs_per_setting: int = 10,
    density: float = 0.4,
    regularity: float = 0.5,
    width: float = 0.6,
    training_config: TrainingConfig | None = None,
    policy_episodes: int = 6,
    seed: int = 0,
    scale_nodes_with_cores: bool = True,
    core_reference_count: int = 4,
    max_scaled_node_count: int | None = None,
) -> List[Dict[str, float | str]]:
    """Runs the structural sweeps requested in the PDF.

    By default, the graph size is coupled to the number of cores.  This makes the
    core-count sweep meaningful: when the platform has more cores, the workload
    also has proportionally more tasks, so energy/utilization changes are not
    hidden by an under-filled platform.
    """

    output_dir = Path(output_dir)
    training_config = training_config or TrainingConfig(learning_rate=5e-3, num_epochs=2, batch_size=1, seed=seed)
    rows: List[Dict[str, float | str]] = []

    for base_node_count in node_counts:
        for core_count in core_counts:
            effective_node_count = (
                scaled_node_count_for_core_sweep(
                    base_node_count=base_node_count,
                    core_count=core_count,
                    reference_core_count=core_reference_count,
                    max_scaled_node_count=max_scaled_node_count,
                )
                if scale_nodes_with_cores
                else base_node_count
            )
            for utilization_level in utilization_levels:
                builder = build_pdf_workload_builder(
                    num_nodes=effective_node_count,
                    density=density,
                    regularity=regularity,
                    width=width,
                    seed=seed + effective_node_count + core_count,
                )
                if utilization_level <= 0.0:
                    raise ValueError("utilization_level must be positive.")

                workload_scale_factor = (
                    float(effective_node_count) / max(1.0, float(base_node_count))
                    if scale_nodes_with_cores
                    else 1.0
                )
                effective_total_utilization = utilization_level * workload_scale_factor

                applications = [
                    builder.build_single_application(
                        application_id=f"sweep_app_{application_index}",
                        total_utilization=effective_total_utilization,
                    )
                    for application_index in range(graphs_per_setting)
                ]
                num_a7 = core_count // 2
                num_a15 = core_count - num_a7
                platform = build_big_little_platform(num_a7=num_a7, num_a15=num_a15)
                suite_output_dir = (
                    output_dir
                    / f"base_nodes_{base_node_count}"
                    / f"nodes_{effective_node_count}"
                    / f"cores_{core_count}"
                    / f"util_{_float_tag(utilization_level)}"
                )
                artifacts = run_baseline_suite(
                    applications=applications,
                    platform=platform,
                    output_dir=suite_output_dir,
                    training_config=training_config,
                    policy_episodes=policy_episodes,
                    nsga_config=NsgaThreeConfig(population_size=8, num_generations=3, reference_divisions=4, seed=seed),
                    train_ratio=0.5,
                    benchmark_learned_model_latency=False,
                )
                for algorithm_name, summary in artifacts.summary_by_algorithm.items():
                    rows.append(
                        {
                            "algorithm": algorithm_name,
                            "base_node_count": float(base_node_count),
                            "node_count": float(effective_node_count),
                            "core_count": float(core_count),
                            "nodes_per_core": float(effective_node_count) / max(1.0, float(core_count)),
                            "utilization_level": utilization_level,
                            "base_total_utilization": utilization_level,
                            "effective_total_utilization": effective_total_utilization,
                            "workload_scale_factor": workload_scale_factor,
                            "scale_nodes_with_cores": 1.0 if scale_nodes_with_cores else 0.0,
                            **summary,
                        }
                    )

    save_rows_as_csv(rows, output_dir / "structural_sweep_summary.csv")
    return rows


def run_full_pdf_pipeline(
    output_root: str | Path,
    dataset_root: str | Path | None = None,
    generate_dataset_first: bool = True,
    smoke_mode: bool = False,
) -> Dict[str, str]:
    """Runs the whole project workflow in a single entry point."""

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_root = Path(dataset_root) if dataset_root is not None else output_root / "datasets" / "pdf_phase_1"
    if generate_dataset_first:
        if smoke_mode:
            generate_pdf_phase_1_dataset(
                output_root=dataset_root,
                graph_counts=[4],
                densities=[0.4],
                regularities=[0.5],
                widths=[0.6],
                num_nodes=20,
                mean_graph_utilization=0.65,
                max_graph_utilization=0.85,
                period=80.0,
                seed=7,
            )
        else:
            generate_pdf_phase_1_dataset(output_root=dataset_root)

    configuration_directories = collect_configuration_directories(dataset_root)
    if not configuration_directories:
        raise RuntimeError("No dataset configurations were found.")

    selected_configuration = configuration_directories[0]
    applications = load_applications_from_configuration(selected_configuration)
    platform = build_big_little_platform(num_a7=4, num_a15=4)

    baseline_artifacts = run_baseline_suite(
        applications=applications,
        platform=platform,
        output_dir=output_root / "results" / "baseline_suite",
        training_config=TrainingConfig(learning_rate=5e-3, num_epochs=2 if smoke_mode else 3, batch_size=1, seed=0),
        policy_episodes=4 if smoke_mode else 8,
        nsga_config=NsgaThreeConfig(population_size=8 if smoke_mode else 10, num_generations=3 if smoke_mode else 4, reference_divisions=4, seed=0),
    )
    uncertainty_artifacts = run_uncertainty_ablation(
        applications=applications,
        platform=platform,
        output_dir=output_root / "results" / "uncertainty_ablation",
        training_config=TrainingConfig(learning_rate=5e-3, num_epochs=2 if smoke_mode else 3, batch_size=1, seed=1),
        policy_episodes=4 if smoke_mode else 8,
    )
    quantization_artifacts = run_quantization_experiment(
        applications=applications,
        platform=platform,
        output_dir=output_root / "results" / "quantization",
        training_config=TrainingConfig(learning_rate=5e-3, num_epochs=1 if smoke_mode else 2, batch_size=1, seed=2),
        benchmark_latency=not smoke_mode,
    )
    if smoke_mode:
        run_structural_sweep(
            output_dir=output_root / "results" / "structural_sweep",
            node_counts=[20],
            core_counts=[4],
            utilization_levels=[0.5],
            graphs_per_setting=3,
            training_config=TrainingConfig(learning_rate=5e-3, num_epochs=1, batch_size=1, seed=3),
            policy_episodes=3,
            seed=3,
        )

    save_json(
        {
            "selected_configuration": str(selected_configuration),
            "baseline_algorithms": list(baseline_artifacts.summary_by_algorithm.keys()),
            "uncertainty_variants": list(uncertainty_artifacts.summary_by_variant.keys()),
            "average_int8_size_reduction": quantization_artifacts.average_relative_size_reduction,
        },
        output_root / "results" / "pipeline_summary.json",
    )

    return {
        "dataset_root": str(dataset_root),
        "selected_configuration": str(selected_configuration),
        "results_root": str(output_root / "results"),
    }
