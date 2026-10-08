"""Simulates dependency-aware list scheduling and records per-task, per-core, schedule, energy, power, and prediction details.

This module is part of the ESRLab defense-ready codebase. It uses snake_case for functions and variables while keeping Python classes in PascalCase, and exposes documented helpers for reproducible experiments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from core.metrics import ScheduleEntry, ScheduleMetrics
from core.random_utils import build_rng
from core.task_graph import TaskGraph
from hardware_model.hardware import HeterogeneousPlatform
from risk_modeling.risk_models import empirical_cvar_upper, predict_mode_switch_probability


DEFAULT_MODE_SWITCH_TIME_OVERHEAD = 0.05
DEFAULT_MODE_SWITCH_POWER_OVERHEAD = 0.25


@dataclass(slots=True)
class DecodedSchedule:
    """Concrete schedule produced by a decoder."""

    entries: List[ScheduleEntry] = field(default_factory=list)
    start_times: Dict[int, float] = field(default_factory=dict)
    finish_times: Dict[int, float] = field(default_factory=dict)
    assigned_cores: Dict[int, int] = field(default_factory=dict)
    assigned_dvfs: Dict[int, int] = field(default_factory=dict)
    durations: Dict[int, float] = field(default_factory=dict)
    mode_switch_count: int = 0
    dropped_lo_tasks: Set[int] = field(default_factory=set)
    runtime_samples_by_task: Dict[int, List[float]] = field(default_factory=dict)


def compute_sub_deadlines(graph: TaskGraph) -> Dict[int, float]:
    """Assign absolute sub-deadlines that increase along each source-to-sink path.

    ``TaskGraph.critical_path_lengths`` returns the remaining LO-WCET critical
    path length from a node to the sink, including the node itself.  A task's
    latest-finish target should be based on the amount of work that has been
    completed *up to and including that task*, not on the remaining work alone.
    Earlier versions used ``deadline * remaining / critical_path`` directly; that
    made source-side tasks receive the largest sub-deadlines and sink-side tasks
    receive the smallest sub-deadlines, which reverses the intended timing order.

    For one task ``i``:
        remaining_after_i = max_{j in succ(i)}(comm(i,j) + longest_path_from_j)
        completed_fraction_i = 1 - remaining_after_i / critical_path_length
        sub_deadline_i = deadline * completed_fraction_i

    The result is an absolute DAG-time deadline in ``[0, deadline]``.  Source
    dummy nodes naturally get a value close to zero, and sink tasks get the DAG
    deadline.  This is the value used by task-level DMR checks.
    """

    deadline = graph.deadline if graph.deadline is not None else graph.critical_path_length(use_hi_wcet=False)
    longest = graph.critical_path_lengths(use_hi_wcet=False)
    if not longest:
        return {}

    root_ids = [task_id for task_id, node in graph.nodes.items() if not node.predecessors]
    if graph.source_id is not None and graph.source_id in longest:
        critical_path_length = longest[graph.source_id]
    elif root_ids:
        critical_path_length = max(longest[task_id] for task_id in root_ids)
    else:
        critical_path_length = max(longest.values())

    denominator = max(critical_path_length, 1e-9)
    sub_deadlines: Dict[int, float] = {}
    for task_id in graph.topological_order():
        node = graph.nodes[task_id]
        if node.successors:
            remaining_after_task = max(
                node.communication_costs.get(successor_id, 0.0) + longest[successor_id]
                for successor_id in node.successors
            )
        else:
            remaining_after_task = 0.0
        completed_fraction = 1.0 - (remaining_after_task / denominator)
        completed_fraction = min(1.0, max(0.0, completed_fraction))
        sub_deadlines[task_id] = deadline * completed_fraction
    return sub_deadlines


def _ready_tasks(graph: TaskGraph, completed: Set[int], excluded: Set[int]) -> List[int]:
    """Returns tasks whose predecessors are all completed or explicitly dropped."""

    ready: List[int] = []
    for task_id in graph.topological_order():
        if task_id in completed or task_id in excluded:
            continue
        node = graph.nodes[task_id]
        if all(predecessor_id in completed or predecessor_id in excluded for predecessor_id in node.predecessors):
            ready.append(task_id)
    return ready




def _mixed_criticality_runtime_work(node, sampled_work: float, hi_mode: bool) -> tuple[float, float, bool]:
    """Returns bounded work, active budget, and mode-switch trigger flag.

    The RTSS'16 IMC semantics use two budgets per task. HI tasks may execute
    beyond their LO budget up to their HI budget and trigger the LO->HI mode
    transition. LO tasks execute up to their LO budget in LO mode and up to their
    reduced HI-mode budget in HI mode.
    """

    if node.is_dummy:
        return 0.0, 0.0, False
    sampled_work = max(0.0, float(sampled_work))
    lo_budget = max(0.0, float(node.lo_wcet))
    if node.criticality == "HI":
        hi_budget = max(lo_budget, float(node.hi_wcet))
        bounded_work = min(sampled_work, hi_budget)
        triggers_switch = (not hi_mode) and sampled_work > lo_budget
        active_budget = hi_budget if hi_mode else lo_budget
        return bounded_work, active_budget, triggers_switch

    hi_budget = min(lo_budget, max(0.0, float(node.hi_wcet)))
    active_budget = hi_budget if hi_mode else lo_budget
    bounded_work = min(sampled_work, active_budget)
    return bounded_work, active_budget, False


def _estimated_budget_for_mode(node, use_predictions: bool, hi_mode: bool) -> float:
    """Returns the budget used for the exported estimated runtime field."""

    if node.is_dummy:
        return 0.0
    if hi_mode:
        if node.criticality == "HI":
            return max(node.lo_wcet, node.hi_wcet)
        return min(node.lo_wcet, max(0.0, node.hi_wcet))
    return node.predicted_lo_wcet if use_predictions else node.lo_wcet


def _drop_remaining_lo_tasks(
    graph: TaskGraph,
    decoded: DecodedSchedule,
    completed: Set[int],
    excluded: Set[int],
    current_time: float,
) -> None:
    """Drops every remaining LO task once the schedule enters HI mode.

    Dropped tasks are counted as service loss, not as task-level DMR.  They still
    receive zero-duration timestamps so dependency release-time calculations and
    detailed CSV exports stay well-defined.
    """

    changed = True
    while changed:
        changed = False
        for task_id in graph.topological_order():
            if task_id in completed or task_id in excluded:
                continue
            node = graph.nodes[task_id]
            if node.criticality != "LO" or node.is_dummy:
                continue
            predecessor_times = [
                decoded.finish_times.get(predecessor_id, current_time)
                for predecessor_id in node.predecessors
                if predecessor_id in completed or predecessor_id in excluded
            ]
            exclusion_time = max(predecessor_times, default=current_time)
            excluded.add(task_id)
            decoded.dropped_lo_tasks.add(task_id)
            decoded.start_times[task_id] = exclusion_time
            decoded.finish_times[task_id] = exclusion_time
            decoded.durations[task_id] = 0.0
            decoded.runtime_samples_by_task[task_id] = [0.0]
            changed = True


def _drop_zero_budget_lo_tasks(
    graph: TaskGraph,
    decoded: DecodedSchedule,
    completed: Set[int],
    excluded: Set[int],
    current_time: float,
) -> None:
    """Drops only LO tasks with zero HI-mode budget.

    This is the classical mixed-criticality special case of the IMC model.  LO
    tasks with a positive reduced HI-mode budget remain schedulable after the
    mode switch and are executed with that reduced budget.
    """

    changed = True
    while changed:
        changed = False
        for task_id in graph.topological_order():
            if task_id in completed or task_id in excluded:
                continue
            node = graph.nodes[task_id]
            if node.criticality != "LO" or node.is_dummy or node.hi_wcet > 0.0:
                continue
            predecessor_times = [
                decoded.finish_times.get(predecessor_id, current_time)
                for predecessor_id in node.predecessors
                if predecessor_id in completed or predecessor_id in excluded
            ]
            exclusion_time = max(predecessor_times, default=current_time)
            excluded.add(task_id)
            decoded.dropped_lo_tasks.add(task_id)
            decoded.start_times[task_id] = exclusion_time
            decoded.finish_times[task_id] = exclusion_time
            decoded.durations[task_id] = 0.0
            decoded.runtime_samples_by_task[task_id] = [0.0]
            changed = True


def decode_priority_schedule(
    graph: TaskGraph,
    platform: HeterogeneousPlatform,
    priority_scores: Dict[int, float],
    core_assignments: Dict[int, int],
    dvfs_assignments: Dict[int, int],
    rng_seed: int = 0,
    use_predictions: bool = False,
    drop_lo_after_mode_switch: bool = False,
    mode_switch_time_overhead: float = DEFAULT_MODE_SWITCH_TIME_OVERHEAD,
    mode_switch_power_overhead: float = DEFAULT_MODE_SWITCH_POWER_OVERHEAD,
) -> DecodedSchedule:
    """Decodes a full schedule from priorities and resource assignments.

    A mixed-criticality mode switch is counted only once for a DAG run: the first
    HI task whose sampled execution demand exceeds its LO budget triggers the
    transition from LO mode to HI mode.  The transition adds a small deterministic
    time overhead and a separate power overhead so output tables can report both
    scheduling delay and reconfiguration energy explicitly.
    """

    rng = build_rng(rng_seed)
    decoded = DecodedSchedule()
    completed: Set[int] = set()
    excluded: Set[int] = set()
    core_available = {core.core_id: 0.0 for core in platform.cores}
    hi_mode = False

    while len(completed) + len(excluded) < len(graph.nodes):
        if hi_mode:
            current_time = max(core_available.values(), default=0.0)
            if drop_lo_after_mode_switch:
                _drop_remaining_lo_tasks(
                    graph=graph,
                    decoded=decoded,
                    completed=completed,
                    excluded=excluded,
                    current_time=current_time,
                )
            else:
                _drop_zero_budget_lo_tasks(
                    graph=graph,
                    decoded=decoded,
                    completed=completed,
                    excluded=excluded,
                    current_time=current_time,
                )
            if len(completed) + len(excluded) >= len(graph.nodes):
                break

        ready = _ready_tasks(graph=graph, completed=completed, excluded=excluded)
        if not ready:
            raise RuntimeError("No ready tasks found during decoding.")
        ready.sort(key=lambda task_id: priority_scores[task_id], reverse=True)
        selected_task = ready[0]
        node = graph.nodes[selected_task]

        if hi_mode and node.criticality == "LO" and not node.is_dummy:
            should_drop_lo = drop_lo_after_mode_switch or node.hi_wcet <= 0.0
            if should_drop_lo:
                excluded.add(selected_task)
                decoded.dropped_lo_tasks.add(selected_task)
                exclusion_time = max((decoded.finish_times.get(predecessor_id, 0.0) for predecessor_id in node.predecessors), default=0.0)
                decoded.start_times[selected_task] = exclusion_time
                decoded.finish_times[selected_task] = exclusion_time
                decoded.durations[selected_task] = 0.0
                decoded.runtime_samples_by_task[selected_task] = [0.0]
                continue

        core_id = core_assignments[selected_task]
        dvfs_level_id = dvfs_assignments[selected_task]
        core = platform.core_by_id(core_id)
        ready_time = graph.predecessor_finish_constraints(
            task_id=selected_task,
            finish_times=decoded.finish_times,
            assigned_cores=decoded.assigned_cores,
            target_core_id=core_id,
        ) if node.predecessors else 0.0
        start_time = max(core_available[core_id], ready_time)

        triggered_mode_switch = False
        switch_time_overhead = 0.0
        switch_energy_overhead = 0.0
        if node.is_dummy:
            estimated_runtime = 0.0
            actual_runtime = 0.0
            runtime_samples = [0.0]
        else:
            estimated_budget = _estimated_budget_for_mode(node=node, use_predictions=use_predictions, hi_mode=hi_mode)
            estimated_runtime = core.execution_time(estimated_budget, dvfs_level_id)
            sampled_runtime_value = node.runtime_distribution.sample(rng) if node.runtime_distribution is not None else node.lo_wcet
            bounded_work, _, triggered_mode_switch = _mixed_criticality_runtime_work(
                node=node,
                sampled_work=sampled_runtime_value,
                hi_mode=hi_mode,
            )
            actual_runtime = core.execution_time(bounded_work, dvfs_level_id)
            runtime_samples = [
                core.execution_time(max(0.0, sample), dvfs_level_id)
                for sample in node.runtime_samples(rng=rng, num_samples=64)
            ]
            if triggered_mode_switch:
                hi_mode = True
                decoded.mode_switch_count += 1
                switch_time_overhead = max(0.0, mode_switch_time_overhead)
                switch_energy_overhead = switch_time_overhead * max(0.0, mode_switch_power_overhead)

        finish_time = start_time + actual_runtime + switch_time_overhead
        entry = ScheduleEntry(
            task_id=selected_task,
            core_id=core_id,
            dvfs_level_id=dvfs_level_id,
            start_time=start_time,
            finish_time=finish_time,
            estimated_runtime=estimated_runtime,
            actual_runtime=actual_runtime,
            criticality=node.criticality,
            mode_switch_time_overhead=switch_time_overhead,
            mode_switch_energy_overhead=switch_energy_overhead,
        )
        decoded.entries.append(entry)
        decoded.start_times[selected_task] = start_time
        decoded.finish_times[selected_task] = finish_time
        decoded.assigned_cores[selected_task] = core_id
        decoded.assigned_dvfs[selected_task] = dvfs_level_id
        decoded.durations[selected_task] = actual_runtime
        decoded.runtime_samples_by_task[selected_task] = runtime_samples
        core_available[core_id] = finish_time
        completed.add(selected_task)

        if hi_mode:
            if drop_lo_after_mode_switch:
                _drop_remaining_lo_tasks(
                    graph=graph,
                    decoded=decoded,
                    completed=completed,
                    excluded=excluded,
                    current_time=finish_time,
                )
            else:
                _drop_zero_budget_lo_tasks(
                    graph=graph,
                    decoded=decoded,
                    completed=completed,
                    excluded=excluded,
                    current_time=finish_time,
                )

    decoded.entries.sort(key=lambda item: item.start_time)
    return decoded


def evaluate_decoded_schedule(
    graph: TaskGraph,
    platform: HeterogeneousPlatform,
    decoded: DecodedSchedule,
) -> ScheduleMetrics:
    """Computes aggregate metrics for a decoded schedule."""

    if decoded.entries:
        makespan = max(entry.finish_time for entry in decoded.entries)
    else:
        makespan = 0.0

    deadline = graph.deadline if graph.deadline is not None else makespan
    deadline_miss = makespan > deadline
    application_deadline_miss_ratio = 1.0 if deadline_miss else 0.0

    real_task_ids = list(graph.real_task_ids)
    real_task_count = len(real_task_ids)
    sub_deadlines = compute_sub_deadlines(graph)
    dropped_real_tasks = set(decoded.dropped_lo_tasks).intersection(real_task_ids)
    completed_real_task_ids = {entry.task_id for entry in decoded.entries if entry.task_id in real_task_ids}
    scheduled_task_misses = sum(
        1
        for task_id in completed_real_task_ids
        if decoded.finish_times.get(task_id, 0.0) > sub_deadlines.get(task_id, deadline)
    )
    task_deadline_miss_ratio = 0.0 if real_task_count == 0 else scheduled_task_misses / real_task_count
    # DMR means Deadline Miss Ratio everywhere. Application-level miss is exposed
    # separately as application_deadline_miss_ratio.
    deadline_miss_ratio = task_deadline_miss_ratio
    busy_runtime = sum(entry.actual_runtime + entry.mode_switch_time_overhead for entry in decoded.entries)
    utilization = 0.0 if makespan <= 0.0 else min(1.0, busy_runtime / (platform.num_cores * makespan))

    energy = 0.0
    mode_switch_time_overhead_total = 0.0
    mode_switch_energy_overhead_total = 0.0
    load_by_core: Dict[int, float] = {}
    for entry in decoded.entries:
        core = platform.core_by_id(entry.core_id)
        energy += core.energy_for_runtime(runtime=entry.actual_runtime, dvfs_level_id=entry.dvfs_level_id)
        energy += entry.mode_switch_energy_overhead
        mode_switch_time_overhead_total += entry.mode_switch_time_overhead
        mode_switch_energy_overhead_total += entry.mode_switch_energy_overhead
        load_by_core[entry.core_id] = load_by_core.get(entry.core_id, 0.0) + (
            entry.actual_runtime + entry.mode_switch_time_overhead
        ) / max(makespan, 1e-9)

    cvar_values: List[float] = []
    mode_switch_probs: List[float] = []
    hi_task_ids = [task_id for task_id in real_task_ids if graph.nodes[task_id].criticality == "HI"]
    lo_task_ids = [task_id for task_id in real_task_ids if graph.nodes[task_id].criticality == "LO"]
    hi_completed = sum(1 for task_id in hi_task_ids if task_id in completed_real_task_ids)
    lo_completed = sum(1 for task_id in lo_task_ids if task_id in completed_real_task_ids)
    lo_dropped = sum(1 for task_id in lo_task_ids if task_id in dropped_real_tasks)
    completed_hi_ratio = 1.0 if not hi_task_ids else hi_completed / len(hi_task_ids)
    completed_lo_ratio = 1.0 if not lo_task_ids else lo_completed / len(lo_task_ids)
    dropped_lo_ratio = 0.0 if not lo_task_ids else lo_dropped / len(lo_task_ids)
    completed_task_ratio = 1.0 if real_task_count == 0 else len(completed_real_task_ids) / real_task_count
    service_loss_ratio = 1.0 - completed_task_ratio

    task_to_entry = {entry.task_id: entry for entry in decoded.entries}
    for task_id in real_task_ids:
        node = graph.nodes[task_id]
        entry = task_to_entry.get(task_id)
        if entry is None:
            samples = decoded.runtime_samples_by_task.get(task_id, [0.0])
            cvar_values.append(empirical_cvar_upper(samples=samples, alpha=0.1))
            continue
        core = platform.core_by_id(entry.core_id)
        samples = decoded.runtime_samples_by_task.get(task_id, [entry.actual_runtime])
        cvar_values.append(empirical_cvar_upper(samples=samples, alpha=0.1))
        if node.criticality == "HI":
            lo_budget_duration = core.execution_time(node.lo_wcet, entry.dvfs_level_id)
            mode_switch_probs.append(
                predict_mode_switch_probability(
                    lo_budget=lo_budget_duration,
                    execution_samples=samples,
                )
            )

    average_cvar = 0.0 if not cvar_values else sum(cvar_values) / len(cvar_values)
    mode_switch_probability = 0.0 if not mode_switch_probs else sum(mode_switch_probs) / len(mode_switch_probs)
    average_reliability = platform.average_reliability(
        core_assignments=decoded.assigned_cores,
        start_times=decoded.start_times,
        durations=decoded.durations,
    )
    aging_index = platform.aggregate_aging_index(time_value=makespan, assigned_loads=load_by_core)

    return ScheduleMetrics(
        makespan=makespan,
        energy=energy,
        utilization=utilization,
        deadline_miss=deadline_miss,
        deadline_miss_ratio=deadline_miss_ratio,
        application_deadline_miss_ratio=application_deadline_miss_ratio,
        task_deadline_miss_ratio=task_deadline_miss_ratio,
        mode_switch_count=decoded.mode_switch_count,
        mode_switch_probability=mode_switch_probability,
        average_cvar=average_cvar,
        aging_index=aging_index,
        average_reliability=average_reliability,
        completed_hi_ratio=completed_hi_ratio,
        completed_lo_ratio=completed_lo_ratio,
        dropped_lo_ratio=dropped_lo_ratio,
        completed_task_ratio=completed_task_ratio,
        service_loss_ratio=service_loss_ratio,
        mode_switch_time_overhead=mode_switch_time_overhead_total,
        mode_switch_energy_overhead=mode_switch_energy_overhead_total,
        entries=decoded.entries,
        task_to_entry=task_to_entry,
    )
