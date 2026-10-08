"""Shared configuration, task graph data structures, metrics, and random utilities.

This module is part of the ESRLab defense-ready codebase. It uses snake_case for functions and variables while keeping Python classes in PascalCase, and exposes documented helpers for reproducible experiments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(slots=True)
class ScheduleEntry:
    """Represents one scheduled task execution."""

    task_id: int
    core_id: int
    dvfs_level_id: int
    start_time: float
    finish_time: float
    estimated_runtime: float
    actual_runtime: float
    criticality: str
    mode_switch_time_overhead: float = 0.0
    mode_switch_energy_overhead: float = 0.0


@dataclass(slots=True)
class ScheduleMetrics:
    """Stores aggregate metrics of one schedule."""

    makespan: float
    energy: float
    utilization: float
    deadline_miss: bool
    deadline_miss_ratio: float
    application_deadline_miss_ratio: float
    task_deadline_miss_ratio: float
    mode_switch_count: int
    mode_switch_probability: float
    average_cvar: float
    aging_index: float
    average_reliability: float
    completed_hi_ratio: float
    completed_lo_ratio: float
    dropped_lo_ratio: float
    completed_task_ratio: float
    service_loss_ratio: float
    mode_switch_time_overhead: float = 0.0
    mode_switch_energy_overhead: float = 0.0
    entries: List[ScheduleEntry] = field(default_factory=list)
    task_to_entry: Dict[int, ScheduleEntry] = field(default_factory=dict)

    @property
    def reward_ready_dict(self) -> Dict[str, float]:
        """Returns a flat dictionary convenient for logging."""

        return {
            "makespan": self.makespan,
            "energy": self.energy,
            "utilization": self.utilization,
            "deadline_miss_ratio": self.deadline_miss_ratio,
            "application_deadline_miss_ratio": self.application_deadline_miss_ratio,
            "task_deadline_miss_ratio": self.task_deadline_miss_ratio,
            "service_loss_ratio": self.service_loss_ratio,
            "dropped_lo_ratio": self.dropped_lo_ratio,
            "completed_lo_ratio": self.completed_lo_ratio,
            "completed_task_ratio": self.completed_task_ratio,
            "mode_switch_probability": self.mode_switch_probability,
            "mode_switch_time_overhead": self.mode_switch_time_overhead,
            "mode_switch_energy_overhead": self.mode_switch_energy_overhead,
            "average_cvar": self.average_cvar,
            "aging_index": self.aging_index,
            "average_reliability": self.average_reliability,
            "completed_hi_ratio": self.completed_hi_ratio,
            "completed_lo_ratio": self.completed_lo_ratio,
            "dropped_lo_ratio": self.dropped_lo_ratio,
            "completed_task_ratio": self.completed_task_ratio,
            "service_loss_ratio": self.service_loss_ratio,
        }
