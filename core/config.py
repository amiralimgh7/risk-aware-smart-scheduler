"""Shared configuration, task graph data structures, metrics, and random utilities.

This module is part of the ESRLab defense-ready codebase. It uses snake_case for functions and variables while keeping Python classes in PascalCase, and exposes documented helpers for reproducible experiments.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class DaggenConfig:
    """Configuration for DAGGEN-style graph generation.

    The project PDF explicitly requires graph generation using UUniFast and DAGGEN,
    and parameter sweeps over graph density, regularity, and width. The ``jump``
    parameter is included because it is part of the standard DAGGEN family; however,
    it defaults to ``1`` so that the implementation stays conservative unless the
    instructor explicitly asks for wider skip edges.
    """

    n: int
    width: float
    regularity: float
    density: float
    jump: int = 1
    min_data: int = 1024
    max_data: int = 64 * 1024
    min_alpha: float = 0.0
    max_alpha: float = 0.3
    ccr_mode: int = 0
    add_dummy_source_sink: bool = True
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate and normalize dataclass fields after initialization."""
        if self.n <= 0:
            raise ValueError("n must be positive.")
        if not 0.0 < self.width <= 1.0:
            raise ValueError("width must be in (0, 1].")
        if not 0.0 <= self.regularity <= 1.0:
            raise ValueError("regularity must be in [0, 1].")
        if not 0.0 <= self.density <= 1.0:
            raise ValueError("density must be in [0, 1].")
        if self.jump <= 0:
            raise ValueError("jump must be positive.")
        if self.min_data <= 0 or self.max_data <= 0:
            raise ValueError("data sizes must be positive.")
        if self.min_data > self.max_data:
            raise ValueError("min_data cannot exceed max_data.")
        if self.min_alpha > self.max_alpha:
            raise ValueError("min_alpha cannot exceed max_alpha.")
        if self.ccr_mode not in (0, 1, 2, 3):
            raise ValueError("ccr_mode must be one of {0, 1, 2, 3}.")


@dataclass(slots=True)
class UUniFastConfig:
    """Configuration for UUniFast and UUniFast-Discard."""

    num_items: int
    total_utilization: float
    utilization_cap_per_item: Optional[float] = None
    max_trials: int = 1_000
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate and normalize dataclass fields after initialization."""
        if self.num_items <= 0:
            raise ValueError("num_items must be positive.")
        if self.total_utilization <= 0.0:
            raise ValueError("total_utilization must be positive.")
        if self.utilization_cap_per_item is not None and self.utilization_cap_per_item <= 0.0:
            raise ValueError("utilization_cap_per_item must be positive when provided.")
        if self.max_trials <= 0:
            raise ValueError("max_trials must be positive.")


@dataclass(slots=True)
class MixedCriticalityConfig:
    """Configuration of mixed-criticality execution budgets and runtime variation."""

    hi_task_ratio: float = 0.3
    hi_wcet_scale_min: float = 1.25
    hi_wcet_scale_max: float = 2.0
    lo_hi_wcet_scale_min: float = 0.5
    lo_hi_wcet_scale_max: float = 0.5
    lo_runtime_sigma: float = 0.12
    hi_runtime_sigma: float = 0.20
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate and normalize dataclass fields after initialization."""
        if not 0.0 <= self.hi_task_ratio <= 1.0:
            raise ValueError("hi_task_ratio must be in [0, 1].")
        if self.hi_wcet_scale_min < 1.0:
            raise ValueError("hi_wcet_scale_min must be at least 1.0.")
        if self.hi_wcet_scale_max < self.hi_wcet_scale_min:
            raise ValueError("hi_wcet_scale_max cannot be smaller than hi_wcet_scale_min.")
        if not 0.0 <= self.lo_hi_wcet_scale_min <= 1.0:
            raise ValueError("lo_hi_wcet_scale_min must be in [0, 1].")
        if not 0.0 <= self.lo_hi_wcet_scale_max <= 1.0:
            raise ValueError("lo_hi_wcet_scale_max must be in [0, 1].")
        if self.lo_hi_wcet_scale_max < self.lo_hi_wcet_scale_min:
            raise ValueError("lo_hi_wcet_scale_max cannot be smaller than lo_hi_wcet_scale_min.")
        if self.lo_runtime_sigma < 0.0 or self.hi_runtime_sigma < 0.0:
            raise ValueError("runtime sigmas must be non-negative.")


@dataclass(slots=True)
class ApplicationTimingConfig:
    """Configuration for periodic DAG application timing."""

    period: float
    relative_deadline: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate and normalize dataclass fields after initialization."""
        if self.period <= 0.0:
            raise ValueError("period must be positive.")
        if self.relative_deadline is not None and self.relative_deadline <= 0.0:
            raise ValueError("relative_deadline must be positive.")


@dataclass(slots=True)
class SimulationConfig:
    """Configuration for simulation and evaluation episodes."""

    num_runtime_samples: int = 64
    cvar_alpha: float = 0.1
    random_seed: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate and normalize dataclass fields after initialization."""
        if self.num_runtime_samples <= 0:
            raise ValueError("num_runtime_samples must be positive.")
        if not 0.0 < self.cvar_alpha <= 1.0:
            raise ValueError("cvar_alpha must be in (0, 1].")


@dataclass(slots=True)
class TrainingConfig:
    """Shared configuration for training loops."""

    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    num_epochs: int = 20
    batch_size: int = 8
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate and normalize dataclass fields after initialization."""
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative.")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
