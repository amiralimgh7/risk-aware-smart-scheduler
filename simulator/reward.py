"""List-scheduling simulation, reward calculation, timing, reliability, and energy behavior.

This module is part of the ESRLab defense-ready codebase. It uses snake_case for functions and variables while keeping Python classes in PascalCase, and exposes documented helpers for reproducible experiments.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RewardWeights:
    """Weights used by the multi-objective reward function."""

    w_u: float = 1.0
    w_s: float = 1.0
    w_r: float = 1.0
    w_d: float = 1.0
    w_a: float = 1.0


def compute_reward(
    utilization: float,
    mode_switch_probability: float,
    cvar_upper_value: float,
    deadline_miss: bool,
    aging_index: float,
    weights: RewardWeights,
) -> float:
    """Computes the project reward function."""

    miss_indicator = 1.0 if deadline_miss else 0.0
    return (
        weights.w_u * utilization
        + weights.w_s * (1.0 - mode_switch_probability)
        - weights.w_r * cvar_upper_value
        - weights.w_d * miss_indicator
        - weights.w_a * aging_index
    )
