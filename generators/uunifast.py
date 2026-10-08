"""DAG, UUniFast utilization, and workload generation utilities.

This module is part of the ESRLab defense-ready codebase. It uses snake_case for functions and variables while keeping Python classes in PascalCase, and exposes documented helpers for reproducible experiments.
"""
from __future__ import annotations

from typing import List, Optional

from core.config import UUniFastConfig
from core.random_utils import build_rng


def uunifast(num_items: int, total_utilization: float, seed: Optional[int] = None) -> List[float]:
    """Generates positive utilization values using the classic UUniFast algorithm."""

    if num_items <= 0:
        raise ValueError("num_items must be positive.")
    if total_utilization <= 0.0:
        raise ValueError("total_utilization must be positive.")

    rng = build_rng(seed)
    values: List[float] = []
    sum_u = total_utilization

    for remaining in range(num_items, 1, -1):
        next_sum_u = sum_u * (rng.random() ** (1.0 / (remaining - 1)))
        values.append(sum_u - next_sum_u)
        sum_u = next_sum_u

    values.append(sum_u)
    return values


def uunifast_discard(config: UUniFastConfig) -> List[float]:
    """Runs UUniFast with rejection sampling when an item cap is required."""

    if config.utilization_cap_per_item is None:
        return uunifast(
            num_items=config.num_items,
            total_utilization=config.total_utilization,
            seed=config.seed,
        )

    if config.total_utilization > config.num_items * config.utilization_cap_per_item:
        raise ValueError("Requested utilization is infeasible under the given cap.")

    seed_rng = build_rng(config.seed)
    for _ in range(config.max_trials):
        candidate_seed = seed_rng.randint(0, 2**31 - 1)
        candidate = uunifast(
            num_items=config.num_items,
            total_utilization=config.total_utilization,
            seed=candidate_seed,
        )
        if max(candidate) <= config.utilization_cap_per_item:
            return candidate

    raise RuntimeError("UUniFast-Discard failed within max_trials.")
