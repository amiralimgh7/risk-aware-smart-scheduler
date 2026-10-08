"""Shared configuration, task graph data structures, metrics, and random utilities.

This module is part of the ESRLab defense-ready codebase. It uses snake_case for functions and variables while keeping Python classes in PascalCase, and exposes documented helpers for reproducible experiments.
"""
from __future__ import annotations

import math
import random
from typing import Optional


def build_rng(seed: Optional[int]) -> random.Random:
    """Builds a dedicated RNG instance."""

    return random.Random(seed)


def get_random_number_between(rng: random.Random, lower: float, upper: float) -> float:
    """Returns a float sampled uniformly in [lower, upper)."""

    if upper < lower:
        raise ValueError("upper must be greater than or equal to lower.")
    if upper == lower:
        return lower
    return rng.uniform(lower, upper)


def get_int_random_number_around(rng: random.Random, center: int, latitude_percent: float) -> int:
    """Returns an integer around a center with percentage latitude."""

    if center <= 0:
        raise ValueError("center must be positive.")
    if latitude_percent < 0.0:
        raise ValueError("latitude_percent must be non-negative.")

    lower = max(1, math.floor(center * (1.0 - latitude_percent / 100.0)))
    upper = max(lower, math.ceil(center * (1.0 + latitude_percent / 100.0)))
    return rng.randint(lower, upper)
