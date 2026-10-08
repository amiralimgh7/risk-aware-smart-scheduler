"""Risk-aware WCET and CVaR calculations used by the scheduler.

This module is part of the ESRLab defense-ready codebase. It uses snake_case for functions and variables while keeping Python classes in PascalCase, and exposes documented helpers for reproducible experiments.
"""
from __future__ import annotations

import math
from statistics import NormalDist
from typing import Iterable, List


def empirical_cvar_upper(samples: Iterable[float], alpha: float) -> float:
    """Computes the upper-tail empirical CVaR."""

    values = sorted(float(value) for value in samples)
    if not values:
        raise ValueError("samples must not be empty.")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1].")

    tail_count = max(1, int(math.ceil(alpha * len(values))))
    tail_values = values[-tail_count:]
    return sum(tail_values) / len(tail_values)


def robust_lo_wcet_from_samples(samples: Iterable[float], alpha: float) -> float:
    """Returns the robust LO-WCET estimate defined through upper-tail CVaR."""

    return empirical_cvar_upper(samples=samples, alpha=alpha)


def predict_mode_switch_probability(lo_budget: float, execution_samples: Iterable[float]) -> float:
    """Estimates the probability of mode switch as P(execution > lo_budget)."""

    values = [float(value) for value in execution_samples]
    if not values:
        raise ValueError("execution_samples must not be empty.")
    exceed_count = sum(1 for value in values if value > lo_budget)
    return exceed_count / len(values)


def gaussian_upper_cvar(mean_value: float, std_value: float, alpha: float) -> float:
    """Returns the upper-tail CVaR of a Gaussian approximation."""

    std_value = max(std_value, 1e-12)
    alpha = min(max(alpha, 1e-6), 1.0)
    z = NormalDist().inv_cdf(1.0 - alpha)
    pdf_z = NormalDist().pdf(z)
    return mean_value + std_value * (pdf_z / alpha)
