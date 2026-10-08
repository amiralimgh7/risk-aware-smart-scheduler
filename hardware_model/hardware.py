"""Defines core types, DVFS states, power models, failure rates, aging state, and reliability calculations.

This module is part of the ESRLab defense-ready codebase. It uses snake_case for functions and variables while keeping Python classes in PascalCase, and exposes documented helpers for reproducible experiments.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(slots=True)
class DvfsLevel:
    """One DVFS operating point.

    Units:
        voltage: Volt (V)
        frequency_ghz: Gigahertz (GHz)

    The project PDF specifies five DVFS levels for each island and gives the
    endpoint ranges:
        - Cortex-A7:  [0.9 V, 0.8 GHz] -> [1.1 V, 1.6 GHz]
        - Cortex-A15: [0.9 V, 1.0 GHz] -> [1.1 V, 2.0 GHz]
    Intermediate levels are the uniform interpolation between those endpoints.
    """

    level_id: int
    voltage: float
    frequency_ghz: float


@dataclass(slots=True)
class CoreSpec:
    """Processing-core model with DVFS, energy, reliability, and aging.

    Important modeling notes:
        1. The PDF provides DVFS voltage/frequency ranges, but does not provide
           measured capacitance, leakage, or board-level power constants.
        2. Therefore, energy is reported as a normalized dynamic-energy proxy,
           not as calibrated physical Joules.
        3. The dynamic-energy model follows the standard CMOS relation:

              P_dynamic ~= C_eff * V^2 * f
              E_dynamic = P_dynamic * runtime

           where ``dynamic_capacitance_coefficient`` is a normalized effective
           coefficient used to distinguish A7 and A15 islands.
        4. Reliability and aging follow the project-level exponential failure-rate
           model used by the scheduler:

              lambda_j(t) = lambda_0_j * exp(kappa_j * t)
              R_j(t, dt) = exp(- integral_t^{t+dt} lambda_j(u) du)
              Lambda_t = sum_j lambda_j(t) * rho_j(t)

           The failure-rate constants are expressed in the simulator's normalized
           time unit. They are calibrated to produce presentation-level reliability
           around two nines for typical application schedules, rather than the
           unrealistically optimistic six-nine values caused by 1e-6-scale rates.
    """

    core_id: int
    core_type: str
    dvfs_levels: List[DvfsLevel]
    base_speed_factor: float
    lambda_0: float
    kappa: float
    dynamic_capacitance_coefficient: float = 1.0
    leakage_power_coefficient: float = 0.0

    def execution_time(self, base_cycles: float, dvfs_level_id: int) -> float:
        """Returns execution time for normalized work units.

        The simulator stores task cost in normalized units. Faster cores and higher
        frequencies reduce execution time by increasing effective speed.
        """

        level = self.dvfs_levels[dvfs_level_id]
        effective_speed = self.base_speed_factor * level.frequency_ghz
        return base_cycles / max(effective_speed, 1e-12)

    def dynamic_power(self, dvfs_level_id: int, switching_factor: float = 1.0) -> float:
        """Returns normalized dynamic power: C_eff * alpha * V^2 * f.

        Unit: relative power unit, because the PDF does not provide a calibrated
        hardware capacitance or measured power table.
        """

        level = self.dvfs_levels[dvfs_level_id]
        return (
            self.dynamic_capacitance_coefficient
            * switching_factor
            * (level.voltage ** 2)
            * level.frequency_ghz
        )

    def leakage_power(self, dvfs_level_id: int) -> float:
        """Returns optional normalized leakage/static power.

        The PDF excerpt does not specify leakage parameters. The default coefficient
        is therefore zero, keeping the energy model faithful to the DVFS dynamic
        power relation unless the experimenter explicitly calibrates leakage.
        """

        level = self.dvfs_levels[dvfs_level_id]
        return self.leakage_power_coefficient * level.voltage

    def total_power(self, dvfs_level_id: int, switching_factor: float = 1.0) -> float:
        """Returns normalized total power used by the energy model."""

        return self.dynamic_power(dvfs_level_id, switching_factor) + self.leakage_power(dvfs_level_id)

    def energy_for_runtime(self, runtime: float, dvfs_level_id: int, switching_factor: float = 1.0) -> float:
        """Returns normalized energy for executing during ``runtime``.

        Formula:
            E = (C_eff * alpha * V^2 * f + P_leak) * runtime

        With the default leakage coefficient, this reduces to:
            E = C_eff * alpha * V^2 * f * runtime
        """

        return self.total_power(dvfs_level_id=dvfs_level_id, switching_factor=switching_factor) * max(runtime, 0.0)

    def failure_rate(self, time_value: float) -> float:
        """Returns time-dependent failure rate lambda_j(t)."""

        return self.lambda_0 * math.exp(min(self.kappa * max(time_value, 0.0), 700.0))

    def reliability_over_interval(self, start_time: float, delta_t: float) -> float:
        """Returns reliability over [start_time, start_time + delta_t].

        For lambda(t) = lambda_0 * exp(kappa*t):
            R = exp(-(lambda_0 / kappa) * (exp(kappa*(t+dt)) - exp(kappa*t)))
        """

        start_time = max(start_time, 0.0)
        delta_t = max(delta_t, 0.0)
        if abs(self.kappa) < 1e-12:
            return math.exp(-self.lambda_0 * delta_t)
        upper = min(self.kappa * (start_time + delta_t), 700.0)
        lower = min(self.kappa * start_time, 700.0)
        exponent = -(self.lambda_0 / self.kappa) * (math.exp(upper) - math.exp(lower))
        return math.exp(max(exponent, -700.0))


@dataclass(slots=True)
class HeterogeneousPlatform:
    """ARM big.LITTLE platform used in the project experiments."""

    cores: List[CoreSpec] = field(default_factory=list)

    @property
    def num_cores(self) -> int:
        """Returns the number of cores."""

        return len(self.cores)

    def core_by_id(self, core_id: int) -> CoreSpec:
        """Returns a core spec by id."""

        return next(core for core in self.cores if core.core_id == core_id)

    def aggregate_aging_index(self, time_value: float, assigned_loads: Dict[int, float]) -> float:
        """Computes the project aging index: Lambda_t = sum_j lambda_j(t) * rho_j(t).

        ``rho_j(t)`` is approximated by the normalized busy fraction of core ``j``
        in the generated schedule. This matches the PDF-level aging formulation,
        while remaining executable in the simulator.
        """

        total = 0.0
        for core in self.cores:
            load_fraction = max(0.0, assigned_loads.get(core.core_id, 0.0))
            total += core.failure_rate(time_value) * load_fraction
        return total

    def average_reliability(
        self,
        core_assignments: Dict[int, int],
        start_times: Dict[int, float],
        durations: Dict[int, float],
    ) -> float:
        """Returns application-level mission reliability for the produced schedule.

        Earlier versions reported the arithmetic mean of per-task reliabilities.
        That made the result look too optimistic because many values close to 1.0
        average back to a value close to 1.0.  For a DAG application, the more
        defensible system-level approximation is a series mission model: every
        scheduled task interval must execute correctly for the application to be
        considered reliable.  Therefore:

            R_application = product_i R_i

        The public field name is kept as ``average_reliability`` for backward
        compatibility with existing CSV/reporting code, but the value is now the
        application-level reliability of the produced schedule.
        """

        if not core_assignments:
            return 1.0
        core_map = {core.core_id: core for core in self.cores}
        log_reliability = 0.0
        for task_id, core_id in core_assignments.items():
            duration = durations.get(task_id, 0.0)
            if duration <= 0.0:
                continue
            interval_reliability = core_map[core_id].reliability_over_interval(
                start_times.get(task_id, 0.0),
                duration,
            )
            log_reliability += math.log(max(interval_reliability, 1e-300))
        return math.exp(max(log_reliability, -700.0))


def _interpolated_dvfs_levels(
    start_voltage: float,
    end_voltage: float,
    start_frequency_ghz: float,
    end_frequency_ghz: float,
    num_levels: int = 5,
) -> List[DvfsLevel]:
    """Builds uniformly interpolated DVFS levels from the PDF endpoint ranges."""

    if num_levels < 2:
        raise ValueError("num_levels must be at least 2.")
    levels: List[DvfsLevel] = []
    for level_id in range(num_levels):
        ratio = level_id / (num_levels - 1)
        voltage = start_voltage + ratio * (end_voltage - start_voltage)
        frequency = start_frequency_ghz + ratio * (end_frequency_ghz - start_frequency_ghz)
        levels.append(DvfsLevel(level_id=level_id, voltage=voltage, frequency_ghz=frequency))
    return levels


def build_big_little_platform(num_a7: int = 4, num_a15: int = 4) -> HeterogeneousPlatform:
    """Builds the PDF-aligned ARM Cortex-A7 / Cortex-A15 platform.

    DVFS levels are exactly the five uniformly interpolated levels over the ranges
    specified in the PDF excerpt:
        A7:  [0.9 V, 0.8 GHz] -> [1.1 V, 1.6 GHz]
        A15: [0.9 V, 1.0 GHz] -> [1.1 V, 2.0 GHz]
    """

    a7_levels = _interpolated_dvfs_levels(
        start_voltage=0.9,
        end_voltage=1.1,
        start_frequency_ghz=0.8,
        end_frequency_ghz=1.6,
        num_levels=5,
    )
    a15_levels = _interpolated_dvfs_levels(
        start_voltage=0.9,
        end_voltage=1.1,
        start_frequency_ghz=1.0,
        end_frequency_ghz=2.0,
        num_levels=5,
    )

    cores: List[CoreSpec] = []
    next_core_id = 0

    for _ in range(num_a7):
        cores.append(
            CoreSpec(
                core_id=next_core_id,
                core_type="A7",
                dvfs_levels=a7_levels,
                base_speed_factor=1.0,
                # Calibrated per normalized simulation-time unit.  The earlier
                # 1e-6 value produced six-nine reliability, which is too optimistic
                # for the scale of these simulated schedules.
                lambda_0=1.0e-3,
                kappa=2e-4,
                dynamic_capacitance_coefficient=0.8,
            )
        )
        next_core_id += 1

    for _ in range(num_a15):
        cores.append(
            CoreSpec(
                core_id=next_core_id,
                core_type="A15",
                dvfs_levels=a15_levels,
                base_speed_factor=1.8,
                # A15 cores are modeled as faster but slightly less reliable
                # under the same normalized mission-time scale.
                lambda_0=1.2e-3,
                kappa=2.5e-4,
                dynamic_capacitance_coefficient=1.2,
            )
        )
        next_core_id += 1

    return HeterogeneousPlatform(cores=cores)
