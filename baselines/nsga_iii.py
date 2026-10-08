"""Classical and learned baseline schedulers used for comparison.

This module is part of the ESRLab defense-ready codebase. It uses snake_case for functions and variables while keeping Python classes in PascalCase, and exposes documented helpers for reproducible experiments.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from core.metrics import ScheduleMetrics
from core.random_utils import build_rng
from core.task_graph import DagApplication, TaskGraph
from hardware_model.hardware import HeterogeneousPlatform
from simulator.list_scheduling import decode_priority_schedule, evaluate_decoded_schedule


@dataclass(slots=True)
class NsgaThreeConfig:
    """Configuration of the operational NSGA-III baseline."""

    population_size: int = 24
    num_generations: int = 12
    reference_divisions: int = 8
    mutation_probability: float = 0.1
    seed: int = 0


@dataclass(slots=True)
class Individual:
    """One evolutionary candidate schedule."""

    priority_scores: Dict[int, float]
    core_assignments: Dict[int, int]
    dvfs_assignments: Dict[int, int]
    metrics: ScheduleMetrics | None = None


class NsgaThreeScheduler:
    """Practical three-objective evolutionary baseline.

    The project PDF names NSGA-III as a baseline without providing the exact
    chromosome layout. This implementation uses a direct task-wise encoding with
    three objectives: makespan, energy, and aging index.
    """

    def __init__(self, config: NsgaThreeConfig | None = None) -> None:
        """Initialize the object with validated configuration and runtime state."""
        self.config = config or NsgaThreeConfig()

    def schedule(self, application: DagApplication, platform: HeterogeneousPlatform) -> ScheduleMetrics:
        """Optimizes a schedule and returns the best evolved candidate."""

        graph = application.clone_graph()
        rng = build_rng(self.config.seed)
        population = [self._random_individual(graph=graph, platform=platform, rng=rng) for _ in range(self.config.population_size)]
        for individual in population:
            individual.metrics = self._evaluate(graph=graph, platform=platform, individual=individual)

        for _ in range(self.config.num_generations):
            offspring: List[Individual] = []
            while len(offspring) < self.config.population_size:
                parent_a = self._tournament_select(population, rng)
                parent_b = self._tournament_select(population, rng)
                child = self._crossover(graph=graph, parent_a=parent_a, parent_b=parent_b, rng=rng)
                self._mutate(individual=child, graph=graph, platform=platform, rng=rng)
                child.metrics = self._evaluate(graph=graph, platform=platform, individual=child)
                offspring.append(child)
            population = self._environmental_selection(population + offspring, target_size=self.config.population_size)

        best = min(population, key=lambda item: self._objective_tuple(item.metrics))
        if best.metrics is None:
            raise RuntimeError("NSGA-III best individual has no metrics.")
        return best.metrics

    def _random_individual(self, graph: TaskGraph, platform: HeterogeneousPlatform, rng) -> Individual:
        """Builds a random chromosome."""

        priority_scores = {task_id: rng.random() for task_id in graph.nodes.keys()}
        core_assignments: Dict[int, int] = {}
        dvfs_assignments: Dict[int, int] = {}
        for task_id in graph.nodes.keys():
            core = rng.choice(platform.cores)
            core_assignments[task_id] = core.core_id
            dvfs_assignments[task_id] = rng.choice(core.dvfs_levels).level_id
        return Individual(
            priority_scores=priority_scores,
            core_assignments=core_assignments,
            dvfs_assignments=dvfs_assignments,
        )

    def _evaluate(self, graph: TaskGraph, platform: HeterogeneousPlatform, individual: Individual) -> ScheduleMetrics:
        """Evaluates one candidate schedule."""

        decoded = decode_priority_schedule(
            graph=graph,
            platform=platform,
            priority_scores=individual.priority_scores,
            core_assignments=individual.core_assignments,
            dvfs_assignments=individual.dvfs_assignments,
            rng_seed=self.config.seed,
        )
        return evaluate_decoded_schedule(graph=graph, platform=platform, decoded=decoded)

    @staticmethod
    def _objective_tuple(metrics: ScheduleMetrics | None) -> Tuple[float, float, float]:
        """Returns the three-objective tuple."""

        if metrics is None:
            return float("inf"), float("inf"), float("inf")
        return metrics.makespan, metrics.energy, metrics.aging_index

    def _dominates(self, a: Individual, b: Individual) -> bool:
        """Checks Pareto dominance for three minimization objectives."""

        a_objectives = self._objective_tuple(a.metrics)
        b_objectives = self._objective_tuple(b.metrics)
        return all(x <= y for x, y in zip(a_objectives, b_objectives)) and any(
            x < y for x, y in zip(a_objectives, b_objectives)
        )

    def _nondominated_sort(self, population: Sequence[Individual]) -> List[List[Individual]]:
        """Builds Pareto fronts."""

        domination_counts: Dict[int, int] = {}
        dominated_sets: Dict[int, List[int]] = {}
        fronts: List[List[int]] = [[]]

        for i, individual_i in enumerate(population):
            domination_counts[i] = 0
            dominated_sets[i] = []
            for j, individual_j in enumerate(population):
                if i == j:
                    continue
                if self._dominates(individual_i, individual_j):
                    dominated_sets[i].append(j)
                elif self._dominates(individual_j, individual_i):
                    domination_counts[i] += 1
            if domination_counts[i] == 0:
                fronts[0].append(i)

        current_front_index = 0
        while current_front_index < len(fronts) and fronts[current_front_index]:
            next_front: List[int] = []
            for i in fronts[current_front_index]:
                for j in dominated_sets[i]:
                    domination_counts[j] -= 1
                    if domination_counts[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            current_front_index += 1

        return [[population[index] for index in front] for front in fronts if front]

    def _environmental_selection(self, population: Sequence[Individual], target_size: int) -> List[Individual]:
        """Selects the next generation using fronts and objective spread."""

        fronts = self._nondominated_sort(population)
        selected: List[Individual] = []
        for front in fronts:
            if len(selected) + len(front) <= target_size:
                selected.extend(front)
                continue
            spread_sorted = sorted(front, key=lambda item: sum(self._objective_tuple(item.metrics)))
            selected.extend(spread_sorted[: target_size - len(selected)])
            break
        return selected

    def _tournament_select(self, population: Sequence[Individual], rng) -> Individual:
        """Selects one parent using binary tournament."""

        a, b = rng.sample(list(population), 2)
        rank_a = self._objective_tuple(a.metrics)
        rank_b = self._objective_tuple(b.metrics)
        return a if rank_a <= rank_b else b

    def _crossover(self, graph: TaskGraph, parent_a: Individual, parent_b: Individual, rng) -> Individual:
        """Creates one child from two parents."""

        priority_scores: Dict[int, float] = {}
        core_assignments: Dict[int, int] = {}
        dvfs_assignments: Dict[int, int] = {}
        for task_id in graph.nodes.keys():
            if rng.random() < 0.5:
                priority_scores[task_id] = parent_a.priority_scores[task_id]
                core_assignments[task_id] = parent_a.core_assignments[task_id]
                dvfs_assignments[task_id] = parent_a.dvfs_assignments[task_id]
            else:
                priority_scores[task_id] = parent_b.priority_scores[task_id]
                core_assignments[task_id] = parent_b.core_assignments[task_id]
                dvfs_assignments[task_id] = parent_b.dvfs_assignments[task_id]
        return Individual(priority_scores=priority_scores, core_assignments=core_assignments, dvfs_assignments=dvfs_assignments)

    def _mutate(self, individual: Individual, graph: TaskGraph, platform: HeterogeneousPlatform, rng) -> None:
        """Mutates one chromosome in-place."""

        for task_id in graph.nodes.keys():
            if rng.random() < self.config.mutation_probability:
                individual.priority_scores[task_id] += rng.uniform(-0.2, 0.2)
            if rng.random() < self.config.mutation_probability:
                core = rng.choice(platform.cores)
                individual.core_assignments[task_id] = core.core_id
                individual.dvfs_assignments[task_id] = rng.choice(core.dvfs_levels).level_id
            elif rng.random() < self.config.mutation_probability:
                selected_core = platform.core_by_id(individual.core_assignments[task_id])
                individual.dvfs_assignments[task_id] = rng.choice(selected_core.dvfs_levels).level_id
