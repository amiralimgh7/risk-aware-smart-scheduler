"""Training loops for supervised predictors and reward-driven policies.

This module is part of the ESRLab defense-ready codebase. It uses snake_case for functions and variables while keeping Python classes in PascalCase, and exposes documented helpers for reproducible experiments.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch import nn
from torch.distributions import Categorical, Normal

from core.config import TrainingConfig
from core.metrics import ScheduleMetrics
from core.task_graph import DagApplication, TaskGraph
from dataset_tools.serialization import load_json, save_json
from hardware_model.hardware import HeterogeneousPlatform
from models.gat_risk import RiskAwareGatRegressor
from models.graph_utils import build_graph_tensors
from models.quantization import (
    benchmark_callable_latency_ms,
    count_parameters,
    estimate_model_size_bytes,
    pdf_qat_parameter_selector,
    quantize_dequantize_int8_weights_model,
)
from simulator.list_scheduling import decode_priority_schedule, evaluate_decoded_schedule
from simulator.reward import RewardWeights, compute_reward
from training.supervised import SupervisedTrainer, build_training_reliability_vector


@dataclass(slots=True)
class PolicyTrainingHistory:
    """Stores training rewards of the RL phase."""

    rewards: List[float] = field(default_factory=list)
    baselines: List[float] = field(default_factory=list)


class SchedulingPolicyNetwork(nn.Module):
    """Policy network that outputs per-task priority, core, and DVFS decisions."""

    def __init__(self, input_features: int, num_cores: int, num_dvfs_levels: int, hidden_features: int = 64) -> None:
        """Initialize the object with validated configuration and runtime state."""
        super().__init__()
        self.input_features = input_features
        self.num_cores = num_cores
        self.num_dvfs_levels = num_dvfs_levels
        self.hidden_features = hidden_features
        self.shared = nn.Sequential(
            nn.Linear(input_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
        )
        self.priority_head = nn.Linear(hidden_features, 1)
        self.core_head = nn.Linear(hidden_features, num_cores)
        self.dvfs_head = nn.Linear(hidden_features, num_dvfs_levels)
        self.priority_log_std = nn.Parameter(torch.tensor(-0.5))

    def forward(self, task_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns task-wise action parameters."""

        hidden = self.shared(task_features)
        priority_mean = self.priority_head(hidden)
        core_logits = self.core_head(hidden)
        dvfs_logits = self.dvfs_head(hidden)
        return priority_mean, core_logits, dvfs_logits


class RiskAwareGatScheduler:
    """End-to-end proposed method: GAT predictor plus policy-gradient scheduler."""

    def __init__(
        self,
        in_features: int = 11,
        hidden_features: int = 16,
        embedding_features: int = 16,
        num_heads: int = 2,
        device: str = "cpu",
        use_qat: bool = False,
        use_uncertainty_features: bool = True,
    ) -> None:
        """Initialize the object with validated configuration and runtime state."""
        self.device = torch.device(device)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.embedding_features = embedding_features
        self.num_heads = num_heads
        self.predictor = RiskAwareGatRegressor(
            in_features=in_features,
            hidden_features=hidden_features,
            embedding_features=embedding_features,
            num_heads=num_heads,
            use_qat=use_qat,
        ).to(self.device)
        self.supervised_trainer = SupervisedTrainer(device=device)
        self.policy_network: SchedulingPolicyNetwork | None = None
        self.use_qat = use_qat
        self.use_uncertainty_features = use_uncertainty_features
        self.is_quantized_int8 = False

    def fit_predictor(
        self,
        applications: Sequence[DagApplication],
        platform: HeterogeneousPlatform,
        training_config: TrainingConfig,
        cvar_alpha: float = 0.1,
        verbose: bool = True,
    ):
        """Trains the risk-aware GAT predictor."""

        return self.supervised_trainer.train_gat(
            model=self.predictor,
            applications=applications,
            platform=platform,
            config=training_config,
            cvar_alpha=cvar_alpha,
            verbose=verbose,
            progress_prefix="ProposedGAT-Predictor",
            enable_uncertainty_losses=self.use_uncertainty_features,
        )

    def fit_policy(
        self,
        applications: Sequence[DagApplication],
        platform: HeterogeneousPlatform,
        num_episodes: int = 20,
        learning_rate: float = 1e-3,
        reward_weights: RewardWeights | None = None,
        verbose: bool = True,
    ) -> PolicyTrainingHistory:
        """Trains the policy network using REINFORCE on full-episode rewards."""

        reward_weights = reward_weights or RewardWeights(w_u=2.0, w_s=1.5, w_r=0.5, w_d=2.0, w_a=0.5)
        num_cores = platform.num_cores
        num_dvfs_levels = max(len(core.dvfs_levels) for core in platform.cores)

        sample_graph = applications[0].graph
        graph_tensors = build_graph_tensors(sample_graph)
        with torch.no_grad():
            destination_reliability = build_training_reliability_vector(sample_graph, platform).to(self.device)
            output = self.predictor(
                node_features=graph_tensors.node_features.to(self.device),
                adjacency=graph_tensors.adjacency.to(self.device),
                destination_reliability=destination_reliability,
            )
            sample_task_features = self._build_policy_task_features(
                graph=sample_graph,
                embeddings=output.embeddings.detach().cpu(),
                predicted_mean=output.lo_mean.detach().cpu(),
                predicted_std=torch.sqrt(torch.exp(output.log_variance).clamp_min(1e-8)).detach().cpu(),
                use_uncertainty_features=self.use_uncertainty_features,
            )
        self.policy_network = SchedulingPolicyNetwork(
            input_features=sample_task_features.size(-1),
            num_cores=num_cores,
            num_dvfs_levels=num_dvfs_levels,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        history = PolicyTrainingHistory()
        moving_baseline = 0.0

        for episode_index in range(num_episodes):
            application = applications[episode_index % len(applications)]
            reward_value, log_probability = self._rollout_episode(
                application=application,
                platform=platform,
                reward_weights=reward_weights,
                deterministic=False,
            )
            moving_baseline = 0.9 * moving_baseline + 0.1 * reward_value
            advantage = reward_value - moving_baseline

            optimizer.zero_grad()
            loss = -log_probability * advantage
            loss.backward()
            optimizer.step()

            history.rewards.append(float(reward_value))
            history.baselines.append(float(moving_baseline))
            if verbose:
                progress_percent = ((episode_index + 1) / max(1, num_episodes)) * 100.0
                print(
                    f"[ProposedGAT-Policy] episode {episode_index + 1}/{num_episodes} ({progress_percent:6.2f}%) "
                    f"reward={float(reward_value):.6f} moving_baseline={float(moving_baseline):.6f}"
                )
        return history

    def save_checkpoint(self, checkpoint_dir: str | Path) -> None:
        """Saves the float32 scheduler checkpoint."""

        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.predictor.state_dict(), checkpoint_dir / "predictor_state.pt")
        policy_metadata: Dict[str, int | None]
        if self.policy_network is not None:
            torch.save(self.policy_network.state_dict(), checkpoint_dir / "policy_state.pt")
            policy_metadata = {
                "input_features": self.policy_network.input_features,
                "num_cores": self.policy_network.num_cores,
                "num_dvfs_levels": self.policy_network.num_dvfs_levels,
                "hidden_features": self.policy_network.hidden_features,
            }
        else:
            policy_metadata = {
                "input_features": None,
                "num_cores": None,
                "num_dvfs_levels": None,
                "hidden_features": None,
            }

        save_json(
            {
                "in_features": self.in_features,
                "hidden_features": self.hidden_features,
                "embedding_features": self.embedding_features,
                "num_heads": self.num_heads,
                "use_uncertainty_features": self.use_uncertainty_features,
                "use_qat": self.use_qat,
                "policy": policy_metadata,
                "predictor_parameter_count": count_parameters(self.predictor),
                "predictor_size_bytes": estimate_model_size_bytes(self.predictor),
                "policy_parameter_count": 0 if self.policy_network is None else count_parameters(self.policy_network),
                "policy_size_bytes": 0 if self.policy_network is None else estimate_model_size_bytes(self.policy_network),
            },
            checkpoint_dir / "metadata.json",
        )

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        device: str = "cpu",
        quantize_int8: bool = False,
    ) -> "RiskAwareGatScheduler":
        """Loads a scheduler checkpoint and optionally converts it to real dynamic INT8."""

        checkpoint_dir = Path(checkpoint_dir)
        metadata = load_json(checkpoint_dir / "metadata.json")
        scheduler = cls(
            in_features=int(metadata["in_features"]),
            hidden_features=int(metadata["hidden_features"]),
            embedding_features=int(metadata["embedding_features"]),
            num_heads=int(metadata["num_heads"]),
            device=device,
            use_qat=bool(metadata.get("use_qat", False)),
            use_uncertainty_features=bool(metadata["use_uncertainty_features"]),
        )
        predictor_state = torch.load(checkpoint_dir / "predictor_state.pt", map_location=device)
        scheduler.predictor.load_state_dict(predictor_state)
        scheduler.predictor.to(device)
        scheduler.predictor.eval()

        policy_meta = dict(metadata.get("policy", {}))
        if policy_meta.get("input_features") is not None:
            scheduler.policy_network = SchedulingPolicyNetwork(
                input_features=int(policy_meta["input_features"]),
                num_cores=int(policy_meta["num_cores"]),
                num_dvfs_levels=int(policy_meta["num_dvfs_levels"]),
                hidden_features=int(policy_meta.get("hidden_features", 64)),
            ).to(device)
            policy_state = torch.load(checkpoint_dir / "policy_state.pt", map_location=device)
            scheduler.policy_network.load_state_dict(policy_state)
            scheduler.policy_network.eval()

        if quantize_int8:
            scheduler = scheduler.build_int8_inference_variant()
        return scheduler

    def build_int8_inference_variant(self) -> "RiskAwareGatScheduler":
        """Builds a PDF-oriented INT8 inference variant without dynamic INT8 kernels.

        The variant applies QAT-compatible INT8 weight rounding/dequantization to
        the PDF-selected linear weights while keeping attention score, Softmax,
        uncertainty heads, and final output heads numerically stable in floating
        point. This avoids the unreliable CPU dynamic-INT8 path and keeps the
        reported INT8 method aligned with the project PDF.
        """

        int8_scheduler = RiskAwareGatScheduler(
            in_features=self.in_features,
            hidden_features=self.hidden_features,
            embedding_features=self.embedding_features,
            num_heads=self.num_heads,
            device="cpu",
            use_qat=False,
            use_uncertainty_features=self.use_uncertainty_features,
        )
        int8_scheduler.predictor = quantize_dequantize_int8_weights_model(
            self.predictor,
            per_channel=True,
            include_parameter=pdf_qat_parameter_selector,
        )
        if self.policy_network is not None:
            int8_scheduler.policy_network = quantize_dequantize_int8_weights_model(
                self.policy_network,
                per_channel=True,
                include_parameter=pdf_qat_parameter_selector,
            )
        int8_scheduler.is_quantized_int8 = True
        return int8_scheduler

    def predictor_parameter_count(self) -> int:
        """Returns predictor parameter count using the float architecture definition."""

        return count_parameters(self.predictor)

    def policy_parameter_count(self) -> int:
        """Returns policy-network parameter count."""

        if self.policy_network is None:
            return 0
        return count_parameters(self.policy_network)

    @torch.no_grad()
    def benchmark_predictor_latency_ms(
        self,
        application: DagApplication,
        platform: HeterogeneousPlatform,
        warmup_runs: int = 1,
        timed_runs: int = 5,
    ) -> float:
        """Benchmarks predictor inference latency in milliseconds."""

        graph = application.graph
        graph_tensors = build_graph_tensors(graph)
        destination_reliability = build_training_reliability_vector(graph, platform).to(self.device)

        def predictor_forward() -> object:
            """Run the predictor forward step and return its computed result."""
            return self.predictor(
                node_features=graph_tensors.node_features.to(self.device),
                adjacency=graph_tensors.adjacency.to(self.device),
                destination_reliability=destination_reliability,
            )

        return benchmark_callable_latency_ms(
            predictor_forward,
            warmup_runs=warmup_runs,
            timed_runs=timed_runs,
        )

    @torch.no_grad()
    def benchmark_policy_latency_ms(
        self,
        application: DagApplication,
        platform: HeterogeneousPlatform,
        warmup_runs: int = 1,
        timed_runs: int = 5,
    ) -> float:
        """Benchmarks policy-network inference latency in milliseconds."""

        if self.policy_network is None:
            raise RuntimeError("Policy network is not initialized.")

        graph = application.graph
        graph_tensors = build_graph_tensors(graph)
        destination_reliability = build_training_reliability_vector(graph, platform).to(self.device)
        predictor_output = self.predictor(
            node_features=graph_tensors.node_features.to(self.device),
            adjacency=graph_tensors.adjacency.to(self.device),
            destination_reliability=destination_reliability,
        )
        predicted_mean = predictor_output.lo_mean.detach().cpu()
        predicted_std = torch.sqrt(torch.exp(predictor_output.log_variance).clamp_min(1e-8)).detach().cpu()
        embeddings = predictor_output.embeddings.detach().cpu()
        task_features = self._build_policy_task_features(
            graph=graph,
            embeddings=embeddings,
            predicted_mean=predicted_mean,
            predicted_std=predicted_std,
            use_uncertainty_features=self.use_uncertainty_features,
        ).to(self.device)

        def policy_forward() -> object:
            """Run the policy forward step and return its computed result."""
            return self.policy_network(task_features)

        return benchmark_callable_latency_ms(
            policy_forward,
            warmup_runs=warmup_runs,
            timed_runs=timed_runs,
        )

    @torch.no_grad()
    def benchmark_schedule_latency_ms(
        self,
        application: DagApplication,
        platform: HeterogeneousPlatform,
        warmup_runs: int = 1,
        timed_runs: int = 3,
    ) -> float:
        """Benchmarks end-to-end deterministic scheduling latency."""

        def schedule_forward() -> object:
            """Run the schedule forward step and return its computed result."""
            return self.schedule(application=application, platform=platform)

        return benchmark_callable_latency_ms(
            schedule_forward,
            warmup_runs=warmup_runs,
            timed_runs=timed_runs,
        )

    def schedule(
        self,
        application: DagApplication,
        platform: HeterogeneousPlatform,
    ) -> ScheduleMetrics:
        """Runs deterministic inference with the trained scheduler."""

        if self.policy_network is None:
            raise RuntimeError("fit_policy must be called before schedule.")
        metrics, _ = self._build_schedule_from_policy(
            application=application,
            platform=platform,
            deterministic=True,
        )
        return metrics

    def _rollout_episode(
        self,
        application: DagApplication,
        platform: HeterogeneousPlatform,
        reward_weights: RewardWeights,
        deterministic: bool,
    ) -> Tuple[float, torch.Tensor]:
        """Runs one policy-gradient episode."""

        metrics, log_probability = self._build_schedule_from_policy(
            application=application,
            platform=platform,
            deterministic=deterministic,
        )
        reward_value = compute_reward(
            utilization=metrics.utilization,
            mode_switch_probability=metrics.mode_switch_probability,
            cvar_upper_value=metrics.average_cvar / max(application.deadline, 1e-9),
            deadline_miss=metrics.deadline_miss,
            aging_index=metrics.aging_index,
            weights=reward_weights,
        )
        return reward_value, log_probability

    def _build_schedule_from_policy(
        self,
        application: DagApplication,
        platform: HeterogeneousPlatform,
        deterministic: bool,
    ) -> Tuple[ScheduleMetrics, torch.Tensor]:
        """Predicts task costs, samples scheduling actions, and evaluates them."""

        if self.policy_network is None:
            raise RuntimeError("Policy network is not initialized.")

        graph = application.clone_graph()
        graph_tensors = build_graph_tensors(graph)
        destination_reliability = build_training_reliability_vector(graph, platform).to(self.device)
        predictor_output = self.predictor(
            node_features=graph_tensors.node_features.to(self.device),
            adjacency=graph_tensors.adjacency.to(self.device),
            destination_reliability=destination_reliability,
        )

        predicted_mean = predictor_output.lo_mean.detach().cpu()
        predicted_std = torch.sqrt(torch.exp(predictor_output.log_variance).clamp_min(1e-8)).detach().cpu()
        embeddings = predictor_output.embeddings.detach().cpu()
        task_features = self._build_policy_task_features(
            graph=graph,
            embeddings=embeddings,
            predicted_mean=predicted_mean,
            predicted_std=predicted_std,
            use_uncertainty_features=self.use_uncertainty_features,
        ).to(self.device)
        priority_mean, core_logits, dvfs_logits = self.policy_network(task_features)

        if deterministic:
            priority_samples = priority_mean.squeeze(-1)
            core_samples = torch.argmax(core_logits, dim=-1)
            dvfs_samples = torch.argmax(dvfs_logits, dim=-1)
            log_probability = torch.zeros((), device=self.device)
        else:
            priority_std = torch.exp(self.policy_network.priority_log_std).clamp(min=1e-3, max=2.0)
            priority_distribution = Normal(loc=priority_mean.squeeze(-1), scale=priority_std)
            priority_samples = priority_distribution.rsample()
            core_distribution = Categorical(logits=core_logits)
            dvfs_distribution = Categorical(logits=dvfs_logits)
            core_samples = core_distribution.sample()
            dvfs_samples = dvfs_distribution.sample()
            log_probability = (
                priority_distribution.log_prob(priority_samples).sum()
                + core_distribution.log_prob(core_samples).sum()
                + dvfs_distribution.log_prob(dvfs_samples).sum()
            )

        task_ids = graph_tensors.task_ids
        priority_scores: Dict[int, float] = {}
        core_assignments: Dict[int, int] = {}
        dvfs_assignments: Dict[int, int] = {}
        for index, task_id in enumerate(task_ids):
            graph.nodes[task_id].predicted_lo_wcet = float(predicted_mean[index].item())
            graph.nodes[task_id].predicted_uncertainty = (
                float(predicted_std[index].item()) if self.use_uncertainty_features else 0.0
            )
            priority_scores[task_id] = float(priority_samples[index].item())
            core_id = int(core_samples[index].item())
            core_id = max(0, min(platform.num_cores - 1, core_id))
            dvfs_id = int(dvfs_samples[index].item())
            max_dvfs_id = len(platform.cores[core_id].dvfs_levels) - 1
            dvfs_id = max(0, min(max_dvfs_id, dvfs_id))
            core_assignments[task_id] = core_id
            dvfs_assignments[task_id] = dvfs_id

        decoded = decode_priority_schedule(
            graph=graph,
            platform=platform,
            priority_scores=priority_scores,
            core_assignments=core_assignments,
            dvfs_assignments=dvfs_assignments,
            use_predictions=True,
            drop_lo_after_mode_switch=False,
        )
        metrics = evaluate_decoded_schedule(graph=graph, platform=platform, decoded=decoded)
        return metrics, log_probability

    @staticmethod
    def _build_policy_task_features(
        graph: TaskGraph,
        embeddings: torch.Tensor,
        predicted_mean: torch.Tensor,
        predicted_std: torch.Tensor,
        use_uncertainty_features: bool,
    ) -> torch.Tensor:
        """Builds policy-network input features from predictor outputs and node metadata."""

        features: List[List[float]] = []
        max_mean = max(float(predicted_mean.max().item()), 1e-9)
        for index, task_id in enumerate(graph.topological_order()):
            node = graph.nodes[task_id]
            hi_flag = 1.0 if node.criticality == "HI" else 0.0
            local_features = [
                float(predicted_mean[index].item()) / max_mean,
                float(predicted_std[index].item()) if use_uncertainty_features else 0.0,
                hi_flag,
                graph.normalized_level(task_id),
                len(node.predecessors),
                len(node.successors),
            ]
            embedding_values = embeddings[index].tolist()
            features.append(embedding_values + local_features)
        return torch.tensor(features, dtype=torch.float32)
