from dataclasses import dataclass


@dataclass(frozen=True)
class AlgorithmConfig:
    num_tasks: int
    multi_task_optimizer = None
    gamma: float = 0.99


@dataclass(frozen=True, kw_only=True)
class TrainingConfig:
    total_steps: int
    evaluation_frequency: int = 200_000 // 500
    compute_network_metrics: bool = True

    # TODO: Maybe put into its own RewardFilterConfig()?
    reward_filter: str | None = None
    reward_filter_sigma: float | None = None
    reward_filter_alpha: float | None = None
    reward_filter_delta: float | None = None
    reward_filter_mode: str | None = None


@dataclass(frozen=True)
class OffPolicyTrainingConfig(TrainingConfig):
    warmstart_steps: int = int(4e3)
    buffer_size: int = int(1e6)
    batch_size: int = 1280


@dataclass(frozen=True)
class OnPolicyTrainingConfig(TrainingConfig):
    rollout_steps: int = 10_000
    num_epochs: int = 16
    num_gradient_steps: int = 32

    compute_advantages: bool = True
    gae_lambda: float = 0.97
    target_kl: float | None = None
