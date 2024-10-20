from dataclasses import dataclass

import gymnasium as gym


@dataclass(frozen=True)
class AlgorithmConfig:
    num_tasks: int
    gamma: float = 0.99
    action_space: gym.Space | None = None


@dataclass(frozen=True, kw_only=True)
class TrainingConfig:
    total_steps: int
    evaluation_frequency: int = 200_000 // 500


@dataclass(frozen=True)
class OffPolicyTrainingConfig(TrainingConfig):
    warmstart_steps: int = int(4e3)
    buffer_size: int = int(1e6)
    batch_size: int = 1280
