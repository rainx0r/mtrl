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
    max_episode_steps: int = 500
    evaluation_frequency: int = 200_000 // 500
    evaluation_num_episodes: int = 50


@dataclass(frozen=True)
class OffPolicyTrainingConfig(TrainingConfig):
    warmstart_steps: int = int(4e3)
    buffer_size: int = int(1e6)
    batch_size: int = 1280
