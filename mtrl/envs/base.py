import abc
from dataclasses import dataclass

import gymnasium as gym

from mtrl.types import Agent


@dataclass(frozen=True)
class EnvConfig(abc.ABC):
    env_id: str
    use_one_hot: bool = True
    max_episode_steps: int = 500
    evaluation_num_episodes: int = 50
    terminate_on_success: bool = False

    @abc.abstractmethod
    def spawn(self) -> gym.vector.VectorEnv: ...

    @abc.abstractmethod
    def evaluate(
        self, envs: gym.vector.VectorEnv, agent: Agent
    ) -> tuple[float, float, dict[str, float]]: ...
