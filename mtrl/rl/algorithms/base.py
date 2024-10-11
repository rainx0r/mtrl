import abc

import jax
import numpy as np
import numpy.typing as npt
from flax import struct

from mtrl.config.rl import AlgorithmConfig, TrainingConfig
from mtrl.types import ReplayBufferSamples, Rollout


class Algorithm(abc.ABC, struct.PyTreeNode):
    """Inspired by https://github.com/kevinzakka/nanorl/blob/main/nanorl/agent.py"""

    @staticmethod
    @abc.abstractmethod
    def initialize(config: AlgorithmConfig) -> "Algorithm": ...

    @abc.abstractmethod
    def update(
        self, data: ReplayBufferSamples | Rollout
    ) -> tuple["Algorithm", dict[str, float]]: ...

    @abc.abstractmethod
    def sample_action(
        self, observations: jax.typing.ArrayLike
    ) -> tuple["Algorithm", npt.NDArray[np.float64]]: ...

    @abc.abstractmethod
    def sample_action_and_log_prob(
        self, observations: jax.typing.ArrayLike
    ) -> tuple["Algorithm", npt.NDArray[np.float64]]: ...

    @abc.abstractmethod
    def eval_action(
        self, observations: jax.typing.ArrayLike
    ) -> npt.NDArray[np.float64]: ...

    @abc.abstractmethod
    def train(self, config: TrainingConfig) -> "Algorithm": ...
