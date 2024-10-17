import abc
from typing import Generic, Self, TypeVar

import time

from flax import struct

from mtrl.config.rl import AlgorithmConfig, TrainingConfig, OffPolicyTrainingConfig
from mtrl.types import (
    Action,
    LogDict,
    Observation,
    ReplayBufferSamples,
    Rollout,
)
from typing import override

AlgorithmConfigType = TypeVar("AlgorithmConfigType", bound=AlgorithmConfig)
TrainingConfigType = TypeVar("TrainingConfigType", bound=TrainingConfig)


class Algorithm(
    abc.ABC, Generic[AlgorithmConfigType, TrainingConfigType], struct.PyTreeNode
):
    """Inspired by https://github.com/kevinzakka/nanorl/blob/main/nanorl/agent.py"""

    num_tasks: int = struct.field(pytree_node=False)

    @staticmethod
    @abc.abstractmethod
    def initialize(config: AlgorithmConfigType) -> "Algorithm": ...

    @abc.abstractmethod
    def update(self, data: ReplayBufferSamples | Rollout) -> tuple[Self, LogDict]: ...

    @abc.abstractmethod
    def sample_action(self, observation: Observation) -> tuple[Self, Action]: ...

    @abc.abstractmethod
    def eval_action(self, observation: Observation) -> Action: ...

    @abc.abstractmethod
    def train(self, config: TrainingConfigType) -> Self: ...


class OffPolicyAlgorithm(
    Algorithm[AlgorithmConfigType, OffPolicyTrainingConfig],
    Generic[AlgorithmConfigType],
):
    @override
    def train(self, config: OffPolicyTrainingConfig) -> Self:
        # start_time = time.time()
        #
        # for global_step in range(start_step, config.total_steps)
        ...
