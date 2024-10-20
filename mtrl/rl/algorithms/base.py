import abc
import time
from collections import deque
from typing import Deque, Generic, Self, TypeVar, override

import numpy as np
import wandb
from flax import struct
from orbax.checkpoint import CheckpointManager

from mtrl.checkpoint import get_checkpoint_save_args
from mtrl.config.rl import AlgorithmConfig, OffPolicyTrainingConfig, TrainingConfig
from mtrl.envs import EnvConfig
from mtrl.rl.buffers import MultiTaskReplayBuffer
from mtrl.types import (
    Action,
    Agent,
    CheckpointMetadata,
    LogDict,
    Observation,
    ReplayBufferCheckpoint,
    ReplayBufferSamples,
    Rollout,
)

AlgorithmConfigType = TypeVar("AlgorithmConfigType", bound=AlgorithmConfig)
TrainingConfigType = TypeVar("TrainingConfigType", bound=TrainingConfig)


class Algorithm(
    abc.ABC, Agent, Generic[AlgorithmConfigType, TrainingConfigType], struct.PyTreeNode
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
    def train(
        self,
        config: TrainingConfigType,
        env_config: EnvConfig,
        seed: int = 1,
        track: bool = True,
        checkpoint_manager: CheckpointManager | None = None,
        checkpoint_metadata: CheckpointMetadata | None = None,
        buffer_checkpoint: ReplayBufferCheckpoint | None = None,
    ) -> Self: ...


class OffPolicyAlgorithm(
    Algorithm[AlgorithmConfigType, OffPolicyTrainingConfig],
    Generic[AlgorithmConfigType],
):
    @override
    def train(
        self,
        config: OffPolicyTrainingConfig,
        env_config: EnvConfig,
        seed: int = 1,
        track: bool = True,
        checkpoint_manager: CheckpointManager | None = None,
        checkpoint_metadata: CheckpointMetadata | None = None,
        buffer_checkpoint: ReplayBufferCheckpoint | None = None,
    ) -> Self:
        global_episodic_return: Deque[float] = deque([], maxlen=20 * self.num_tasks)
        global_episodic_length: Deque[int] = deque([], maxlen=20 * self.num_tasks)

        envs = env_config.spawn()

        obs, _ = envs.reset()
        has_autoreset = np.full((envs.num_envs,), False)
        start_step, episodes_ended = 0, 0

        if checkpoint_metadata is not None:
            start_step = checkpoint_metadata["step"]
            episodes_ended = checkpoint_metadata["episodes"]

        replay_buffer = MultiTaskReplayBuffer(
            total_capacity=config.buffer_size,
            num_tasks=self.num_tasks,
            envs=envs,
            seed=seed,
        )

        if buffer_checkpoint is not None:
            replay_buffer.load_checkpoint(buffer_checkpoint)

        start_time = time.time()

        for global_step in range(start_step, config.total_steps // envs.num_envs):
            total_steps = global_step * envs.num_envs

            if global_step < config.warmstart_steps:
                actions = np.array(
                    [envs.single_action_space.sample() for _ in range(envs.num_envs)]
                )
            else:
                self, actions = self.sample_action(obs)

            next_obs, rewards, terminations, truncations, infos = envs.step(actions)

            if not has_autoreset.any():
                replay_buffer.add(obs, next_obs, actions, rewards, terminations)
            elif has_autoreset.any() and not has_autoreset.all():
                # TODO: handle the case where only some envs have autoreset
                raise NotImplementedError(
                    "Only some envs resetting isn't implemented at the moment."
                )

            has_autoreset = np.logical_or(terminations, truncations)

            for i, env_ended in enumerate(has_autoreset):
                if env_ended:
                    global_episodic_return.append(infos["episode"]["r"][i])
                    global_episodic_length.append(infos["episode"]["l"][i])
                    episodes_ended += 1

            obs = next_obs

            if global_step % 500 == 0 and global_episodic_return:
                print(
                    f"global_step={total_steps}, mean_episodic_return={np.mean(list(global_episodic_return))}"
                )
                if track:
                    wandb.log(
                        {
                            "charts/mean_episodic_return": np.mean(
                                list(global_episodic_return)
                            ),
                            "charts/mean_episodic_length": np.mean(
                                list(global_episodic_length)
                            ),
                        },
                        step=total_steps,
                    )

            if global_step > config.warmstart_steps:
                # Update the agent with data
                data = replay_buffer.sample(config.batch_size)
                self, logs = self.update(data)

                # Logging
                if global_step % 100 == 0:
                    sps_steps = (global_step - start_step) * envs.num_envs
                    sps = int(sps_steps / (time.time() - start_time))
                    print("SPS:", sps)

                    if track:
                        wandb.log({"charts/SPS": sps} | logs, step=total_steps)

                # Evaluation
                if (
                    config.evaluation_frequency > 0
                    and episodes_ended % config.evaluation_frequency == 0
                    and has_autoreset.any()
                    and global_step > 0
                ):
                    mean_success_rate, mean_returns, mean_success_per_task = (
                        env_config.evaluate(envs, self)
                    )
                    eval_metrics = {
                        "charts/mean_success_rate": float(mean_success_rate),
                        "charts/mean_evaluation_return": float(mean_returns),
                    } | {
                        f"charts/{task_name}_success_rate": float(success_rate)
                        for task_name, success_rate in mean_success_per_task.items()
                    }
                    print(
                        f"total_steps={total_steps}, mean evaluation success rate: {mean_success_rate:.4f}"
                        + f" return: {mean_returns:.4f}"
                    )

                    if track:
                        wandb.log(eval_metrics, step=total_steps)

                    # Reset envs again to exit eval mode
                    obs, _ = envs.reset()

                    # Checkpointing
                    if checkpoint_manager is not None:
                        if not has_autoreset.all():
                            raise NotImplementedError(
                                "Checkpointing currently doesn't work for the case where evaluation is run before all envs have finished their episodes / are about to be reset."
                            )

                        checkpoint_manager.save(
                            total_steps,
                            args=get_checkpoint_save_args(
                                self,
                                envs,
                                global_step,
                                episodes_ended,
                                buffer=replay_buffer,
                            ),
                            metrics=eval_metrics,
                        )
        return self
