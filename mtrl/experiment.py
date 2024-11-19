"""Based on https://github.com/kevinzakka/nanorl/blob/main/nanorl/infra/experiment.py"""

import pathlib
import random
from dataclasses import dataclass

import orbax.checkpoint as ocp
import numpy as np
import wandb

from mtrl.checkpoint import (
    Checkpoint,
    get_checkpoint_restore_args,
    load_env_checkpoints,
)
from mtrl.config.rl import AlgorithmConfig, OffPolicyTrainingConfig, TrainingConfig
from mtrl.envs import EnvConfig
from mtrl.rl.algorithms import (
    Algorithm,
    OffPolicyAlgorithm,
    get_algorithm_for_config,
)
from mtrl.types import CheckpointMetadata


@dataclass
class Experiment:
    exp_name: str
    seed: int
    data_dir: pathlib.Path

    env: EnvConfig
    algorithm: AlgorithmConfig
    training_config: TrainingConfig

    checkpoint: bool = True
    max_checkpoints_to_keep: int = 5
    best_checkpoint_metric: str = "charts/mean_success_rate"
    resume: bool = False

    def __post_init__(self) -> None:
        self._wandb_enabled = False

    def enable_wandb(self, **wandb_kwargs) -> None:
        self._wandb_enabled = True

        wandb.init(dir=str(self.data_dir), **wandb_kwargs)

    def run(self) -> None:
        envs = self.env.spawn(seed=self.seed)

        algorithm_cls = get_algorithm_for_config(self.algorithm)
        algorithm: Algorithm
        algorithm = algorithm_cls.initialize(self.algorithm, self.env, seed=self.seed)
        is_off_policy = isinstance(algorithm, OffPolicyAlgorithm)

        buffer_checkpoint = None
        checkpoint_manager = None
        checkpoint_metadata = None
        envs_checkpoint = None

        random.seed(self.seed)
        np.random.seed(self.seed)

        if self.checkpoint:
            checkpoint_items = (
                "agent",
                "env_states",
                "rngs",
                "metadata",
            )
            if is_off_policy:
                checkpoint_items += ("buffer",)

            checkpoint_manager = ocp.CheckpointManager(
                pathlib.Path(self.data_dir / "checkpoints").absolute(),
                item_names=checkpoint_items,
                options=ocp.CheckpointManagerOptions(
                    max_to_keep=self.max_checkpoints_to_keep,
                    create=True,
                    best_fn=lambda x: x[self.best_checkpoint_metric],
                ),
            )

            if self.resume and checkpoint_manager.latest_step() is not None:
                if is_off_policy:
                    assert isinstance(self.training_config, OffPolicyTrainingConfig)
                    rb = algorithm.spawn_replay_buffer(
                        self.env,
                        self.training_config,
                    )
                else:
                    rb = None
                ckpt: Checkpoint = checkpoint_manager.restore(  # pyright: ignore [reportAssignmentType]
                    checkpoint_manager.latest_step(),
                    args=get_checkpoint_restore_args(algorithm, rb),
                )
                algorithm = ckpt["agent"]

                if is_off_policy:
                    buffer_checkpoint = ckpt["buffer"]  # pyright: ignore [reportTypedDictNotRequiredAccess]

                envs_checkpoint = ckpt["env_states"]
                load_env_checkpoints(envs, envs_checkpoint)

                random.setstate(ckpt["rngs"]["python_rng_state"])
                np.random.set_state(ckpt["rngs"]["global_numpy_rng_state"])

                checkpoint_metadata: CheckpointMetadata | None = ckpt["metadata"]
                assert checkpoint_metadata is not None

                print(f"Loaded checkpoint at step {checkpoint_metadata['step']}")

        # Track number of params
        if self._wandb_enabled:
            wandb.config.update(algorithm.get_num_params())

        algorithm.train(
            config=self.training_config,
            envs=envs,
            env_config=self.env,
            seed=self.seed,
            track=self._wandb_enabled,
            checkpoint_manager=checkpoint_manager,
            checkpoint_metadata=checkpoint_metadata,
            buffer_checkpoint=buffer_checkpoint,
        )
