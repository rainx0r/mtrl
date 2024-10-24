import jax
import jax.numpy as jnp
from dataclasses import dataclass

from pathlib import Path

from mtrl.config.optim import OptimizerConfig
from mtrl.config.rl import OffPolicyTrainingConfig
from mtrl.experiment import Experiment
from mtrl.config.networks import ContinuousActionPolicyConfig, QValueFunctionConfig
from mtrl.config.nn import SoftModulesConfig
from mtrl.envs import MetaworldConfig
from mtrl.rl.algorithms import MTSACConfig

import tyro


@dataclass(frozen=True)
class Args:
    seed: int
    experiment_name: str
    wandb_project: str
    wandb_entity: str


def main() -> None:
    args = tyro.cli(Args)

    experiment = Experiment(
        exp_name=args.experiment_name,
        seed=args.seed,
        data_dir=Path(f"./experiments/{args.experiment_name}"),
        env=MetaworldConfig(
            env_id="MT10",
            terminate_on_success=False,
        ),
        algorithm=MTSACConfig(
            num_tasks=10,
            gamma=0.99,
            actor_config=ContinuousActionPolicyConfig(
                network_config=SoftModulesConfig(
                    num_tasks=10, depth=2, optimizer=OptimizerConfig(max_grad_norm=1.0)
                )
            ),
            critic_config=QValueFunctionConfig(
                network_config=SoftModulesConfig(
                    num_tasks=10, depth=2, optimizer=OptimizerConfig(max_grad_norm=1.0)
                )
            ),
            num_critics=2,
            use_task_weights=True,
        ),
        training_config=OffPolicyTrainingConfig(
            total_steps=int(2e7),
            buffer_size=int(1e6),
            batch_size=1280,
        ),
        checkpoint=True,
        resume=True,
    )

    experiment.enable_wandb(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=experiment,
        name=experiment.exp_name,
        id=f"{experiment.exp_name}_{experiment.seed}",
        resume="allow",
    )

    experiment.run()


if __name__ == "__main__":
    main()
