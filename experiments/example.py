import jax
import jax.numpy as jnp
from dataclasses import dataclass

from pathlib import Path

from mtrl.config.optim import OptimizerConfig
from mtrl.config.rl import OffPolicyTrainingConfig
from mtrl.experiment import Experiment
from mtrl.config.networks import ContinuousActionPolicyConfig, QValueFunctionConfig
from mtrl.config.nn import MultiHeadConfig
from mtrl.envs import MetaworldConfig
from mtrl.rl.algorithms import MTSACConfig

from time import localtime, strftime
from typing import Optional
import tyro


@dataclass(frozen=True)
class Args:
    # TODO: Make a common set of experiment args, then a set of args per alg? ie experiment args: seed, exp_name, wandb, info. Algorithm args: MTSAC: gamma, num_tasks, etc
    seed: Optional[int] = 1
    experiment_name: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None


def main() -> None:
    args = tyro.cli(Args)

    experiment = Experiment(
        exp_name=args.experiment_name,
        seed=args.seed,
        data_dir=Path(f"./experiment_results/{args.experiment_name}"),
        env=MetaworldConfig(
            env_id="MT10",
            terminate_on_success=False,
            # num_eval_episodes=1,
        ),
        algorithm=MTSACConfig(
            num_tasks=10,
            gamma=0.99,
            actor_config=ContinuousActionPolicyConfig(
                network_config=MultiHeadConfig(
                    num_tasks=10, depth=2, optimizer=OptimizerConfig(max_grad_norm=1.0)
                )
            ),
            critic_config=QValueFunctionConfig(
                network_config=MultiHeadConfig(
                    num_tasks=10, depth=2, optimizer=OptimizerConfig(max_grad_norm=1.0)
                )
            ),
            num_critics=2,
            use_task_weights=True,
        ),
        training_config=OffPolicyTrainingConfig(
            #warmstart_steps=0,
            total_steps=int(2e7),
            buffer_size=int(1e6),
            batch_size=1280,
            #evaluation_frequency=1
        ),
        checkpoint=True,
        resume=False,
    )


    if args.experiment_name:
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
