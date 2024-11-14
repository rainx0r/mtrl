from dataclasses import dataclass
from pathlib import Path

import tyro

from mtrl.config.networks import ContinuousActionPolicyConfig, QValueFunctionConfig
from mtrl.config.nn import MultiHeadConfig
from mtrl.config.optim import OptimizerConfig
from mtrl.config.rl import OffPolicyTrainingConfig
from mtrl.envs import MetaworldConfig
from mtrl.experiment import Experiment
from mtrl.rl.algorithms import MTSACConfig


@dataclass(frozen=True)
class Args:
    experiment_name: str
    seed: int = 1
    track: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    data_dir: Path = Path("./experiment_results")


def main() -> None:
    args = tyro.cli(Args)

    experiment = Experiment(
        exp_name=args.experiment_name,
        seed=args.seed,
        data_dir=args.data_dir / args.experiment_name,
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
                    num_tasks=10, optimizer=OptimizerConfig(max_grad_norm=1.0)
                )
            ),
            critic_config=QValueFunctionConfig(
                network_config=MultiHeadConfig(
                    num_tasks=10, optimizer=OptimizerConfig(max_grad_norm=1.0)
                )
            ),
            num_critics=2,
            use_task_weights=True,
        ),
        training_config=OffPolicyTrainingConfig(
            warmstart_steps=0,
            total_steps=int(2e7),
            buffer_size=int(1e6),
            batch_size=1280,
            # evaluation_frequency=1
        ),
        checkpoint=True,
        resume=False,
    )

    if args.track:
        assert args.wandb_project is not None and args.wandb_entity is not None
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
