from dataclasses import dataclass
from pathlib import Path

import tyro

from mtrl.config.networks import ContinuousActionPolicyConfig, QValueFunctionConfig
from mtrl.config.nn import SoftModulesConfig
from mtrl.config.optim import OptimizerConfig
from mtrl.config.rl import OffPolicyTrainingConfig
from mtrl.envs import MetaworldConfig
from mtrl.experiment import Experiment
from mtrl.rl.algorithms import MTSACConfig


@dataclass(frozen=True)
class Args:
    seed: int = 1
    track: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    data_dir: Path = Path("./experiment_results")
    resume: bool = False


def main() -> None:
    args = tyro.cli(Args)
    NUM_TASKS = 50

    experiment = Experiment(
        exp_name="mt50_softmodules_mooore_params",
        seed=args.seed,
        data_dir=args.data_dir,
        env=MetaworldConfig(
            env_id="MT50",
            terminate_on_success=False,
        ),
        algorithm=MTSACConfig(
            num_tasks=NUM_TASKS,
            gamma=0.99,
            actor_config=ContinuousActionPolicyConfig(
                network_config=SoftModulesConfig(
                    num_tasks=NUM_TASKS,
                    optimizer=OptimizerConfig(max_grad_norm=1.0),
                    depth=4,
                    num_modules=4,
                    width=305,
                    module_width=305,
                )
            ),
            critic_config=QValueFunctionConfig(
                network_config=SoftModulesConfig(
                    num_tasks=NUM_TASKS,
                    optimizer=OptimizerConfig(max_grad_norm=1.0),
                    depth=4,
                    num_modules=4,
                    width=305,
                    module_width=305,
                )
            ),
            num_critics=2,
            use_task_weights=True,
        ),
        training_config=OffPolicyTrainingConfig(
            total_steps=int(2_000_000 * NUM_TASKS),  # 1e8
            buffer_size=int(100_000 * NUM_TASKS),  # 5M
            batch_size=(128 * NUM_TASKS),  # 6400
            evaluation_frequency=int(1_000_000 // 500),
        ),
        checkpoint=True,
        resume=args.resume,
    )

    if args.track:
        assert args.wandb_project is not None and args.wandb_entity is not None
        experiment.enable_wandb(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=experiment,
            resume="allow",
        )

    experiment.run()


if __name__ == "__main__":
    main()
