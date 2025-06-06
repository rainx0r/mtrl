import os

os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                           "intra_op_parallelism_threads=4")

os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREAD"] = "4"

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
    seed: int = 1
    track: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    data_dir: Path = Path("./experiment_results")
    resume: bool = False


def main() -> None:
    args = tyro.cli(Args)

    experiment = Experiment(
        exp_name="mt10_mtmhsac_paco_params_v2_3_layers_use_task_weights",
        seed=args.seed,
        data_dir=args.data_dir,
        env=MetaworldConfig(
            env_id="MT10",
            terminate_on_success=False,
        ),
        algorithm=MTSACConfig(
            num_tasks=10,
            gamma=0.99,
            actor_config=ContinuousActionPolicyConfig(
                network_config=MultiHeadConfig(
                    num_tasks=10,
                    optimizer=OptimizerConfig(max_grad_norm=1.0),
                    depth=3,
                    width=885
                )
            ),
            critic_config=QValueFunctionConfig(
                network_config=MultiHeadConfig(
                    num_tasks=10,
                    optimizer=OptimizerConfig(max_grad_norm=1.0),
                    depth=3,
                    width=885
                )
            ),
            use_task_weights=True,
            num_critics=2,
        ),
        training_config=OffPolicyTrainingConfig(
            total_steps=int(2e7),
            buffer_size=int(1e6),
            batch_size=1280,
        ),
        checkpoint=False,
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
