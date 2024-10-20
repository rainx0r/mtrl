# pyright: reportCallIssue=false, reportAttributeAccessIssue=false
from pathlib import Path
import random
from typing import TypedDict, NotRequired
import gymnasium as gym

import numpy as np
import orbax.checkpoint as ocp


from mtrl.rl.algorithms.base import Algorithm
from mtrl.rl.buffers import MultiTaskReplayBuffer
from mtrl.types import (
    CheckpointMetadata,
    EnvCheckpoint,
    RNGCheckpoint,
    ReplayBufferCheckpoint,
)


class Checkpoint(TypedDict):
    agent: Algorithm
    buffer: NotRequired[ReplayBufferCheckpoint]
    envs: EnvCheckpoint
    rngs: RNGCheckpoint
    metadata: CheckpointMetadata


def checkpoint_envs(envs: gym.vector.VectorEnv) -> list[tuple[str, dict]]:
    return envs.call("get_checkpoint")


def load_env_checkpoints(envs: gym.vector.VectorEnv, env_ckpts: list[tuple[str, dict]]):
    envs.call("load_checkpoint", env_ckpts)


def get_checkpoint_save_args(
    agent: Algorithm,
    envs: gym.vector.VectorEnv,
    total_steps: int,
    episodes_ended: int,
    buffer: MultiTaskReplayBuffer | None = None,
) -> ocp.args.CheckpointArgs:
    if buffer is not None:
        rb_ckpt = buffer.checkpoint()
        buffer_args = ocp.args.Composite(
            data=ocp.args.PyTreeSave(rb_ckpt["data"]),
            rng_state=ocp.args.JsonSave(rb_ckpt["rng_state"]),
        )
    else:
        buffer_args = None
    return ocp.args.Composite(
        agent=ocp.args.PyTreeSave(agent),
        buffer=buffer_args,
        env_states=ocp.args.JsonSave(checkpoint_envs(envs)),
        rngs=ocp.args.Composite(
            python_rng_state=ocp.args.PyTreeSave(random.getstate()),
            global_numpy_rng_state=ocp.args.NumpyRandomKeySave(np.random.get_state()),
        ),
        metadata=ocp.args.JsonSave(
            {
                "step": total_steps,
                "episodes_ended": episodes_ended,
            }
        ),
    )


def get_checkpoint_restore_args(
    agent: Algorithm, buffer: MultiTaskReplayBuffer | None = None
):
    if buffer is not None:
        rb_ckpt = buffer.checkpoint()
        buffer_args = ocp.args.Composite(
            data=ocp.args.PyTreeRestore(rb_ckpt["data"]),
            rng_state=ocp.args.JsonRestore(),
        )
    else:
        buffer_args = None

    return ocp.args.Composite(
        agent=ocp.args.PyTreeRestore(agent),
        buffer=buffer_args,
        env_states=ocp.args.JsonRestore(),
        rngs=ocp.args.Composite(
            python_rng_state=ocp.args.PyTreeRestore(random.getstate()),
            global_numpy_rng_state=ocp.args.NumpyRandomKeyRestore(),
        ),
        metadata=ocp.args.JsonRestore(),
    )
