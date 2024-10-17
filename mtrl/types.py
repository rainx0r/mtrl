from typing import NamedTuple

import numpy as np

from jaxtyping import Float

type LogDict = dict[str, float]

Action = Float[np.ndarray, "... action_dim"]
LogProb = Float[np.ndarray, "... 1"]
Observation = Float[np.ndarray, "... obs_dim"]


class ReplayBufferSamples(NamedTuple):
    observations: Float[Observation, " batch"]
    actions: Float[Action, " batch"]
    next_observations: Float[Observation, " batch"]
    dones: Float[np.ndarray, "batch 1"]
    rewards: Float[np.ndarray, "batch 1"]


class Rollout(NamedTuple):
    # Standard timestep data
    observations: Float[Observation, "task timestep"]
    actions: Float[Action, "task timestep"]
    rewards: Float[np.ndarray, "task timestep 1"]
    dones: Float[np.ndarray, "task timestep 1"]

    # Auxiliary policy outputs
    log_probs: Float[LogProb, "task timestep"] | None = None
    means: Float[np.ndarray, "task timestep 1"] | None = None
    stds: Float[np.ndarray, "task timestep 1"] | None = None

    # Computed statistics about observed rewards
    returns: Float[np.ndarray, "task timestep 1"] | None = None
    advantages: Float[np.ndarray, "task timestep 1"] | None = None
    episode_returns: Float[np.ndarray, "task timestep 1"] | None = None
