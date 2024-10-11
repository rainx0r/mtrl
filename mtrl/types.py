from typing import NamedTuple

import numpy.typing as npt


class ReplayBufferSamples(NamedTuple):
    observations: npt.NDArray
    actions: npt.NDArray
    next_observations: npt.NDArray
    dones: npt.NDArray
    rewards: npt.NDArray


class Rollout(NamedTuple):
    # Standard timestep data
    observations: npt.NDArray
    actions: npt.NDArray
    rewards: npt.NDArray
    dones: npt.NDArray

    # Auxiliary policy outputs
    log_probs: npt.NDArray | None = None
    means: npt.NDArray | None = None
    stds: npt.NDArray | None = None

    # Computed statistics about observed rewards
    returns: npt.NDArray | None = None
    advantages: npt.NDArray | None = None
    episode_returns: npt.NDArray | None = None
