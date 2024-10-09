from collections.abc import Callable
from typing import NamedTuple

import gymnasium as gym
import numpy as np
import numpy.typing as npt

import scipy


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


class MultiTaskReplayBuffer:
    """Replay buffer for the multi-task benchmarks.

    Each sampling step, it samples a batch for each tasks, returning a batch of shape (batch_size, num_tasks).
    """

    obs: npt.NDArray
    actions: npt.NDArray
    rewards: npt.NDArray
    next_obs: npt.NDArray
    dones: npt.NDArray
    pos: int

    def __init__(
        self,
        total_capacity: int,
        num_tasks: int,
        envs: gym.vector.VectorEnv,
        seed: int | None = None,
    ):
        assert (
            total_capacity % num_tasks == 0
        ), "Total capacity must be divisible by the number of tasks."
        self.capacity = total_capacity // num_tasks
        self.num_tasks = num_tasks
        self._rng = np.random.default_rng(seed)
        self._obs_shape = np.array(envs.single_observation_space.shape).prod()
        self._action_shape = np.array(envs.single_action_space.shape).prod()
        self.full = False
        self.reset()  # Init buffer

    def reset(self):
        """Reinitialize the buffer."""
        self.obs = np.zeros(
            (self.capacity, self.num_tasks, self._obs_shape), dtype=np.float32
        )
        self.actions = np.zeros(
            (self.capacity, self.num_tasks, self._action_shape), dtype=np.float32
        )
        self.rewards = np.zeros((self.capacity, self.num_tasks, 1), dtype=np.float32)
        self.next_obs = np.zeros(
            (self.capacity, self.num_tasks, self._obs_shape), dtype=np.float32
        )
        self.dones = np.zeros((self.capacity, self.num_tasks, 1), dtype=np.float32)
        self.pos = 0

    def checkpoint(self) -> dict:
        return {
            "data": {
                "obs": self.obs,
                "actions": self.actions,
                "rewards": self.rewards,
                "next_obs": self.next_obs,
                "dones": self.dones,
                "pos": self.pos,
                "full": self.full,
            },
            "rng_state": self._rng.__getstate__(),
        }

    def load_checkpoint(self, ckpt: dict) -> None:
        for key in ["data", "rng_state"]:
            assert key in ckpt

        for key in ["obs", "actions", "rewards", "next_obs", "dones", "pos", "full"]:
            assert key in ckpt["data"]
            setattr(self, key, ckpt["data"][key])

        self._rng.__setstate__(ckpt["rng_state"])

    def add(
        self,
        obs: npt.NDArray,
        next_obs: npt.NDArray,
        action: npt.NDArray,
        reward: npt.NDArray,
        done: npt.NDArray,
    ):
        """Add a batch of samples to the buffer.

        It is assumed that the observation has a one-hot task embedding as its suffix.
        """
        task_idx = obs[:, -self.num_tasks :].argmax(1)

        self.obs[self.pos, task_idx] = obs.copy()
        self.actions[self.pos, task_idx] = action.copy()
        self.rewards[self.pos, task_idx] = reward.copy().reshape(-1, 1)
        self.next_obs[self.pos, task_idx] = next_obs.copy()
        self.dones[self.pos, task_idx] = done.copy().reshape(-1, 1)

        self.pos = self.pos + 1
        if self.pos == self.capacity:
            self.full = True

        self.pos = self.pos % self.capacity

    def single_task_sample(self, task_idx: int, batch_size: int) -> ReplayBufferSamples:
        assert task_idx < self.num_tasks, "Task index out of bounds."

        sample_idx = self._rng.integers(
            low=0,
            high=max(self.pos if not self.full else self.capacity, batch_size),
            size=(batch_size,),
        )

        batch = (
            self.obs[sample_idx][task_idx],
            self.actions[sample_idx][task_idx],
            self.next_obs[sample_idx][task_idx],
            self.dones[sample_idx][task_idx],
            self.rewards[sample_idx][task_idx],
        )

        return ReplayBufferSamples(*batch)

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        """Sample a batch of size `single_task_batch_size` for each task.

        Args:
            batch_size (int): The total batch size. Must be divisible by number of tasks

        Returns:
            ReplayBufferSamples: A batch of samples of batch shape (batch_size,).
        """
        assert (
            batch_size % self.num_tasks == 0
        ), "Batch size must be divisible by the number of tasks."
        single_task_batch_size = batch_size // self.num_tasks

        sample_idx = self._rng.integers(
            low=0,
            high=max(
                self.pos if not self.full else self.capacity, single_task_batch_size
            ),
            size=(single_task_batch_size,),
        )

        batch = (
            self.obs[sample_idx],
            self.actions[sample_idx],
            self.next_obs[sample_idx],
            self.dones[sample_idx],
            self.rewards[sample_idx],
        )

        mt_batch_size = single_task_batch_size * self.num_tasks
        batch = map(lambda x: x.reshape(mt_batch_size, *x.shape[2:]), batch)

        return ReplayBufferSamples(*batch)


class MultiTaskRolloutBuffer:
    """A buffer to accumulate rollouts for multiple tasks.
    Useful for on-policy MTRL algorithms.

    In Metaworld, all episodes are as long as the time limit (typically 500), thus in this buffer we assume
    fixed-length episodes and leverage that for optimisations."""

    rollouts: list[list[Rollout]]

    def __init__(
        self,
        num_tasks: int,
        rollouts_per_task: int,
        max_episode_steps: int,
    ):
        self.num_tasks = num_tasks
        self._rollouts_per_task = rollouts_per_task
        self._max_episode_steps = max_episode_steps

        self.reset()

    def reset(self):
        """Reset the buffer."""
        self.rollouts = [[] for _ in range(self.num_tasks)]
        self._running_rollouts = [[] for _ in range(self.num_tasks)]

    @property
    def ready(self) -> bool:
        """Returns whether or not a full batch of rollouts for each task has been sampled."""
        return all(len(t) == self._rollouts_per_task for t in self.rollouts)

    def _get_returns(self, rewards: npt.NDArray, discount: float):
        """Discounted cumulative sum.

        See https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html#difference-equation-filtering
        """
        # From garage, modified to work on multi-dimensional arrays, and column reward vectors
        reshape = rewards.shape[-1] == 1
        if reshape:
            rewards = rewards.reshape(rewards.shape[:-1])
        returns = scipy.signal.lfilter(
            [1], [1, float(-discount)], rewards[..., ::-1], axis=-1
        )[..., ::-1]
        return returns if not reshape else returns.reshape(*returns.shape, 1)

    def _compute_advantage(
        self,
        rewards: npt.NDArray,
        baselines: npt.NDArray,
        gamma: float,
        gae_lambda: float,
    ):
        assert (
            rewards.shape == baselines.shape
        ), "Rewards and baselines must have the same shape."
        reshape = rewards.shape[-1] == 1
        if reshape:
            rewards = rewards.reshape(rewards.shape[:-1])
            baselines = baselines.reshape(baselines.shape[:-1])

        # From ProMP's advantage computation, modified to work on multi-dimensional arrays
        baselines = np.append(baselines, np.zeros((*baselines.shape[:-1], 1)), axis=-1)
        deltas = rewards + gamma * baselines[..., 1:] - baselines[..., :-1]
        advantages = self._get_returns(deltas, gamma * gae_lambda)
        return advantages if not reshape else advantages.reshape(*advantages.shape, 1)

    def _normalize_advantages(self, advantages: npt.NDArray) -> npt.NDArray:
        axis = (
            tuple(np.arange(advantages.ndim)[1:])
            if (advantages.ndim > 2 and advantages.shape[-1] == 1)
            else None
        )
        mean = np.mean(advantages, axis=axis, keepdims=axis is not None)
        var = np.var(advantages, axis=axis, keepdims=axis is not None)

        return (advantages - mean) / (var + 1e-8)

    def get_single_task(
        self,
        task_idx: int,
        as_is: bool = False,
        gamma: float | None = None,
        gae_lambda: float | None = None,
        baseline: Callable | None = None,
        fit_baseline: Callable | None = None,
        normalize_advantages: bool = False,
    ) -> Rollout:
        """Compute returns and advantages for the collected rollouts.

        Returns a Rollout tuple for a single task where each array has the batch dimensions (Timestep,).
        The timesteps are multiple rollouts flattened into one time dimension."""
        assert task_idx < self.num_tasks, "Task index out of bounds."

        task_rollouts = Rollout(
            *map(lambda *xs: np.stack(xs), *self.rollouts[task_idx])
        )

        assert task_rollouts.observations.shape[:2] == (
            self._rollouts_per_task,
            self._max_episode_steps,
        ), "Buffer does not have the expected amount of data before sampling."

        if as_is:
            return task_rollouts

        assert (
            gamma is not None and gae_lambda is not None
        ), "Gamma and gae_lambda must be provided if GAE computation is not disabled through the `as_is` flag."

        # 0) Get episode rewards for logging
        task_rollouts = task_rollouts._replace(
            episode_returns=np.sum(task_rollouts.rewards, axis=1)
        )

        # 1) Get returns
        task_rollouts = task_rollouts._replace(
            returns=self._get_returns(task_rollouts.rewards, gamma)
        )

        # 2.1) (Optional) Fit baseline
        if fit_baseline is not None:
            baseline = fit_baseline(task_rollouts)

        # 2.2) Apply baseline
        # NOTE: baseline is responsible for any data conversions / moving to the GPU
        assert (
            baseline is not None
        ), "You must provide a baseline function, or a fit_baseline that returns one."
        baselines = baseline(task_rollouts)

        # 3) Compute advantages
        advantages = self._compute_advantage(
            task_rollouts.rewards, baselines, gamma, gae_lambda
        )
        task_rollouts = task_rollouts._replace(advantages=advantages)

        # 3.1) (Optional) Normalize advantages
        if normalize_advantages:
            assert task_rollouts.advantages is not None
            task_rollouts = task_rollouts._replace(
                advantages=self._normalize_advantages(task_rollouts.advantages)
            )

        # 4) Flatten rollout and time dimensions
        task_rollouts = Rollout(
            *map(
                lambda x: x.reshape(-1, *x.shape[2:]) if x is not None else x,
                task_rollouts,
            )  # type: ignore[reportArgumentType]
        )

        return task_rollouts

    def get(
        self,
        as_is: bool = False,
        gamma: float | None = None,
        gae_lambda: float | None = None,
        baseline: Callable | None = None,
        fit_baseline: Callable | None = None,
        normalize_advantages: bool = False,
    ) -> Rollout:
        """Compute returns and advantages for the collected rollouts.

        Returns a Rollout tuple where each array has the batch dimensions (Task,Timestep,).
        The timesteps are multiple rollouts flattened into one time dimension."""
        rollouts_per_task = [
            Rollout(*map(lambda *xs: np.stack(xs), *t)) for t in self.rollouts
        ]
        all_rollouts = Rollout(*map(lambda *xs: np.stack(xs), *rollouts_per_task))
        assert all_rollouts.observations.shape[:3] == (
            self.num_tasks,
            self._rollouts_per_task,
            self._max_episode_steps,
        ), "Buffer does not have the expected amount of data before sampling."

        if as_is:
            return all_rollouts

        assert (
            gamma is not None and gae_lambda is not None
        ), "Gamma and gae_lambda must be provided if GAE computation is not disabled through the `as_is` flag."

        # 0) Get episode rewards for logging
        all_rollouts = all_rollouts._replace(
            episode_returns=np.sum(all_rollouts.rewards, axis=2)
        )

        # 1) Get returns
        all_rollouts = all_rollouts._replace(
            returns=self._get_returns(all_rollouts.rewards, gamma)
        )

        # 2.1) (Optional) Fit baseline
        if fit_baseline is not None:
            baseline = fit_baseline(all_rollouts)

        # 2.2) Apply baseline
        # NOTE: baseline is responsible for any data conversions / moving to the GPU
        assert (
            baseline is not None
        ), "You must provide a baseline function, or a fit_baseline that returns one."
        baselines = baseline(all_rollouts)

        # 3) Compute advantages
        advantages = self._compute_advantage(
            all_rollouts.rewards, baselines, gamma, gae_lambda
        )
        all_rollouts = all_rollouts._replace(advantages=advantages)

        # 3.1) (Optional) Normalize advantages
        if normalize_advantages:
            assert all_rollouts.advantages is not None
            all_rollouts = all_rollouts._replace(
                advantages=self._normalize_advantages(all_rollouts.advantages)
            )

        # 4) Flatten rollout and time dimensions
        all_rollouts = Rollout(
            *map(lambda x: x.reshape(self.num_tasks, -1, *x.shape[3:]), all_rollouts)  # type: ignore[reportArgumentType]
        )

        return all_rollouts

    def push(
        self,
        obs: npt.NDArray,
        action: npt.NDArray,
        reward: npt.NDArray,
        done: npt.NDArray,
        log_prob: npt.NDArray | None = None,
        mean: npt.NDArray | None = None,
        std: npt.NDArray | None = None,
    ):
        """Add a batch of timesteps to the buffer. Multiple batch dims are supported, but they
        need to multiply to the buffer's meta batch size.

        If an episode finishes here for any of the envs, pop the full rollout into the rollout buffer.
        """
        assert np.prod(reward.shape) == self.num_tasks

        obs = obs.copy()
        action = action.copy()
        assert obs.ndim == action.ndim
        if (
            obs.ndim > 2 and action.ndim > 2
        ):  # Flatten outer batch dims only if they exist
            obs = obs.reshape(-1, *obs.shape[2:])
            action = action.reshape(-1, *action.shape[2:])

        reward = reward.reshape(-1, 1).copy()
        done = done.reshape(-1, 1).copy()
        if log_prob is not None:
            log_prob = log_prob.reshape(-1, 1).copy()
        if mean is not None:
            mean = mean.copy()
            if mean.ndim > 2:
                mean = mean.reshape(-1, *mean.shape[2:])
        if std is not None:
            std = std.copy()
            if std.ndim > 2:
                std = std.reshape(-1, *std.shape[2:])

        for i in range(self.num_tasks):
            timestep = (obs[i], action[i], reward[i], done[i])
            if log_prob is not None:
                timestep += (log_prob[i],)
            if mean is not None:
                timestep += (mean[i],)
            if std is not None:
                timestep += (std[i],)
            self._running_rollouts[i].append(timestep)

            if done[i]:  # pop full rollouts into the rollouts buffer
                rollout = Rollout(
                    *map(lambda *xs: np.stack(xs), *self._running_rollouts[i])
                )
                self.rollouts[i].append(rollout)
                self._running_rollouts[i] = []
