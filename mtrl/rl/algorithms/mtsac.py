"""Inspired by https://github.com/kevinzakka/robopianist-rl/blob/main/sac.py"""

import dataclasses
from functools import cached_property
from typing import Self, override

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from flax.core import FrozenDict
from flax.training.train_state import TrainState
from jaxtyping import Array, Float, PRNGKeyArray
import optax

from mtrl.config.networks import ContinuousActionPolicyConfig, QValueFunctionConfig
from mtrl.config.optim import OptimizerConfig
from mtrl.config.rl import AlgorithmConfig
from mtrl.types import (
    Action,
    LogDict,
    Observation,
    ReplayBufferSamples,
    Rollout,
)

from .base import OffPolicyAlgorithm


class MultiTaskTemperature(nn.Module):
    num_tasks: int
    initial_temperature: float = 1.0

    def setup(self):
        self.log_alpha = self.param(
            "alpha",
            init_fn=lambda _: jnp.full(
                (self.num_tasks,), jnp.log(self.initial_temperature)
            ),
        )

    def __call__(
        self, task_ids: Float[Array, "... num_tasks"]
    ) -> Float[Array, "... 1"]:
        return jnp.exp(task_ids @ self.log_alpha.reshape(-1, 1))


class CriticTrainState(TrainState):
    target_params: FrozenDict | None = None


@jax.jit
def _sample_action(
    actor: TrainState, observation: Observation, key: PRNGKeyArray
) -> tuple[Float[Array, "... action_dim"], PRNGKeyArray]:
    key, action_key = jax.random.split(key)
    dist = actor.apply_fn(actor.params, observation)
    action = dist.sample(seed=action_key)
    return action, key


@jax.jit
def _eval_action(
    actor: TrainState, observation: Observation
) -> Float[Array, "... action_dim"]:
    return actor.apply_fn(actor.params, observation).mode()


def extract_task_weights(
    alpha_params: FrozenDict, task_ids: Float[np.ndarray, "... num_tasks"]
) -> Float[Array, "... 1"]:
    log_alpha: jax.Array
    task_weights: jax.Array

    # TODO: check that this access works
    log_alpha = alpha_params["log_alpha"]  # pyright: ignore [reportAssignmentType]
    task_weights = jax.nn.softmax(-log_alpha)  # NOTE 6
    task_weights = task_ids @ task_weights.reshape(-1, 1)  # pyright: ignore [reportAssignmentType]
    task_weights *= log_alpha.shape[0]  # NOTE 6
    return task_weights


@dataclasses.dataclass(frozen=True)
class MTSACConfig(AlgorithmConfig):
    actor_config: ContinuousActionPolicyConfig = ContinuousActionPolicyConfig()
    critic_config: QValueFunctionConfig = QValueFunctionConfig()
    temperature_optimizer_config: OptimizerConfig = OptimizerConfig()
    num_critics: int = 2
    tau: float = 0.005
    use_task_weights: bool = False

    @cached_property
    def target_entropy(self) -> float:
        assert self.action_space is not None and self.action_space.shape is not None
        return -np.prod(self.action_space.shape).item()


class MTSAC(OffPolicyAlgorithm[MTSACConfig]):
    actor: TrainState
    critic: CriticTrainState
    alpha: TrainState
    key: PRNGKeyArray
    gamma: float = struct.field(pytree_node=False)
    tau: float = struct.field(pytree_node=False)
    target_entropy: float = struct.field(pytree_node=False)
    use_task_weights: bool = struct.field(pytree_node=False)

    @override
    @staticmethod
    def initialize(config: MTSACConfig) -> "MTSAC": ...

    @override
    def sample_action(self, observation: Observation) -> tuple[Self, Action]:
        action, key = _sample_action(self.actor, observation, self.key)
        return self.replace(key=key), action

    @override
    def eval_action(self, observation: Observation) -> Action:
        return _eval_action(self.actor, observation)

    @jax.jit
    def _update_inner(self, data: ReplayBufferSamples) -> tuple[Self, LogDict]:
        task_ids = data.observations[..., -self.num_tasks :]

        # --- Critic loss ---
        key, actor_loss_key, critic_loss_key = jax.random.split(self.key, 3)

        def update_critic(
            _critic: CriticTrainState,
            alpha_val: Float[Array, "batch 1"],
            task_weights: Float[Array, "batch 1"] | None = None,
        ) -> tuple[CriticTrainState, LogDict]:
            # Sample a'
            next_actions, next_action_log_probs = self.actor.apply_fn(
                self.actor.params, data.next_observations
            ).sample_and_log_prob(seed=critic_loss_key)
            # Compute target Q values
            q_values = self.critic.apply_fn(
                self.critic.target_params, data.next_observations, next_actions
            )

            def critic_loss(
                params: FrozenDict,
            ) -> tuple[Float[Array, ""], Float[Array, ""]]:
                # next_action_log_probs is (B,) shaped because of the sum(axis=1), while Q values are (B, 1)
                min_qf_next_target = jnp.min(
                    q_values, axis=0
                ) - alpha_val * next_action_log_probs.reshape(-1, 1)
                next_q_value = jax.lax.stop_gradient(
                    data.rewards + (1 - data.dones) * self.gamma * min_qf_next_target
                )

                q_pred = critic.apply_fn(params, data.observations, data.actions)
                if self.use_task_weights:
                    assert task_weights is not None
                    loss = (
                        0.5
                        * (task_weights * (q_pred - next_q_value) ** 2)
                        .mean(axis=1)
                        .sum()
                    )
                else:
                    loss = 0.5 * ((q_pred - next_q_value) ** 2).mean(axis=1).sum()
                return loss, q_pred.mean()

            (critic_loss_value, qf_values), critic_grads = jax.value_and_grad(
                critic_loss, has_aux=True
            )(_critic.params)
            _critic = _critic.apply_gradients(grads=critic_grads)
            return _critic, {
                "losses/qf_values": qf_values,
                "losses/qf_loss": critic_loss_value,
            }

        # --- Alpha loss ---

        def update_alpha(
            _alpha: TrainState, log_probs: Float[Array, " batch"]
        ) -> tuple[
            TrainState, Float[Array, "batch 1"], Float[Array, "batch 1"] | None, LogDict
        ]:
            def alpha_loss(params: FrozenDict) -> Float[Array, ""]:
                log_alpha: jax.Array
                log_alpha = task_ids @ params["log_alpha"].reshape(-1, 1)  # pyright: ignore [reportAttributeAccessIssue]
                return (
                    -log_alpha * (log_probs.reshape(-1, 1) + self.target_entropy)
                ).mean()

            alpha_loss_value, alpha_grads = jax.value_and_grad(alpha_loss)(
                _alpha.params, log_probs
            )
            _alpha = _alpha.apply_gradients(grads=alpha_grads)
            alpha_vals = _alpha.apply_fn(_alpha.params, task_ids)
            if self.use_task_weights:
                task_weights = extract_task_weights(_alpha.params, task_ids)
            else:
                task_weights = None

            return (
                _alpha,
                alpha_vals,
                task_weights,
                {
                    "losses/alpha_loss": alpha_loss_value,
                    "alpha": jnp.exp(_alpha.params).sum(),  # pyright: ignore [reportReturnType,reportArgumentType]
                },
            )

        # --- Actor loss --- & calls for the other losses
        def actor_loss(params: FrozenDict):
            action_samples, log_probs = self.actor.apply_fn(
                params, data.observations
            ).sample_and_log_prob(seed=actor_loss_key)

            # HACK: Putting the other losses / grad updates inside this function for performance,
            # so we can reuse the action_samples / log_probs while also doing alpha loss first
            _alpha, _alpha_val, task_weights, alpha_logs = update_alpha(
                self.alpha, log_probs
            )
            _alpha_val = jax.lax.stop_gradient(_alpha_val)
            if task_weights is not None:
                task_weights = jax.lax.stop_gradient(task_weights)
            _critic, critic_logs = update_critic(critic, _alpha_val, task_weights)
            logs = {**alpha_logs, **critic_logs}

            q_values = _critic.apply_fn(
                _critic.params, data.observations, action_samples
            )
            min_qf_values = jnp.min(q_values, axis=0)
            if task_weights is not None:
                loss = (
                    task_weights
                    * (_alpha_val * log_probs.reshape(-1, 1) - min_qf_values)
                ).mean()
            else:
                loss = (_alpha_val * log_probs.reshape(-1, 1) - min_qf_values).mean()
            return loss, (_alpha, _critic, logs)

        (actor_loss_value, (alpha, critic, logs)), actor_grads = jax.value_and_grad(
            actor_loss, has_aux=True
        )(self.actor.params)
        actor = self.actor.apply_gradients(grads=actor_grads)

        self = self.replace(
            key=key,
            actor=actor,
            critic=critic,
            alpha=alpha,
        )

        qf_state = self.critic.replace(
            target_params=optax.incremental_update(
                self.critic.params,
                self.critic.target_params,  # pyright: ignore [reportArgumentType]
                self.tau,
            )
        )
        self = self.replace(critic=qf_state)

        return (self, {**logs, "losses/actor_loss": actor_loss_value})

    @override
    def update(self, data: ReplayBufferSamples | Rollout) -> tuple[Self, LogDict]:
        assert isinstance(data, ReplayBufferSamples), "MTSAC does not support rollouts"
        return self._update_inner(data)
