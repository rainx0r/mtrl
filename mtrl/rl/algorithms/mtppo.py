from dataclasses import dataclass
from typing import Self, override

import chex
import distrax
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from flax.core import FrozenDict
from jaxtyping import Array, Float, PRNGKeyArray

from mtrl.config.networks import ContinuousActionPolicyConfig, ValueFunctionConfig
from mtrl.config.rl import AlgorithmConfig
from mtrl.envs import EnvConfig
from mtrl.monitoring.metrics import (
    compute_srank,
    extract_activations,
    get_dormant_neuron_logs,
)
from mtrl.optim.pcgrad import PCGradState
from mtrl.rl.networks import ContinuousActionPolicy, ValueFunction
from mtrl.types import (
    Action,
    Intermediates,
    LogDict,
    LogProb,
    Observation,
    ReplayBufferSamples,
    Rollout,
    Value,
)

from .base import OnPolicyAlgorithm
from .utils import TrainState


@jax.jit
def _sample_action(
    policy: TrainState, observation: Observation, key: PRNGKeyArray
) -> tuple[Float[Array, "... action_dim"], PRNGKeyArray]:
    key, action_key = jax.random.split(key)
    dist: distrax.Distribution
    dist = policy.apply_fn(policy.params, observation)
    action = dist.sample(seed=action_key)
    return action, key


@jax.jit
def _eval_action(
    policy: TrainState, observation: Observation
) -> Float[Array, "... action_dim"]:
    dist: distrax.Distribution
    dist = policy.apply_fn(policy.params, observation)
    return dist.mode()


@jax.jit
def _sample_action_dist_and_value(
    policy: TrainState,
    value_function: TrainState,
    observation: Observation,
    key: PRNGKeyArray,
) -> tuple[
    Float[Array, "... action_dim"],
    Float[Array, "..."],
    Float[Array, "... action_dim"],
    Float[Array, "... action_dim"],
    Float[Array, "..."],
    PRNGKeyArray,
]:
    dist: distrax.Distribution
    key, action_key = jax.random.split(key)
    dist = policy.apply_fn(policy.params, observation)
    action, action_log_prob = dist.sample_and_log_prob(seed=action_key)
    value = value_function.apply_fn(value_function.params, observation)
    return action, action_log_prob, dist.mode(), dist.stddev(), value, key  # pyright: ignore[reportReturnType]


@dataclass(frozen=True)
class MTPPOConfig(AlgorithmConfig):
    policy_config: ContinuousActionPolicyConfig = ContinuousActionPolicyConfig()
    vf_config: ValueFunctionConfig = ValueFunctionConfig()
    clip_eps: float = 0.2
    clip_vf_loss: bool = True
    entropy_coefficient: float = 5e-3
    vf_coefficient: float = 0.001
    normalize_advantages: bool = True


class MTPPO(OnPolicyAlgorithm[MTPPOConfig]):
    policy: TrainState
    value_function: TrainState
    key: PRNGKeyArray
    gamma: float = struct.field(pytree_node=False)
    clip_eps: float = struct.field(pytree_node=False)
    clip_vf_loss: bool = struct.field(pytree_node=False)
    entropy_coefficient: float = struct.field(pytree_node=False)
    vf_coefficient: float = struct.field(pytree_node=False)
    normalize_advantages: bool = struct.field(pytree_node=False)
    split_policy_losses: bool = struct.field(pytree_node=False)
    split_vf_losses: bool = struct.field(pytree_node=False)

    @override
    @staticmethod
    def initialize(
        config: MTPPOConfig, env_config: EnvConfig, seed: int = 1
    ) -> "MTPPO":
        assert isinstance(env_config.action_space, gym.spaces.Box), (
            "Non-box spaces currently not supported."
        )
        assert isinstance(env_config.observation_space, gym.spaces.Box), (
            "Non-box spaces currently not supported."
        )

        master_key = jax.random.PRNGKey(seed)
        algorithm_key, actor_init_key, vf_init_key = jax.random.split(master_key, 3)
        dummy_obs = jnp.array(
            [env_config.observation_space.sample() for _ in range(config.num_tasks)]
        )

        policy_net = ContinuousActionPolicy(
            int(np.prod(env_config.action_space.shape)), config=config.policy_config
        )
        policy = TrainState.create(
            apply_fn=policy_net.apply,
            params=policy_net.init(actor_init_key, dummy_obs),
            tx=config.policy_config.network_config.optimizer.spawn(),
        )

        print("Policy Arch:", jax.tree_util.tree_map(jnp.shape, policy.params))
        print("Policy Params:", sum(x.size for x in jax.tree.leaves(policy.params)))

        vf_net = ValueFunction(config.vf_config)
        value_function = TrainState.create(
            apply_fn=vf_net.apply,
            params=vf_net.init(vf_init_key, dummy_obs),
            tx=config.vf_config.network_config.optimizer.spawn(),
        )

        print("Vf Arch:", jax.tree_util.tree_map(jnp.shape, value_function.params))
        print("Vf Params:", sum(x.size for x in jax.tree.leaves(value_function.params)))

        return MTPPO(
            num_tasks=config.num_tasks,
            policy=policy,
            value_function=value_function,
            key=algorithm_key,
            gamma=config.gamma,
            clip_eps=config.clip_eps,
            clip_vf_loss=config.clip_vf_loss,
            entropy_coefficient=config.entropy_coefficient,
            vf_coefficient=config.vf_coefficient,
            normalize_advantages=config.normalize_advantages,
            split_policy_losses=config.policy_config.network_config.optimizer.requires_split_task_losses,
            split_vf_losses=config.vf_config.network_config.optimizer.requires_split_task_losses,
        )

    @override
    def get_num_params(self) -> dict[str, int]:
        return {
            "policy_num_params": sum(
                x.size for x in jax.tree.leaves(self.policy.params)
            ),
            "vf_num_params": sum(
                x.size for x in jax.tree.leaves(self.value_function.params)
            ),
        }

    @override
    def sample_action(self, observation: Observation) -> tuple[Self, Action]:
        action, key = _sample_action(self.policy, observation, self.key)
        return self.replace(key=key), jax.device_get(action)

    @override
    def sample_action_dist_and_value(
        self, observation: Observation
    ) -> tuple[Self, Action, LogProb, Action, Action, Value]:
        action, log_prob, mean, std, value, key = _sample_action_dist_and_value(
            self.policy, self.value_function, observation, self.key
        )
        return (
            self.replace(key=key),
            *jax.device_get((action, log_prob, mean, std, value)),
        )

    @override
    def eval_action(self, observations: Observation) -> Action:
        return jax.device_get(_eval_action(self.policy, observations))

    def update_policy(self, data: Rollout) -> tuple[Self, LogDict]:
        key, policy_loss_key = jax.random.split(self.key, 2)

        def policy_loss(
            params: FrozenDict, _data: Rollout
        ) -> tuple[Float[Array, ""], LogDict]:
            action_dist: distrax.Distribution
            new_log_probs: Float[Array, " batch_size"]

            action_dist = self.policy.apply_fn(params, _data.observations)
            _, new_log_probs = action_dist.sample_and_log_prob(seed=policy_loss_key)  # pyright: ignore[reportAssignmentType]
            log_ratio = new_log_probs.reshape(-1, 1) - _data.log_probs
            ratio = jnp.exp(log_ratio)

            # For logs
            approx_kl = jax.lax.stop_gradient(((ratio - 1) - log_ratio).mean())
            clip_fracs = jax.lax.stop_gradient(
                jnp.array(
                    jnp.abs(ratio - 1.0) > self.clip_eps,
                    dtype=jnp.float32,
                ).mean()
            )

            if self.normalize_advantages:
                advantages = (_data.advantages - jnp.mean(_data.advantages)) / (  # pyright: ignore[reportArgumentType]
                    jnp.std(_data.advantages) + 1e-8  # pyright: ignore[reportArgumentType]
                )
            else:
                advantages = _data.advantages

            pg_loss1 = -advantages * ratio  # pyright: ignore[reportOptionalOperand]
            pg_loss2 = -advantages * jnp.clip(  # pyright: ignore[reportOptionalOperand]
                ratio, 1 - self.clip_eps, 1 + self.clip_eps
            )

            pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()
            entropy_loss = action_dist.entropy().mean()

            return pg_loss - self.entropy_coefficient * entropy_loss, {
                "losses/entropy_loss": entropy_loss,
                "losses/policy_loss": pg_loss,
                "losses/approx_kl": approx_kl,
                "losses/clip_fracs": clip_fracs,
            }

        if self.split_policy_losses:
            (_, logs), policy_grads = jax.vmap(
                jax.value_and_grad(policy_loss, has_aux=True),
                in_axes=(None, 0),
                out_axes=0,
            )(self.policy.params, data)
            policy = self.policy.apply_gradients(grads=policy_grads)
        else:
            (_, logs), policy_grads = jax.value_and_grad(policy_loss, has_aux=True)(
                self.policy.params, data
            )
            policy = self.policy.apply_gradients(grads=policy_grads)

        return self.replace(policy=policy, key=key), logs

    def update_value_function(self, data: Rollout) -> tuple[Self, LogDict]:
        def value_function_loss(params: FrozenDict) -> tuple[Float[Array, ""], LogDict]:
            new_values: Float[Array, "batch_size 1"]
            new_values = self.value_function.apply_fn(params, data.observations)
            chex.assert_equal_shape(new_values, data.returns)

            if self.clip_vf_loss:
                vf_loss_unclipped = (new_values - data.returns) ** 2
                v_clipped = data.values + jnp.clip(
                    new_values - data.values, -self.clip_eps, self.clip_eps
                )
                vf_loss_clipped = (v_clipped - data.returns) ** 2  # pyright: ignore[reportOperatorIssue]
                vf_loss = 0.5 * jnp.maximum(vf_loss_unclipped, vf_loss_clipped).mean()
            else:
                vf_loss = 0.5 * ((new_values - data.returns) ** 2).mean()

            return self.vf_coefficient * vf_loss, {
                "losses/value_function": vf_loss,
                "losses/values": new_values.mean(),
            }

        if self.split_vf_losses:
            (_, vf_grads), logs = jax.vmap(
                jax.value_and_grad(value_function_loss, has_aux=True),
                in_axes=(None, 0),
                out_axes=0,
            )(self.value_function.params, data)
            value_function = self.value_function.apply_gradients(grads=vf_grads)
        else:
            (_, logs), vf_grads = jax.value_and_grad(value_function_loss, has_aux=True)(
                self.value_function.params, data
            )
            value_function = self.value_function.apply_gradients(grads=vf_grads)

        return self.replace(value_function=value_function), logs

    @jax.jit
    def _update_inner(self, data: Rollout) -> tuple[Self, LogDict]:
        self, policy_logs = self.update_policy(data)
        self, vf_logs = self.update_value_function(data)

        # HACK: PCGrad logs
        assert isinstance(self.value_function.opt_state, tuple)
        assert isinstance(self.policy.opt_state, tuple)
        vf_optim_logs = (
            {
                f"metrics/vf_{key}": value
                for key, value in self.value_function.opt_state[0]._asdict().items()
            }
            if isinstance(self.value_function.opt_state[0], PCGradState)
            else {}
        )
        policy_optim_logs = (
            {
                f"metrics/policy_{key}": value
                for key, value in self.policy.opt_state[0]._asdict().items()
            }
            if isinstance(self.policy.opt_state[0], PCGradState)
            else {}
        )

        return self, policy_logs | vf_logs | vf_optim_logs | policy_optim_logs

    @override
    def update(self, data: ReplayBufferSamples | Rollout) -> tuple[Self, LogDict]:
        assert isinstance(data, Rollout), (
            "MTPPO does not support replay buffer samples."
        )
        assert data.log_probs is not None, (
            "Rollout policy log probs must have been recorded."
        )
        assert data.advantages is not None, "GAE must be enabled for MTPPO."
        assert data.returns is not None, "Returns must be computed for MTPPO."
        return self._update_inner(data)

    @jax.jit
    def _get_intermediates(self, data: Rollout) -> tuple[Intermediates, Intermediates]:
        batch_size = data.observations.shape[0]

        _, policy_state = self.policy.apply_fn(
            self.policy.params, data.observations, capture_intermediates=True
        )

        _, vf_state = self.value_function.apply_fn(
            self.value_function.params,
            data.observations,
            capture_intermediates=True,
        )

        actor_intermediates = jax.tree.map(
            lambda x: x.reshape(batch_size, -1), policy_state["intermediates"]
        )
        critic_intermediates = jax.tree.map(
            lambda x: x.reshape(batch_size, -1), vf_state["intermediates"]
        )

        return actor_intermediates, critic_intermediates

    @override
    def get_metrics(self, data: Rollout) -> tuple[Self, LogDict]:
        policy_intermediates, vf_intermediates = self._get_intermediates(data)

        policy_acts = extract_activations(policy_intermediates)
        vf_acts = extract_activations(vf_intermediates)

        metrics: LogDict
        metrics = {}
        metrics.update(
            {
                f"metrics/dormant_neurons_policy_{log_name}": log_value
                for log_name, log_value in get_dormant_neuron_logs(policy_acts).items()
            }
        )
        for key, value in policy_acts.items():
            metrics[f"metrics/srank_policy_{key}"] = compute_srank(value)

        metrics.update(
            {
                f"metrics/dead_neurons_vf_{log_name}": log_value
                for log_name, log_value in get_dormant_neuron_logs(vf_acts).items()
            }
        )
        for key, value in vf_acts.items():
            metrics[f"metrics/srank_vf_{key}"] = compute_srank(value)

        return self, metrics
