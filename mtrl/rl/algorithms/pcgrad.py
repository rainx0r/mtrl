"""Inspired by https:self.//github.com/kevinzakka/robopianist-rl/blob/main/sac.py"""
import time
import numpy as np

# from jax.experimental.static_array import static_args
import dataclasses
from functools import partial
from typing import Self, override

import distrax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.flatten_util as flatten_util
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.core import FrozenDict
from flax.training.train_state import TrainState
from jaxtyping import Array, Float, PRNGKeyArray

from mtrl.config.networks import ContinuousActionPolicyConfig, QValueFunctionConfig
from mtrl.config.optim import OptimizerConfig
from mtrl.config.rl import AlgorithmConfig
from mtrl.envs import EnvConfig
from mtrl.monitoring.metrics import (
    compute_srank,
    extract_activations,
    get_dormant_neuron_logs,
)
from mtrl.rl.networks import ContinuousActionPolicy, Ensemble, QValueFunction
from mtrl.types import (
    Action,
    AuxPolicyOutputs,
    Intermediates,
    LayerActivationsDict,
    LogDict,
    Observation,
    ReplayBufferSamples,
)

from .base import OffPolicyAlgorithm


class MultiTaskTemperature(nn.Module):
    num_tasks: int
    initial_temperature: float = 1.0

    def setup(self):
        self.log_alpha = self.param(
            "log_alpha",
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

    log_alpha = alpha_params["params"]["log_alpha"]  # pyright: ignore [reportAssignmentType]
    task_weights = jax.nn.softmax(-log_alpha)
    task_weights = task_ids @ task_weights.reshape(-1, 1)  # pyright: ignore [reportAssignmentType]
    task_weights *= log_alpha.shape[0]
    return task_weights


@dataclasses.dataclass(frozen=True)
class PCGradConfig(AlgorithmConfig):
    actor_config: ContinuousActionPolicyConfig = ContinuousActionPolicyConfig()
    critic_config: QValueFunctionConfig = QValueFunctionConfig()
    temperature_optimizer_config: OptimizerConfig = OptimizerConfig(max_grad_norm=None)
    initial_temperature: float = 1.0
    num_critics: int = 2
    tau: float = 0.005
    use_task_weights: bool = False


class PCGrad(OffPolicyAlgorithm[PCGradConfig]):
    actor: TrainState
    critic: CriticTrainState
    alpha: TrainState
    key: PRNGKeyArray
    gamma: float = struct.field(pytree_node=False)
    tau: float = struct.field(pytree_node=False)
    target_entropy: float = struct.field(pytree_node=False)
    use_task_weights: bool = struct.field(pytree_node=False)
    num_critics: int = struct.field(pytree_node=False)

    @override
    @staticmethod
    def initialize(
        config: PCGradConfig, env_config: EnvConfig, seed: int = 1
    ) -> "PCGrad":
        assert isinstance(
            env_config.action_space, gym.spaces.Box
        ), "Non-box spaces currently not supported."
        assert isinstance(
            env_config.observation_space, gym.spaces.Box
        ), "Non-box spaces currently not supported."

        master_key = jax.random.PRNGKey(seed)
        algorithm_key, actor_init_key, critic_init_key, alpha_init_key = (
            jax.random.split(master_key, 4)
        )

        actor_net = ContinuousActionPolicy(
            int(np.prod(env_config.action_space.shape)), config=config.actor_config
        )
        dummy_obs = jnp.array(
            [env_config.observation_space.sample() for _ in range(config.num_tasks)]
        )
        actor = TrainState.create(
            apply_fn=actor_net.apply,
            params=actor_net.init(actor_init_key, dummy_obs),
            tx=config.actor_config.network_config.optimizer.spawn(),
        )

        print("Actor Arch:", jax.tree_util.tree_map(jnp.shape, actor.params))
        print("Actor Params:", sum(x.size for x in jax.tree.leaves(actor.params)))

        critic_cls = partial(QValueFunction, config=config.critic_config)
        critic_net = Ensemble(critic_cls, num=config.num_critics)
        dummy_action = jnp.array(
            [env_config.action_space.sample() for _ in range(config.num_tasks)]
        )
        critic_init_params = critic_net.init(critic_init_key, dummy_obs, dummy_action)
        critic = CriticTrainState.create(
            apply_fn=critic_net.apply,
            params=critic_init_params,
            target_params=critic_init_params,
            tx=config.critic_config.network_config.optimizer.spawn(),
        )

        print("Critic Arch:", jax.tree_util.tree_map(jnp.shape, critic.params))
        print("Critic Params:", sum(x.size for x in jax.tree.leaves(critic.params)))

        alpha_net = MultiTaskTemperature(config.num_tasks, config.initial_temperature)
        dummy_task_ids = jnp.array(
            [np.ones((config.num_tasks,)) for _ in range(config.num_tasks)]
        )
        alpha = TrainState.create(
            apply_fn=alpha_net.apply,
            params=alpha_net.init(alpha_init_key, dummy_task_ids),
            tx=config.temperature_optimizer_config.spawn(),
        )

        target_entropy = -np.prod(env_config.action_space.shape).item()

        return PCGrad(
            num_tasks=config.num_tasks,
            actor=actor,
            critic=critic,
            alpha=alpha,
            key=algorithm_key,
            gamma=config.gamma,
            tau=config.tau,
            target_entropy=target_entropy,
            use_task_weights=config.use_task_weights,
            num_critics=config.num_critics,
        )

    @override
    def get_num_params(self) -> dict[str, int]:
        return {
            "actor_num_params": sum(x.size for x in jax.tree.leaves(self.actor.params)),
            "critic_num_params": sum(
                x.size for x in jax.tree.leaves(self.critic.params)
            ),
        }

    @override
    def sample_action(self, observation: Observation) -> tuple[Self, Action]:
        action, key = _sample_action(self.actor, observation, self.key)
        return self.replace(key=key), jax.device_get(action)

    @override
    def eval_action(self, observation: Observation) -> tuple[Action, AuxPolicyOutputs]:
        return jax.device_get(_eval_action(self.actor, observation)), {}


    @staticmethod
    @partial(jax.jit, static_argnums=(4,))  # Specify the index of num_tasks argument
    def compute_pcgrad(
        critic: CriticTrainState,
        data: ReplayBufferSamples, 
        task_ids: Float[Array, "batch num_tasks"],
        # alpha_val: Float[Array, "batch 1"],
        next_q_value: Float[Array, "batch 1"],
        num_tasks: int,
        ) -> tuple[Array, LogDict]:

        def get_task_grad(task_idx: int) -> Array:
            task_mask = task_ids[:, task_idx] == 1
            
            def task_loss(params: FrozenDict) -> Float[Array, ""]:
                q_pred = critic.apply_fn(params, data.observations, data.actions)
                loss = 0.5 * ((q_pred - next_q_value) ** 2 * task_mask[:, None]).mean()
                return loss
                
            grad = jax.grad(task_loss)(critic.params)
            flat_grad, _ = jax.flatten_util.ravel_pytree(grad)
            return flat_grad


        # Get gradients for each task using vmap
        task_grads = jax.vmap(get_task_grad)(jnp.arange(num_tasks))

        # PCGrad projection
        def project_grads(xy) -> jax.Array:
            x, y = xy
            dot = jnp.dot(x, y)
            grad_conflicts = dot < 0
            return jnp.where(grad_conflicts, x - (dot * y) / (jnp.sum(y**2) + 1e-8), x), grad_conflicts.sum()  # pyright: ignore[reportReturnType]

        # Project gradients
        def pcgrad_vmap(num_tasks, task_grads): # Remove num_tasks?
            @partial(jax.vmap, in_axes=(0, 0, None), out_axes=0)
            def p_grads(
                task_gradient: Float[Array, " gradient_dim"],
                i: Float[Array, ""],
                all_grads: Float[Array, " num_tasks gradient_dim"],
            ) -> Float[Array, " gradient_dim"]:

                g_i_pc = task_gradient
                total = 0
                for j in range(all_grads.shape[0]):
                   g_i_pc, confs = jax.lax.cond(i != j, (g_i_pc, all_grads[j]), project_grads, g_i_pc, lambda x: (x,0))
                   total += confs
                return g_i_pc, total

            # (num_tasks, gradient_dim)
            res, total = p_grads(task_grads, jnp.arange(num_tasks), task_grads)
            total = total.sum() / 2
            return res, total
        
        final_grads, total_grad_conflicts = pcgrad_vmap(num_tasks, task_grads)

        # Average projected gradients
        avg_grad = jnp.mean(final_grads, axis=0)
        
        # Unravel back to pytree
        _, unravel_fn = jax.flatten_util.ravel_pytree(critic.params)
        final_grad = unravel_fn(avg_grad)

        # Metrics 
        def vmap_cos_sim(grads, num_tasks):
            def calc_cos_sim(selected_grad, grads):

                new_cos_sim  = jnp.array([ # Removed the jnp.mean
                                jnp.sum(selected_grad * grads, axis=1) / (
                                        jnp.linalg.norm(selected_grad) * jnp.linalg.norm(grads, axis=1) + 1e-12
                                        )
                                ])

                return new_cos_sim

            cos_sim_mat = jax.vmap(calc_cos_sim, in_axes=(0,None), out_axes=-1)(grads, grads)
            mask = jnp.triu(jnp.ones((num_tasks, num_tasks)), k=1) # Get upper triangle
            num_unique = jnp.sum(mask)

            masked_cos_sim = mask * cos_sim_mat
            avg_cos_sim = jnp.sum(masked_cos_sim.flatten()) / num_unique # n in upper triangle
            return avg_cos_sim
        
        avg_cos_sim = vmap_cos_sim(task_grads, num_tasks)
        new_cos_sim = vmap_cos_sim(final_grads, num_tasks)

        metrics = {
            "metrics/n_grad_conflicts": total_grad_conflicts,
            "metrics/avg_critic_grad_magnitude": jnp.mean(jnp.linalg.norm(final_grads, axis=1)),
            "metrics/avg_critic_grad_magnitude_before_grad_surgery": jnp.mean(jnp.linalg.norm(task_grads, axis=1)),
            "metrics/avg_cosine_similarity":  avg_cos_sim,
        "metrics/avg_cosine_similarity_diff": avg_cos_sim - new_cos_sim
        }       
        return final_grad, metrics

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

            q_values = self.critic.apply_fn(
                self.critic.target_params, data.next_observations, next_actions
            )

            min_qf_next_target = jnp.min(
                q_values, axis=0
            ) - alpha_val * next_action_log_probs.reshape(-1, 1)
            next_q_value = jax.lax.stop_gradient(
                data.rewards + (1 - data.dones) * self.gamma * min_qf_next_target
            )

            # Compute PCGrad update
            critic_grads, metrics = PCGrad.compute_pcgrad(
                _critic, 
                data,
                jnp.array(task_ids),
                # alpha_val,
                next_q_value,
                self.num_tasks
            )

            _critic = _critic.apply_gradients(grads=critic_grads)
            
            # For metrics - TODO: Improve performance
            q_pred = _critic.apply_fn(_critic.params, data.observations, data.actions)
            qf_loss = 0.5 * ((q_pred - next_q_value) ** 2).mean()

            metrics.update({
                "losses/qf_values": q_pred, # Sort these out later
                "losses/qf_loss": qf_loss,
                # "metrics/critic_grad_magnitude": jnp.linalg.norm(flat_grads),
            })
    
            return _critic , metrics

        # --- Alpha loss ---

        def update_alpha(
            _alpha: TrainState, log_probs: Float[Array, " batch"]
        ) -> tuple[
            TrainState, Float[Array, "batch 1"], Float[Array, "batch 1"] | None, LogDict
        ]:
            def alpha_loss(params: FrozenDict) -> Float[Array, ""]:
                log_alpha: jax.Array
                log_alpha = task_ids @ params["params"]["log_alpha"].reshape(-1, 1)  # pyright: ignore [reportAttributeAccessIssue]
                return (
                    -log_alpha * (log_probs.reshape(-1, 1) + self.target_entropy)
                ).mean()

            alpha_loss_value, alpha_grads = jax.value_and_grad(alpha_loss)(
                _alpha.params
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
                    "alpha": jnp.exp(_alpha.params["params"]["log_alpha"]).sum(),  # pyright: ignore [reportReturnType,reportArgumentType]
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
            _critic, critic_logs = update_critic(self.critic, _alpha_val, task_weights)
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

        flat_grads, _ = flatten_util.ravel_pytree(actor_grads)
        logs["metrics/actor_grad_magnitude"] = jnp.linalg.norm(flat_grads)

        flat_params_act, _ = flatten_util.ravel_pytree(self.actor.params)
        logs["metrics/actor_params_norm"] = jnp.linalg.norm(flat_params_act)

        flat_params_crit, _ = flatten_util.ravel_pytree(self.critic.params)
        logs["metrics/critic_params_norm"] = jnp.linalg.norm(flat_params_crit)

        critic: CriticTrainState
        critic = critic.replace(
            target_params=optax.incremental_update(
                critic.params,
                critic.target_params,  # pyright: ignore [reportArgumentType]
                self.tau,
            )
        )

        self = self.replace(
            key=key,
            actor=actor,
            critic=critic,
            alpha=alpha,
        )

        return (self, {**logs, "losses/actor_loss": actor_loss_value})

    @override
    def update(self, data: ReplayBufferSamples) -> tuple[Self, LogDict]:
        return self._update_inner(data)

    def _split_critic_activations(
        self, critic_acts: LayerActivationsDict
    ) -> tuple[LayerActivationsDict, ...]:
        return tuple(
            {key: value[i] for key, value in critic_acts.items()}
            for i in range(self.num_critics)
        )

    @jax.jit
    def _get_intermediates(
        self, data: ReplayBufferSamples
    ) -> tuple[Self, Intermediates, Intermediates]:
        key, critic_activations_key = jax.random.split(self.key, 2)

        actions_dist: distrax.Distribution
        batch_size = data.observations.shape[0]
        actions_dist, actor_state = self.actor.apply_fn(
            self.actor.params, data.observations, mutable="intermediates"
        )
        actions = actions_dist.sample(seed=critic_activations_key)

        _, critic_state = self.critic.apply_fn(
            self.critic.params, data.observations, actions, mutable="intermediates"
        )

        actor_intermediates = jax.tree.map(
            lambda x: x.reshape(batch_size, -1), actor_state["intermediates"]
        )
        critic_intermediates = jax.tree.map(
            lambda x: x.reshape(self.num_critics, batch_size, -1),
            critic_state["intermediates"]["VmapQValueFunction_0"],
        )

        self = self.replace(key=key)

        # HACK: Explicitly using the generated name of the Vmap Critic module here.
        return (
            self,
            actor_intermediates,
            critic_intermediates,
        )

    @override
    def get_metrics(self, data: ReplayBufferSamples) -> tuple[Self, LogDict]:
        self, actor_intermediates, critic_intermediates = self._get_intermediates(data)

        actor_acts = extract_activations(actor_intermediates)
        critic_acts = extract_activations(critic_intermediates)
        critic_acts = self._split_critic_activations(critic_acts)

        # TODO: None of the dormant neuron logs / srank compute are jitted at the top level
        metrics: LogDict
        metrics = {}
        metrics.update(
            {
                f"metrics/dormant_neurons_actor_{log_name}": log_value
                for log_name, log_value in get_dormant_neuron_logs(actor_acts).items()
            }
        )
        for key, value in actor_acts.items():
            metrics[f"metrics/srank_actor_{key}"] = compute_srank(value)

        for i, acts in enumerate(critic_acts):
            metrics.update(
                {
                    f"metrics/dormant_neurons_critic_{i}_{log_name}": log_value
                    for log_name, log_value in get_dormant_neuron_logs(acts).items()
                }
            )
            for key, value in acts.items():
                metrics[f"metrics/srank_critic_{i}_{key}"] = compute_srank(value)

        return self, metrics

