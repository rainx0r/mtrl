"""Inspired by https://github.com/kevinzakka/robopianist-rl/blob/main/sac.py"""

import dataclasses
from functools import partial
from typing import Self, override, Dict
import copy

import gymnasium as gym
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from flax.core import FrozenDict
from flax.training.train_state import TrainState
from jaxtyping import Array, Float, PRNGKeyArray
import optax
import jax.flatten_util as fu


from mtrl.config.networks import ContinuousActionPolicyConfig, QValueFunctionConfig
from mtrl.config.optim import OptimizerConfig
from mtrl.config.rl import AlgorithmConfig
from mtrl.envs import EnvConfig
from mtrl.rl.networks import ContinuousActionPolicy, Ensemble, QValueFunction
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
    task_weights = jax.nn.softmax(-log_alpha)  # NOTE 6
    task_weights = task_ids @ task_weights.reshape(-1, 1)  # pyright: ignore [reportAssignmentType]
    task_weights *= log_alpha.shape[0]  # NOTE 6
    return task_weights

def get_critic_params(params):
    critic1_params = {
    'MultiHeadNetwork_0': {
        'VmapDense_0': {
            'bias': params['params']['VmapQValueFunction_0']['MultiHeadNetwork_0']['VmapDense_0']['bias'][0],
            'kernel': params['params']['VmapQValueFunction_0']['MultiHeadNetwork_0']['VmapDense_0']['kernel'][0]
        },
        'layer_0': {
            'bias': params['params']['VmapQValueFunction_0']['MultiHeadNetwork_0']['layer_0']['bias'][0],
            'kernel': params['params']['VmapQValueFunction_0']['MultiHeadNetwork_0']['layer_0']['kernel'][0]
        },
        'layer_1': {
            'bias': params['params']['VmapQValueFunction_0']['MultiHeadNetwork_0']['layer_1']['bias'][0],
            'kernel': params['params']['VmapQValueFunction_0']['MultiHeadNetwork_0']['layer_1']['kernel'][0]
            }
        }
    }

    critic2_params = {
    'MultiHeadNetwork_0': {
        'VmapDense_0': {
            'bias': params['params']['VmapQValueFunction_0']['MultiHeadNetwork_0']['VmapDense_0']['bias'][1],
            'kernel': params['params']['VmapQValueFunction_0']['MultiHeadNetwork_0']['VmapDense_0']['kernel'][1]
        },
        'layer_0': {
            'bias': params['params']['VmapQValueFunction_0']['MultiHeadNetwork_0']['layer_0']['bias'][1],
            'kernel': params['params']['VmapQValueFunction_0']['MultiHeadNetwork_0']['layer_0']['kernel'][1]
        },
        'layer_1': {
            'bias': params['params']['VmapQValueFunction_0']['MultiHeadNetwork_0']['layer_1']['bias'][1],
            'kernel': params['params']['VmapQValueFunction_0']['MultiHeadNetwork_0']['layer_1']['kernel'][1]
        }
        }
    }
    return critic1_params, critic2_params

def get_dead_neuron_count(intermediate_act, dead_neuron_threshold=0.1):
    all_layers_score = {}
    dead_neurons = {}  # To store both mask and count for each layer

    for act_key, act_value in intermediate_act.items():
        act = act_value
        neurons_score = jnp.mean(jnp.abs(act), axis=0)
        neurons_score = neurons_score / (jnp.mean(neurons_score) + 1e-9)
        all_layers_score[act_key] = neurons_score

        mask = jnp.where(
            neurons_score <= dead_neuron_threshold,
            jnp.ones_like(neurons_score, dtype=jnp.int32),
            jnp.zeros_like(neurons_score, dtype=jnp.int32)
        )
        num_dead_neurons = jnp.sum(mask)

        dead_neurons[act_key] = {
            'mask': mask,
            'count': num_dead_neurons
        }


    total_dead_neurons = 0
    total_hidden_count = 0
    for layer_count, (layer_name, layer_score) in enumerate(all_layers_score.items()):
        num_dead_neurons = dead_neurons[layer_name]['count']
        total_dead_neurons += num_dead_neurons
        total_hidden_count += layer_score.shape[0]

    return np.array((total_dead_neurons / total_hidden_count) * 100)


@dataclasses.dataclass(frozen=True)
class MTSACConfig(AlgorithmConfig):
    actor_config: ContinuousActionPolicyConfig = ContinuousActionPolicyConfig()
    critic_config: QValueFunctionConfig = QValueFunctionConfig()
    temperature_optimizer_config: OptimizerConfig = OptimizerConfig(max_grad_norm=None)
    initial_temperature: float = 1.0
    num_critics: int = 2
    tau: float = 0.005
    use_task_weights: bool = False


class MTSAC(OffPolicyAlgorithm[MTSACConfig]):
    actor: TrainState
    critic: CriticTrainState
    alpha: TrainState
    key: PRNGKeyArray
    gamma: float = struct.field(pytree_node=False)
    tau: float = struct.field(pytree_node=False)
    target_entropy: float = struct.field(pytree_node=False)
    use_task_weights: bool = struct.field(pytree_node=False)

    initial_actor: TrainState | None = None
    initial_critic: CriticTrainState | None = None

    @override
    @staticmethod
    def initialize(
        config: MTSACConfig, env_config: EnvConfig, seed: int = 1
    ) -> "MTSAC":
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

        return MTSAC(
            num_tasks=config.num_tasks,
            actor=actor,
            critic=critic,
            alpha=alpha,
            key=algorithm_key,
            gamma=config.gamma,
            tau=config.tau,
            target_entropy=target_entropy,
            use_task_weights=config.use_task_weights,
            initial_actor=actor,
            initial_critic=critic
        )

    @override
    def get_initial_parameters(self,) -> tuple[Dict, Dict]:
        return self.initial_actor, self.initial_critic

    @override
    def sample_action(self, observation: Observation) -> tuple[Self, Action]:
        action, key = _sample_action(self.actor, observation, self.key)
        return self.replace(key=key), jax.device_get(action)

    @override
    def eval_action(self, observation: Observation) -> Action:
        return jax.device_get(_eval_action(self.actor, observation))

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

                q_pred = self.critic.apply_fn(params, data.observations, data.actions)
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
            flat_grads, _ = fu.ravel_pytree(critic_grads)
            return _critic, {
                "losses/qf_values": qf_values,
                "losses/qf_loss": critic_loss_value,
                "critic_grad_magnitude": jnp.linalg.norm(flat_grads)
            }

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

        flat_grads, _ = fu.ravel_pytree(actor_grads)
        logs['actor_grad_mag'] = jnp.linalg.norm(flat_grads)

        flat_params_act, _ = fu.ravel_pytree(self.actor.params)
        logs['actor_params_norm'] = jnp.linalg.norm(flat_params_act)

        flat_params_crit, _ = fu.ravel_pytree(self.critic.params)
        logs['critic_params_norm'] = jnp.linalg.norm(flat_params_crit)

        #critic: CriticTrainState
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
    def update(self, data: ReplayBufferSamples | Rollout) -> tuple[Self, LogDict]:
        assert isinstance(data, ReplayBufferSamples), "MTSAC does not support rollouts"
        return self._update_inner(data)

    @jax.jit
    def target_data(self, x): # this generates target data for the different models
        q_values_mean_preds = self.critic.apply_fn(self.critic.params, x.next_observations, x.actions).mean(axis=0)
        return q_values_mean_preds + jnp.sin(1e5 * self.critic.apply_fn(self.initial_critic.params, x.next_observations, x.actions))

    @jax.jit
    def mod_critic_loss(self, params: FrozenDict, data, target) -> Float[Array, ""]: # This is the MSE between the predicted q_values, and the generated data
        q_pred = self.critic.apply_fn(params, data.observations, data.actions)
        loss = 0.5 * ((q_pred - target) ** 2).mean(axis=1).sum()
        return loss

    @override
    #@partial(jax.jit, static_argnames=["q_network"])
    def get_activations(self, data, q_network, replay_buffer, batch_size) -> tuple[Dict, Dict]:
        key, critic_samp_key = jax.random.split(self.key, 2)

        next_actions, next_actions_log_probs = self.actor.apply_fn(
            self.actor.params, data.next_observations
        ).sample_and_log_prob(seed=critic_samp_key)

        cr1, cr2 = get_critic_params(self.critic.params)
        _, act1 = q_network.apply({'params':cr1}, data.next_observations, next_actions, capture_intermediates=True)
        _, act2 = q_network.apply({'params':cr2}, data.next_observations, next_actions, capture_intermediates=True)

        _, act_acts = self.actor.apply_fn(
            self.actor.params, data.next_observations, capture_intermediates=True
        )

        q_values_mean_preds = self.critic.apply_fn(self.critic.params, data.next_observations, next_actions).mean(axis=0) # This generates alpha for plasticity

        init_params = copy.deepcopy(self.initial_critic)
        curr_params = copy.deepcopy(self.critic)

        final_init_loss = None
        final_curr_loss = None

        for i in range(200):
            data_sample = replay_buffer.sample(batch_size)
            data_sample = self.target_data(data_sample)

            critic_loss_value_init, critic_grads = jax.value_and_grad(
                self.mod_critic_loss,
            )(init_params.params, data, data_sample)
            init_params = init_params.apply_gradients(grads=critic_grads)

            critic_loss_value, critic_grads = jax.value_and_grad(
                self.mod_critic_loss,
            )(curr_params.params, data, data_sample)
            curr_params = curr_params.apply_gradients(grads=critic_grads)

            b = np.var(data_sample)

            final_init_loss = b - critic_loss_value_init
            final_crit_loss = b - critic_loss_value

        self = self.replace(key=key)
        return act1['intermediates'], act2['intermediates'], act_acts['intermediates'], {'plasticity_crit_loss': final_crit_loss, 'plasticity_init_loss': final_init_loss, 'plasticity': final_crit_loss - final_init_loss}


    @override
    def get_metrics(self, data, config, replay_buffer, batch_size): # TODO: This calculation is based on a static number of layers, there's probably a way to do things more intelligently using the config or something
        metrics = dict()
        q_network = QValueFunction(config=config.critic_config)
 
        crit1_acts, crit2_acts, actor_acts, plasticities = self.get_activations(data, q_network, replay_buffer, batch_size)

        network = actor_acts[list(actor_acts.keys())[0]]
        layer0_act = network['layer_0']['__call__'][0]  # First layer
        layer1_act = network['layer_1']['__call__'][0]  # Second layer
        final_act = network['__call__'][0] 

        intermediate_act = {'l1': layer0_act, 'l2': layer1_act, 'l3': final_act}
        metrics['dead_neurons_actor'] = get_dead_neuron_count(intermediate_act)


        dense0_acts = crit1_acts['MultiHeadNetwork_0']['layer_0']['__call__'][0]
        layer0_acts = crit1_acts['MultiHeadNetwork_0']['layer_1']['__call__'][0]
        intermediate_act = {'l1': dense0_acts, 'l2': layer0_acts}
        metrics['dead_neurons_critic_1'] = get_dead_neuron_count(intermediate_act)

        dense0_acts = crit2_acts['MultiHeadNetwork_0']['layer_0']['__call__'][0]
        layer0_acts = crit2_acts['MultiHeadNetwork_0']['layer_1']['__call__'][0]
        intermediate_act = {'l1': dense0_acts, 'l2': layer0_acts}
        metrics['dead_neurons_critic_2'] = get_dead_neuron_count(intermediate_act)

        metrics.update(plasticities)

        metrics['srank_crit1'] = compute_srank(crit1_acts['MultiHeadNetwork_0']['layer_1']['__call__'][0])
        metrics['stank_crti2'] = compute_srank(crit2_acts['MultiHeadNetwork_0']['layer_1']['__call__'][0])

        return metrics

def compute_srank(feature_matrix, delta=0.01):
    """Compute effective rank (srank) of a feature matrix.
    Args:
        feature_matrix: Matrix of shape [num_features, feature_dim]
        delta: Threshold parameter (default: 0.01)
    Returns:
        Effective rank (srank) value
    """
    s = jnp.linalg.svd(feature_matrix, compute_uv=False)
    cumsum = jnp.cumsum(s)
    total = jnp.sum(s)
    ratios = cumsum / total
    mask = ratios >= (1.0 - delta)
    srank = jnp.argmax(mask) + 1
    return srank
