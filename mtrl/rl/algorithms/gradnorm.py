"""Inspired by https:self.//github.com/kevinzakka/robopianist-rl/blob/main/sac.py"""
import time
import numpy as np

# from jax.experimental.static_array import static_args
import dataclasses
from dataclasses import field
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
from flax.core import FrozenDict, freeze, unfreeze
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

from collections import deque
from typing import Deque, Generic, Self, TypeVar, override

import orbax.checkpoint as ocp
import wandb
from flax import struct
import flax

from mtrl.checkpoint import get_checkpoint_save_args
from mtrl.config.rl import (
    AlgorithmConfig,
    OnPolicyTrainingConfig,
    OffPolicyTrainingConfig,
    TrainingConfig,
)
from mtrl.envs import EnvConfig
from mtrl.rl.buffers import MultiTaskReplayBuffer, MultiTaskRolloutBuffer
from mtrl.types import (
    Action,
    Agent,
    AuxPolicyOutputs,
    CheckpointMetadata,
    LogDict,
    LogProb,
    Observation,
    ReplayBufferCheckpoint,
    ReplayBufferSamples,
    Rollout,
    Value,
)

AlgorithmConfigType = TypeVar("AlgorithmConfigType", bound=AlgorithmConfig)
TrainingConfigType = TypeVar("TrainingConfigType", bound=TrainingConfig)
DataType = TypeVar("DataType", ReplayBufferSamples, Rollout)

# GRADNORM ALGORITHM:
#
# Initialize $w_i(0)=1 \forall i$
# Initialize network weights $\mathcal{W}$
# Pick value for $\alpha>0$ and pick the weights $W$ (usually the
#     final layer of weights which are shared between tasks)
# for $t=0$ to max_train_steps $^{-10}$
#     Input batch $x_i$ to compute $L_i(t) \forall i$ and
#         $L(t)=\sum_i w_i(t) L_i(t)$ [standard forward pass]
#     Compute $G_W^{(i)}(t)$ and $r_i(t) \forall i$
#     Compute $\bar{G}_W(t)$ by averaging the $G_W^{(i)}(t)$
#     Compute $L_{\text {grad }}=\sum_i\left|G_W^{(i)}(t)-\bar{G}_W(t) \times\left[r_i(t)\right]^\alpha\right|_1$
#     Compute GradNorm gradients $\nabla_{w_i} L_{\text {grad }}$, keeping
#         targets $\bar{G}_W(t) \times\left[r_i(t)\right]^\alpha$ constant
#     Compute standard gradients $\nabla_{\mathcal{W}} L(t)$
#     Update $w_i(t) \mapsto w_i(t+1)$ using $\nabla_{w_i} L_{\text {grad }}$
#     Update $\mathcal{W}(t) \mapsto \mathcal{W}(t+1)$ using $\nabla_{\mathcal{W}} L(t)$ [standard
#         backward pass]
#     Renormalize $w_i(t+1)$ so that $\sum_i w_i(t+1)=T$
# end for

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

class GradNormWeights(nn.Module):
    num_tasks: int

    def setup(self): # gn weights are essentially task weights using GradNorm loss
        self.gn_weights = self.param(
            "gn_weights",
            nn.initializers.ones,
            (self.num_tasks,)
            )

    def __call__(
        self, task_ids: Float[Array, "... num_tasks"]
    ) -> Float[Array, "... 1"]:
        return task_ids @ self.gn_weights.reshape(-1, 1) # (10,1) x (10,1) for mt10


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
    task_weights: FrozenDict, task_ids: Float[np.ndarray, "... num_tasks"]
) -> Float[Array, "... 1"]:
    log_alpha: jax.Array
    task_weights: jax.Array

    log_tw = task_weights["params"]["gn_weights"]  # pyright: ignore [reportAssignmentType]
    task_weights = jax.nn.softmax(-log_tw)
    task_weights = task_ids @ task_weights.reshape(-1, 1)  # pyright: ignore [reportAssignmentType]
    task_weights *= log_tw.shape[0]
    return task_weights


@dataclasses.dataclass(frozen=True)
class GradNormConfig(AlgorithmConfig):
    actor_config: ContinuousActionPolicyConfig = ContinuousActionPolicyConfig()
    critic_config: QValueFunctionConfig = QValueFunctionConfig()
    temperature_optimizer_config: OptimizerConfig = OptimizerConfig(max_grad_norm=None)
    gn_optimizer_config: OptimizerConfig = OptimizerConfig(max_grad_norm=None)
    initial_temperature: float = 1.0
    num_critics: int = 2
    tau: float = 0.005
    asymmetry: float = 0.12 # Called alpha in paper (recommended range 0 < a < 3)
    use_task_weights: bool = True


class GradNorm(OffPolicyAlgorithm[GradNormConfig]):
    actor: TrainState
    critic: CriticTrainState
    alpha: TrainState
    gn_state: TrainState
    key: PRNGKeyArray
    gamma: float = struct.field(pytree_node=False)
    tau: float = struct.field(pytree_node=False)
    target_entropy: float = struct.field(pytree_node=False)
    use_task_weights: bool = struct.field(pytree_node=False)
    num_critics: int = struct.field(pytree_node=False)
    asymmetry: float = struct.field(pytree_node=False)

    @override
    @staticmethod
    def initialize(
        config: GradNormConfig, env_config: EnvConfig, seed: int = 1
    ) -> "GradNorm":
        assert isinstance(
            env_config.action_space, gym.spaces.Box
        ), "Non-box spaces currently not supported."
        assert isinstance(
            env_config.observation_space, gym.spaces.Box
        ), "Non-box spaces currently not supported."

        master_key = jax.random.PRNGKey(seed)
        algorithm_key, actor_init_key, critic_init_key, alpha_init_key, gradnorm_init_key = (
            jax.random.split(master_key, 5)
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
        gn_layer = GradNormWeights(config.num_tasks)
        gn_state = TrainState.create(
            apply_fn=gn_layer.apply,
            params=gn_layer.init(gradnorm_init_key, jnp.ones(config.num_tasks)),
            tx=config.gn_optimizer_config.spawn(),
        )

        target_entropy = -np.prod(env_config.action_space.shape).item()

        return GradNorm(
            num_tasks=config.num_tasks,
            actor=actor,
            critic=critic,
            alpha=alpha,
            gn_state=gn_state,
            key=algorithm_key,
            gamma=config.gamma,
            tau=config.tau,
            target_entropy=target_entropy,
            use_task_weights=config.use_task_weights,
            num_critics=config.num_critics,
            asymmetry=config.asymmetry
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

    # @staticmethod
    # @jax.jit
    # def renormalise_weights(train_state: TrainState, num_tasks: int): # general but mainly used for critic
    #     '''The original paper renormalises the weights of gradnormed layer'''
    #     last_layer_name = train_state.params["params"]["VmapQValueFunction_0"]["MultiHeadNetwork_0"].keys()[-1]
    #     last_layer = _critic.params["params"]["VmapQValueFunction_0"]["MultiHeadNetwork_0"][last_layer_name]
    #     weights, unravel_fn = jax.flatten_util.ravel_pytree(last_layer)
    #     uf_ll = unfreeze(weights)[last_layer_name]
    #     # uf_ll[last_layer_name] = 
    #
    #     # We want weights for each task to avg to 1 so total weights should equal num tasks w(i+1) = Tw(i)/sum{W(i)}
    #     new_params = unravel_fn( (weights / (jnp.sum(weights) + 1e-12) ) * num_tasks)
    #     train_state = train_state.replace(params=new_params)
    #     return train_state


    @jax.jit
    def _update_inner(self, data: ReplayBufferSamples, original_losses) -> tuple[Self, LogDict]:
        task_ids = data.observations[..., -self.num_tasks :]
        
        # --- Critic loss ---
        key, actor_loss_key, critic_loss_key = jax.random.split(self.key, 3)

        def update_critic(
            _critic: CriticTrainState,
            alpha_val: Float[Array, "batch 1"],
            # task_weights: Float[Array, "batch 1"] | None = None,
            gn_state: TrainState,
            num_tasks,
            task_ids
        ) -> tuple[CriticTrainState, LogDict]:

            # Sample a'
            next_actions, next_action_log_probs = self.actor.apply_fn(
                self.actor.params, data.next_observations
            ).sample_and_log_prob(seed=critic_loss_key)

            assert self.use_task_weights, "Task weights are required for GradNorm"
            if self.use_task_weights:
                task_weights = extract_task_weights(gn_state.params, task_ids)
            else:
                task_weights = None

            # Compute target Q values
            q_values = self.critic.apply_fn(
                self.critic.target_params, data.next_observations, next_actions
            )
            min_qf_next_target = jnp.min(
                q_values, axis=0
            ) - alpha_val * next_action_log_probs.reshape(-1, 1)
            next_q_value = jax.lax.stop_gradient(
                data.rewards + (1 - data.dones) * self.gamma * min_qf_next_target
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
                        * (jax.lax.stop_gradient(task_weights) * (q_pred - next_q_value) ** 2)
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
            flat_grads, _ = flatten_util.ravel_pytree(critic_grads)

            # --- GradNorm Weight Update ---
            _gn_state, task_weights, _original_losses, gn_metrics = update_gn_weights(gn_state,
                                                                          original_losses,
                                                                          _critic,
                                                                          num_tasks,
                                                                          task_ids, 
                                                                          next_q_value,
                                                                          self.asymmetry)
            
            return _critic, {
                "losses/qf_values": qf_values,
                "losses/qf_loss": critic_loss_value,
                "metrics/critic_grad_magnitude": jnp.linalg.norm(flat_grads),
            } | gn_metrics, _gn_state, task_weights, _original_losses

        # --- Alpha loss ---

        def update_alpha(
            _alpha: TrainState,
            log_probs: Float[Array, " batch"],
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

            return (
                _alpha,
                alpha_vals,
                {
                    "losses/alpha_loss": alpha_loss_value,
                    "alpha": jnp.exp(_alpha.params["params"]["log_alpha"]).sum(),  # pyright: ignore [reportReturnType,reportArgumentType]
                },
            )
        # asymmetry/restoring force is given as alpha in paper
        def update_gn_weights(gn_state, original_losses, critic, num_tasks, task_ids, next_q_value, asymmetry=0.12): 

            # Get grads and loss for each task
            def get_task_grad(task_idx: int, task_weights: Array) -> Array:
                task_mask = task_ids[:, task_idx] == 1
                
                def task_loss(params: FrozenDict) -> Float[Array, ""]:
                    q_pred = critic.apply_fn(params, data.observations, data.actions)
                    loss = (
                        0.5
                        * (task_weights * (q_pred - next_q_value) ** 2)
                        .mean(axis=1)
                        .sum()
                    )
                    return loss

                # Get gradients for each task using vmap
                loss, grad = jax.value_and_grad(task_loss)(critic.params)
                flat_grad, _ = jax.flatten_util.ravel_pytree(grad)
                return flat_grad, loss

            def gn_loss(params): # Could also try original losses
                task_weights = extract_task_weights(params, task_ids) # Get normalised task weights

                # Compute task specific losses and grads
                task_grads, task_losses = jax.vmap(get_task_grad, in_axes=(0,None))(jnp.arange(num_tasks).reshape(-1,1), task_weights)
                _original_losses = jax.lax.select(jnp.all(jnp.isnan(original_losses)), 
                                                  jax.lax.stop_gradient(task_losses),
                                                  original_losses)

                improvement = task_losses / (_original_losses + 1e-12) # \tilde{L}_i(t)  || (num_tasks,)
                avg_impr = jnp.mean(improvement, axis=0) # E_{task}[\tilde{L}_i(t)]  || Scalar
                rit = improvement / avg_impr # (num_tasks,)
                grad_norms = jnp.linalg.norm(task_grads, axis=1, ord=2)  # G_W  || (num_tasks, params) -> (num_tasks,)
                avg_grad_norm = jnp.mean(grad_norms)  # \bar{G}_W(t)  || Scalar
                constant = avg_grad_norm * rit ** asymmetry # (num_tasks,)
                l_grad = jnp.sum(jnp.abs(grad_norms - constant)) # Scalar
                return l_grad, (_original_losses, task_weights, task_losses)

            (loss, (_original_losses, task_weights, task_losses)), grads = jax.value_and_grad(gn_loss, has_aux=True)(gn_state.params)
            _gn_state = gn_state.apply_gradients(grads=grads)

            # Weight normalisation
            _gn_state.params['params']['gn_weights'] = (_gn_state.params['params']['gn_weights'] / _gn_state.params['params']['gn_weights'].sum())\
                    * num_tasks

            # def get_order(arr):
            #     sorted_indices = np.argsort(arr)
            #
            #     ranks = np.empty_like(sorted_indices)
            #     ranks[sorted_indices] = np.arange(len(arr))
            #     return ranks


            return _gn_state, task_weights, _original_losses, {'metrics/gradnorm_task_weights_std': jnp.std(task_weights),
                                                               'metrics/gradnorm_loss': loss,
                                                               'metrics/asymmetry_const': asymmetry,
                                                               'metrics/gn_weights_max': jnp.argmax(gn_state.params['params']['gn_weights'])}

        # --- Actor loss --- & calls for the other losses
        def actor_loss(params: FrozenDict):
            action_samples, log_probs = self.actor.apply_fn(
                params, data.observations
            ).sample_and_log_prob(seed=actor_loss_key)

            # HACK: Putting the other losses / grad updates inside this function for performance,
            # so we can reuse the action_samples / log_probs while also doing alpha loss first
            _alpha, _alpha_val, alpha_logs = update_alpha(
                self.alpha, log_probs
            )

            assert self.use_task_weights, "GradNorm requires task weights"
            _alpha_val = jax.lax.stop_gradient(_alpha_val)

            _critic, critic_logs, _gn_state, task_weights, _original_losses = update_critic(self.critic,
                                             _alpha_val,
                                             self.gn_state,
                                             self.num_tasks,
                                             jnp.array(task_ids))
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
            
            return loss, (_alpha, _critic, logs, _original_losses, _gn_state)

        (actor_loss_value, (alpha, critic, logs, original_losses, gn_state)), actor_grads = jax.value_and_grad(
            actor_loss, has_aux=True
        )(self.actor.params)
        actor = self.actor.apply_gradients(grads=actor_grads)

        flat_grads, _ = flatten_util.ravel_pytree(actor_grads)
        logs["metrics/actor_grad_magnitude"] = jnp.linalg.norm(flat_grads)

        flat_params_act, _ = flatten_util.ravel_pytree(self.actor.params)
        logs["metrics/actor_params_norm"] = jnp.linalg.norm(flat_params_act)

        flat_params_crit, _ = flatten_util.ravel_pytree(self.critic.params)
        logs["metrics/critic_params_norm"] = jnp.linalg.norm(flat_params_crit)

        # logs["metrics/gn_weights"] = wandb.Histogram(logs["metrics/gn_weights"])
        # logs["metrics/task_losses"] = wandb.Histogram(logs["metrics/task_losses"])

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
            gn_state=gn_state
        )

        return (self, {**logs, "losses/actor_loss": actor_loss_value}, original_losses)

    @override
    def update(self, data: ReplayBufferSamples, original_losses) -> tuple[Self, LogDict]:
        return self._update_inner(data, original_losses)

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
    
    @override
    def train(
        self,
        config: OffPolicyTrainingConfig,
        envs: gym.vector.VectorEnv,
        env_config: EnvConfig,
        run_timestamp: str,
        seed: int = 1,
        track: bool = True,
        checkpoint_manager: ocp.CheckpointManager | None = None,
        checkpoint_metadata: CheckpointMetadata | None = None,
        buffer_checkpoint: ReplayBufferCheckpoint | None = None,
    ) -> Self:
        global_episodic_return: Deque[float] = deque([], maxlen=20 * self.num_tasks)
        global_episodic_length: Deque[int] = deque([], maxlen=20 * self.num_tasks)

        obs, _ = envs.reset()

        has_autoreset = np.full((envs.num_envs,), False)
        start_step, episodes_ended = 0, 0

        if checkpoint_metadata is not None:
            start_step = checkpoint_metadata["step"]
            episodes_ended = checkpoint_metadata["episodes_ended"]

        replay_buffer = self.spawn_replay_buffer(env_config, config, seed)
        if buffer_checkpoint is not None:
            replay_buffer.load_checkpoint(buffer_checkpoint)

        start_time = time.time()
        original_losses = jnp.full((self.num_tasks,), jnp.nan)
        ### Handle trainstate here?

        for global_step in range(start_step, config.total_steps // envs.num_envs):
            total_steps = global_step * envs.num_envs

            if global_step < config.warmstart_steps:
                actions = np.array(
                    [envs.single_action_space.sample() for _ in range(envs.num_envs)]
                )
            else:
                self, actions = self.sample_action(obs)

            next_obs, rewards, terminations, truncations, infos = envs.step(actions)

            if not has_autoreset.any():
                replay_buffer.add(obs, next_obs, actions, rewards, terminations)
            elif has_autoreset.any() and not has_autoreset.all():
                # TODO: handle the case where only some envs have autoreset
                raise NotImplementedError(
                    "Only some envs resetting isn't implemented at the moment."
                )

            has_autoreset = np.logical_or(terminations, truncations)

            for i, env_ended in enumerate(has_autoreset):
                if env_ended:
                    global_episodic_return.append(infos["episode"]["r"][i])
                    global_episodic_length.append(infos["episode"]["l"][i])
                    episodes_ended += 1

            obs = next_obs

            if global_step % 500 == 0 and global_episodic_return:
                print(
                    f"global_step={total_steps}, mean_episodic_return={np.mean(list(global_episodic_return))}"
                )
                if track:
                    wandb.log(
                        {
                            "charts/mean_episodic_return": np.mean(
                                list(global_episodic_return)
                            ),
                            "charts/mean_episodic_length": np.mean(
                                list(global_episodic_length)
                            ),
                        },
                        step=total_steps,
                    )

            if global_step > config.warmstart_steps:
                # Update the agent with data
                data = replay_buffer.sample(config.batch_size)
                self, logs, original_losses = self.update(data, original_losses)


                # Logging
                if global_step % 100 == 0:
                    sps_steps = (global_step - start_step) * envs.num_envs
                    sps = int(sps_steps / (time.time() - start_time))
                    print("SPS:", sps)

                    if track:
                        wandb.log({"charts/SPS": sps} | logs, step=total_steps)

                # Evaluation
                if (
                    config.evaluation_frequency > 0
                    and episodes_ended % config.evaluation_frequency == 0
                    and has_autoreset.any()
                    and global_step > 0
                ):
                    mean_success_rate, mean_returns, mean_success_per_task = (
                        env_config.evaluate(envs, self)
                    )
                    eval_metrics = {
                        "charts/mean_success_rate": float(mean_success_rate),
                        "charts/mean_evaluation_return": float(mean_returns),
                    } | {
                        f"charts/{task_name}_success_rate": float(success_rate)
                        for task_name, success_rate in mean_success_per_task.items()
                    }
                    print(
                        f"total_steps={total_steps}, mean evaluation success rate: {mean_success_rate:.4f}"
                        + f" return: {mean_returns:.4f}"
                    )

                    if track:
                        wandb.log(eval_metrics, step=total_steps)

                    if config.compute_network_metrics:
                        self, network_metrics = self.get_metrics(data)

                        if track:
                            wandb.log(network_metrics, step=total_steps)

                    # Reset envs again to exit eval mode
                    obs, _ = envs.reset()

                    # Checkpointing
                    if checkpoint_manager is not None:
                        if not has_autoreset.all():
                            raise NotImplementedError(
                                "Checkpointing currently doesn't work for the case where evaluation is run before all envs have finished their episodes / are about to be reset."
                            )

                        checkpoint_manager.save(
                            total_steps,
                            args=get_checkpoint_save_args(
                                self,
                                envs,
                                global_step,
                                episodes_ended,
                                run_timestamp,
                                buffer=replay_buffer,
                            ),
                            metrics=eval_metrics,
                        )
        return self


# '''
#     @staticmethod
#     @partial(jax.jit, static_argnums=(4,))  # Specify the index of num_tasks argument
#     def compute_gradnorm(
#         critic: CriticTrainState,
#         data: ReplayBufferSamples, 
#         task_ids: Float[Array, "batch num_tasks"],
#         # alpha_val: Float[Array, "batch 1"],
#         next_q_value: Float[Array, "batch 1"],
#         num_tasks: int,
#         original_losses: Array
#         ) -> tuple[Array, LogDict]:
#
#         def get_task_grad(task_idx: int) -> Array:
#             task_mask = task_ids[:, task_idx] == 1
#             
#             def task_loss(params: FrozenDict) -> Float[Array, ""]:
#                 q_pred = critic.apply_fn(params, data.observations, data.actions)
#                 loss = 0.5 * ((q_pred - next_q_value) ** 2 * task_mask[:, None]).mean()
#                 return loss
#                 
#             loss, grad = jax.value_and_grad(task_loss)(critic.params)
#             flat_grad, _ = jax.flatten_util.ravel_pytree(grad)
#             return flat_grad, loss
#
#
#         # Get gradients for each task using vmap
#         task_grads, task_losses = jax.vmap(get_task_grad)(jnp.arange(num_tasks))
#    
#         # GRADNORM ALGORITHM:
#         #     - Save original loss [X]
#         #     - Calculate improvement fraction as: (initial loss/loss now) remember div by 0 so add 1e-15 or something
#         #     - Normalise losses by weighting them based on if they're improving too quickly or not quick enough
#         #     - Use parameter alpha to determine how much to weight by
#         #     - Change fixed alpha to being an actual hyperparameter, maybe choose a different letter?
#         #
#         #     \tilde{L}_i(t) = L_i(t) / L_i(0) # Loss ratio, aka inverse training rate
#         #     r_i(t) = \tilde{L}_i(t) / E_{task}[\tilde{L}_i(t)] # The *relative* inverse training rate
#         #     ... Define G (task grad) and \bar{G} (avg grad) similarly
#         #
#         # Initialize $w_i(0)=1 \forall i$
#         # Initialize network weights $\mathcal{W}$
#         # Pick value for $\alpha>0$ and pick the weights $W$ (usually the
#         #     final layer of weights which are shared between tasks)
#         # for $t=0$ to max_train_steps $^{-10}$
#         #     Input batch $x_i$ to compute $L_i(t) \forall i$ and
#         #         $L(t)=\sum_i w_i(t) L_i(t)$ [standard forward pass]
#         #     Compute $G_W^{(i)}(t)$ and $r_i(t) \forall i$
#         #     Compute $\bar{G}_W(t)$ by averaging the $G_W^{(i)}(t)$
#         #     Compute $L_{\text {grad }}=\sum_i\left|G_W^{(i)}(t)-\bar{G}_W(t) \times\left[r_i(t)\right]^\alpha\right|_1$
#         #     Compute GradNorm gradients $\nabla_{w_i} L_{\text {grad }}$, keeping
#         #         targets $\bar{G}_W(t) \times\left[r_i(t)\right]^\alpha$ constant
#         #     Compute standard gradients $\nabla_{\mathcal{W}} L(t)$
#         #     Update $w_i(t) \mapsto w_i(t+1)$ using $\nabla_{w_i} L_{\text {grad }}$
#         #     Update $\mathcal{W}(t) \mapsto \mathcal{W}(t+1)$ using $\nabla_{\mathcal{W}} L(t)$ [standard
#         #         backward pass]
#         #     Renormalize $w_i(t+1)$ so that $\sum_i w_i(t+1)=T$
#         # end for
#
#         @jax.jit
#         def gradnorm(alpha=0.1):
#             # Should only be nan at the start, could replace this with flag for added robustness
#             _original_losses = jax.lax.select(jnp.all(jnp.isnan(original_losses)), jax.lax.stop_gradient(task_losses), original_losses)
#
#             def loss_fn(params):
#                 improvement = task_losses / (_original_losses + 1e-12) # \tilde{L}_i(t)
#
#                 grad_norms = jnp.linalg.norm(task_grads, axis=1, ord=1)  # G_W
#                 avg_grad_norm = jnp.mean(grad_norms)  # \bar{G}_W(t)
#                 avg_impr = jnp.mean(improvement, axis=0) # E_{task}[\tilde{L}_i(t)]
#
#                 rit = improvement / avg_impr
#                 l_grad = jnp.sum(jnp.linalg.norm(grad_norms - jax.lax.stop_gradient(avg_grad_norm * (rit)**alpha), ord=1))# jnp.sum(task_losses - (avg_grad_norm * (rit)**alpha ) )
#                 return l_grad
#
#             loss, grad = jax.value_and_grad(loss_fn)(critic.params)
#             return loss, grad, _original_losses
#         
#         loss, final_grad, _original_losses = gradnorm(alpha=0.)
#         print(loss, _original_losses.mean())
#
#         # Unravel back to pytree
#         # _, unravel_fn = jax.flatten_util.ravel_pytree(critic.params)
#         # final_grad = unravel_fn(grad)
#         # final_grad = grad
#             
#         # Metrics 
#         def vmap_cos_sim(grads, num_tasks):
#             def calc_cos_sim(selected_grad, grads):
#
#                 new_cos_sim  = jnp.array([ # Removed the jnp.mean
#                                 jnp.sum(selected_grad * grads, axis=1) / (
#                                         jnp.linalg.norm(selected_grad) * jnp.linalg.norm(grads, axis=1) + 1e-12
#                                         )
#                                 ])
#
#                 return new_cos_sim
#
#             cos_sim_mat = jax.vmap(calc_cos_sim, in_axes=(0,None), out_axes=-1)(grads, grads)
#             mask = jnp.triu(jnp.ones((num_tasks, num_tasks)), k=1) # Get upper triangle
#             num_unique = jnp.sum(mask)
#
#             masked_cos_sim = mask * cos_sim_mat
#             avg_cos_sim = jnp.sum(masked_cos_sim.flatten()) / num_unique # n in upper triangle
#             return avg_cos_sim
#         
#         # avg_cos_sim = vmap_cos_sim(task_grads, num_tasks)
#         # new_cos_sim = vmap_cos_sim(final_grads, num_tasks)
#         metrics = {
#             # "metrics/n_grad_conflicts": total_grad_conflicts,
#             # "metrics/avg_critic_grad_magnitude": jnp.mean(jnp.linalg.norm(final_grads, axis=1)),
#             # "metrics/avg_critic_grad_magnitude_before_grad_surgery": jnp.mean(jnp.linalg.norm(task_grads, axis=1)),
#         #     "metrics/avg_cosine_similarity":  avg_cos_sim,
#         # "metrics/avg_cosine_similarity_diff": avg_cos_sim - new_cos_sim
#             
#         }       
#         return final_grad, metrics, _original_losses
#
#
# '''
