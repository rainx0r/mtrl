from collections.abc import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp

from .base import MLP
from .initializers import uniform

# NOTE: the paper is missing quite a lot of details that are in the official code
#
# 1) there is an extra embedding layer for the task embedding after z and f have been combined
#    that downsizes the embedding from D to 256 (in both the deep and shallow versions of the network)
# 2) the obs embedding is activated before it's passed into the layers
# 3) p_l+1 is not dependent on just p_l but on all p_<l with skip connections
# 4) ReLU is applied after the weighted sum in forward computation, not before as in Eq. 8 in the paper
# 5) there is an extra p_L+1 that is applied as a dot product over the final module outputs
# 6) the task weights take the softmax over log alpha, not actual alpha.
#    And they're also multiplied by the number of tasks
#
# These are marked with "NOTE: <number>"


class BasePolicyNetworkLayer(nn.Module):
    """A Flax Module to represent a single layer of modules of the Base Policy Network"""

    num_modules: int  # n
    module_dim: int  # d
    kernel_init: jax.nn.initializers.Initializer = jax.nn.initializers.he_normal()
    bias_init: jax.nn.initializers.Initializer = jax.nn.initializers.zeros

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        modules = nn.vmap(
            nn.Dense,
            variable_axes={"params": 0},  # Different params per module
            split_rngs={"params": True},  # Different initialization per module
            in_axes=-2,
            out_axes=-2,
            axis_size=self.num_modules,
        )(self.module_dim, kernel_init=self.kernel_init, bias_init=self.bias_init)

        # NOTE: 4, activation *should* be here according to the paper, but it's after the weighted sum
        return modules(x)


class RoutingNetworkLayer(nn.Module):
    """A Flax Module to represent a single layer of the Routing Network"""

    embedding_dim: int  # D
    num_modules: int
    activation_fn: Callable[[jax.typing.ArrayLike], jax.Array] = jax.nn.relu
    kernel_init: jax.nn.initializers.Initializer = jax.nn.initializers.he_normal()
    bias_init: jax.nn.initializers.Initializer = jax.nn.initializers.zeros
    last: bool = False  # NOTE: 5

    def setup(self):
        self.prob_embedding_fc = nn.Dense(
            self.embedding_dim,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )  # W_u^l
        # NOTE: 5
        prob_output_dim = (
            self.num_modules if self.last else self.num_modules * self.num_modules
        )
        self.prob_output_fc = nn.Dense(
            prob_output_dim,
            kernel_init=uniform(1e-3),
            bias_init=jax.nn.initializers.zeros,
        )  # W_d^l

    def __call__(
        self, task_embedding: jax.Array, prev_probs: jax.Array | None = None
    ) -> jax.Array:
        if prev_probs is not None:  # Eq 5-only bit
            task_embedding *= self.prob_embedding_fc(prev_probs)
        x = self.prob_output_fc(self.activation_fn(task_embedding))
        if not self.last:  # NOTE: 5
            x = x.reshape(*x.shape[:-1], self.num_modules, self.num_modules)
        x = jax.nn.softmax(x, axis=-1)  # Eq. 7
        return x


class SoftModularizationNetwork(nn.Module):
    """A Flax Module to represent the Base Policy Network and the Routing Network simultaneously,
    since their layers are so intertwined.

    Corresponds to `ModularGatedCascadeCondNet` in the official implementation."""

    embedding_dim: int  # D
    module_dim: int  # d
    num_layers: int
    num_modules: int
    output_dim: int  # o, 1 for Q networks and 2 * action_dim for policy networks

    activation_fn: Callable[[jax.typing.ArrayLike], jax.Array] = jax.nn.relu
    kernel_init: jax.nn.initializers.Initializer = jax.nn.initializers.he_normal()
    bias_init: jax.nn.initializers.Initializer = jax.nn.initializers.zeros

    head_kernel_init: jax.nn.initializers.Initializer = jax.nn.initalizers.he_normal()
    head_bias_init: jax.nn.initializers.Initializer = jax.nn.initializers.zeros

    routing_skip_connections: bool = True  # NOTE: 3

    def setup(self) -> None:
        # Base policy network layers
        self.f = MLP(
            depth=1,
            output_dim=self.embedding_dim,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            activation_fn=self.activation_fn,
        )
        self.layers = [
            BasePolicyNetworkLayer(
                self.num_modules,
                self.module_dim,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
            for _ in range(self.num_layers)
        ]
        self.output_head = nn.Dense(
            self.output_dim,
            kernel_init=self.head_kernel_init,
            bias_init=self.head_bias_init,
        )

        # Routing network layers
        self.z = MLP(
            depth=0,
            output_dim=self.embedding_dim,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )
        self.task_embedding_fc = MLP(
            depth=1,
            width=256,
            output_dim=256,
            activation_fn=self.activation_fn,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )  # NOTE: 1
        self.prob_fcs = [
            RoutingNetworkLayer(
                embedding_dim=256,
                num_modules=self.num_modules,
                last=i == self.num_layers - 1,
                activation_fn=self.activation_fn,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
            for i in range(self.num_layers)  # NOTE: 5
        ]

    def __call__(self, x: jax.Array, task_idx: jax.Array) -> jax.Array:
        # Feature extraction
        obs_embedding = self.f(x)
        task_embedding = self.z(task_idx) * obs_embedding
        task_embedding = self.task_embedding_fc(
            self.activation_fn(task_embedding)
        )  # NOTE: 1

        # Initial layer inputs
        prev_probs = None
        obs_embedding = self.activation_fn(obs_embedding)  # NOTE: 2
        module_ins = jnp.stack(
            [obs_embedding for _ in range(self.num_modules)], axis=-2
        )
        weights = None

        if self.routing_skip_connections:  # NOTE: 3
            weights = []

        for i in range(
            self.num_layers - 1
        ):  # Equation 8, holds for all layers except L
            probs = self.prob_fcs[i](task_embedding, prev_probs)
            module_outs = self.activation_fn(
                probs @ self.layers[i](module_ins)
            )  # NOTE: 4

            # Post processing
            probs = probs.reshape(probs.shape[:-1], self.num_modules * self.num_modules)
            if weights is not None and self.routing_skip_connections:  # NOTE: 3
                weights.append(probs)
                prev_probs = jnp.concatenate(weights, axis=-1)
            else:
                prev_probs = probs
            module_ins = module_outs

        # Last layer L, Equation 9
        module_outs = self.layers[-1](module_ins)
        probs = jnp.expand_dims(
            self.prob_fcs[-1](task_embedding, prev_probs), axis=-1
        )  # NOTE: 5
        output_embedding = self.activation_fn(jnp.sum(module_outs * probs, axis=-2))
        return self.output_head(output_embedding)
