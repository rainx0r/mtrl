import flax.linen as nn
import jax
import jax.numpy as jnp

from .base import MLP

from mtrl.config.nn import PaCoConfig


class PaCoNetwork(nn.Module):
    config: PaCoConfig

    head_dim: int  # o, 1 for Q networks and 2 * action_dim for policy networks
    head_kernel_init: jax.nn.initializers.Initializer = jax.nn.initializers.he_normal()
    head_bias_init: jax.nn.initializers.Initializer = jax.nn.initializers.zeros

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        assert self.config.num_tasks is not None, "Number of tasks must be provided."
        task_idx = x[..., -self.config.num_tasks :]
        x = x[..., : -self.config.num_tasks]

        # Task ID embedding
        w_tau = nn.Dense(
            self.config.num_parameter_sets,
            use_bias=self.config.use_bias,
            kernel_init=self.config.kernel_init(),
            bias_init=self.config.bias_init(),
        )(task_idx)

        for _ in range(self.config.depth):
            x = nn.vmap(
                nn.Dense,
                variable_axes={"params": 0},
                split_rngs={
                    "params": True,
                    "dropout": True,
                },  # TODO: Check that init is different
                in_axes=None,  # pyright: ignore [reportArgumentType]
                out_axes=-2,
                axis_size=self.config.num_parameter_sets,
            )(
                self.config.width,
                use_bias=self.config.use_bias,
                kernel_init=self.config.kernel_init(),
                bias_init=self.config.bias_init(),
            )(x)
            x = jnp.einsum("bkn,bk->bn", x, w_tau)
            x = self.config.activation(x)

        x = nn.vmap(
            nn.Dense,
            variable_axes={"params": 0},
            split_rngs={
                "params": True,
                "dropout": True,
            },  # TODO: Check that init is different
            in_axes=None,  # pyright: ignore [reportArgumentType]
            out_axes=-2,
            axis_size=self.config.num_parameter_sets,
        )(
            self.head_dim,
            use_bias=self.config.use_bias,
            kernel_init=self.head_kernel_init,
            bias_init=self.head_bias_init,
        )(x)
        x = jnp.einsum("bkn,bk->bn", x, w_tau)

        return x
