from collections.abc import Callable

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp


class MultiHeadNetwork(nn.Module):
    num_heads: int
    output_dim: int
    activate_last: bool = False
    # TODO: support variable width?
    width: int = 400
    depth: int = 3
    activation_fn: Callable[[jax.Array], jax.Array] = jax.nn.relu
    kernel_init: Callable = jax.nn.initializers.he_uniform
    bias_init: Callable = lambda: jax.nn.initializers.constant(0.1)

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        task_idx: jax.Array,
    ) -> jax.Array:
        batch_dim, obs_dim = x.shape
        task_idx_dim = task_idx.shape[-1]

        # 1) Forward both the obs and the task idx through an MLP
        x = jnp.concatenate((x, task_idx), axis=-1)
        chex.assert_shape(x, (batch_dim, obs_dim + task_idx_dim))

        for i in range(self.depth):
            x = nn.Dense(
                self.width,
                name=f"layer_{i}",
                kernel_init=self.kernel_init(),
                bias_init=self.bias_init(),
            )(x)
            x = self.activation_fn(x)

        # 2) Create a head for each task. Pass *every* input through *every* head
        # because we assume the batch dim is not necessarily a task dimension

        # TODO: runtime of this can be reduced significantly if we assume that
        # instead of a batch dim we have a task dim. In that case, `in_axes=0`, `out_axes=0`.
        # however this would mess up inference where we might want to have batch size 1 and select the right head
        # with task_idx.
        x = nn.vmap(
            nn.Dense,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,  # type: ignore[reportArgumentType]
            out_axes=1,
            axis_size=self.num_heads,
        )(self.output_dim, kernel_init=self.kernel_init(), bias_init=self.bias_init())(
            x
        )
        chex.assert_shape(x, (batch_dim, self.num_heads, self.output_dim))

        # 3) Collect the output from the appropriate head for each input
        task_indices = task_idx.argmax(axis=-1)
        x = x[:, task_indices]
        chex.assert_shape(x, (batch_dim, self.output_dim))

        if self.activate_last:
            x = self.activation_fn(x)

        return x
