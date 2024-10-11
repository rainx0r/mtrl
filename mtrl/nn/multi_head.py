import flax.linen as nn
import jax
import jax.numpy as jnp

from mtrl.config.nn import MultiHeadConfig


class MultiHeadNetwork(nn.Module):
    config: MultiHeadConfig

    head_dim: int
    head_kernel_init: jax.nn.initializers.Initializer = jax.nn.initializers.he_normal()
    head_bias_init: jax.nn.initializers.Initializer = jax.nn.initializers.zeros

    activate_last: bool = False

    # TODO: support variable width?

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
    ) -> jax.Array:
        batch_dim = x.shape[0]
        assert self.config.num_tasks is not None, "Number of tasks must be provided."
        task_idx = x[..., -self.config.num_tasks :]

        for i in range(self.config.depth):
            x = nn.Dense(
                self.config.width,
                name=f"layer_{i}",
                kernel_init=self.config.kernel_init(),
                bias_init=self.config.bias_init(),
                use_bias=self.config.use_bias,
            )(x)
            x = self.config.activation(x)

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
            axis_size=self.config.num_tasks,
        )(
            self.head_dim,
            kernel_init=self.head_kernel_init,
            bias_init=self.head_bias_init,
            use_bias=self.config.use_bias,
        )(x)

        # 3) Collect the output from the appropriate head for each input
        task_indices = task_idx.argmax(axis=-1)
        x = x[jnp.arange(batch_dim), task_indices]

        if self.activate_last:
            x = self.config.activation(x)

        return x
