from collections.abc import Callable

import flax.linen as nn
import jax


class MLP(nn.Module):
    """A Flax Module to represent an MLP feature extractor."""

    output_dim: int
    depth: int = 3
    # TODO: Support variable width?
    width: int = 400
    activation_fn: Callable[[jax.typing.ArrayLike], jax.Array] = jax.nn.relu
    kernel_init: jax.nn.initializers.Initializer = jax.nn.initializers.he_uniform()
    bias_init: jax.nn.initializers.Initializer = jax.nn.initializers.constant(0.1)
    activate_last: bool = False

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for i in range(self.depth):
            x = nn.Dense(
                self.width,
                name=f"layer_{i}",
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )(x)
            x = self.activation_fn(x)
        x = nn.Dense(self.output_dim, name=f"layer_{self.depth}")(x)
        if self.activate_last:
            x = self.activation_fn(x)
        return x
