import chex
import flax.linen as nn
import jax
import jax.numpy as jnp

from mtrl.config.nn import CAREConfig

from .base import MLP


class CARENetwork(nn.Module):
    config: CAREConfig

    head_dim: int  # o, 1 for Q networks and 2 * action_dim for policy networks
    head_kernel_init: jax.nn.initializers.Initializer = jax.nn.initializers.he_normal()
    head_bias_init: jax.nn.initializers.Initializer = jax.nn.initializers.zeros

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        assert self.config.num_tasks is not None, "Number of tasks must be provided."
        task_idx = x[..., -self.config.num_tasks :]
        x = x[..., : -self.config.num_tasks]

        # Encoder
        # TODO: This should be replaced with a pretrained NLP model eventually
        # for language description embeddings
        task_embedding = nn.Embed(
            num_embeddings=self.config.num_tasks,
            features=self.config.embedding_dim,
            name="task_idx_embedding",
        )(task_idx.argmax(-1))
        chex.assert_shape(
            task_embedding, (*task_idx.shape[:-1], self.config.embedding_dim)
        )

        attention_weights = MLP(
            width=self.config.width,
            depth=self.config.depth,
            head_dim=self.config.num_experts,
            kernel_init=self.config.kernel_init(),
            bias_init=self.config.bias_init(),
            use_bias=self.config.use_bias,
            activation_fn=self.config.activation,
        )(task_embedding)
        chex.assert_shape(
            attention_weights, (*task_idx.shape[:-1], self.config.num_experts)
        )
        attention_weights = jax.nn.softmax(
            attention_weights / self.temperature, axis=-1
        )

        moe_out = nn.vmap(
            MLP,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,  # pyright: ignore [reportArgumentType]
            out_axes=-2,
            axis_size=self.config.num_experts,
        )(
            width=self.config.width,
            depth=self.config.depth,
            head_dim=self.config.embedding_dim,
            kernel_init=self.config.kernel_init(),
            bias_init=self.config.bias_init(),
            use_bias=self.config.use_bias,
            activation_fn=self.config.activation,
            name="moe_mlp",
        )(x)
        chex.assert_shape(
            moe_out, (self.config.num_experts, *x.shape[:-1], self.config.embedding_dim)
        )

        encoder_out = jnp.einsum("bne,bn->be", moe_out, attention_weights)
        chex.assert_shape(encoder_out, (*x.shape[:-1], self.config.embedding_dim))

        # Main network forward pass
        torso_input = jnp.concatenate((encoder_out, task_embedding), axis=-1)
        chex.assert_shape(
            torso_input,
            (*x.shape[:-1], self.config.embedding_dim + self.config.embedding_dim),
        )

        torso_output = MLP(
            head_dim=self.head_dim,
            depth=self.config.depth,
            width=self.config.width,
            activation_fn=self.config.activation,
            kernel_init=self.config.kernel_init(),
            bias_init=self.config.bias_init(),
            use_bias=self.config.use_bias,
            head_kernel_init=self.head_kernel_init,
            head_bias_init=self.head_bias_init,
        )(torso_input)
        chex.assert_shape(torso_output, (*x.shape[:-1], self.head_dim))

        return torso_output