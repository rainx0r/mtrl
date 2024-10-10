import enum

import flax.linen
import jax

import mtrl.nets.initializers


class Initializer(enum.Enum):
    ZEROS = lambda: jax.nn.initializers.zeros  # noqa: E731
    HE_NORMAL = jax.nn.initializers.he_normal
    HE_UNIFORM = jax.nn.initializers.he_uniform
    XAVIER_NORMAL = jax.nn.initializers.xavier_normal
    XAVIER_UNIFORM = jax.nn.initializers.xavier_uniform
    CONSTANT = jax.nn.initializers.constant
    UNIFORM = mtrl.nets.initializers.uniform


class Activation(enum.Enum):
    ReLU = jax.nn.relu
    Tanh = jax.nn.tanh
    LeakyReLU = jax.nn.leaky_relu
    PReLU = lambda x: flax.linen.PReLU()(x)  # noqa: E731
    ReLU6 = jax.nn.relu6
    SiLU = jax.nn.silu
    GELU = jax.nn.gelu
    GLU = jax.nn.glu
    # TODO: Add SwiGLU
