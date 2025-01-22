import enum

import flax.linen
import jax
import optax


def _uniform_init(bound: float) -> jax.nn.initializers.Initializer:
    import mtrl.nn.initializers

    return mtrl.nn.initializers.uniform(bound)


class Initializer(enum.Enum):
    ZEROS = enum.member(lambda: jax.nn.initializers.zeros)  # noqa: E731
    HE_NORMAL = enum.member(jax.nn.initializers.he_normal)
    HE_UNIFORM = enum.member(jax.nn.initializers.he_uniform)
    XAVIER_NORMAL = enum.member(jax.nn.initializers.xavier_normal)
    XAVIER_UNIFORM = enum.member(jax.nn.initializers.xavier_uniform)
    CONSTANT = enum.member(jax.nn.initializers.constant)
    UNIFORM = enum.member(_uniform_init)

    def __call__(self, *args):
        return self.value(*args)


class Activation(enum.Enum):
    ReLU = enum.member(jax.nn.relu)
    Tanh = enum.member(jax.nn.tanh)
    LeakyReLU = enum.member(jax.nn.leaky_relu)
    PReLU = enum.member(lambda x: flax.linen.PReLU()(x))  # noqa: E731
    ReLU6 = enum.member(jax.nn.relu6)
    SiLU = enum.member(jax.nn.silu)
    GELU = enum.member(jax.nn.gelu)
    GLU = enum.member(jax.nn.glu)
    # TODO: Add SwiGLU

    def __call__(self, *args):
        return self.value(*args)


class Optimizer(enum.Enum):
    Adam = enum.member(optax.adam)
    AdamW = enum.member(optax.adamw)
    RMSProp = enum.member(optax.rmsprop)
    SGD = enum.member(optax.sgd)

    def __call__(self, learning_rate: float, **kwargs):
        return self.value(learning_rate, **kwargs)


class Metrics(enum.Enum):
    NONE = 0
    DORMANT_NEURONS = 1
    SRANK = 2
    ALL = 3

    def is_enabled(self, other: "Metrics") -> bool:
        return self.value & other.value == other.value
