import enum
from dataclasses import dataclass

from .utils import Activation, Initializer


@dataclass(frozen=True)
class NeuralNetworkConfig:
    width: int = 400
    """The number of neurons in the hidden layers."""

    depth: int = 3
    """The number of hidden layers."""

    kernel_init: Initializer = Initializer.HE_UNIFORM
    """The initializer to use for hidden layer weights."""
    # TODO: How to pass arguments to kernel_init

    bias_init: Initializer = Initializer.ZEROS
    """The initializer to use for hidden layer biases."""
    # TODO: How to pass arguments to bias_init

    use_bias: bool = True
    """Whether or not to use bias terms across the network."""

    activation: Activation = Activation.ReLU
    """The activation function to use."""


@dataclass(frozen=True)
class MultiHeadConfig(NeuralNetworkConfig):
    num_tasks: int | None = None
    """The number of tasks (which in turn determines the number of heads)."""


@dataclass(frozen=True)
class SoftModulesConfig(NeuralNetworkConfig):
    num_tasks: int | None = None
    """The number of tasks (which in turn determines the number of heads)."""

    width: int = 256
    """The number of neurons in the Dense layers around the network."""

    module_width: int = 256
    """The number of neurons in each module in the Base Policy Network. `d` in the paper."""

    num_modules: int = 2
    """The number of modules to use in each Base Policy Network layer."""

    embedding_dim: int = 400
    """The dimension of the observation / task index embedding. `D` in the paper."""


# class Architecture(enum.Enum):
#     SoftModules = enum.auto()
#     MultiHead = enum.auto()
#     Vanilla = enum.auto()
#
#     def verify_config(self, config: NeuralNetworkConfig):
#         match self:
#             case Architecture.Vanilla:
#                 assert isinstance(config, NeuralNetworkConfig)
#             case Architecture.MultiHead:
#                 assert isinstance(config, NeuralNetworkConfig)
#             case Architecture.SoftModules:
#                 assert isinstance(config, SoftModulesConfig)
