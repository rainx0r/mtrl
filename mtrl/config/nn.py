from dataclasses import dataclass

from mtrl.config.optim import OptimizerConfig

from .utils import Activation, Initializer


@dataclass(frozen=True, kw_only=True)
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

    optimizer: OptimizerConfig = OptimizerConfig()
    """The optimizer to use for the network."""


@dataclass(frozen=True, kw_only=True)
class MultiHeadConfig(NeuralNetworkConfig):
    num_tasks: int
    """The number of tasks (used for extracting the task IDs & to determine the number of heads)."""


@dataclass(frozen=True, kw_only=True)
class SoftModulesConfig(NeuralNetworkConfig):
    num_tasks: int
    """The number of tasks (used for extracting the task IDs)."""

    width: int = 256
    """The number of neurons in the Dense layers around the network."""

    module_width: int = 256
    """The number of neurons in each module in the Base Policy Network. `d` in the paper."""

    num_modules: int = 2
    """The number of modules to use in each Base Policy Network layer."""

    embedding_dim: int = 400
    """The dimension of the observation / task index embedding. `D` in the paper."""


@dataclass(frozen=True, kw_only=True)
class PaCoConfig(NeuralNetworkConfig):
    num_tasks: int
    """The number of tasks (used for extracting the task IDs)."""

    num_parameter_sets: int = 5
    """The number of parameter sets. `K` in the paper."""
