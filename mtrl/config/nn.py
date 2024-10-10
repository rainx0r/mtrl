from dataclasses import dataclass
from .utils import Initializer, Activation


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

    activation: Activation = Activation.ReLU
    """The activation function to use."""


@dataclass(frozen=True)
class SoftModulesConfig(NeuralNetworkConfig):
    module_width: int = 256
    """The number of neurons in each module."""

    num_modules: int = 2
    """The number of modules to use in each layer."""
