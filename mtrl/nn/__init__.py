import mtrl.config.nn

import flax.linen as nn

from .base import MLP as VanillaNetwork
from .multi_head import MultiHeadNetwork
from .soft_modules import SoftModularizationNetwork


def get_nn_arch_for_config(
    config: mtrl.config.nn.NeuralNetworkConfig,
) -> type[nn.Module]:
    if isinstance(config, mtrl.config.nn.MultiHeadConfig):
        return MultiHeadNetwork
    elif isinstance(config, mtrl.config.nn.SoftModulesConfig):
        return SoftModularizationNetwork
    elif isinstance(config, mtrl.config.nn.NeuralNetworkConfig):
        return VanillaNetwork


__all__ = ["VanillaNetwork", "MultiHeadNetwork", "SoftModularizationNetwork"]
