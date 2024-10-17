from dataclasses import dataclass
from .utils import Optimizer
import optax


@dataclass(frozen=True, kw_only=True)
class OptimizerConfig:
    lr: float = 3e-4
    optimizer: Optimizer = Optimizer.Adam
    clip_grad_norm: float | None = None

    def instantiate(self) -> optax.GradientTransformation:
        # TODO: Clip grad norm
        return self.optimizer(learning_rate=self.lr)
