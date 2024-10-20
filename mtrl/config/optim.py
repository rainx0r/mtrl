from dataclasses import dataclass
from .utils import Optimizer
import optax


@dataclass(frozen=True, kw_only=True)
class OptimizerConfig:
    lr: float = 3e-4
    optimizer: Optimizer = Optimizer.Adam
    max_grad_norm: float | None = None

    def spawn(self) -> optax.GradientTransformation:
        optim = self.optimizer(learning_rate=self.lr)
        if self.max_grad_norm is not None:
            optim = optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optim,
            )
        return optim
