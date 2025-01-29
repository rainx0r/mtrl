import jax
import flax.struct
import numpy.typing as npt
from jaxtyping import Float


class Histogram(flax.struct.PyTreeNode):
    data: Float[npt.NDArray, "..."]
