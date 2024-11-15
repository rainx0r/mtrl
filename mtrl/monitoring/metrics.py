import chex
import jax.numpy as jnp
from jaxtyping import Array, Float

from mtrl.types import Intermediates, LayerActivationsDict


def compute_srank(
    feature_matrix: Float[Array, "num_features feature_dim"], delta: float = 0.01
) -> Float[Array, ""]:
    """Compute effective rank (srank) of a feature matrix.

    Args:
        feature_matrix: Matrix of shape [num_features, feature_dim]
        delta: Threshold parameter (default: 0.01)

    Returns:
        Effective rank (srank) value
    """
    s = jnp.linalg.svd(feature_matrix, compute_uv=False)
    cumsum = jnp.cumsum(s)
    total = jnp.sum(s)
    ratios = cumsum / total
    mask = ratios >= (1.0 - delta)
    srank = jnp.argmax(mask) + 1
    return srank


def extract_activations(
    network_dict: Intermediates, activation_key: str = "__call__"
) -> LayerActivationsDict:
    def recursive_extract(
        d: Intermediates, current_path: list[str] = []
    ) -> LayerActivationsDict:
        activations = {}
        if isinstance(d, dict):
            # If this dictionary has '__call__', store its activation
            # but only if it's not the top-level / 'fina' object.
            if activation_key in d:
                if len(current_path) != 0:
                    layer_name = "_".join(current_path)
                    layer_activations = d[activation_key]
                    assert isinstance(layer_activations, tuple)

                    # HACK: assume only 1 output, might need to be changed if the networks
                    # start returning multiple outputs
                    activations[layer_name] = layer_activations[0]
            # Recurse through other keys
            for k, v in d.items():
                if k != activation_key:  # Skip '__call__' in recursion
                    assert isinstance(v, dict)
                    sub_activations = recursive_extract(v, current_path + [k])
                    activations.update(sub_activations)
        return activations

    return recursive_extract(network_dict)


def get_dead_neuron_ratio(
    layer_activations: LayerActivationsDict, dead_neuron_threshold: float = 0.1
) -> Float[Array, ""]:
    """Compute the dormant neuron ratio per layer using Equation 1 from "The Dormant Neuron Phenomenon in Deep Reinforcement Learning" (Sokar et al., 2023; https://proceedings.mlr.press/v202/sokar23a/sokar23a.pdf).

    Adapted from https://github.com/google/dopamine/blob/master/dopamine/labs/redo/tfagents/sac_train_eval.py#L563"""

    all_layers_score: LayerActivationsDict = {}
    dead_neurons = {}  # To store both mask and count for each layer

    for act_key, act_value in layer_activations.items():
        chex.assert_rank(act_value, 2)
        neurons_score = jnp.mean(jnp.abs(act_value), axis=0)
        neurons_score = neurons_score / (jnp.mean(neurons_score) + 1e-9)
        all_layers_score[act_key] = neurons_score

        mask = jnp.where(
            neurons_score <= dead_neuron_threshold,
            jnp.ones_like(neurons_score, dtype=jnp.int32),
            jnp.zeros_like(neurons_score, dtype=jnp.int32),
        )
        num_dead_neurons = jnp.sum(mask)

        dead_neurons[act_key] = {"mask": mask, "count": num_dead_neurons}

    total_dead_neurons = 0
    total_hidden_count = 0
    for layer_name, layer_score in all_layers_score.items():
        num_dead_neurons = dead_neurons[layer_name]["count"]
        total_dead_neurons += num_dead_neurons
        total_hidden_count += layer_score.shape[0]

    return jnp.array((total_dead_neurons / total_hidden_count) * 100)
