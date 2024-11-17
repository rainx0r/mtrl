import chex
import jax.numpy as jnp
from jaxtyping import Array, Float

from mtrl.types import Intermediates, LayerActivationsDict, LogDict


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


def extract_activations(network_dict: Intermediates) -> LayerActivationsDict:
    def recursive_extract(
        d: Intermediates, current_path: list[str] = []
    ) -> LayerActivationsDict:
        activations = {}
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, dict):
                    sub_activations = recursive_extract(v, current_path + [k])
                    activations.update(sub_activations)
                else:
                    assert isinstance(v, tuple)
                    # HACK: assuming every module only has 1 output
                    activations[k] = v[0]
        return activations

    return recursive_extract(network_dict)


def get_dormant_neuron_logs(
    layer_activations: LayerActivationsDict, dormant_neuron_threshold: float = 0.1
) -> LogDict:
    """Compute the dormant neuron ratio per layer using Equation 1 from
    "The Dormant Neuron Phenomenon in Deep Reinforcement Learning" (Sokar et al., 2023; https://proceedings.mlr.press/v202/sokar23a/sokar23a.pdf).

    Adapted from https://github.com/google/dopamine/blob/master/dopamine/labs/redo/tfagents/sac_train_eval.py#L563"""

    all_layers_score: LayerActivationsDict = {}
    dormant_neurons = {}  # To store both mask and count for each layer

    for act_key, act_value in layer_activations.items():
        chex.assert_rank(act_value, 2)
        neurons_score = jnp.mean(jnp.abs(act_value), axis=0)
        neurons_score = neurons_score / (jnp.mean(neurons_score) + 1e-9)
        all_layers_score[act_key] = neurons_score

        mask = jnp.where(
            neurons_score <= dormant_neuron_threshold,
            jnp.ones_like(neurons_score, dtype=jnp.int32),
            jnp.zeros_like(neurons_score, dtype=jnp.int32),
        )
        num_dormant_neurons = jnp.sum(mask)

        dormant_neurons[act_key] = {"mask": mask, "count": num_dormant_neurons}

    logs = {}

    total_dead_neurons = 0
    total_hidden_count = 0
    for layer_name, layer_score in all_layers_score.items():
        num_dormant_neurons = dormant_neurons[layer_name]["count"]
        logs[f"{layer_name}_ratio"] = (num_dormant_neurons / layer_score.shape[0]) * 100
        logs[f"{layer_name}_count"] = num_dormant_neurons
        total_dead_neurons += num_dormant_neurons
        total_hidden_count += layer_score.shape[0]

    logs.update(
        {
            "total_ratio": jnp.array((total_dead_neurons / total_hidden_count) * 100),
            "total_count": total_dead_neurons,
        }
    )

    return logs
