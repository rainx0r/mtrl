import jax
import jax.numpy as jnp
import numpy as np


def compute_srank(feature_matrix, delta=0.01):
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


def return_net_layers(config):
    if isinstance(config.actor_config.network_config, SoftModulesConfig):
        return 'SoftModularizationNetwork_0', 'f', 'layers_0', 'layers_1'


def extract_params_at_index(params_dict, index):
    def recursive_extract(d):
        if isinstance(d, dict):
            return {k: recursive_extract(v) for k, v in d.items()}
        else:
            return d[index]
    # Extract only the relevant part of the params dictionary
    network_params = params_dict['params']['VmapQValueFunction_0']
    return recursive_extract(network_params)


def extract_activations(network_dict, activation_key='__call__'):
    def recursive_extract(d, current_path=[]):
        activations = {}
        if isinstance(d, dict):
            # If this dictionary has '__call__', store its activation
            if activation_key in d:
                layer_name = '_'.join(current_path) if current_path else 'final'
                activations[layer_name] = d[activation_key][0]
            # Recurse through other keys
            for k, v in d.items():
                if k != activation_key:  # Skip '__call__' in recursion
                    sub_activations = recursive_extract(v, current_path + [k])
                    activations.update(sub_activations)
        return activations

    return recursive_extract(network_dict)


def get_dead_neuron_count(intermediate_act, dead_neuron_threshold=0.1):
    all_layers_score = {}
    dead_neurons = {}  # To store both mask and count for each layer

    for act_key, act_value in intermediate_act.items():
        act = act_value
        neurons_score = jnp.mean(jnp.abs(act), axis=0)
        neurons_score = neurons_score / (jnp.mean(neurons_score) + 1e-9)
        all_layers_score[act_key] = neurons_score

        mask = jnp.where(
            neurons_score <= dead_neuron_threshold,
            jnp.ones_like(neurons_score, dtype=jnp.int32),
            jnp.zeros_like(neurons_score, dtype=jnp.int32)
        )
        num_dead_neurons = jnp.sum(mask)

        dead_neurons[act_key] = {
            'mask': mask,
            'count': num_dead_neurons
        }


    total_dead_neurons = 0
    total_hidden_count = 0
    for layer_count, (layer_name, layer_score) in enumerate(all_layers_score.items()):
        num_dead_neurons = dead_neurons[layer_name]['count']
        total_dead_neurons += num_dead_neurons
        total_hidden_count += layer_score.shape[0]

    return np.array((total_dead_neurons / total_hidden_count) * 100)
