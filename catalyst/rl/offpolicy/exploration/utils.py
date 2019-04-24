import numpy as np
import torch


def get_network_weights(network, exclude_norm=False):
    """
    Args:
        network: torch.nn.Module, neural network, e.g. actor or critic
        exclude_norm: True if layers corresponding to norm will be excluded
    Returns:
        state_dict: dictionary which contains neural network parameters
    """
    state_dict = network.state_dict()
    if exclude_norm:
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if all(x not in key for x in ["norm", "lstm"])
        }
    state_dict = {key: value.clone() for key, value in state_dict.items()}
    return state_dict


def set_network_weights(network, weights, strict=True):
    network.load_state_dict(weights, strict=strict)


def set_params_noise(actor, states, noise_delta=0.2, tol=1e-3, max_steps=1000):
    """
    Perturbs parameters of the policy represented by the actor network.
    Binary search is employed to find the appropriate magnitude of the noise
    corresponding to the desired distance measure (noise_delta) between
    non-perturbed and perturbed policy.

    Args:
        actor: torch.nn.Module, neural network which represents actor
        states: batch of states to estimate the distance measure between the
            non-perturbed and perturbed policy
        noise_delta: float, parameter noise threshold value
        tol: float, controls the tolerance of binary search
        max_steps: maximum number of steps in binary search
    """

    if states is None:
        return noise_delta

    exclude_norm = True
    orig_weights = get_network_weights(actor, exclude_norm=exclude_norm)
    orig_actions = actor(states)

    sigma_min = 0.
    sigma_max = 100.
    sigma = sigma_max

    for step in range(max_steps):
        dist = torch.distributions.normal.Normal(0, sigma)
        weights = {
            key: w.clone() + dist.sample(w.shape)
            for key, w in orig_weights.items()
        }
        set_network_weights(actor, weights, strict=not exclude_norm)

        new_actions = actor(states)
        dist = (new_actions - orig_actions).pow(2).sum(1).sqrt().mean().item()

        dist_mismatch = dist - noise_delta

        # the difference between current dist and desired dist is too small
        if np.abs(dist_mismatch) < tol:
            break
        # too big sigma
        if dist_mismatch > 0:
            sigma_max = sigma
        # too small sigma
        else:
            sigma_min = sigma
        sigma = sigma_min + (sigma_max - sigma_min) / 2

    return dist
