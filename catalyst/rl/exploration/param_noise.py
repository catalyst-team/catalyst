import numpy as np
import torch
from catalyst.rl.core import ExplorationStrategy
from catalyst.rl.utils import get_network_weights, set_network_weights

EPS = 1e-6


def _set_params_noise(
    actor, states, noise_delta=0.2, tol=1e-3, max_steps=1000
):
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
        noise_dist = torch.distributions.normal.Normal(0, sigma)
        weights = {
            key: w.clone() + noise_dist.sample(w.shape)
            for key, w in orig_weights.items()
        }
        set_network_weights(actor, weights, strict=not exclude_norm)

        new_actions = actor(states)
        distance = \
            (new_actions - orig_actions).pow(2).sum(1).sqrt().mean().item()

        distance_mismatch = distance - noise_delta

        # the difference between current distance
        # and desired distance is too small
        if np.abs(distance_mismatch) < tol:
            break
        # too big sigma
        if distance_mismatch > 0:
            sigma_max = sigma
        # too small sigma
        else:
            sigma_min = sigma
        sigma = sigma_min + (sigma_max - sigma_min) / 2

    return distance


class ParameterSpaceNoise(ExplorationStrategy):
    """
    For continuous environments only.
    At the beginning of the episode, perturbs the weights of actor network
    forcing it to produce more diverse actions.
    Paper: https://arxiv.org/abs/1706.01905
    """

    def __init__(self, target_sigma, tolerance=1e-3, max_steps=1000):
        super().__init__()

        self.target_sigma = target_sigma
        self.tol = tolerance
        self.max_steps = max_steps

    def set_power(self, value):
        super().set_power(value)
        self.target_sigma *= self._power

    def update_actor(self, actor, states):
        return _set_params_noise(
            actor, states, self.target_sigma, self.tol, self.max_steps
        )

    def get_action(self, action):
        return action


__all__ = ["ParameterSpaceNoise"]
