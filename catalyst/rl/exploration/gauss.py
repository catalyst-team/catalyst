import numpy as np
from catalyst.rl.core import ExplorationStrategy


class NoExploration(ExplorationStrategy):
    """
    For continuous environments only.
    Returns action produced by the actor network without changes.
    """

    def get_action(self, action):
        return action


class GaussNoise(ExplorationStrategy):
    """
    For continuous environments only.
    Adds spherical Gaussian noise to the action produced by actor.
    """

    def __init__(self, sigma):
        super().__init__()

        self.sigma = sigma

    def set_power(self, value):
        super().set_power(value)
        self.sigma *= self._power

    def get_action(self, action):
        noisy_action = np.random.normal(action, self.sigma)
        return noisy_action


class OrnsteinUhlenbeckProcess(ExplorationStrategy):
    """
    For continuous environments only.
    Adds temporally correlated Gaussian noise generated with
    Ornstein-Uhlenbeck process.
    Paper: https://arxiv.org/abs/1509.02971
    """

    def __init__(self, sigma, theta, dt=1e-2):
        super().__init__()

        self.sigma = sigma
        self.theta = theta
        self.dt = dt

    def set_power(self, value):
        super().set_power(value)
        self.sigma *= self._power

    def reset_state(self, action_size):
        self.x_prev = np.zeros(action_size)

    def get_action(self, action):
        mu = self.x_prev * (1 - self.theta * self.dt)
        sigma = self.sigma * np.sqrt(self.dt)
        x = np.random.normal(mu, sigma)
        noisy_action = action + x
        self.x_prev = x
        return noisy_action


__all__ = ["NoExploration", "GaussNoise", "OrnsteinUhlenbeckProcess"]
