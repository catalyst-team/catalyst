import numpy as np
from .utils import set_params_noise


class ExplorationStrategy:

    def __init__(self, power=1.0):
        self._power = power

    def set_power(self, value):
        assert 0. <= value <= 1.0
        self._power = value

    def update_action(self, action):
        return action


class Greedy(ExplorationStrategy):
    pass


class EpsilonGreedy(ExplorationStrategy):
    def __init__(self, eps_init, eps_final, annealing_steps, num_actions):
        super().__init__()

        self.eps_init = eps_init
        self.eps_final = eps_final
        self.num_steps = annealing_steps
        self.delta_eps = (self.eps_init - self.eps_final) / self.num_steps
        self.eps = eps_init
        self.num_actions = num_actions

    def set_power(self, value):
        super().set_power(value)
        self.eps_init *= self._power
        self.eps_final *= self._power
        self.delta_eps = (self.eps_init - self.eps_final) / self.num_steps
        self.eps = self.eps_init

    def update_action(self, action):
        if np.random.random() < self.eps:
            action = np.random.randint(self.num_actions)
        self.eps = max(self.eps_final, self.eps - self.delta_eps)
        return action


class GaussNoise(ExplorationStrategy):
    def __init__(self, sigma):
        super().__init__()

        self.sigma = sigma

    def set_power(self, value):
        super().set_power(value)
        self.sigma *= self._power

    def update_action(self, action):
        noisy_action = np.random.normal(action, self.sigma)
        return noisy_action


class ParameterSpaceNoise(ExplorationStrategy):
    def __init__(self, target_sigma, tolerance=1e-3, max_steps=1000):
        super().__init__()

        self.target_sigma = target_sigma
        self.tol = tolerance
        self.max_steps = max_steps

    def set_power(self, value):
        super().set_power(value)
        self.target_sigma *= self._power

    def update_actor(self, actor, states):
        return set_params_noise(
            actor, states, self.target_sigma, self.tol, self.max_steps
        )
