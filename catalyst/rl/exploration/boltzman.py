import numpy as np
from catalyst.rl.core import ExplorationStrategy
from catalyst.utils.numpy import np_softmax

EPS = 1e-6


class Boltzmann(ExplorationStrategy):
    """
    For discrete environments only.
    Selects soft maximum action (softmax_a [Q(s,a)/t]).
    Temperature parameter t usually decreases during the course of
    training. Importantly, the effective range of t depends on the
    magnitutdes of environment rewards.
    """
    def __init__(self, temp_init, temp_final, annealing_steps, temp_min=0.01):
        super().__init__()

        self.temp_init = max(temp_init, temp_min)
        self.temp_final = max(temp_final, temp_min)
        self.num_steps = annealing_steps
        self.delta_temp = (self.temp_init - self.temp_final) / self.num_steps
        self.temperature = temp_init
        self.temp_min = temp_min

    def set_power(self, value):
        super().set_power(value)
        self.temp_init *= self._power
        self.temp_init = max(self.temp_init, self.temp_min)
        self.temp_final *= self._power
        self.temp_final = max(self.temp_final, self.temp_min)
        self.delta_temp = (self.temp_init - self.temp_final) / self.num_steps
        self.temperature = self.temp_init

    def get_action(self, q_values):
        probs = np_softmax(q_values + EPS / self.temperature)
        action = np.random.choice(np.arange(len(probs)), p=probs)
        self.temperature = max(
            self.temp_final, self.temperature - self.delta_temp
        )
        return action


__all__ = ["Boltzmann"]
