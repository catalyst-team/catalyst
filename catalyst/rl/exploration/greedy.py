import numpy as np
from catalyst.rl.core import ExplorationStrategy


class Greedy(ExplorationStrategy):
    """
    For discrete environments only.
    Selects greedy action (argmax_a Q(s,a)).
    """

    def get_action(self, q_values):
        action = np.argmax(q_values)
        return action


class EpsilonGreedy(ExplorationStrategy):
    """
    For discrete environments only.
    Selects random action with probability eps and greedy action
    (argmax_a Q(s,a)) with probability 1-eps.
    Random action selection probability eps usually decreases
    from 1 to 0.01-0.05 during the course of training.
    """

    def __init__(self, eps_init, eps_final, annealing_steps, eps_min=0.01):
        super().__init__()

        self.eps_init = max(eps_init, eps_min)
        self.eps_final = max(eps_final, eps_min)
        self.num_steps = annealing_steps
        self.delta_eps = (self.eps_init - self.eps_final) / self.num_steps
        self.eps = eps_init
        self.eps_min = eps_min

    def set_power(self, value):
        super().set_power(value)
        self.eps_init *= self._power
        self.eps_final *= self._power
        self.eps_final = max(self.eps_final, self.eps_min)
        self.delta_eps = (self.eps_init - self.eps_final) / self.num_steps
        self.eps = self.eps_init

    def get_action(self, q_values):
        if np.random.random() < self.eps:
            action = np.random.randint(len(q_values))
        else:
            action = np.argmax(q_values)
        self.eps = max(self.eps_final, self.eps - self.delta_eps)
        return action


__all__ = ["Greedy", "EpsilonGreedy"]
