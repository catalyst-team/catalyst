import numpy as np
from .utils import np_softmax, set_params_noise


class ExplorationStrategy:
    """
    Base class for working with various exploration strategies.
    In discrete case must contain method get_action(q_values).
    In continuous case must contain method get_action(action).
    """
    def __init__(self, power=1.0):
        self._power = power

    def set_power(self, value):
        assert 0. <= value <= 1.0
        self._power = value


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
    def __init__(self, eps_init, eps_final, annealing_steps):
        super().__init__()

        self.eps_init = eps_init
        self.eps_final = eps_final
        self.num_steps = annealing_steps
        self.delta_eps = (self.eps_init - self.eps_final) / self.num_steps
        self.eps = eps_init

    def set_power(self, value):
        super().set_power(value)
        self.eps_init *= self._power
        self.eps_final *= self._power
        self.delta_eps = (self.eps_init - self.eps_final) / self.num_steps
        self.eps = self.eps_init

    def get_action(self, q_values):
        if np.random.random() < self.eps:
            action = np.random.randint(len(q_values))
        else:
            action = np.argmax(q_values)
        self.eps = max(self.eps_final, self.eps - self.delta_eps)
        return action


class Boltzmann(ExplorationStrategy):
    """
    For discrete environments only.
    Selects soft maximum action (softmax_a [Q(s,a)/t]).
    Temperature parameter t usually decreases during the course of
    training. Importantly, the effective range of t depends on the
    magnitutdes of environment rewards.
    """
    def __init__(self, temp_init, temp_final, annealing_steps):
        super().__init__()

        self.temp_init = temp_init
        self.temp_final = temp_final
        self.num_steps = annealing_steps
        self.delta_temp = (self.temp_init - self.temp_final) / self.num_steps
        self.temperature = temp_init

    def set_power(self, value):
        super().set_power(value)
        self.temp_init *= self._power
        self.temp_final *= self._power
        self.delta_temp = (self.temp_init - self.temp_final) / self.num_steps
        self.temperature = self.temp_init

    def get_action(self, q_values):
        probs = np_softmax(q_values / self.temperature)
        action = np.random.choice(np.arange(len(probs)), p=probs)
        self.temperature = max(
            self.temp_final, self.temperature - self.delta_temp
        )
        return action


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
        return set_params_noise(
            actor, states, self.target_sigma, self.tol, self.max_steps
        )

    def get_action(self, action):
        return action
