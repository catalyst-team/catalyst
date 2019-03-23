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
    step = 0

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


class Explorator:
    def __init__(self, config):
        from catalyst.contrib.registry import Registry

        config_ = config.copy()
        self.strategies = []
        self.probs = []
        for expl in config_["exploration"]:
            probability = expl["probability"]
            expl_params = expl["params"] or {}
            strategy = Registry.get_exploration(
                strategy=expl["strategy"], **expl_params)
            self.strategies.append(strategy)
            self.probs.append(probability)
        self.num_strategies = len(self.probs)

    def get_exploration_strategy(self):
        strategy_idx = np.random.choice(self.num_strategies, p=self.probs)
        strategy = self.strategies[strategy_idx]
        return strategy


class ExplorationStrategy:
    def __init__(self, **params):
        pass

    def _explore(self, action):
        return action

    def _run(self):
        pass


class ParameterSpaceNoise(ExplorationStrategy):
    def __init__(self, target_sigma, tolerance=1e-3, max_steps=1000):
        self.target_sigma = target_sigma
        self.tol = tolerance
        self.max_steps = max_steps

    def _run(self, actor, states):
        set_params_noise(
            actor, states, self.target_sigma, self.tol, self.max_steps
        )


class GaussNoise(ExplorationStrategy):
    def __init__(self, sigma):
        self.sigma = sigma

    def _explore(self, action):
        noisy_action = np.random.normal(action, self.sigma)
        return noisy_action


class Greedy(ExplorationStrategy):
    def __init__(self, **kwargs):
        pass


class EpsilonGreedy(ExplorationStrategy):
    def __init__(self, eps_init, eps_final, annealing_steps, num_actions):
        self.eps = eps_init
        self.eps_final = eps_final
        self.delta_eps = (eps_init - eps_final) / annealing_steps
        self.num_actions = num_actions
        
    def _explore(self, action):
        if np.random.random() < self.eps:
            action = np.random.randint(self.num_actions)
        self.eps = max(self.eps_final, self.eps - self.delta_eps)
        return action
            