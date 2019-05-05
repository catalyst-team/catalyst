from typing import List
from copy import deepcopy
import numpy as np
from gym.spaces import Discrete
from catalyst.rl.registry import EXPLORATION
from catalyst.rl.environments.core import EnvironmentSpec
from .strategies import ExplorationStrategy, EpsilonGreedy


class ExplorationHandler:
    def __init__(self, *exploration_params, env: EnvironmentSpec):
        params = deepcopy(exploration_params)
        self.exploration_strategies: List[ExplorationStrategy] = []
        self.probs = []

        for params_ in params:
            exploration_name = params_.pop("exploration")
            probability = params_.pop("probability")
            strategy_fn = EXPLORATION.get(exploration_name)

            if issubclass(strategy_fn, EpsilonGreedy):
                assert isinstance(env.action_space, Discrete)
                params_["num_actions"] = env.action_space.n

            strategy = strategy_fn(**params_)
            self.exploration_strategies.append(strategy)
            self.probs.append(probability)

        self.num_strategies = len(self.probs)
        assert np.isclose(np.sum(self.probs), 1.0)

    def set_power(self, value):
        assert 0. <= value <= 1.0
        for exploration in self.exploration_strategies:
            exploration.set_power(value=value)

    def get_exploration_strategy(self):
        strategy_idx = np.random.choice(self.num_strategies, p=self.probs)
        strategy = self.exploration_strategies[strategy_idx]
        return strategy
