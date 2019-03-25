from typing import List
import numpy as np
from .strategies import ExplorationStrategy


class ExplorationHandler:
    def __init__(self, **params):
        from catalyst.contrib.registry import Registry

        config_ = params.copy()
        self.strategies: List[ExplorationStrategy] = []
        self.probs = []

        for key, expl in config_.items():
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
