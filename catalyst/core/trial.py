from typing import Dict
from abc import ABC


# Optuna, Ray, Hyperopt
class ITrial(ABC):
    """
    An abstraction that syncs experiment run with
    different hyperparameter-search systems.
    """

    pass


# could it be a Union[supported trials?]
class Trial(ITrial):
    pass


def get_trial_by_params(trial_params: Dict):
    return Trial()
