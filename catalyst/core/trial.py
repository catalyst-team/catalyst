from abc import ABC


# Optuna, Ray, Hyperopt
class ITrial(ABC):
    """An abstraction that syncs experiment run with different hyperparameter search systems."""

    pass


__all__ = ["ITrial"]
