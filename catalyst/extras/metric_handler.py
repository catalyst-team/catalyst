# flake8: noqa
from functools import partial


def _is_better_min(score, best, min_delta):
    return score <= (best - min_delta)


def _is_better_max(score, best, min_delta):
    return score >= (best + min_delta)


class MetricHandler:
    """Docs.

    Args:
        minimize: Docs
        min_delta: Docs
    """

    def __init__(self, minimize: bool = True, min_delta: float = 1e-6):
        """Init."""
        self.minimize = minimize
        self.min_delta = min_delta
        if self.minimize:
            self.is_better = partial(_is_better_min, min_delta=min_delta)
        else:
            self.is_better = partial(_is_better_max, min_delta=min_delta)

    def __call__(self, score, best_score):
        """Docs."""
        return self.is_better(score, best_score)


__all__ = ["MetricHandler"]
