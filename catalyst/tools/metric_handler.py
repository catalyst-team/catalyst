# flake8: noqa
# @TODO: we also can make it BestScoreHanlder and store best score inside
from functools import partial


def _is_better_min(score, best, min_delta):
    return score <= (best - min_delta)


def _is_better_max(score, best, min_delta):
    return score >= (best + min_delta)


class MetricHandler:
    """@TODO: docs.

    Args:
        minimize: @TODO: docs
        min_delta: @TODO: docs
    """

    def __init__(self, minimize: bool = True, min_delta: float = 1e-6):
        """Init."""
        self.minimize = minimize
        self.min_delta = min_delta
        # self.best_score = None

        if self.minimize:
            self.is_better = partial(_is_better_min, min_delta=min_delta)
        else:
            self.is_better = partial(_is_better_max, min_delta=min_delta)

    def __call__(self, score, best_score):
        """@TODO: docs."""
        # if self.best_score is None or self.is_better(score, self.best_score):
        #     self.best_score = score
        #     return True
        # else:
        #     return False
        return self.is_better(score, best_score)


__all__ = ["MetricHandler"]
