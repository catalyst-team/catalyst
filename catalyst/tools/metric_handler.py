# @TODO: we also can make it BestScoreHanlder and store best score inside
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

            def _is_better(score, best):
                return score <= (best - self.min_delta)

        else:

            def _is_better(score, best):
                return score >= (best + self.min_delta)

        self.is_better = _is_better

    def __call__(self, score, best_score):
        """@TODO: docs."""
        # if self.best_score is None or self.is_better(score, self.best_score):
        #     self.best_score = score
        #     return True
        # else:
        #     return False
        return self.is_better(score, best_score)


__all__ = ["MetricHandler"]
