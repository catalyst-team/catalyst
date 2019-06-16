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


class NoExploration(ExplorationStrategy):
    """
    For continuous environments only.
    Returns action produced by the actor network without changes.
    """
    def get_action(self, action):
        return action
