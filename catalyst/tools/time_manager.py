from time import time


class TimeManager(object):
    """@TODO: Docs. Contribution is welcome."""

    def __init__(self):
        """Initialization"""
        self._starts = {}
        self.elapsed = {}

    def start(self, name: str) -> None:
        """Starts timer ``name``.

        Args:
            name: name of a timer
        """
        self._starts[name] = time()

    def stop(self, name: str) -> None:
        """Stops timer ``name``.

        Args:
            name: name of a timer
        """
        assert name in self._starts, f"Timer '{name}' wasn't started"

        self.elapsed[name] = time() - self._starts[name]
        del self._starts[name]

    def reset(self) -> None:
        """Reset all previous timers."""
        self.elapsed = {}
        self._starts = {}


__all__ = ["TimeManager"]
