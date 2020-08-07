"""
Meters provide a way to keep track of important statistics in an online manner.
"""


class Meter(object):
    """
    This class is abstract, but provides a standard interface for all meters to
    follow.
    """

    def reset(self):
        """Resets the meter to default settings."""
        pass

    def add(self, value):
        """Log a new value to the meter.

        Args:
            value: Next result to include.

        """
        pass

    def value(self):
        """Get the value of the meter in the current state."""
        pass


__all__ = ["Meter"]
