"""
Frozen class.
Example of usage can be found in :py:class:`catalyst.core.state.State`.
"""


class FrozenClass:
    """Class which prohibit ``__setattr__`` on existing attributes.

    Examples:
        >>> class State(FrozenClass):
    """

    __isfrozen = False

    def __setattr__(self, key, value):
        """@TODO: Docs. Contribution is welcome."""
        if self.__isfrozen and not hasattr(self, key):
            raise TypeError("%r is a frozen class" % self)
        object.__setattr__(self, key, value)

    def _freeze(self):
        self.__isfrozen = True
