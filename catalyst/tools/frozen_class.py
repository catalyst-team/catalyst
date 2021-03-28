class FrozenClass:
    """Class which prohibit ``__setattr__`` on existing attributes.

    Examples:
        >>> class IRunner(FrozenClass):
    """

    __is_frozen = False

    def __setattr__(self, key, value):
        """@TODO: Docs. Contribution is welcome."""
        if self.__is_frozen and not hasattr(self, key):
            raise TypeError("%r is a frozen class for key %s" % (self, key))
        object.__setattr__(self, key, value)

    def _freeze(self):
        self.__is_frozen = True

    def _unfreeze(self):
        self.__is_frozen = False


__all__ = ["FrozenClass"]
