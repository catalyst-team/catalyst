from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple, Type, Union
import collections
import functools
import inspect
import warnings

Factory = Union[Type, Callable[..., Any]]
LateAddCallbak = Callable[["Registry"], None]
MetaFactory = Callable[[Factory, Tuple, Mapping], Any]


def _default_meta_factory(factory: Factory, args: Tuple, kwargs: Mapping):
    if inspect.isfunction(factory):
        return functools.partial(factory, *args, **kwargs)
    return factory(*args, **kwargs)


class RegistryException(Exception):
    """Exception class for all registry errors."""

    def __init__(self, message):
        """
        Init.

        Args:
            message: exception message
        """
        super().__init__(message)


class Registry(collections.MutableMapping):
    """
    Universal class allowing to add and access various factories by name.

    Args:
        default_meta_factory: default object
            that calls factory. Optional. Default just calls factory.
    """

    def __init__(self, default_meta_factory: MetaFactory = _default_meta_factory):
        """Init."""
        self.meta_factory = default_meta_factory
        self._factories: Dict[str, Factory] = {}
        self._late_add_callbacks: List[LateAddCallbak] = []

    @staticmethod
    def _get_factory_name(f, provided_name=None) -> str:
        if not provided_name:
            provided_name = getattr(f, "__name__", None)
            if not provided_name:
                raise RegistryException(
                    f"Factory {f} has no __name__ and no " f"name was provided"
                )
            if provided_name == "<lambda>":
                raise RegistryException("Name for lambda factories must be provided")
        return provided_name

    def _do_late_add(self):
        if self._late_add_callbacks:
            for cb in self._late_add_callbacks:
                cb(self)
            self._late_add_callbacks = []

    def add(
        self,
        factory: Factory = None,
        *factories: Factory,
        name: str = None,
        **named_factories: Factory,
    ) -> Factory:
        """
        Adds factory to registry with it's ``__name__`` attribute or provided
        name. Signature is flexible.

        Args:
            factory: Factory instance
            factories: More instances
            name: Provided name for first instance. Use only when pass
                single instance.
            named_factories: Factory and their names as kwargs

        Returns:
            Factory: First factory passed

        Raises:
            RegistryException: if factory with provided name is already present
        """
        if len(factories) > 0 and name is not None:
            raise RegistryException("Multiple factories with single name are not allowed")

        if factory is not None:
            named_factories[self._get_factory_name(factory, name)] = factory

        if len(factories) > 0:
            new = {self._get_factory_name(f): f for f in factories}
            named_factories.update(new)

        if len(named_factories) == 0:
            warnings.warn("No factories were provided!")

        for name, f in named_factories.items():
            # self._factories[name] != f is a workaround for
            # https://github.com/catalyst-team/catalyst/issues/135
            if name in self._factories and self._factories[name] != f:
                raise RegistryException(
                    f"Factory with name '{name}' is already present\n"
                    f"Already registered: '{self._factories[name]}'\n"
                    f"New: '{f}'"
                )

        self._factories.update(named_factories)

        return factory

    def late_add(self, cb: LateAddCallbak):
        """
        Allows to prevent cycle imports by delaying some imports till next
        registry query.

        Args:
            cb: Callback receives registry and must call it's methods to
                register factories
        """
        self._late_add_callbacks.append(cb)

    def add_from_module(self, module, prefix: Union[str, List[str]] = None) -> None:
        """
        Adds all factories present in module.
        If ``__all__`` attribute is present, takes ony what mentioned in it.

        Args:
            module: module to scan
            prefix (Union[str, List[str]]): prefix string for all the module's
                factories. If prefix is a list, all values will be treated
                as aliases.

        Raises:
            TypeError: if prefix is not a list or a string
        """
        factories = {
            k: v for k, v in module.__dict__.items() if inspect.isclass(v) or inspect.isfunction(v)
        }

        # Filter by __all__ if present
        names_to_add = getattr(module, "__all__", list(factories.keys()))

        if prefix is None:
            prefix = [""]
        elif isinstance(prefix, str):
            prefix = [prefix]
        elif isinstance(prefix, list):
            if any((not isinstance(p, str)) for p in prefix):
                raise TypeError("All prefix in list must be strings.")
        else:
            raise TypeError(f"Prefix must be a list or a string, got {type(prefix)}.")

        to_add = {f"{p}{name}": factories[name] for p in prefix for name in names_to_add}
        self.add(**to_add)

    def get(self, name: str) -> Optional[Factory]:
        """
        Retrieves factory, without creating any objects with it
        or raises error.

        Args:
            name: factory name

        Returns:
            Factory: factory by name

        Raises:
            RegistryException: if no factory with provided name was registered
        """
        self._do_late_add()

        if name is None:
            return None

        res = self._factories.get(name, None)

        if not res:
            raise RegistryException(f"No factory with name '{name}' was registered")

        return res

    def get_if_str(self, obj: Union[str, Factory]):
        """
        Returns object from the registry if ``obj`` type is string.
        """
        if type(obj) is str:
            return self.get(obj)
        return obj

    def get_instance(self, name: str, *args, meta_factory=None, **kwargs):
        """
        Creates instance by calling specified factory
        with ``instantiate_fn``.

        Args:
            name: factory name
            meta_factory: Function that calls factory the right way.
                If not provided, default is used
            args: args to pass to the factory
            **kwargs: kwargs to pass to the factory

        Returns:
            created instance

        Raises:
            RegistryException: if could not create object instance
        """
        meta_factory = meta_factory or self.meta_factory
        f = self.get(name)

        try:
            if hasattr(f, "get_from_params"):
                return f.get_from_params(*args, **kwargs)
            return meta_factory(f, args, kwargs)
        except Exception as e:
            raise RegistryException(
                f"Factory '{name}' call failed: args={args} kwargs={kwargs}"
            ) from e

    def get_from_params(
        self, *, meta_factory=None, **kwargs
    ) -> Union[Any, Tuple[Any, Mapping[str, Any]]]:
        """
        Creates instance based in configuration dict with ``instantiation_fn``.
        If ``config[name_key]`` is None, None is returned.

        Args:
            meta_factory: Function that calls factory the right way.
                If not provided, default is used.
            **kwargs: additional kwargs for factory

        Returns:
            result of calling ``instantiate_fn(factory, **config)``
        """
        name = kwargs.pop("_target_", None)
        if name:
            return self.get_instance(name, meta_factory=meta_factory, **kwargs)

    def all(self) -> List[str]:
        """
        Returns:
            list of names of registered items
        """
        self._do_late_add()
        result = list(self._factories.keys())

        return result

    def len(self) -> int:
        """
        Returns:
            length of registered items
        """
        return len(self._factories)

    def __str__(self) -> str:
        """Returns a string of registered items."""
        return self.all().__str__()

    def __repr__(self) -> str:
        """Returns a string representation of registered items."""
        return self.all().__str__()

    # mapping methods
    def __len__(self) -> int:
        """Returns length of registered items."""
        self._do_late_add()
        return self.len()

    def __getitem__(self, name: str) -> Optional[Factory]:
        """Returns a value from the registry by name."""
        return self.get(name)

    def __iter__(self) -> Iterator[str]:
        """Iterates over all registered items."""
        self._do_late_add()
        return self._factories.__iter__()

    def __contains__(self, name: str):
        """Check if a particular name was registered."""
        self._do_late_add()
        return self._factories.__contains__(name)

    def __setitem__(self, name: str, factory: Factory) -> None:
        """Add a new factory by giving name."""
        self.add(factory, name=name)

    def __delitem__(self, name: str) -> None:
        """Removes a factory by giving name."""
        self._factories.pop(name)


__all__ = ["Registry", "RegistryException"]
