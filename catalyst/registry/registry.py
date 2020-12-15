from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from catalyst.registry.subregistry import (
    Factory,
    LateAddCallbak,
    RegistryException,
    SubRegistry,
)


class Registry:
    def __init__(self, sep: str = ":"):
        self.sep = sep

        self._registries: Dict[str, SubRegistry] = {}

    def _prepare_name_subregistry(
        self, name: str, subregistry: Optional[str] = None
    ) -> Tuple[str, SubRegistry]:
        if self.sep in name:
            if (
                name.startswith(self.sep)
                or name.endswith(self.sep)
                or len(name.split(self.sep)) > 2
            ):
                raise RegistryException(f"Invalid name `{name}`")

            name, subregistry = name.split(self.sep)
        elif subregistry is None:
            raise RegistryException(
                f"SubRegistry `{subregistry}` does is not exist"
            )

        return name, self._registries[subregistry]

    def get_subregistry(self, key: str) -> SubRegistry:
        if key not in self._registries:
            self._registries[key] = SubRegistry(key)

        registry = self._registries[key]

        return registry

    def add_subregistry(self, key: str, **kwargs):
        new_subregistry = SubRegistry(default_name_key=key, **kwargs)
        self[key] = new_subregistry

        return new_subregistry

    def late_add(self, key: str, cb: LateAddCallbak):
        return self.get_subregistry(key).late_add(cb)

    def add_from_module(
        self, key: str, module, prefix: Union[str, List[str]] = None
    ) -> None:
        registry = self.get_subregistry(key)
        return registry.add_from_module(module=module, prefix=prefix)

    def get(
        self, name: str, subregistry: Optional[str] = None
    ) -> Optional[Factory]:
        name, registry = self._prepare_name_subregistry(
            name=name, subregistry=subregistry
        )
        return registry.get(name)

    def get_instance(
        self,
        name: str,
        *args,
        subregistry: Optional[str] = None,
        meta_factory=None,
        **kwargs,
    ):
        name, registry = self._prepare_name_subregistry(
            name=name, subregistry=subregistry
        )
        return registry.get_instance(
            name, *args, meta_factory=meta_factory, **kwargs
        )

    def get_from_params(
        self, *, subregistry: Optional[str] = None, meta_factory=None, **kwargs
    ) -> Union[Any, Tuple[Any, Mapping[str, Any]]]:
        if not subregistry:
            common_keys = set(self._registries.keys()) & set(kwargs.keys())

            if len(common_keys) != 1:
                raise RegistryException("Please, specify registry to use")

            subregistry = next(iter(common_keys))

        name = kwargs.get(subregistry)
        if name:
            name, registry = self._prepare_name_subregistry(
                name=name, subregistry=subregistry
            )
            kwargs[subregistry] = name

            return registry.get_from_params(
                meta_factory=meta_factory, **kwargs
            )

    def __len__(self) -> int:
        return sum(len(value) for key, value in self._registries.items())

    def __getitem__(self, key: str) -> SubRegistry:
        return self._registries.get(key)

    def __setitem__(self, key: str, value: SubRegistry):
        if key in self._registries and self._registries[key].all():
            raise RegistryException(
                "Multiple subregistries with single name are not allowed"
            )

        self._registries[key] = value

    def __delitem__(self, key: str) -> None:
        self._registries.pop(key)


__all__ = ["Registry"]
