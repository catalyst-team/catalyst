import collections
from typing import Dict, Union, List, Optional, Tuple

from catalyst.registry.subregistry import SubRegistry, RegistryException, LateAddCallbak, Factory


class Registry:
    def __init__(self):
        self._registries: Dict[str, SubRegistry] = collections.defaultdict(lambda x: SubRegistry(x))

    def _prepare_name_subregistry(self, name: str, subregistry: Optional[str] = None) -> Tuple[str, SubRegistry]:
        if ':' in name:
            if name.startswith(':') or name.endswith(':') or len(name.split(':')) > 2:
                raise RegistryException(f'invalid name `{name}`')

            name, subregistry = name.split(':')
        elif subregistry is None:
            raise RegistryException('subregistry is not defined')

        return name, self._registries[subregistry]

    def add_subregistry(self, key: str, **kwargs):
        new_subregistry = SubRegistry(default_name_key=key, **kwargs)
        self[key] = new_subregistry

        return new_subregistry

    def __len__(self) -> int:
        return sum(len(value) for key, value in self._registries.items())

    def __getitem__(self, key: str) -> SubRegistry:
        return self._registries[key]

    def __setitem__(self, key: str, value: SubRegistry):
        if key in self._registries and self._registries[key].all():
            raise RegistryException(
                "Multiple subregistries with single name are not allowed"
            )

        self._registries[key] = value

    def __delitem__(self, key: str) -> None:
        self._registries.pop(key)

    # def add(self, factory: Factory = None, *factories: Factory, name: str = None, **named_factories: Factory) -> Factory:

    def late_add(self, key: str, cb: LateAddCallbak):
        return self._registries[key].late_add(cb)

    def add_from_module(
        self, key: str, module, prefix: Union[str, List[str]] = None
    ) -> None:
        return self._registries[key].add_from_module(module=module, prefix=prefix)

    def get(self, name: str, subregistry: Optional[str] = None) -> Optional[Factory]:
        name, subregistry_ = self._prepare_name_subregistry(name=name, subregistry=subregistry)
        return subregistry_.get(name)

    def get_instance(self, name: str, *args, subregistry: Optional[str] = None, meta_factory=None, **kwargs):
        name, subregistry_ = self._prepare_name_subregistry(name=name, subregistry=subregistry)
        return subregistry_.get_instance(name, *args, meta_factory=meta_factory, **kwargs)

    def get_from_params(self, *, meta_factory=None, **kwargs):
        subregistry_name = kwargs.pop('subregistry')
        if subregistry_name:
            subregistry = self._registries[subregistry_name]
        else:
            common_keys = set(self._registries.keys()) & set(kwargs.keys())

            if len(common_keys) != 1:
                raise RegistryException('')

            name_key = next(iter(common_keys))
            name, subregistry = self._prepare_name_subregistry(kwargs[name_key])
            kwargs[name_key] = name

        subregistry.get_from_params(meta_factory=meta_factory, **kwargs)
