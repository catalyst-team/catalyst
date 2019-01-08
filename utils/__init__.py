from typing import Type, Union, Callable, List

from catalyst.contrib import optimizers, criterion, models
from catalyst.dl import callbacks

_REGISTERS = {
    "optimizer": optimizers.OPTIMIZERS,
    "criterion": criterion.CRITERION,
    "callback": callbacks.CALLBACKS
}

Factory = Union[Type, Callable]


def register(register_type: str):
    """Add object type or factory method to global
        object list to make it available in config
        Can be called or used as decorator.
        :param register_type: one of ['optimizers', 'criterion', 'callbacks']
        :returns: callable/decorator which requires object_factory (method or type)
    """

    def inner_register(*object_factories: Factory
                       ) -> Union[Factory, List[Factory]]:

        for cf in object_factories:
            registers = _REGISTERS[register_type]
            registers[cf.__name__] = cf

        if len(object_factories) == 1:
            return object_factories[0]
        return object_factories

    return inner_register
