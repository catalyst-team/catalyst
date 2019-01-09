from typing import Type, List, Union, Callable


Factory = Union[Type, Callable]


def register(register_type: str):
    """Add object type or factory method to global
        object list to make it available in config
        Can be called or used as decorator.
        :param register_type: one of
        ['optimizers', 'criterion', 'callbacks']
        :returns: callable/decorator
        which requires object_factory (method or type)
    """
    # hack to prevent cycle imports
    from catalyst.dl import callbacks
    from catalyst.contrib import optimizers, criterion, models

    _REGISTERS = {
        "callback": callbacks.CALLBACKS,
        "model": models.MODELS,
        "optimizer": optimizers.OPTIMIZERS,
        "criterion": criterion.CRITERION
    }

    def inner_register(
        *object_factories: Factory
    ) -> Union[Factory, List[Factory]]:

        for factory in object_factories:
            registers = _REGISTERS[register_type]
            registers[factory.__name__] = factory

        if len(object_factories) == 1:
            return object_factories[0]
        return object_factories

    return inner_register
