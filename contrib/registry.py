from typing import Type, Union, Callable, List

from catalyst.contrib import optimizers, criterion, models
from catalyst.dl import callbacks

Factory = Union[Type, Callable]


class Registry:
    _REGISTERS = {
        "optimizer": optimizers.OPTIMIZERS,
        "criterion": criterion.CRITERION,
        "callback": callbacks.CALLBACKS,
        "model": models.MODELS
    }

    @staticmethod
    def _inner_register(
        register_type: str, *object_factories: Factory
    ) -> Union[Factory, List[Factory]]:

        for factory in object_factories:
            registers = Registry._REGISTERS[register_type]
            registers[factory.__name__] = factory

        if len(object_factories) == 1:
            return object_factories[0]
        return object_factories

    @staticmethod
    def optimizer(
        *optimizer_factories: Factory
    ) -> Union[Factory, List[Factory]]:
        """Add optimizer type or factory method to global
            optimizer list to make it available in config
            Can be called or used as decorator
            :param: optimizer_factories
                Required optimizer factory (method or type)
            :returns: single optimizer factory or list of them
        """
        return Registry._inner_register("optimizer", *optimizer_factories)

    @staticmethod
    def criterion(
        *criterion_factories: Factory
    ) -> Union[Factory, List[Factory]]:
        """Add criterion type or factory method to global
            criterion list to make it available in config
            Can be called or used as decorator
            :param: criterion_factories
                Required criterion factory (method or type)
            :returns: single criterion factory or list of them
        """
        return Registry._inner_register("criterion", *criterion_factories)

    @staticmethod
    def callback(
        *callback_factories: Factory
    ) -> Union[Factory, List[Factory]]:
        """Add callback type or factory method to global
            callback list to make it available in config
            Can be called or used as decorator
            :param: callback_factories
                Required criterion factory (method or type)
            :returns: single callback factory or list of them
        """
        return Registry._inner_register("callback", *callback_factories)

    @staticmethod
    def model(*models_factories: Factory) -> Union[Factory, List[Factory]]:
        """Add model type or factory method to global
            model list to make it available in config
            Can be called or used as decorator
            :param: models_factories
                Required model factory (method or type)
            :returns: single model factory or list of them
        """
        return Registry._inner_register("model", *models_factories)
