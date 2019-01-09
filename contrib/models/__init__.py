from typing import List, Union
from catalyst.utils import Factory
from .resnet_encoder import ResnetEncoder
from .sequential import SequentialNet
from . import segmentation

__all__ = ["ResnetEncoder", "SequentialNet"]

MODELS = {
    **{
        "ResnetEncoder": ResnetEncoder,
        "SequentialNet": SequentialNet
    },
    **segmentation.__dict__
}


def register_model(
    *models_factories: Factory
) -> Union[Factory, List[Factory]]:
    """Add model type or factory method to global
        model list to make it available in config
        Can be called or used as decorator
        :param: models_factories Required model factory (method or type)
        :returns: single model factory or list of them
    """

    for mf in models_factories:
        MODELS[mf.__name__] = mf

    if len(models_factories) == 1:
        return models_factories[0]
    return models_factories
