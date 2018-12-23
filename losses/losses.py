import torch
from typing import Type, List, Union, Callable

import catalyst.losses.unet as unet_loss
import catalyst.losses.center_loss as center_loss
import catalyst.losses.contrastive as contrastive_loss
import catalyst.losses.huber as huber_loss
import catalyst.losses.ce as ce
import catalyst.losses.focal_loss as focal_loss
import catalyst.losses.dice as dice

LOSSES = {
    **torch.nn.__dict__,
    **unet_loss.__dict__,
    **center_loss.__dict__,
    **contrastive_loss.__dict__,
    **huber_loss.__dict__,
    **ce.__dict__,
    **focal_loss.__dict__,
    **dice.__dict__,
}


def register_criterion(
        criterion_factory: Union[Type, Callable],
        *criterion_factories: Union[Type, Callable]
) -> Union[Union[Type, Callable], List[Union[Type, Callable]]]:
    """Add criterion type or factory method to global
        criterion list to make it available in config
        Can be called or used as decorator
        :param: criterion_factory Required criterion factory (method or type)
        :param: criterion_factories Some optional factories
        :returns: single criterion factory or list of them
    """
    criterion_factories = [criterion_factory, *criterion_factories]

    for cf in criterion_factories:
        LOSSES[cf.__name__] = cf

    if len(criterion_factories) == 1:
        return criterion_factory
    return criterion_factories
