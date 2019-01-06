import torch
from typing import List, Union
from catalyst.utils import Factory

from . import unet as unet_loss
from . import center_loss
from . import contrastive as contrastive_loss
from . import huber as huber_loss
from . import ce
from . import bcece
from . import focal_loss
from . import dice

CRITERION = {
    **torch.nn.__dict__,
    **unet_loss.__dict__,
    **center_loss.__dict__,
    **contrastive_loss.__dict__,
    **huber_loss.__dict__,
    **ce.__dict__,
    **bcece.__dict__,
    **focal_loss.__dict__,
    **dice.__dict__,
}


def register_criterion(*criterion_factories: Factory
                       ) -> Union[Factory, List[Factory]]:
    """Add criterion type or factory method to global
        criterion list to make it available in config
        Can be called or used as decorator
        :param: criterion_factories Required criterion factory (method or type)
        :returns: single criterion factory or list of them
    """

    for cf in criterion_factories:
        CRITERION[cf.__name__] = cf

    if len(criterion_factories) == 1:
        return criterion_factories[0]
    return criterion_factories
