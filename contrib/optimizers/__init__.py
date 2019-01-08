from typing import List, Union
import torch

from catalyst.utils import Factory

OPTIMIZERS = {
    **torch.optim.__dict__,
}


def register_optimizer(*optimizer_factories: Factory
                       ) -> Union[Factory, List[Factory]]:
    """Add optimizer type or factory method to global
        optimizer list to make it available in config
        Can be called or used as decorator
        :param: optimizer_factories Required optimizer factory (method or type)
        :returns: single optimizer factory or list of them
    """

    for cf in optimizer_factories:
        OPTIMIZERS[cf.__name__] = cf

    if len(optimizer_factories) == 1:
        return optimizer_factories[0]
    return optimizer_factories
