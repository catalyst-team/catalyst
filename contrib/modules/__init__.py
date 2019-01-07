import torch.nn as nn
from . import common
from . import pooling
from . import noisy

MODULES = {
    **nn.__dict__,
    **common.__dict__,
    **pooling.__dict__,
    **noisy.__dict__
}


def name2nn(name):
    if name is None:
        return None
    elif isinstance(name, nn.Module):
        return name
    elif isinstance(name, str):
        return MODULES[name]
    else:
        return name
