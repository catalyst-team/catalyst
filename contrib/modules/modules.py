import torch.nn as nn
import catalyst.contrib.modules.common as common
import catalyst.contrib.modules.pooling as pooling
import catalyst.contrib.modules.noisy as noisy
import catalyst.contrib.models.sequential as sequential

MODULES = {
    **nn.__dict__,
    **common.__dict__,
    **pooling.__dict__,
    **noisy.__dict__,
    **sequential.__dict__
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
