import torch
import torch.nn as nn
from collections import OrderedDict
from itertools import tee


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class SequentialNet(nn.Module):
    def __init__(
            self, hiddens,
            layer_fn=nn.Linear, bias=True,
            norm_fn=None,
            activation_fn=nn.ReLU,
            dropout=None):
        super().__init__()
        # hack to prevent cycle imports
        from catalyst.modules.modules import name2nn

        layer_fn = name2nn(layer_fn)
        activation_fn = name2nn(activation_fn)
        norm_fn = name2nn(norm_fn)

        net = []

        for i, (f_in, f_out) in enumerate(pairwise(hiddens)):
            net.append((f"layer_{i}", layer_fn(f_in, f_out, bias=bias)))
            if norm_fn is not None:
                net.append((f"norm_{i}", norm_fn(f_out)))
            if dropout is not None:
                net.append((f"drop_{i}", nn.Dropout(dropout)))
            if activation_fn is not None:
                net.append((f"activation_{i}", activation_fn()))

        self.net = torch.nn.Sequential(OrderedDict(net))

    def forward(self, x):
        x = self.net.forward(x)
        return x
