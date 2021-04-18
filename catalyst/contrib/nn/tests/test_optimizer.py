from torch import nn, optim

from catalyst.contrib.nn import optimizers as module
from catalyst.contrib.nn.optimizers import Lookahead


def test_optimizer_init():
    """@TODO: Docs. Contribution is welcome."""
    model = nn.Linear(10, 10)
    for name, module_class in module.__dict__.items():
        if isinstance(module_class, type):
            if name in ["Optimizer", "SparseAdam"]:
                # @TODO: add test for SparseAdam
                continue
            elif module_class == Lookahead:
                instance = optim.SGD(model.parameters(), lr=1e-3)
                instance = module_class(instance)
            else:
                instance = module_class(model.parameters(), lr=1e-3)
            assert instance is not None
