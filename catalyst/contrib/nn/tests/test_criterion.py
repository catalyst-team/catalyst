from catalyst.contrib.nn import criterion as module
from catalyst.contrib.nn.criterion import CircleLoss


def test_criterion_init():
    """@TODO: Docs. Contribution is welcome."""
    for module_class in module.__dict__.values():
        if isinstance(module_class, type):
            if module_class == CircleLoss:
                instance = module_class(margin=0.25, gamma=256)
            else:
                instance = module_class()
            assert instance is not None
