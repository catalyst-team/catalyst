from catalyst.contrib.nn import criterion as module
from catalyst.contrib.nn.criterion import CircleLoss


def test_criterion_init():
    """@TODO: Docs. Contribution is welcome."""
    for cls in module.__dict__.values():
        if isinstance(cls, type):
            if cls == CircleLoss:
                instance = cls(margin=0.25, gamma=256)
            else:
                instance = cls()
            assert instance is not None
