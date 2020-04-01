from catalyst.contrib.nn import criterion as module


def test_criterion_init():
    """@TODO: Docs. Contribution is welcome."""
    for cls in module.__dict__.values():
        if isinstance(cls, type):
            instance = cls()
            assert instance is not None
