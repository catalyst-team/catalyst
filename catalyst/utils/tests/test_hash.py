# flake8: noqa
from catalyst.utils.misc import get_hash


def test_hash():
    """Docs? Contribution is welcome."""
    a = get_hash({"a": "foo"})
    print(a)
