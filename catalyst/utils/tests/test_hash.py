from catalyst.utils.hash import get_hash


def test_hash():
    """@TODO: Docs. Contribution is welcome."""
    a = get_hash({"a": "foo"})
    print(a)
