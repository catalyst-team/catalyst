from ..hash import get_hash


def test_hash():
    a = get_hash({"a": "foo"})
    print(a)
