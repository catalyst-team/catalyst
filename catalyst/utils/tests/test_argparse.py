# flake8: noqa
from catalyst.utils.argparse import args_are_not_none


def test_args_are_not_none():
    assert args_are_not_none(1, 2, 3, "")
    assert not args_are_not_none(-8, "", None, True)
    assert not args_are_not_none(None)
