# flake8: noqa
from catalyst import utils


def test_args_are_not_none():
    """@TODO: Docs. Contribution is welcome."""
    assert utils.args_are_not_none(1, 2, 3, "")
    assert not utils.args_are_not_none(-8, "", None, True)
    assert not utils.args_are_not_none(None)
