import pytest
from catalyst.utils import parse


def test_folds_to_list():
    assert parse.folds_to_list("1,2,1,3,4,2,4,6") == [1, 2, 3, 4, 6]
    assert parse.folds_to_list([1, 2, 3.0, 5, 2, 1]) == [1, 2, 3, 5]
    assert parse.folds_to_list([]) == []

    with pytest.raises(ValueError):
        parse.folds_to_list([1, "True", 3.0, None, 2, 1])
