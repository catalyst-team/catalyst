import pytest
from catalyst.utils import pandas


def test_folds_to_list():
    assert pandas.folds_to_list("1,2,1,3,4,2,4,6") == [1, 2, 3, 4, 6]
    assert pandas.folds_to_list([1, 2, 3.0, 5, 2, 1]) == [1, 2, 3, 5]
    assert pandas.folds_to_list([]) == []

    with pytest.raises(ValueError):
        pandas.folds_to_list([1, "True", 3.0, None, 2, 1])
