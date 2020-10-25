import pytest
from catalyst.metrics import f1_score, fbeta_score, precision, recall


def test_precision_recall_f_binary_single_class() -> None:
    # Test precision, recall and F-scores behave with a single positive
    assert 1. == precision([1, 1], [1, 1])
    assert 1. == recall([1, 1], [1, 1])
    assert 1. == f1_score([1, 1], [1, 1])
    assert 1. == fbeta_score([1, 1], [1, 1], 0)