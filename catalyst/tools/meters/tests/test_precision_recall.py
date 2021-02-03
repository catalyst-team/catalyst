import pytest

from catalyst.tools.meters.ppv_tpr_f1_meter import precision, recall

EPS = 1e-5


@pytest.mark.parametrize(
    "tp,fp,zero_division,true_precision",
    (
        (4, 0, 0, 1),
        (0, 3, 0, 0),
        (4, 0, 1, 1),
        (0, 3, 1, 0),
        (0, 0, 0, 0),
        (0, 0, 1, 1),
        (1, 1, 0, 0.5),
    ),
)
def test_precision(tp, fp, zero_division, true_precision):
    assert (
        abs(
            precision(tp=tp, fp=fp, zero_division=zero_division)
            - true_precision
        )
        < EPS
    )


@pytest.mark.parametrize(
    "tp,fn,zero_division,true_recall",
    (
        (4, 0, 0, 1),
        (0, 3, 0, 0),
        (4, 0, 1, 1),
        (0, 3, 1, 0),
        (0, 0, 0, 0),
        (0, 0, 1, 1),
        (1, 1, 0, 0.5),
    ),
)
def test_recall(tp, fn, zero_division, true_recall):
    assert (
        abs(recall(tp=tp, fn=fn, zero_division=zero_division) - true_recall)
        < EPS
    )
