import torch

from catalyst.utils import meters
from catalyst.utils.meters.ppv_tpr_f1_meter import f1score, precision, recall


def precision_recall_f1(tp, fp, fn):
    """Calculates precision, recall, and f1 score.

    Args:
        tp: number of true positives
        fp: number of false positives
        fn: number of false negatives

    Returns:
        precision value (0-1), recall_value (0-1), f1score (0-1)
    """
    precision_value = round(precision(tp, fp), 3)
    recall_value = round(recall(tp, fn), 3)
    f1_value = round(f1score(precision_value, recall_value), 3)
    return (precision_value, recall_value, f1_value)


def test_precision_recall_f1score():
    """Sanity checks for the `precision`, `recall`, `f1score` functions."""
    # case 1
    tp, fp, fn = (10, 0, 0)
    ppv, tpr, f1 = precision_recall_f1(tp, fp, fn)
    assert ppv == tpr == f1 == 1, "No fp and fn means everything should be =1"

    # case 2
    tp, fp, fn = (0, 0, 0)
    ppv, tpr, f1 = precision_recall_f1(tp, fp, fn)
    assert (
        ppv == tpr == f1 == 1
    ), "No tp, fp and fn means there weren't any objects (everything =1)"

    # case 3
    tp, fp, fn = (10, 10, 10)
    ppv, tpr, f1 = precision_recall_f1(tp, fp, fn)
    assert ppv == tpr == 0.5, "Example where ppv and tpr should be =0.5."

    # case 4
    tp, fp, fn = (0, 1, 1)
    ppv, tpr, f1 = precision_recall_f1(tp, fp, fn)
    assert ppv == tpr == f1 == 0, "No tp means everything should be =0"


def create_dummy_tensors_single():
    """Binary: 1 actual, 1 predicted (tp: 1, fp: 0, fn: 0)."""
    label = torch.tensor([1])
    pred = torch.tensor([1])
    return (label, pred)


def create_dummy_tensors_batched(batch_size=16):
    """Binary: 1 actual, 1 predicted (tp: 1, fp: 0, fn: 0)."""
    label = torch.ones((batch_size, 1))
    pred = torch.ones((batch_size, 1))
    return (label, pred)


def create_dummy_tensors_seg(batch_size=16, channels=1):
    """Binary: 1 actual, 1 predicted (tp: 1, fp: 0, fn: 0)."""
    base_shape = (channels, 15, 15)
    label = torch.ones((batch_size,) + base_shape)
    pred = torch.ones((batch_size,) + base_shape)
    return (label, pred)


def runs_tests_on_meter_counts_and_value(meter, num_tp_check=16):
    """
    Tests the meter's counts and values (ppv, tpr, f1). Assumes there are no
    fp and fn (everything is tp).
    """
    counts_dict = meter.tp_fp_fn_counts
    assert counts_dict["tp"] == num_tp_check
    assert (
        counts_dict["fp"] == 0 and counts_dict["fn"] == 0
    ), "There should be no fp and fn for this test case."
    ppv, tpr, f1 = meter.value()
    ppv, tpr, f1 = map(lambda x: round(x, 3), [ppv, tpr, f1])
    assert (
        ppv == tpr == f1 == 1
    ), "No fp and fn means that all metrics should be =1."


def test_meter():
    """
    Tests:
        * .reset()
        * .add()
        * .value()
    """
    meter = meters.PrecisionRecallF1ScoreMeter()
    # tests the .reset() method, which happens to be called in initialization
    for key in ["tp", "fp", "fn"]:
        assert (
            meter.tp_fp_fn_counts[key] == 0
        ), "Counts should be initialized to 0."

    # testing .add() and .value() with tensors w/no batch size dim
    binary_y, binary_pred = create_dummy_tensors_single()
    meter.add(binary_pred, binary_y)
    runs_tests_on_meter_counts_and_value(meter, num_tp_check=1)

    # testing .add() and .value() with tensors w/the batch size dim
    meter.reset()
    batch_size = 16
    binary_y, binary_pred = create_dummy_tensors_batched(batch_size)
    meter.add(binary_pred, binary_y)
    runs_tests_on_meter_counts_and_value(meter, num_tp_check=batch_size)

    # testing with seg; shape (batch_size, n_channels, h, w)
    meter.reset()
    batch_size = 16
    binary_y, binary_pred = create_dummy_tensors_seg(batch_size)
    meter.add(binary_pred, binary_y)
    runs_tests_on_meter_counts_and_value(
        meter, num_tp_check=batch_size * 15 * 15
    )
