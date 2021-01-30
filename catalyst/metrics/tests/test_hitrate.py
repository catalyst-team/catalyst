import numpy as np
import torch

from catalyst.metrics.functional import wrap_topk_metric2dict
from catalyst.metrics.hitrate import hitrate


def test_hitrate():
    """
    Tests for catalyst.metrics.hitrate metric.
    """
    y_pred = [0.5, 0.2]
    y_true = [1.0, 0.0]
    k = [1, 2]

    hitrate_at1, hitrate_at2 = hitrate(torch.Tensor([y_pred]), torch.Tensor([y_true]), k)
    assert hitrate_at1 == 1.0
    assert hitrate_at2 == 0.5

    # check 1 simple case
    y_pred = [0.5, 0.2]
    y_true = [0.0, 0.0]
    k = [2]

    hitrate_at2 = hitrate(torch.Tensor([y_pred]), torch.Tensor([y_true]), k)[0]
    assert hitrate_at2 == 0.0


def test_wrapper_metrics():
    """
    Tests for wrapper for metrics
    """
    y_pred = [0.5, 0.2]
    y_true = [1.0, 0.0]
    topk_args = [1, 2]

    outputs = torch.Tensor([y_pred])
    targets = torch.Tensor([y_true])

    hitrate_wrapper = wrap_topk_metric2dict(hitrate, topk_args)
    hitrate_dict = hitrate_wrapper(outputs, targets)

    hitrate_at1 = hitrate_dict["01"]
    hitrate_at2 = hitrate_dict["02"]

    true_hitrate_at1 = 1.0
    true_hitrate_at2 = 0.5
    assert np.isclose(true_hitrate_at1, hitrate_at1)
    assert np.isclose(true_hitrate_at2, hitrate_at2)
