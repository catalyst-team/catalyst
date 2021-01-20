import math

import numpy as np
import torch

from catalyst import metrics
from catalyst.metrics.functional import wrap_topk_metric2dict


def test_dcg():
    """
    Tests for catalyst.metrics.dcg metric.
    Tests from Stanforf course
    """
    y_true = [2.0, 1.0, 2.0, 0.0]
    y_pred = np.arange(3, -1, -1)

    dcg_at4 = torch.sum(
        metrics.dcg(torch.tensor([y_pred]), torch.tensor([y_true]), gain_function="linear_rank",)
    )
    assert torch.isclose(dcg_at4, torch.tensor(4.261), atol=0.05)

    y_true = [2.0, 2.0, 1.0, 0.0]
    y_pred = np.arange(3, -1, -1)

    dcg_at4 = torch.sum(
        metrics.dcg(torch.tensor([y_pred]), torch.tensor([y_true]), gain_function="linear_rank",)
    )
    assert torch.isclose(dcg_at4, torch.tensor(4.631), atol=0.05)

    y_true = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    y_pred = np.arange(9, -1, -1)

    dcg_at10 = torch.sum(
        metrics.dcg(torch.tensor([y_pred]), torch.tensor([y_true]), gain_function="linear_rank",)
    )

    assert torch.isclose(dcg_at10, torch.tensor(9.61), atol=0.05)


def test_zero_ndcg():
    """
    Tests for catalyst.metrics.ndcg metric.
    """
    y_pred = [6, 5, 4, 3, 2, 1, 0]
    y_true = [0, 0, 0, 0, 0, 0, 1]

    ndcg_at1, ndcg_at3, ndcg_at7 = metrics.ndcg(
        torch.tensor([y_pred]), torch.tensor([y_true]), topk=[1, 3, 7]
    )

    assert torch.isclose(ndcg_at1, torch.tensor(0.0))
    assert torch.isclose(ndcg_at3, torch.tensor(0.0))
    assert torch.isclose(ndcg_at7, torch.tensor(0.3), atol=0.05)


def test_sample_ndcg():
    """
    Tests for catalyst.metrics.ndcg metric.
    """
    y_pred = [0.5, 0.2, 0.1]
    y_true = [1.0, 0.0, 1.0]

    outputs = torch.Tensor([y_pred])
    targets = torch.Tensor([y_true])

    true_ndcg_at2 = 1.0 / (1.0 + 1 / math.log2(3))
    comp_ndcg_at2 = metrics.ndcg(outputs, targets, topk=[2])[0]

    assert np.isclose(true_ndcg_at2, comp_ndcg_at2)

    y_pred1 = [0.5, 0.2, 0.1]
    y_pred2 = [0.5, 0.2, 0.1]
    y_true1 = [1.0, 0.0, 1.0]
    y_true2 = [1.0, 0.0, 1.0]

    outputs = torch.Tensor([y_pred1, y_pred2])
    targets = torch.Tensor([y_true1, y_true2])

    true_ndcg_at2 = 1.0 / (1.0 + 1 / math.log2(3))
    comp_ndcg_at2 = metrics.ndcg(outputs, targets, topk=[2])[0]

    assert np.isclose(true_ndcg_at2, comp_ndcg_at2)


def test_wrapper_metrics():
    """
    Tests for wrapper for metrics
    """
    y_pred1 = [0.5, 0.2, 0.1]
    y_pred2 = [0.5, 0.2, 0.1]
    y_true1 = [1.0, 0.0, 1.0]
    y_true2 = [1.0, 0.0, 1.0]

    outputs = torch.Tensor([y_pred1, y_pred2])
    targets = torch.Tensor([y_true1, y_true2])

    topk_args = [1, 2]
    ndcg_wrapper = wrap_topk_metric2dict(metrics.ndcg, topk_args)
    ndcg_dict = ndcg_wrapper(outputs, targets)

    ndcg_at1 = ndcg_dict["01"]
    ndcg_at2 = ndcg_dict["02"]

    true_ndcg_at1 = 1.0
    true_ndcg_at2 = 1.0 / (1.0 + 1 / math.log2(3))
    assert np.isclose(true_ndcg_at1, ndcg_at1)
    assert np.isclose(true_ndcg_at2, ndcg_at2)
