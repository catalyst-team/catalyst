import torch

from catalyst.utils import metrics


def test_mrr():
    """
    Tests for catalyst.utils.metrics.mrr metric.
    """

    # check 0 simple case
    y_pred = [0.5, 0.2]
    y_true = [1.0, 0.0]

    mrr = metrics.mrr(torch.Tensor(y_pred), torch.Tensor(y_true))
    assert mrr == 1

    # check 1 simple case
    y_pred = [0.5, 0.2]
    y_true = [0.0, 1.0]

    mrr = metrics.mrr(torch.Tensor(y_pred), torch.Tensor(y_true))
    assert mrr == 0.5

    # check 2 simple case
    y_pred = [0.2, 0.5]
    y_true = [0.0, 1.0]

    mrr = metrics.mrr(torch.Tensor(y_pred), torch.Tensor(y_true))
    assert mrr == 1.0

    #test batched slates
    y_pred_1 = [0.2, 0.5]
    y_pred_05 = [0.5, 0.2]
    y_true = [0.0, 1.0]

    mrr = metrics.mrr(torch.Tensor([y_pred_1, y_pred_05]), torch.Tensor([y_true, y_true]))
    assert mrr[0][0] == 1.0
    assert mrr[1][0] == 0.5
