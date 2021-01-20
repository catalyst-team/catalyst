import torch

from catalyst import metrics
from catalyst.metrics.functional import wrap_topk_metric2dict


def test_reciprocal_rank():
    """
    Tests for catalyst.metrics.mrr metric.
    """
    # # check 0 simple case
    y_pred = [0.5, 0.2]
    y_true = [1.0, 0.0]
    k = 2
    value = metrics.reciprocal_rank(torch.Tensor([y_pred]), torch.Tensor([y_true]), k)
    assert value[0][0] == 1

    # check 1 simple case
    y_pred = [0.5, 0.2]
    y_true = [0.0, 1.0]
    k = 2

    value = metrics.reciprocal_rank(torch.Tensor([y_pred]), torch.Tensor([y_true]), k)

    assert value[0][0] == 0.5

    # check 2 simple case
    y_pred = [0.2, 0.5]
    y_true = [0.0, 1.0]
    k = 2

    value = metrics.reciprocal_rank(torch.Tensor([y_pred]), torch.Tensor([y_true]), k)
    assert value[0][0] == 1.0

    # check 3 test multiple users
    y_pred1 = [0.2, 0.5]
    y_pred05 = [0.5, 0.2]
    y_true = [0.0, 1.0]
    k = 2

    value = metrics.reciprocal_rank(
        torch.Tensor([y_pred1, y_pred05]), torch.Tensor([y_true, y_true]), k
    )
    assert value[0][0] == 1.0
    assert value[1][0] == 0.5

    # check 4 test with k
    y_pred1 = [4.0, 2.0, 3.0, 1.0]
    y_pred2 = [1.0, 2.0, 3.0, 4.0]
    y_true1 = [0, 0, 1.0, 1.0]
    y_true2 = [0, 0, 1.0, 1.0]
    k = 3

    y_pred_torch = torch.Tensor([y_pred1, y_pred2])
    y_true_torch = torch.Tensor([y_true1, y_true2])

    value = metrics.reciprocal_rank(y_pred_torch, y_true_torch, k=k)

    assert value[0][0] == 0.5
    assert value[1][0] == 1.0

    # check 5 test with k

    y_pred1 = [4.0, 2.0, 3.0, 1.0]
    y_pred2 = [1.0, 2.0, 3.0, 4.0]
    y_true1 = [0, 0, 1.0, 1.0]
    y_true2 = [0, 0, 1.0, 1.0]
    k = 1

    y_pred_torch = torch.Tensor([y_pred1, y_pred2])
    y_true_torch = torch.Tensor([y_true1, y_true2])

    value = metrics.reciprocal_rank(y_pred_torch, y_true_torch, k=k)

    assert value[0][0] == 0.0
    assert value[1][0] == 1.0


def test_mrr():
    """
    Test mrr
    """
    y_pred1 = [4.0, 2.0, 3.0, 1.0]
    y_pred2 = [1.0, 2.0, 3.0, 4.0]
    y_true1 = [0, 0, 1.0, 1.0]
    y_true2 = [0, 0, 1.0, 1.0]
    k_list = [1, 3]

    y_pred_torch = torch.Tensor([y_pred1, y_pred2])
    y_true_torch = torch.Tensor([y_true1, y_true2])

    mrr_results = metrics.mrr(y_pred_torch, y_true_torch, k_list)

    mrr_at1 = mrr_results[0]
    mrr_at3 = mrr_results[1]

    assert mrr_at1 == 0.5
    assert mrr_at3 == 0.75


def wrapper_mrr():
    """
    Tets wrapper
    """
    y_pred1 = [4.0, 2.0, 3.0, 1.0]
    y_pred2 = [1.0, 2.0, 3.0, 4.0]
    y_true1 = [0, 0, 1.0, 1.0]
    y_true2 = [0, 0, 1.0, 1.0]

    topk_args = [1, 3]

    output = torch.Tensor([y_pred1, y_pred2])
    target = torch.Tensor([y_true1, y_true2])

    mrr_wrapper = wrap_topk_metric2dict(metrics.mrr, topk_args)
    mrr_dict = mrr_wrapper(output, target)

    mrr_at1 = mrr_dict["01"]
    mrr_at3 = mrr_dict["03"]

    assert mrr_at1 == 0.5
    assert mrr_at3 == 0.75
