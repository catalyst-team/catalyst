import torch

from catalyst.metrics.mrr import mrr
from catalyst.metrics.mrr import reciprocal_rank_at_k


def test_reciprocal_rank():
    """
    Tests for catalyst.metrics.mrr metric.
    """
    # # check 0 simple case
    y_pred = [0.5, 0.2]
    y_true = [1.0, 0.0]
    k = 2
    value = reciprocal_rank_at_k(
        torch.Tensor([y_pred]),
        torch.Tensor([y_true]),
        k
    )
    assert value[0][0] == 1

    # check 1 simple case
    y_pred = [0.5, 0.2]
    y_true = [0.0, 1.0]
    k = 2

    value = reciprocal_rank_at_k(
        torch.Tensor([y_pred]),
        torch.Tensor([y_true]),
        k
    )

    assert value[0][0] == 0.5


    # check 2 simple case
    y_pred = [0.2, 0.5]
    y_true = [0.0, 1.0]
    k = 2

    value = reciprocal_rank_at_k(
        torch.Tensor([y_pred]),
        torch.Tensor([y_true]),
        k
    )
    assert value[0][0] == 1.0

    # check 3 test multiple users
    y_pred1 = [0.2, 0.5]
    y_pred05 = [0.5, 0.2]
    y_true = [0.0, 1.0]
    k = 2

    value = reciprocal_rank_at_k(
        torch.Tensor([y_pred1, y_pred05]),
        torch.Tensor([y_true, y_true]),
        k
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

    value = reciprocal_rank_at_k(
        y_pred_torch,
        y_true_torch,
        k=k
    )

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

    value = reciprocal_rank_at_k(
        y_pred_torch,
        y_true_torch,
        k=1
    )

    assert value[0][0] == 0.0
    assert value[1][0] == 1.0


def test_mrr():
    '''
    Test mrr
    '''

    y_pred1 = [4.0, 2.0, 3.0, 1.0]
    y_pred2 = [1.0, 2.0, 3.0, 4.0]
    y_true1 = [0, 0, 1.0, 1.0]
    y_true2 = [0, 0, 1.0, 1.0]
    k_list = [1, 3]

    y_pred_torch = torch.Tensor([y_pred1, y_pred2])
    y_true_torch = torch.Tensor([y_true1, y_true2])

    mrr_at_1, mrr_at_3 = mrr(
        y_pred_torch,
        y_true_torch,
        k_list
    )

    assert mrr_at_1 == 0.5
    assert mrr_at_3 == 0.75
