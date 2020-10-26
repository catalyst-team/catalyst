import torch

from catalyst.utils import metrics


def test_zero_ndcg():
    """
    Tests for catalyst.utils.metrics.ndcg metric.
    """
    ndcg_at3 = metrics.ndcg(
        torch.tensor([6, 5, 4, 3, 2, 1, 0]),
        torch.tensor([0, 0, 0, 0, 0, 0, 1]),
        k=3
    )

    ndcg_at7 = metrics.ndcg(
        torch.tensor([6, 5, 4, 3, 2, 1, 0]),
        torch.tensor([0, 0, 0, 0, 0, 0, 1]),
        k=1
    )

    ndcg_at1 = metrics.ndcg(
        torch.tensor([6, 5, 4, 3, 2, 1, 0]),
        torch.tensor([0, 0, 0, 0, 0, 0, 1]),
        k=1
    )

    assert torch.isclose(ndcg_at1, torch.tensor(0.0))
    assert torch.isclose(ndcg_at3, torch.tensor(0.0))
    assert torch.isclose(ndcg_at7, torch.tensor(3.0))

def test_zero_ndcg():
    """
    Tests for catalyst.utils.metrics.ndcg metric.
    """
    ndcg_at1, ndcg_at3, ndcg_at7 = metrics.ndcg(
        torch.tensor([6, 5, 4, 3, 2, 1, 0]),
        torch.tensor([0, 0, 0, 0, 0, 0, 1]),
        topk=(1, 3, 7),
    )
    assert torch.isclose(ndcg_at1, torch.tensor(0.0))
    assert torch.isclose(ndcg_at3, torch.tensor(0.0))
    assert torch.isclose(ndcg_at7, torch.tensor(3.0))


def test_ndcg_ordering_invariance():
    """
    Tests for catalyst.utils.metrics.ndcg metric.
    """
    [in_order] = metrics.ndcg(
        torch.tensor([2, 1, 0]), torch.tensor([1, 0, 0]), topk=(1,)
    )
    [first_last] = metrics.ndcg(
        torch.tensor([0, 1, 2]), torch.tensor([0, 0, 1]), topk=(1,)
    )
    [first_middle] = metrics.ndcg(
        torch.tensor([1, 2, 0]), torch.tensor([0, 1, 0]), topk=(1,)
    )
    assert torch.isclose(in_order, first_last)
    assert torch.isclose(first_last, first_middle)
