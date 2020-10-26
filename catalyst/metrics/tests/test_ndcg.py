import torch

from catalyst.utils import metrics


def test_zero_ndcg():
    """
    Tests for catalyst.utils.metrics.ndcg metric.
    """
    y_pred = [6, 5, 4, 3, 2, 1, 0]
    y_true = [0, 0, 0, 0, 0, 0, 1]

    ndcg_at3 = metrics.ndcg(
        torch.tensor([y_pred]),
        torch.tensor([y_true]),
        k=3
    )

    ndcg_at1 = metrics.ndcg(
        torch.tensor([y_pred]),
        torch.tensor([y_true]),
        k=1
    )

    ndcg_at7 = metrics.ndcg(
        torch.tensor([y_pred]),
        torch.tensor([y_true]),
        k=7
    )

    assert torch.isclose(ndcg_at1, torch.tensor(0.0))
    assert torch.isclose(ndcg_at3, torch.tensor(0.0))
    assert torch.isclose(ndcg_at7, torch.tensor(0.3), atol=0.05)


def test_sample_ndcg():
    y_pred = [0.5, 0.2, 0.1]
    y_true = [1.0, 0.0, 1.0]

    outputs = torch.Tensor([y_pred])
    targets = torch.Tensor([y_true])

    ndcg_at_2 = torch.tensor(1.0 / (1.0 + 1 / math.log2(3)))

    assert torch.isclose(ndcg_at_2, metrics.ndcg(outputs, targets, k=2))