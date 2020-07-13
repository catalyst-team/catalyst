import numpy as np

import torch

from catalyst.utils import metrics


def test_ndcg():
    """
    Tests for catalyst.utils.metrics.ndcg metric.
    """
    
    # check 0: common values
    assert metrics.ndcg(torch.tensor([2, 1, 0]), torch.tensor([1, 0, 0])) == 1.0
    assert np.allclose(metrics.ndcg(torch.tensor([2, 1, 0]), torch.tensor([0, 1, 0])), 0.63093)
    assert np.allclose(metrics.ndcg(torch.tensor([2, 1, 0]), torch.tensor([0, 0, 1])), 0.50000)
    assert np.allclose(metrics.ndcg(torch.tensor([2, 1, 0]), torch.tensor([1, 0, 1])), 0.91972)
    assert np.allclose(metrics.ndcg(torch.tensor([2, 1, 0]), torch.tensor([1, 0, 1]), k=2), 0.61315)
    
    # check 1: ordering invariance
    assert np.allclose(metrics.ndcg(torch.tensor([2, 1, 0]), torch.tensor([0, 0, 1])), 
                    metrics.ndcg(torch.tensor([0, 1, 2]), torch.tensor([1, 0, 0])))
    assert np.allclose(metrics.ndcg(torch.tensor([2, 1, 0]), torch.tensor([0, 0, 1])), 
                    metrics.ndcg(torch.tensor([2, 0, 1]), torch.tensor([0, 1, 0])))
    
    # check2: zero ndcg
    assert metrics.ndcg(torch.tensor([2, 1, 0]), torch.tensor([0, 0, 0])) == 0.0
    assert metrics.ndcg(torch.tensor([2, 1, 0]), torch.tensor([0, 0, 1]), k=2) == 0.0