# flake8: noqa
import numpy as np
import pytest

import torch
from torch import nn

from catalyst import dl

try:
    from torch.nn.utils.prune import l1_unstructured
    from catalyst.core import PruningCallback

    PRUNING_AVAILABLE = True
except ImportError:
    PRUNING_AVAILABLE = False


def pruning_factor(model):
    num_parameters_before = 0
    num_parameters_after = 0

    for _name, module in model.named_modules():
        try:
            n_pruned = int(torch.sum(module.weight == 0))
            num_parameters_before += module.weight.nelement()
            num_parameters_after += module.weight.nelement() - n_pruned
        except AttributeError:
            pass
    return num_parameters_after / num_parameters_before


def prepare_experiment():
    features = torch.randn((100, 100))
    labels = torch.distributions.Categorical(
        probs=torch.tensor([1 / 10 for _ in range(10)])
    ).sample((100,))
    dataset = torch.utils.data.TensorDataset(features, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)
    return dataloader


@pytest.mark.skipif(not PRUNING_AVAILABLE, reason="torch version too low")
def test_pruning():
    dataloader = prepare_experiment()
    model = nn.Linear(100, 10, bias=False)
    runner = dl.SupervisedRunner()
    criterion = nn.CrossEntropyLoss()
    runner.train(
        model=model,
        optimizer=torch.optim.Adam(model.parameters()),
        criterion=criterion,
        loaders={"train": dataloader},
        callbacks=[PruningCallback(l1_unstructured)],
        num_epochs=1,
    )
    assert np.isclose(pruning_factor(model), 0.5)


@pytest.mark.skipif(not PRUNING_AVAILABLE, reason="torch version too low")
def test_parametrization():
    dataloader = prepare_experiment()
    model = nn.Linear(100, 10, bias=False)
    runner = dl.SupervisedRunner()
    criterion = nn.CrossEntropyLoss()
    runner.train(
        model=model,
        optimizer=torch.optim.Adam(model.parameters()),
        criterion=criterion,
        loaders={"train": dataloader},
        callbacks=[
            PruningCallback(l1_unstructured, remove_reparametrization=False)
        ],
        num_epochs=1,
    )
    assert np.isclose(pruning_factor(model), 0.5)
    try:
        _mask = model.weight_mask
        mask_applied = True
    except AttributeError:
        mask_applied = False
    assert mask_applied
