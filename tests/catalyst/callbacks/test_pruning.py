# flake8: noqa
import numpy as np
import pytest
import torch
from torch import nn

from catalyst import dl
from catalyst.settings import SETTINGS

if SETTINGS.pruning_required:
    from torch.nn.utils.prune import l1_unstructured

    from catalyst.dl import PruningCallback


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


@pytest.mark.skipif(not SETTINGS.pruning_required, reason="torch version too low")
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
        callbacks=[PruningCallback(l1_unstructured, amount=0.5)],
        num_epochs=1,
    )
    assert np.isclose(pruning_factor(model), 0.5)


@pytest.mark.skipif(not SETTINGS.pruning_required, reason="torch version too low")
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
            PruningCallback(
                l1_unstructured, amount=0.5, remove_reparametrization_on_stage_end=False
            )
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


@pytest.mark.skipif(not SETTINGS.pruning_required, reason="torch version too low")
def test_pruning_str_unstructured():
    dataloader = prepare_experiment()
    model = nn.Linear(100, 10, bias=False)
    runner = dl.SupervisedRunner()
    criterion = nn.CrossEntropyLoss()
    runner.train(
        model=model,
        optimizer=torch.optim.Adam(model.parameters()),
        criterion=criterion,
        loaders={"train": dataloader},
        callbacks=[PruningCallback("l1_unstructured", amount=0.5)],
        num_epochs=1,
    )
    assert np.isclose(pruning_factor(model), 0.5)


@pytest.mark.skipif(not SETTINGS.pruning_required, reason="torch version too low")
def test_pruning_str_structured():
    dataloader = prepare_experiment()
    model = nn.Linear(100, 10, bias=False)
    runner = dl.SupervisedRunner()
    criterion = nn.CrossEntropyLoss()
    runner.train(
        model=model,
        optimizer=torch.optim.Adam(model.parameters()),
        criterion=criterion,
        loaders={"train": dataloader},
        callbacks=[PruningCallback("ln_structured", amount=0.5, dim=1, l_norm=2)],
        num_epochs=1,
    )
    assert np.isclose(pruning_factor(model), 0.5)


@pytest.mark.skipif(not SETTINGS.pruning_required, reason="torch version too low")
@pytest.mark.xfail(raises=Exception)
def test_pruning_str_structured_f():
    dataloader = prepare_experiment()
    model = nn.Linear(100, 10, bias=False)
    runner = dl.SupervisedRunner()
    criterion = nn.CrossEntropyLoss()
    runner.train(
        model=model,
        optimizer=torch.optim.Adam(model.parameters()),
        criterion=criterion,
        loaders={"train": dataloader},
        callbacks=[PruningCallback("ln_structured", amount=0.5, dim=1)],
        num_epochs=1,
    )
    assert np.isclose(pruning_factor(model), 0.5)


@pytest.mark.skipif(not SETTINGS.pruning_required, reason="torch version too low")
@pytest.mark.xfail(raises=Exception)
def test_pruning_str_random_structured_f():
    dataloader = prepare_experiment()
    model = nn.Linear(100, 10, bias=False)
    runner = dl.SupervisedRunner()
    criterion = nn.CrossEntropyLoss()
    runner.train(
        model=model,
        optimizer=torch.optim.Adam(model.parameters()),
        criterion=criterion,
        loaders={"train": dataloader},
        callbacks=[PruningCallback("random_structured", amount=0.5)],
        num_epochs=1,
    )
    assert np.isclose(pruning_factor(model), 0.5)
