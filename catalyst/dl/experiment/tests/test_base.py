from collections import OrderedDict

import torch

from catalyst.dl.experiment.base import BaseDLExperiment


def test_defaults():
    """
    Test on defaults for BaseDLExperiment. It will be useful if we decide to
    change anything in those values as it could make breaking change.
    """
    model = torch.nn.Module()
    dataset = torch.utils.data.Dataset()
    dataloader = torch.utils.data.DataLoader(dataset)
    loaders = OrderedDict()
    loaders["train"] = dataloader

    exp = BaseDLExperiment(model=model, loaders=loaders)

    assert exp.initial_seed == 42
    assert exp.logdir is None
    assert exp.stages == ["train"]
    assert exp.distributed_params == {}
    assert exp.monitoring_params == {}
    assert exp.get_state_params("") == {
        "logdir": None,
        "num_epochs": 1,
        "valid_loader": "valid",
        "main_metric": "loss",
        "verbose": False,
        "minimize_metric": True,
        "checkpoint_data": {},
    }
    assert exp.get_model("") == model
    assert exp.get_criterion("") is None
    assert exp.get_optimizer("", model) is None
    assert exp.get_scheduler("") is None
    assert exp.get_callbacks("") == OrderedDict()
    assert exp.get_loaders("") == loaders
