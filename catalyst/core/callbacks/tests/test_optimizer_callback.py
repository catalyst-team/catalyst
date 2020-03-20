from catalyst.dl.experiment.supervised import SupervisedExperiment
from catalyst.dl.runner import SupervisedRunner
import torch.nn as nn
import torch.optim as optim

from ._model import SimpleNet
from ..optimizer import OptimizerCallback
from ...callbacks.criterion import CriterionCallback


def test_save_model_grads():
    runner = SupervisedRunner()
    model = SimpleNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    runner.train()
    callbacks = [CriterionCallback(), OptimizerCallback()]
