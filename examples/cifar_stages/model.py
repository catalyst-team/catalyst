import torch
import torch.nn as nn
import torch.nn.functional as F
from catalyst.dl.runner import ClassificationRunner
from catalyst.contrib.registry import Registry


@Registry.model
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ModelRunner(ClassificationRunner):
    @staticmethod
    def prepare_stage_model(*, model, stage, **kwargs):
        ClassificationRunner.prepare_stage_model(
            model=model, stage=stage, **kwargs
        )
        model_ = model
        if isinstance(model, torch.nn.DataParallel):
            model_ = model_.module

        if stage == "stage2":
            for key in ["conv1", "pool", "conv2"]:
                layer = getattr(model_, key)
                for param in layer.parameters():
                    param.requires_grad = False
