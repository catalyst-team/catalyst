import os

import pytest  # noqa

import torch

from catalyst.utils.quantization import quantize_model
from catalyst.contrib.nn import Flatten


def test_api():
    model = torch.nn.Sequential(
        Flatten(),
        torch.nn.Linear(28*28, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.Linear(64, 10)
    )
    q_model = quantize_model(model)
    torch.save(model.state_dict(), "model.pth")
    torch.save(q_model.state_dict(), "q_model.pth")
    model_size = os.path.getsize("model.pth")
    q_model_size = os.path.getsize("q_model.pth")
    assert q_model_size * 3.8 < model_size
    os.remove("model.pth")
    os.remove("q_model.pth")
