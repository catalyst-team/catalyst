import os

import torch
from torch import nn

from catalyst.utils.onnx import onnx_export


def test_api():
    """Tests if API is working. Minimal example."""
    model = nn.Sequential(nn.Linear(768, 128), nn.ReLU(), nn.Linear(128, 10))
    onnx_export(model, batch=torch.randn((1, 768)), file="model.onnx")
    assert os.path.isfile("model.onnx")
    os.remove("model.onnx")
