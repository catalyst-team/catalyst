import os

import pytest  # noqa: F401
from torch import nn

from catalyst.utils.onnx import convert_to_onnx


def test_api():
    """Tests if API is working. Minimal example."""
    model = nn.Sequential(nn.Linear(768, 128), nn.ReLU(), nn.Linear(128, 10))
    convert_to_onnx(model, input_shape=(1, 768))
    assert os.path.isfile("model.onnx")
    os.remove("model.onnx")
