from typing import Optional, Union
import logging

import torch
from torch import quantization

from catalyst.settings import IS_ONNX_AVAILABLE

if IS_ONNX_AVAILABLE:
    from catalyst.utils.onnx import quantize_onnx_model  # noqa: F401


def quantize_model(
        model,
        qconfig_spec=None,
        dtype: Union[str, Optional[torch.dtype]] = "qint8",
):
    if isinstance(dtype, str):
        type_mapping = {
            "qint8": torch.qint8,
            "quint8": torch.quint8
        }
    quantized_model = quantization.quantize_dynamic(
        model.cpu(), qconfig_spec=qconfig_spec, dtype=type_mapping[dtype],
    )
    return quantized_model


__all__ = [
    "quantize_model",
]

if IS_ONNX_AVAILABLE:
    __all__.append("quantize_onnx_model")
