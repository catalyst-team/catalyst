from typing import Dict, Optional, Union

import torch
from torch import quantization

from catalyst.settings import IS_ONNX_AVAILABLE
from catalyst.typing import Model

if IS_ONNX_AVAILABLE:
    from catalyst.utils.onnx import quantize_onnx_model  # noqa: F401


def quantize_model(
    model: Model, qconfig_spec: Dict = None, dtype: Union[str, Optional[torch.dtype]] = "qint8",
) -> Model:
    """Function to quantize model weights.

    Args:
        model (Model): model to quantize
        qconfig_spec (Dict, optional): quantization config in PyTorch format. Defaults to None.
        dtype (Union[str, Optional[torch.dtype]], optional): Type of weights after quantization.
            Defaults to "qint8".

    Returns:
        Model: quantized model
    """
    if isinstance(dtype, str):
        type_mapping = {"qint8": torch.qint8, "quint8": torch.quint8}
    try:
        quantized_model = quantization.quantize_dynamic(
            model.cpu(), qconfig_spec=qconfig_spec, dtype=type_mapping[dtype],
        )
    except RuntimeError:
        torch.backends.quantized.engine = "qnnpack"
        quantized_model = quantization.quantize_dynamic(
            model.cpu(), qconfig_spec=qconfig_spec, dtype=type_mapping[dtype],
        )

    return quantized_model


__all__ = [
    "quantize_model",
]

if IS_ONNX_AVAILABLE:
    __all__.append("quantize_onnx_model")
