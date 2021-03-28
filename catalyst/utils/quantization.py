from typing import Dict, Optional, Union

import torch
from torch import quantization

from catalyst.typing import Model
from catalyst.utils.torch import get_nn_from_ddp_module


def quantize_model(
    model: Model, qconfig_spec: Dict = None, dtype: Union[str, Optional[torch.dtype]] = "qint8",
) -> Model:
    """Function to quantize model weights.

    Args:
        model: model to be quantized
        qconfig_spec (Dict, optional): quantization config in PyTorch format. Defaults to None.
        dtype (Union[str, Optional[torch.dtype]], optional): Type of weights after quantization.
            Defaults to "qint8".

    Returns:
        Model: quantized model
    """
    nn_model = get_nn_from_ddp_module(model)
    if isinstance(dtype, str):
        type_mapping = {"qint8": torch.qint8, "quint8": torch.quint8}
    try:
        quantized_model = quantization.quantize_dynamic(
            nn_model.cpu(), qconfig_spec=qconfig_spec, dtype=type_mapping[dtype],
        )
    except RuntimeError:
        torch.backends.quantized.engine = "qnnpack"
        quantized_model = quantization.quantize_dynamic(
            nn_model.cpu(), qconfig_spec=qconfig_spec, dtype=type_mapping[dtype],
        )

    return quantized_model


__all__ = ["quantize_model"]
