import torch
from typing import Set, Optional, Dict, Union

from catalyst.core import IRunner
from catalyst.core.callback import CallbackOrder, Callback
from torch import quantization


class DynamicQuantizationCallback(Callback):
    """
    Dynamic Quantization Caallback

    This callback applying dynamic quantization
    of the model at the end of the stage.

    >>> runner.model
    {"original": ..., "quantized": ...}
    """
    def __init__(
            self,
            qconfig_spec: Optional[Union[Set, Dict]] = None,
            dtype: Optional[torch.dtype] = torch.qint8
    ):
        """
        Init method for callback
        Args:
            qconfig_spec: torch.quantization.quantize_dynamic parameter
                you can define layers to be quantize
            dtype: type of the model parameters, default int8
        """
        super().__init__(CallbackOrder.External)
        self.qconfig_spec = qconfig_spec
        self.dtype = dtype

    def on_stage_end(self, runner: "IRunner") -> None:
        """
        On stage end action.
        We are applying quantization here
        Args:
            runner: runner of your exeriment
        """
        quantized_model = quantization.quantize_dynamic(
            runner.model.cpu(),
            qconfig_spec=self.qconfig_spec,
            dtype=self.dtype,
        )
        runner.model = {"original": runner.model.cpu(), "quantized": quantized_model}
