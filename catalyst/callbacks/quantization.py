from typing import Dict, Optional, TYPE_CHECKING, Union
from pathlib import Path

import torch

from catalyst.core import Callback, CallbackNode, CallbackOrder
from catalyst.utils import quantize_model

if TYPE_CHECKING:
    from catalyst.core import IRunner


class QuantizationCallback(Callback):
    """
    Callback for model quantiztion.

    Args:
        logdir: path to folder for saving
        filename: filename
        qconfig_spec (Dict, optional): quantization config in PyTorch format. Defaults to None.
        dtype (Union[str, Optional[torch.dtype]], optional): Type of weights after quantization.
            Defaults to "qint8".
    """

    def __init__(
        self,
        logdir: Union[str, Path] = None,
        filename: str = "quantized.pth",
        qconfig_spec: Dict = None,
        dtype: Union[str, Optional[torch.dtype]] = "qint8",
    ):
        """
        Callback for model quantiztion.

        Args:
            logdir: path to folder for saving
            filename: filename
            qconfig_spec (Dict, optional): quantization config in PyTorch format.
                Defaults to None.
            dtype (Union[str, Optional[torch.dtype]], optional):
                Type of weights after quantization.
                Defaults to "qint8".
        """
        super().__init__(
            order=CallbackOrder.ExternalExtra, node=CallbackNode.master
        )  # External Extra for applying
        # after CheckpointCallback; node master for saving.
        self.qconfig_spec = qconfig_spec
        self.dtype = dtype
        if logdir is not None:
            self.filename = Path(logdir) / filename
        else:
            self.filename = filename

    def on_stage_end(self, runner: "IRunner") -> None:
        model = runner.model.cpu()
        q_model = quantize_model(model.cpu(), qconfig_spec=self.qconfig_spec, dtype=self.dtype)
        torch.save(q_model.state_dict(), self.filename)


__all__ = ["QuantizationCallback"]
