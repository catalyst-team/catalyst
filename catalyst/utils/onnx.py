from typing import Dict, Iterable, List, Union
import os
from pathlib import Path

import torch

from catalyst.extras.forward_wrapper import ModelForwardWrapper
from catalyst.settings import SETTINGS
from catalyst.utils.distributed import get_nn_from_ddp_module

if SETTINGS.onnx_required:
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType


def onnx_export(
    model: torch.nn.Module,
    batch: torch.Tensor,
    file: str,
    method_name: str = "forward",
    input_names: Iterable = None,
    output_names: List[str] = None,
    dynamic_axes: Union[Dict[str, int], Dict[str, Dict[str, int]]] = None,
    opset_version: int = 9,
    do_constant_folding: bool = False,
    return_model: bool = False,
    verbose: bool = False,
) -> Union[None, "onnx"]:
    """Converts model to onnx runtime.

    Args:
        model: model
        batch: inputs
        file: file to save. Defaults to "model.onnx".
        method_name: Forward pass method to be converted. Defaults to "forward".
        input_names: name of inputs in graph. Defaults to None.
        output_names: name of outputs in graph. Defaults to None.
        dynamic_axes: axes
            with dynamic shapes. Defaults to None.
        opset_version: Defaults to 9.
        do_constant_folding: If True, the constant-folding optimization
            is applied to the model during export. Defaults to False.
        return_model: If True then returns onnxruntime model (onnx required).
            Defaults to False.
        verbose: if specified, we will print out a debug
            description of the trace being exported.

    Example:
        .. code-block:: python

           import torch

           from catalyst.utils import convert_to_onnx

           class LinModel(torch.nn.Module):
               def __init__(self):
                   super().__init__()
                   self.lin1 = torch.nn.Linear(10, 10)
                   self.lin2 = torch.nn.Linear(2, 10)

               def forward(self, inp_1, inp_2):
                   return self.lin1(inp_1), self.lin2(inp_2)

               def first_only(self, inp_1):
                   return self.lin1(inp_1)

           lin_model = LinModel()
           convert_to_onnx(
               model, batch=torch.randn((1, 10)),
               file="model.onnx",
               method_name="first_only"
           )

    Raises:
        ImportError: when ``return_model`` is True, but onnx is not installed.

    Returns:
        Union[None, "onnx"]: onnx model if return_model set to True.
    """
    nn_model = get_nn_from_ddp_module(model)
    if method_name != "forward":
        nn_model = ModelForwardWrapper(model=nn_model, method_name=method_name)
    torch.onnx.export(
        nn_model,
        batch,
        file,
        verbose=verbose,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=do_constant_folding,
        opset_version=opset_version,
    )
    if return_model:
        if not SETTINGS.onnx_required:
            raise ImportError(
                "To use onnx model you should install it with ``pip install onnx``"
            )
        return onnx.load(file)


def quantize_onnx_model(
    onnx_model_path: Union[Path, str],
    quantized_model_path: Union[Path, str],
    qtype: str = "qint8",
    verbose: bool = False,
) -> None:
    """Takes model converted to onnx runtime and applies pruning.

    Args:
        onnx_model_path: path to onnx model.
        quantized_model_path: path to quantized model.
        qtype: Type of weights in quantized model.
            Can be `quint8` or `qint8`. Defaults to "qint8".
        verbose: If set to True prints model size before
            and after quantization. Defaults to False.

    Raises:
        ValueError: If qtype is not understood.
    """
    type_mapping = {
        "qint8": QuantType.QInt8,
        "quint8": QuantType.QUInt8,
    }
    if qtype not in type_mapping.keys():
        raise ValueError(
            "type should be string one of 'quint8' or 'qint8'. Got {}".format(qtype)
        )
    quantize_dynamic(
        onnx_model_path, quantized_model_path, weight_type=type_mapping[qtype]
    )
    if verbose:
        v_str = (
            "Model size before quantization (MB):"
            f"{os.path.getsize(onnx_model_path) / 2**20:.2f}\n"
            "Model size after quantization (MB):"
            f"{os.path.getsize(quantized_model_path) / 2**20:.2f}"
        )
        print("Done.")
        print(v_str)
        print(f"Quantized model saved to {quantized_model_path}.")


__all__ = ["onnx_export", "quantize_onnx_model"]
