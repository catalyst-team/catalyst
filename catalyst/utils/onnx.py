from typing import Dict, Iterable, List, Tuple, Union
import os
from pathlib import Path

import torch

from catalyst.settings import SETTINGS
from catalyst.tools.forward_wrapper import ModelForwardWrapper

if SETTINGS.onnx_required:
    from onnxruntime.quantization import quantize_dynamic, QuantType


def convert_to_onnx(
    model: torch.nn.Module,
    input_shape: Union[List, Tuple, torch.Size],
    method_name: str = "forward",
    input_names: Iterable = None,
    output_names: List[str] = None,
    file="model.onnx",
    dynamic_axes: Union[Dict[str, int], Dict[str, Dict[str, int]]] = None,
    opset_version: int = 9,
    do_constant_folding: bool = False,
):
    """@TODO: docs.

    Args:
        model (torch.nn.Module): [description]
        input_shape (Union[List, Tuple, torch.Size]): [description]
        method_name (str, optional): Forward pass method to be converted. Defaults to "forward".
        input_names (Iterable, optional): [description]. Defaults to None.
        output_names (List[str], optional): [description]. Defaults to None.
        file (str, optional): [description]. Defaults to "model.onnx".
        dynamic_axes (Union[Dict[str, int], Dict[str, Dict[str, int]]], optional): [description].
            Defaults to None.
        opset_version (int, optional): [description]. Defaults to 9.
        do_constant_folding (bool, optional): [description]. Defaults to False.
    """
    if method_name != "forward":
        model = ModelForwardWrapper(model=model, method_name=method_name)
    torch.onnx.export(
        model,
        input_shape,
        file,
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=do_constant_folding,
        opset_version=opset_version,
    )


def quantize_onnx_model(
    onnx_model_path: Union[Path, str],
    quantized_model_path: Union[Path, str],
    qtype: str = "qint8",
    verbose: bool = False,
) -> None:
    """Takes model converted to onnx runtime and applies pruning.

    Args:
        onnx_model_path (Union[Path, str]): path to onnx model.
        quantized_model_path (Union[Path, str]): path to quantized model.
        qtype (str, optional): Type of weights in quantized model.
            Can be `quint8` or `qint8`. Defaults to "qint8".
        verbose (bool, optional): If set to True prints model size before
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
            "type should be string one of 'quint8' or 'qint8'. Got {}".format(
                qtype
            )
        )
    quantize_dynamic(
        onnx_model_path, quantized_model_path, weight_type=type_mapping[qtype]
    )
    if verbose:
        v_str = (
            "Model size before quantization (MB):"
            f"{os.path.getsize(onnx_model_path) / 2**20:.2f}\n"
            "Model size after quantization (MB): "
            f"{os.path.getsize(quantized_model_path) / 2**20:.2f}"
        )
        print("Done.")
        print(v_str)
        print(f"Quantized model saved to {quantized_model_path}.")


__all__ = ["convert_to_onnx", "quantize_onnx_model"]
