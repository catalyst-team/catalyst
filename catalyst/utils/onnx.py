import os

from typing import Union, List, Tuple, Iterable, Dict

import torch

from catalyst.settings import IS_ONNX_AVAILABLE

from catalyst.utils.torch import ForwardOverrideModel

if IS_ONNX_AVAILABLE:
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
    if method_name != "forward":
        model = ForwardOverrideModel(model=model, method_name=method_name)
    torch.onnx.export(
        model,
        input_shape,
        file,
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=do_constant_folding,
        opset_version=opset_version
    )


def quantize_onnx_model(onnx_model_path, quantized_model_path, qtype: str="qint8", verbose: bool = False):
    type_mapping = {
        "qint8": QuantType.QInt8,
        "quint8": QuantType.QUInt8,
    }
    if qtype not in type_mapping.keys():
        raise ValueError(
            "type should be string one of 'quint8' or 'qint8'. Got {}".format(qtype)
        )
    quantize_dynamic(onnx_model_path,
                     quantized_model_path,
                     weight_type=type_mapping[qtype])
    if verbose:
        v_str =\
            f"Model size before quantization (MB): {os.path.getsize(onnx_model_path) / 2**20:.2f}\n"\
            f"Model size after quantization (MB): {os.path.getsize(quantized_model_path) / 2**20:.2f}"
        print("Done.")
        print(v_str)
        print(f"Quantized model saved to {quantized_model_path}.")
