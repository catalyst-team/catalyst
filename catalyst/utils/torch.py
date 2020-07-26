# flake8: noqa
# @TODO: code formatting issue for 20.07 release
from typing import Dict, Iterable, List, Union
import collections
import os
import re

import numpy as np

import torch
from torch import nn, Tensor
import torch.backends
from torch.backends import cudnn

from catalyst.tools.settings import IS_XLA_AVAILABLE
from catalyst.tools.typing import Device, Model, Optimizer
from catalyst.utils.dict import merge_dicts


def get_optimizable_params(model_or_params):
    """Returns all the parameters that requires gradients."""
    params: Iterable[torch.Tensor] = model_or_params
    if isinstance(model_or_params, nn.Module):
        params = model_or_params.parameters()

    master_params = [p for p in params if p.requires_grad]
    return master_params


def get_optimizer_momentum(optimizer: Optimizer) -> float:
    """Get momentum of current optimizer.

    Args:
        optimizer: PyTorch optimizer

    Returns:
        float: momentum at first param group
    """
    betas = optimizer.param_groups[0].get("betas", None)
    momentum = optimizer.param_groups[0].get("momentum", None)
    return betas[0] if betas is not None else momentum


def set_optimizer_momentum(optimizer: Optimizer, value: float, index: int = 0):
    """Set momentum of ``index`` 'th param group of optimizer to ``value``.

    Args:
        optimizer: PyTorch optimizer
        value (float): new value of momentum
        index (int, optional): integer index of optimizer's param groups,
            default is 0
    """
    betas = optimizer.param_groups[0].get("betas", None)
    momentum = optimizer.param_groups[0].get("momentum", None)
    if betas is not None:
        _, beta = betas
        optimizer.param_groups[index]["betas"] = (value, beta)
    elif momentum is not None:
        optimizer.param_groups[index]["momentum"] = value


def get_device() -> torch.device:
    """Simple returning the best available device (TPU > GPU > CPU)."""
    is_available_gpu = torch.cuda.is_available()
    device = "cpu"
    if IS_XLA_AVAILABLE:
        import torch_xla.core.xla_model as xm

        device = xm.xla_device()
    elif is_available_gpu:
        device = "cuda"
    return torch.device(device)


def get_available_gpus():
    """Array of available GPU ids.

    Examples:
        >>> os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
        >>> get_available_gpus()
        [0, 2]

        >>> os.environ["CUDA_VISIBLE_DEVICES"] = "0,-1,1"
        >>> get_available_gpus()
        [0]

        >>> os.environ["CUDA_VISIBLE_DEVICES"] = ""
        >>> get_available_gpus()
        []

        >>> os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        >>> get_available_gpus()
        []

    Returns:
        iterable: available GPU ids
    """
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        result = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        result = [id_ for id_ in result if id_ != ""]
        # invisible GPUs
        # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars
        if -1 in result:
            index = result.index(-1)
            result = result[:index]
    elif torch.cuda.is_available():
        result = list(range(torch.cuda.device_count()))
    else:
        result = []
    return result


def get_activation_fn(activation: str = None):
    """Returns the activation function from ``torch.nn`` by its name."""
    if activation is None or activation.lower() == "none":
        activation_fn = lambda x: x  # noqa: E731
    else:
        activation_fn = torch.nn.__dict__[activation]()
    return activation_fn


def any2device(value, device: Device):
    """
    Move tensor, list of tensors, list of list of tensors,
    dict of tensors, tuple of tensors to target device.

    Args:
        value: Object to be moved
        device (Device): target device ids

    Returns:
        Same structure as value, but all tensors and np.arrays moved to device
    """
    if isinstance(value, dict):
        return {k: any2device(v, device) for k, v in value.items()}
    elif isinstance(value, (tuple, list)):
        return [any2device(v, device) for v in value]
    elif torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    elif (
        isinstance(value, (np.ndarray, np.void))
        and value.dtype.fields is not None
    ):
        return {
            k: any2device(value[k], device) for k in value.dtype.fields.keys()
        }
    elif isinstance(value, np.ndarray):
        return torch.Tensor(value).to(device)
    return value


def prepare_cudnn(deterministic: bool = None, benchmark: bool = None) -> None:
    """
    Prepares CuDNN benchmark and sets CuDNN
    to be deterministic/non-deterministic mode

    Args:
        deterministic (bool): deterministic mode if running in CuDNN backend.
        benchmark (bool): If ``True`` use CuDNN heuristics to figure out
            which algorithm will be most performant
            for your model architecture and input.
            Setting it to ``False`` may slow down your training.
    """
    if torch.cuda.is_available():
        # CuDNN reproducibility
        # https://pytorch.org/docs/stable/notes/randomness.html#cudnn
        if deterministic is None:
            deterministic = (
                os.environ.get("CUDNN_DETERMINISTIC", "True") == "True"
            )
        cudnn.deterministic = deterministic

        # https://discuss.pytorch.org/t/how-should-i-disable-using-cudnn-in-my-code/38053/4
        if benchmark is None:
            benchmark = os.environ.get("CUDNN_BENCHMARK", "True") == "True"
        cudnn.benchmark = benchmark


def process_model_params(
    model: Model,
    layerwise_params: Dict[str, dict] = None,
    no_bias_weight_decay: bool = True,
    lr_scaling: float = 1.0,
) -> List[Union[torch.nn.Parameter, dict]]:
    """Gains model parameters for ``torch.optim.Optimizer``.

    Args:
        model (torch.nn.Module): Model to process
        layerwise_params (Dict): Order-sensitive dict where
            each key is regex pattern and values are layer-wise options
            for layers matching with a pattern
        no_bias_weight_decay (bool): If true, removes weight_decay
            for all ``bias`` parameters in the model
        lr_scaling (float): layer-wise learning rate scaling,
            if 1.0, learning rates will not be scaled

    Returns:
        iterable: parameters for an optimizer

    Example::

        >>> model = catalyst.contrib.models.segmentation.ResnetUnet()
        >>> layerwise_params = collections.OrderedDict([
        >>>     ("conv1.*", dict(lr=0.001, weight_decay=0.0003)),
        >>>     ("conv.*", dict(lr=0.002))
        >>> ])
        >>> params = process_model_params(model, layerwise_params)
        >>> optimizer = torch.optim.Adam(params, lr=0.0003)

    """
    params = list(model.named_parameters())
    layerwise_params = layerwise_params or collections.OrderedDict()

    model_params = []
    for name, parameters in params:
        options = {}
        for pattern, pattern_options in layerwise_params.items():
            if re.match(pattern, name) is not None:
                # all new LR rules write on top of the old ones
                options = merge_dicts(options, pattern_options)

        # no bias decay from https://arxiv.org/abs/1812.01187
        if no_bias_weight_decay and name.endswith("bias"):
            options["weight_decay"] = 0.0

        # lr linear scaling from https://arxiv.org/pdf/1706.02677.pdf
        if "lr" in options:
            options["lr"] *= lr_scaling

        model_params.append({"params": parameters, **options})

    return model_params


def get_requires_grad(model: Model):
    """Gets the ``requires_grad`` value for all model parameters.

    Example::

        >>> model = SimpleModel()
        >>> requires_grad = get_requires_grad(model)

    Args:
        model (torch.nn.Module): model

    Returns:
        requires_grad (Dict[str, bool]): value
    """
    requires_grad = {}
    for name, param in model.named_parameters():
        requires_grad[name] = param.requires_grad
    return requires_grad


def set_requires_grad(
    model: Model, requires_grad: Union[bool, Dict[str, bool]]
):
    """Sets the ``requires_grad`` value for all model parameters.

    Example::

        >>> model = SimpleModel()
        >>> set_requires_grad(model, requires_grad=True)
        >>> # or
        >>> model = SimpleModel()
        >>> set_requires_grad(model, requires_grad={""})

    Args:
        model (torch.nn.Module): model
        requires_grad (Union[bool, Dict[str, bool]]): value
    """
    if isinstance(requires_grad, dict):
        for name, param in model.named_parameters():
            assert (
                name in requires_grad
            ), f"Parameter `{name}` does not exist in requires_grad"
            param.requires_grad = requires_grad[name]
    else:
        requires_grad = bool(requires_grad)
        for param in model.parameters():
            param.requires_grad = requires_grad


def get_network_output(net: Model, *input_shapes_args, **input_shapes_kwargs):
    """# noqa: D202
    For each input shape returns an output tensor

    Examples:
        >>> net = nn.Linear(10, 5)
        >>> utils.get_network_output(net, (1, 10))
        tensor([[[-0.2665,  0.5792,  0.9757, -0.5782,  0.1530]]])

    Args:
        net (Model): the model
        *input_shapes_args: variable length argument list of shapes
        **input_shapes_kwargs: key-value arguemnts of shapes

    Returns:
        tensor with network output
    """

    def _rand_sample(
        input_shape,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if isinstance(input_shape, dict):
            input_t = {
                key: torch.Tensor(torch.randn((1,) + key_input_shape))
                for key, key_input_shape in input_shape.items()
            }
        else:
            input_t = torch.Tensor(torch.randn((1,) + input_shape))
        return input_t

    input_args = [
        _rand_sample(input_shape) for input_shape in input_shapes_args
    ]
    input_kwargs = {
        key: _rand_sample(input_shape)
        for key, input_shape in input_shapes_kwargs.items()
    }

    output_t = net(*input_args, **input_kwargs)
    return output_t


def detach(tensor: torch.Tensor) -> np.ndarray:
    """Detach a pytorch tensor from graph and
    convert it to numpy array

    Args:
        tensor: PyTorch tensor

    Returns:
        numpy ndarray
    """
    return tensor.cpu().detach().numpy()


def trim_tensors(tensors):
    """
    Trim padding off of a batch of tensors to the smallest possible length.
    Should be used with `catalyst.data.DynamicLenBatchSampler`.

    Adapted from `Dynamic minibatch trimming to improve BERT training speed`_.

    Args:
        tensors ([torch.tensor]): list of tensors to trim.

    Returns:
        List[torch.tensor]: list of trimmed tensors.

    .. _`Dynamic minibatch trimming to improve BERT training speed`:
        https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/94779
    """
    max_len = torch.max(torch.sum((tensors[0] != 0), 1))
    if max_len > 2:
        tensors = [tsr[:, :max_len] for tsr in tensors]
    return tensors


def normalize(samples: Tensor) -> Tensor:
    """
    Args:
        samples: tensor with shape of [n_samples, features_dim]

    Returns:
        normalized tensor with the same shape
    """
    norms = torch.norm(samples, p=2, dim=1).unsqueeze(1)
    samples = samples / (norms + torch.finfo(torch.float32).eps)
    return samples


__all__ = [
    "get_optimizable_params",
    "get_optimizer_momentum",
    "set_optimizer_momentum",
    "get_device",
    "get_available_gpus",
    "get_activation_fn",
    "any2device",
    "prepare_cudnn",
    "process_model_params",
    "get_requires_grad",
    "set_requires_grad",
    "get_network_output",
    "detach",
    "trim_tensors",
    "normalize",
]
