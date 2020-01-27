from typing import Dict, Iterable, List, Tuple, Union  # isort:skip
import collections
import copy
import os
import re

import numpy as np
import safitty

import torch
from torch import nn
import torch.backends.cudnn as cudnn

from catalyst import utils
from catalyst.utils.tools.typing import (
    Criterion, Device, Model, Optimizer, Scheduler
)


def ce_with_logits(logits, target):
    """Returns cross entropy for giving logits"""
    return torch.sum(-target * torch.log_softmax(logits, -1), -1)


def log1p_exp(x):
    """
    Computationally stable function for computing log(1+exp(x)).
    """
    x_ = x * x.ge(0).to(torch.float32)
    res = x_ + torch.log1p(torch.exp(-torch.abs(x)))
    return res


def normal_sample(mu, sigma):
    """
    Sample from multivariate Gaussian distribution z ~ N(z|mu,sigma)
    while supporting backpropagation through its mean and variance.
    """
    return mu + sigma * torch.randn_like(sigma)


def normal_logprob(mu, sigma, z):
    """
    Probability density function of multivariate Gaussian distribution
    N(z|mu,sigma).
    """
    normalization_constant = (-sigma.log() - 0.5 * np.log(2 * np.pi))
    square_term = -0.5 * ((z - mu) / sigma)**2
    logprob_vec = normalization_constant + square_term
    logprob = logprob_vec.sum(1)
    return logprob


def soft_update(target, source, tau):
    """Updates the target data with smoothing by ``tau``"""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def get_optimizable_params(model_or_params):
    """
    Returns all the parameters that requires gradients
    """
    params: Iterable[torch.Tensor] = model_or_params
    if isinstance(model_or_params, nn.Module):
        params = model_or_params.parameters()

    master_params = [p for p in params if p.requires_grad]
    return master_params


def get_optimizer_momentum(optimizer: Optimizer) -> float:
    """
    Get momentum of current optimizer.

    Args:
        optimizer: PyTorch optimizer

    Returns:
        float: momentum at first param group
    """
    beta = safitty.get(optimizer.param_groups, 0, "betas", 0)
    momentum = safitty.get(optimizer.param_groups, 0, "momentum")
    return beta if beta is not None else momentum


def set_optimizer_momentum(optimizer: Optimizer, value: float, index: int = 0):
    """
    Set momentum of ``index`` 'th param group of optimizer to ``value``

    Args:
        optimizer: PyTorch optimizer
        value (float): new value of momentum
        index (int, optional): integer index of optimizer's param groups,
            default is 0
    """
    betas = safitty.get(optimizer.param_groups, index, "betas")
    momentum = safitty.get(optimizer.param_groups, index, "momentum")
    if betas is not None:
        _, beta = betas
        safitty.set(
            optimizer.param_groups, index, "betas", value=(value, beta)
        )
    elif momentum is not None:
        safitty.set(optimizer.param_groups, index, "momentum", value=value)


def assert_fp16_available() -> None:
    """
    Asserts for installed and available Apex FP16
    """
    assert torch.backends.cudnn.enabled, \
        "fp16 mode requires cudnn backend to be enabled."

    try:
        import apex  # noqa: F401
        from apex import amp  # noqa: F401
    except ImportError:
        assert False, \
            "NVidia Apex package must be installed. " \
            "See https://github.com/NVIDIA/apex."


def get_device() -> torch.device:
    """
    Simple returning the best available device (GPU or CPU)
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_available_gpus():
    """
    Array of available GPU ids
    Returns:
        iterable: available GPU ids
    Examples:
        >>> os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
        >>> get_available_gpus()
        >>> [0, 2]

        >>> os.environ["CUDA_VISIBLE_DEVICES"] = "0,-1,1"
        >>> get_available_gpus()
        >>> [0]

        >>> os.environ["CUDA_VISIBLE_DEVICES"] = ""
        >>> get_available_gpus()
        >>> []

        >>> os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        >>> get_available_gpus()
        >>> []
    """
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        result = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        result = [int(id_) for id_ in result if id_ != ""]
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
    """
    Returns the activation function from ``torch.nn`` by its name
    """
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
        return dict((k, any2device(v, device)) for k, v in value.items())
    elif isinstance(value, (tuple, list)):
        return list(any2device(v, device) for v in value)
    elif torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    elif isinstance(value, (np.ndarray, np.void)) \
            and value.dtype.fields is not None:
        return dict(
            (k, any2device(value[k], device))
            for k in value.dtype.fields.keys()
        )
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
            deterministic = \
                os.environ.get("CUDNN_DETERMINISTIC", "True") == "True"
        cudnn.deterministic = deterministic

        # https://discuss.pytorch.org/t/how-should-i-disable-using-cudnn-in-my-code/38053/4
        if benchmark is None:
            benchmark = os.environ.get("CUDNN_BENCHMARK", "True") == "True"
        cudnn.benchmark = benchmark


def process_model_params(
    model: Model,
    layerwise_params: Dict[str, dict] = None,
    no_bias_weight_decay: bool = True,
    lr_scaling: float = 1.0
) -> List[Union[torch.nn.Parameter, dict]]:
    """
    Gains model parameters for ``torch.optim.Optimizer``

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

    Examples:
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
        for pattern, options_ in layerwise_params.items():
            if re.match(pattern, name) is not None:
                # all new LR rules write on top of the old ones
                options = utils.merge_dicts(options, options_)

        # no bias decay from https://arxiv.org/abs/1812.01187
        if no_bias_weight_decay and name.endswith("bias"):
            options["weight_decay"] = 0.0

        # lr linear scaling from https://arxiv.org/pdf/1706.02677.pdf
        if "lr" in options:
            options["lr"] *= lr_scaling

        model_params.append({"params": parameters, **options})

    return model_params


def process_components(
    model: Model,
    criterion: Criterion = None,
    optimizer: Optimizer = None,
    scheduler: Scheduler = None,
    distributed_params: Dict = None,
    device: Device = None,
) -> Tuple[Model, Criterion, Optimizer, Scheduler, Device]:
    """
    Returns the processed model, criterion, optimizer, scheduler and device

    Args:
        model (Model): torch model
        criterion (Criterion): criterion function
        optimizer (Optimizer): optimizer
        scheduler (Scheduler): scheduler
        distributed_params (dict, optional): dict with the parameters
            for distributed and FP16 methond
        device (Device, optional): device
    """
    distributed_params = distributed_params or {}
    distributed_params = copy.deepcopy(distributed_params)
    if device is None:
        device = utils.get_device()

    model: Model = utils.maybe_recursive_call(model, "to", device=device)

    if utils.is_wrapped_with_ddp(model):
        pass
    elif len(distributed_params) > 0:
        assert isinstance(model, nn.Module)
        distributed_rank = distributed_params.pop("rank", -1)
        syncbn = distributed_params.pop("syncbn", False)

        if distributed_rank > -1:
            torch.cuda.set_device(distributed_rank)
            torch.distributed.init_process_group(
                backend="nccl", init_method="env://"
            )

        if "opt_level" in distributed_params:
            utils.assert_fp16_available()
            from apex import amp

            amp_result = amp.initialize(model, optimizer, **distributed_params)
            if optimizer is not None:
                model, optimizer = amp_result
            else:
                model = amp_result

            if distributed_rank > -1:
                from apex.parallel import DistributedDataParallel
                model = DistributedDataParallel(model)

                if syncbn:
                    from apex.parallel import convert_syncbn_model
                    model = convert_syncbn_model(model)

        if distributed_rank <= -1 and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
    elif torch.cuda.device_count() > 1:
        if isinstance(model, nn.Module):
            model = torch.nn.DataParallel(model)
        elif isinstance(model, dict):
            model = {k: torch.nn.DataParallel(v) for k, v in model.items()}

    model: Model = utils.maybe_recursive_call(model, "to", device=device)

    return model, criterion, optimizer, scheduler, device


def set_requires_grad(model: Model, requires_grad: bool):
    """
    Sets the ``requires_grad`` value for all model parameters.

    Args:
        model (torch.nn.Module): Model
        requires_grad (bool): value

    Examples:
        >>> model = SimpleModel()
        >>> set_requires_grad(model, requires_grad=True)
    """
    requires_grad = bool(requires_grad)
    for param in model.parameters():
        param.requires_grad = requires_grad


def get_network_output(net: Model, *input_shapes):
    """
    For each input shape returns an output tensor

    Args:
        net (Model): the model
        *args: variable length argument list of shapes
    """
    inputs = []
    for input_shape in input_shapes:
        if isinstance(input_shape, dict):
            input_t = {}
            for key, input_shape_ in input_shape.items():
                input_t[key] = torch.Tensor(torch.randn((1, ) + input_shape_))
        else:
            input_t = torch.Tensor(torch.randn((1, ) + input_shape))
        inputs.append(input_t)
    output_t = net(*inputs)
    return output_t


def detach(tensor: torch.Tensor) -> np.ndarray:
    """
    Detaches the input tensor to a numpy array
    """
    return tensor.detach().cpu().numpy()


__all__ = [
    "ce_with_logits", "log1p_exp", "normal_sample", "normal_logprob",
    "soft_update", "get_optimizable_params", "get_optimizer_momentum",
    "set_optimizer_momentum", "assert_fp16_available", "get_device",
    "get_available_gpus", "get_activation_fn", "any2device", "prepare_cudnn",
    "process_model_params", "set_requires_grad", "get_network_output", "detach"
]
