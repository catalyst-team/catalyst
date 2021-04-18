from typing import Any, Callable, Dict, Iterable, List, Mapping, TYPE_CHECKING, Union
import collections
import os
import re

import numpy as np
import torch
from torch import nn, Tensor
import torch.backends
from torch.backends import cudnn

from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES, SETTINGS
from catalyst.typing import (
    Device,
    Model,
    Optimizer,
    RunnerCriterion,
    RunnerModel,
    RunnerOptimizer,
    RunnerScheduler,
)
from catalyst.utils.distributed import get_nn_from_ddp_module
from catalyst.utils.misc import maybe_recursive_call, merge_dicts

if TYPE_CHECKING:
    from catalyst.core.engine import IEngine

# TODO: move to global registry with activation functions
ACTIVATIONS = {  # noqa: WPS407
    None: "sigmoid",
    nn.Sigmoid: "sigmoid",
    nn.Tanh: "tanh",
    nn.ReLU: "relu",
    nn.LeakyReLU: "leaky_relu",
    nn.ELU: "relu",
}


def _nonlinearity2name(nonlinearity):
    if isinstance(nonlinearity, nn.Module):
        nonlinearity = nonlinearity.__class__
    nonlinearity = ACTIVATIONS.get(nonlinearity, nonlinearity)
    nonlinearity = nonlinearity.lower()
    return nonlinearity


def get_optimal_inner_init(nonlinearity: nn.Module, **kwargs) -> Callable[[nn.Module], None]:
    """
    Create initializer for inner layers
    based on their activation function (nonlinearity).

    Args:
        nonlinearity: non-linear activation
        **kwargs: extra kwargs

    Returns:
        optimal initialization function

    Raises:
        NotImplementedError: if nonlinearity is out of
            `sigmoid`, `tanh`, `relu, `leaky_relu`
    """
    nonlinearity: str = _nonlinearity2name(nonlinearity)
    assert isinstance(nonlinearity, str)

    if nonlinearity in ["sigmoid", "tanh"]:
        weignt_init_fn = nn.init.xavier_uniform_
        init_args = kwargs
    elif nonlinearity in ["relu", "leaky_relu"]:
        weignt_init_fn = nn.init.kaiming_normal_
        init_args = {**{"nonlinearity": nonlinearity}, **kwargs}
    else:
        raise NotImplementedError

    def inner_init(layer):
        if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            weignt_init_fn(layer.weight.data, **init_args)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias.data)

    return inner_init


def outer_init(layer: nn.Module) -> None:
    """
    Initialization for output layers of policy and value networks typically
    used in deep reinforcement learning literature.

    Args:
        layer: torch nn.Module instance
    """
    if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        v = 3e-3
        nn.init.uniform_(layer.weight.data, -v, v)
        if layer.bias is not None:
            nn.init.uniform_(layer.bias.data, -v, v)


def reset_weights_if_possible(module: nn.Module):
    """
    Resets module parameters if possible.

    Args:
        module: Module to reset.
    """
    try:
        module.reset_parameters()
    except AttributeError:
        pass


def get_optimizable_params(model_or_params):
    """Returns all the parameters that requires gradients."""
    params: Iterable[torch.Tensor] = model_or_params
    if isinstance(model_or_params, torch.nn.Module):
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


def get_optimizer_momentum_list(optimizer: Optimizer) -> List[Union[float, None]]:
    """Get list of optimizer momentums (for each param group)

    Args:
        optimizer: PyTorch optimizer

    Returns:
        momentum_list (List[Union[float, None]]): momentum for each param group
    """
    result = []

    for param_group in optimizer.param_groups:
        betas = param_group.get("betas", None)
        momentum = param_group.get("momentum", None)
        result.append(betas[0] if betas is not None else momentum)

    return result


def set_optimizer_momentum(optimizer: Optimizer, value: float, index: int = 0):
    """Set momentum of ``index`` 'th param group of optimizer to ``value``.

    Args:
        optimizer: PyTorch optimizer
        value: new value of momentum
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
    if SETTINGS.xla_required:
        import torch_xla.core.xla_model as xm

        device = xm.xla_device()
    elif is_available_gpu:
        device = "cuda"
    return torch.device(device)


def get_available_engine(
    fp16: bool = False, ddp: bool = False, amp: bool = False, apex: bool = False
) -> "IEngine":
    """Returns available engine based on given arguments.

    Args:
        fp16 (bool): option to use fp16 for training. Default is `False`.
        ddp (bool): option to use DDP for training. Default is `False`.
        amp (bool): option to use APEX for training. Default is `False`.
        apex (bool): option to use APEX for training. Default is `False`.

    Returns:
        IEngine which match requirements.
    """
    from catalyst.engines.torch import (
        DataParallelEngine,
        DeviceEngine,
        DistributedDataParallelEngine,
    )

    if fp16 and not amp and not apex:
        amp = SETTINGS.amp_required or (SETTINGS.amp_required and SETTINGS.apex_required)
        apex = SETTINGS.apex_required and (not SETTINGS.amp_required)

    if amp:
        assert (
            SETTINGS.amp_required
        ), "catalyst[amp] is not available, to install it, run `pip install catalyst[amp]`."
        assert not apex, "Could not use both apex and amp engines"
        from catalyst.engines.amp import (
            AMPEngine,
            DataParallelAMPEngine,
            DistributedDataParallelAMPEngine,
        )

    if apex:
        assert (
            SETTINGS.apex_required
        ), "catalyst[apex] is not available, to install it, run `pip install catalyst[apex]`."
        assert not amp, "Could not use both apex and amp engines"
        from catalyst.engines.apex import (
            APEXEngine,
            DataParallelApexEngine,
            DistributedDataParallelApexEngine,
        )

    is_multiple_gpus = NUM_CUDA_DEVICES > 1
    if not IS_CUDA_AVAILABLE:
        return DeviceEngine("cpu")
    elif is_multiple_gpus:
        if ddp:
            if amp:
                return DistributedDataParallelAMPEngine()
            elif apex:
                return DistributedDataParallelApexEngine()
            else:
                return DistributedDataParallelEngine()
        else:
            if amp:
                return DataParallelAMPEngine()
            elif apex:
                return DataParallelApexEngine()
            else:
                return DataParallelEngine()
    else:
        if amp:
            return AMPEngine()
        elif apex:
            return APEXEngine()
        else:
            return DeviceEngine("cuda")


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


def any2device(value, device: Device):
    """
    Move tensor, list of tensors, list of list of tensors,
    dict of tensors, tuple of tensors to target device.

    Args:
        value: Object to be moved
        device: target device ids

    Returns:
        Same structure as value, but all tensors and np.arrays moved to device
    """
    if isinstance(value, dict):
        return {k: any2device(v, device) for k, v in value.items()}
    elif isinstance(value, (tuple, list)):
        return [any2device(v, device) for v in value]
    elif torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    elif isinstance(value, (np.ndarray, np.void)) and value.dtype.fields is not None:
        return {k: any2device(value[k], device) for k in value.dtype.fields.keys()}
    elif isinstance(value, np.ndarray):
        return torch.tensor(value, device=device)
    elif isinstance(value, nn.Module):
        return value.to(device)
    return value


def prepare_cudnn(deterministic: bool = None, benchmark: bool = None) -> None:
    """
    Prepares CuDNN benchmark and sets CuDNN
    to be deterministic/non-deterministic mode

    Args:
        deterministic: deterministic mode if running in CuDNN backend.
        benchmark: If ``True`` use CuDNN heuristics to figure out
            which algorithm will be most performant
            for your model architecture and input.
            Setting it to ``False`` may slow down your training.
    """
    if torch.cuda.is_available():
        # CuDNN reproducibility
        # https://pytorch.org/docs/stable/notes/randomness.html#cudnn
        if deterministic is None:
            deterministic = os.environ.get("CUDNN_DETERMINISTIC", "True") == "True"
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
        model: Model to process
        layerwise_params: Order-sensitive dict where
            each key is regex pattern and values are layer-wise options
            for layers matching with a pattern
        no_bias_weight_decay: If true, removes weight_decay
            for all ``bias`` parameters in the model
        lr_scaling: layer-wise learning rate scaling,
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
        model: model

    Returns:
        requires_grad: value
    """
    requires_grad = {}
    for name, param in model.named_parameters():
        requires_grad[name] = param.requires_grad
    return requires_grad


def set_requires_grad(model: Model, requires_grad: Union[bool, Dict[str, bool]]):
    """Sets the ``requires_grad`` value for all model parameters.

    Example::

        >>> model = SimpleModel()
        >>> set_requires_grad(model, requires_grad=True)
        >>> # or
        >>> model = SimpleModel()
        >>> set_requires_grad(model, requires_grad={""})

    Args:
        model: model
        requires_grad: value
    """
    if isinstance(requires_grad, dict):
        for name, param in model.named_parameters():
            assert name in requires_grad, f"Parameter `{name}` does not exist in requires_grad"
            param.requires_grad = requires_grad[name]
    else:
        requires_grad = bool(requires_grad)
        for param in model.parameters():
            param.requires_grad = requires_grad


def get_network_output(net: Model, *input_shapes_args, **input_shapes_kwargs):
    """For each input shape returns an output tensor

    Examples:
        >>> net = nn.Linear(10, 5)
        >>> utils.get_network_output(net, (1, 10))
        tensor([[[-0.2665,  0.5792,  0.9757, -0.5782,  0.1530]]])

    Args:
        net: the model
        *input_shapes_args: variable length argument list of shapes
        **input_shapes_kwargs: key-value arguemnts of shapes

    Returns:
        tensor with network output
    """

    def _rand_sample(input_shape) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if isinstance(input_shape, dict):
            input_t = {
                key: torch.Tensor(torch.randn((1,) + key_input_shape))
                for key, key_input_shape in input_shape.items()
            }
        else:
            input_t = torch.Tensor(torch.randn((1,) + input_shape))
        return input_t

    input_args = [_rand_sample(input_shape) for input_shape in input_shapes_args]
    input_kwargs = {
        key: _rand_sample(input_shape) for key, input_shape in input_shapes_kwargs.items()
    }

    output_t = net(*input_args, **input_kwargs)
    return output_t


def detach_tensor(tensor: torch.Tensor) -> np.ndarray:
    """Detach a pytorch tensor from graph and
    convert it to numpy array

    Args:
        tensor: PyTorch tensor

    Returns:
        numpy ndarray
    """
    return tensor.cpu().detach().numpy()


def trim_tensors(tensors: Tensor) -> List[torch.Tensor]:
    """
    Trim padding off of a batch of tensors to the smallest possible length.
    Should be used with `catalyst.data.DynamicLenBatchSampler`.

    Adapted from `Dynamic minibatch trimming to improve BERT training speed`_.

    Args:
        tensors: list of tensors to trim.

    Returns:
        List[torch.tensor]: list of trimmed tensors.

    .. _`Dynamic minibatch trimming to improve BERT training speed`:
        https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/94779
    """
    max_len = torch.max(torch.sum((tensors[0] != 0), 1))
    if max_len > 2:
        tensors = [tsr[:, :max_len] for tsr in tensors]
    return tensors


def pack_checkpoint(
    model: RunnerModel = None,
    criterion: RunnerCriterion = None,
    optimizer: RunnerOptimizer = None,
    scheduler: RunnerScheduler = None,
    **kwargs,
) -> Dict:
    """
    Packs ``model``, ``criterion``, ``optimizer``, ``scheduler``
    and some extra info ``**kwargs`` to torch-based checkpoint.

    Args:
        model: torch model
        criterion: torch criterion
        optimizer: torch optimizer
        scheduler: torch scheduler
        **kwargs: some extra info to pack

    Returns:
        torch-based checkpoint with ``model_state_dict``,
        ``criterion_state_dict``, ``optimizer_state_dict``,
        ``scheduler_state_dict`` keys.
    """
    checkpoint = kwargs

    if isinstance(model, dict):
        for key, value in model.items():
            model_module = get_nn_from_ddp_module(value)
            checkpoint[f"model_{key}_state_dict"] = maybe_recursive_call(
                model_module, "state_dict"
            )
    else:
        model_module = get_nn_from_ddp_module(model)
        checkpoint["model_state_dict"] = maybe_recursive_call(model_module, "state_dict")

    for dict2save, name2save in zip(
        [criterion, optimizer, scheduler], ["criterion", "optimizer", "scheduler"],
    ):
        if dict2save is None:
            continue
        if isinstance(dict2save, dict):
            for key, value in dict2save.items():
                if value is not None:
                    state_dict2save = name2save + "_" + str(key)
                    # checkpoint[name2save_] = value
                    state_dict2save = state_dict2save + "_state_dict"
                    checkpoint[state_dict2save] = value.state_dict()
        else:
            # checkpoint[name2save] = dict2save
            name2save = name2save + "_state_dict"
            checkpoint[name2save] = dict2save.state_dict()
    return checkpoint


def unpack_checkpoint(
    checkpoint: Dict,
    model: RunnerModel = None,
    criterion: RunnerCriterion = None,
    optimizer: RunnerOptimizer = None,
    scheduler: RunnerScheduler = None,
) -> None:
    """Load checkpoint from file and unpack the content to a model
    (if not None), criterion (if not None), optimizer (if not None),
    scheduler (if not None).

    Args:
        checkpoint: checkpoint to load
        model: model where should be updated state
        criterion: criterion where should be updated state
        optimizer: optimizer where should be updated state
        scheduler: scheduler where should be updated state
    """
    if model is not None:
        model = get_nn_from_ddp_module(model)
        maybe_recursive_call(
            model, "load_state_dict", recursive_args=checkpoint["model_state_dict"],
        )

    for dict2load, name2load in zip(
        [criterion, optimizer, scheduler], ["criterion", "optimizer", "scheduler"],
    ):
        if dict2load is None:
            continue

        if isinstance(dict2load, dict):
            for key, value in dict2load.items():
                if value is not None:
                    state_dict2load = f"{name2load}_{key}_state_dict"
                    value.load_state_dict(checkpoint[state_dict2load])
        else:
            name2load = f"{name2load}_state_dict"
            dict2load.load_state_dict(checkpoint[name2load])


def save_checkpoint(checkpoint: Mapping[str, Any], path: str):
    """Saves checkpoint to a file.

    Args:
        checkpoint: data to save.
        path: filepath where checkpoint should be stored.
    """
    torch.save(checkpoint, path)


def load_checkpoint(path: str):
    """Load checkpoint from path.

    Args:
        path: checkpoint file to load

    Returns:
        checkpoint content
    """
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    return checkpoint


__all__ = [
    "get_optimizable_params",
    "get_optimizer_momentum",
    "get_optimizer_momentum_list",
    "set_optimizer_momentum",
    "get_device",
    "get_available_gpus",
    "any2device",
    "prepare_cudnn",
    "process_model_params",
    "get_requires_grad",
    "set_requires_grad",
    "get_network_output",
    "get_available_engine",
    "detach_tensor",
    "trim_tensors",
    "get_optimal_inner_init",
    "outer_init",
    "reset_weights_if_possible",
    "pack_checkpoint",
    "unpack_checkpoint",
    "save_checkpoint",
    "load_checkpoint",
]
