from typing import Any, Dict, List, Mapping, Tuple, TYPE_CHECKING, Union
import os

import numpy as np

import torch
from torch import nn
import torch.backends
from torch.backends import cudnn

from catalyst.settings import SETTINGS
from catalyst.typing import (
    RunnerCriterion,
    RunnerDevice,
    RunnerModel,
    RunnerOptimizer,
    RunnerScheduler,
    TorchModel,
    TorchOptimizer,
)
from catalyst.utils.distributed import get_nn_from_ddp_module

if SETTINGS.xla_required:
    import torch_xla.core.xla_model as xm

if TYPE_CHECKING:
    from catalyst.core.engine import Engine


def get_optimizer_momentum(optimizer: TorchOptimizer) -> float:
    """Get momentum of current optimizer.

    Args:
        optimizer: PyTorch optimizer

    Returns:
        float: momentum at first param group
    """
    betas = optimizer.param_groups[0].get("betas", None)
    momentum = optimizer.param_groups[0].get("momentum", None)
    return betas[0] if betas is not None else momentum


def get_optimizer_momentum_list(optimizer: TorchOptimizer) -> List[Union[float, None]]:
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


def set_optimizer_momentum(optimizer: TorchOptimizer, value: float, index: int = 0):
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
    device = torch.device("cpu")
    if SETTINGS.xla_required:
        device = xm.xla_device()
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    return device


def get_available_engine(
    cpu: bool = False,
    fp16: bool = False,
    ddp: bool = False,
) -> "Engine":
    """Returns available engine based on given arguments.

    Args:
        cpu (bool): option to use cpu for training. Default is `False`.
        ddp (bool): option to use DDP for training. Default is `False`.
        fp16 (bool): option to use APEX for training. Default is `False`.

    Returns:
        Engine which match requirements.
    """
    from catalyst.engines.torch import (
        CPUEngine,
        DataParallelEngine,
        DistributedDataParallelEngine,
        GPUEngine,
    )

    if SETTINGS.xla_required:
        from catalyst.engines.torch import DistributedXLAEngine

        return DistributedXLAEngine()

    if cpu or not torch.cuda.is_available():
        return CPUEngine()
    is_multiple_gpus = torch.cuda.device_count() > 1
    if is_multiple_gpus:
        if ddp:
            return DistributedDataParallelEngine(fp16=fp16)
        else:
            return DataParallelEngine(fp16=fp16)
    else:
        return GPUEngine(fp16=fp16)


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


def any2device(
    value: Union[Dict, List, Tuple, np.ndarray, torch.Tensor, nn.Module],
    device: RunnerDevice,
) -> Union[Dict, List, Tuple, torch.Tensor, nn.Module]:
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

        # http://discuss.pytorch.org/t/how-should-i-disable-using-cudnn-in-my-code/38053
        if benchmark is None:
            benchmark = os.environ.get("CUDNN_BENCHMARK", "True") == "True"
        cudnn.benchmark = benchmark


def get_requires_grad(model: TorchModel):
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


def set_requires_grad(model: TorchModel, requires_grad: Union[bool, Dict[str, bool]]):
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
            assert (
                name in requires_grad
            ), f"Parameter `{name}` does not exist in requires_grad"
            param.requires_grad = requires_grad[name]
    else:
        requires_grad = bool(requires_grad)
        for param in model.parameters():
            param.requires_grad = requires_grad


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

    for dict2save, name2save in zip(
        [model, criterion, optimizer, scheduler],
        ["model", "criterion", "optimizer", "scheduler"],
    ):
        if dict2save is None:
            continue
        if isinstance(dict2save, dict):
            for key, value in dict2save.items():
                if value is not None:
                    state_dict2save = name2save + "_" + str(key) + "_state_dict"
                    value = get_nn_from_ddp_module(value)
                    checkpoint[state_dict2save] = value.state_dict()
        else:
            # checkpoint[name2save] = dict2save
            name2save = name2save + "_state_dict"
            dict2save = get_nn_from_ddp_module(dict2save)
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
    for dict2load, name2load in zip(
        [model, criterion, optimizer, scheduler],
        ["model", "criterion", "optimizer", "scheduler"],
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


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Updates the `target` data with the `source` one
    smoothing by ``tau`` (inplace operation).

    Args:
        target: nn.Module to update
        source: nn.Module for updating
        tau: smoothing parametr

    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def mixup_batch(
    batch: List[torch.Tensor], alpha: float = 0.2, mode: str = "replace"
) -> List[torch.Tensor]:
    """

    Args:
        batch: batch to which you want to apply augmentation
        alpha: beta distribution a=b parameters.
            Must be >=0. The closer alpha to zero the less effect of the mixup.
        mode: algorithm used for muxup: ``"replace"`` | ``"add"``. If "replace"
            then replaces the batch with a mixed one, while the batch size is not changed
            If "add", concatenates mixed examples to the current ones,
            the batch size increases by 2 times.

    Returns:
        augmented batch

    """
    assert alpha >= 0, "alpha must be>=0"
    assert mode in ("add", "replace"), f"mode must be in 'add', 'replace', get: {mode}"

    batch_size = batch[0].shape[0]
    beta = np.random.beta(alpha, alpha, batch_size).astype(np.float32)
    indexes = np.arange(batch_size)
    # index shift by 1
    indexes_2 = (indexes + 1) % batch_size
    for idx, targets in enumerate(batch):
        device = targets.device
        targets_shape = [batch_size] + [1] * len(targets.shape[1:])
        key_beta = torch.as_tensor(beta.reshape(targets_shape), device=device)
        targets = targets * key_beta + targets[indexes_2] * (1 - key_beta)

        if mode == "replace":
            batch[idx] = targets
        else:
            # mode == 'add'
            batch[idx] = torch.cat([batch[idx], targets])
    return batch


__all__ = [
    "get_optimizer_momentum",
    "get_optimizer_momentum_list",
    "set_optimizer_momentum",
    "get_device",
    "get_available_gpus",
    "any2device",
    "prepare_cudnn",
    "get_requires_grad",
    "set_requires_grad",
    "get_available_engine",
    # "pack_checkpoint",
    # "unpack_checkpoint",
    # "save_checkpoint",
    # "load_checkpoint",
    "soft_update",
    "mixup_batch",
]
