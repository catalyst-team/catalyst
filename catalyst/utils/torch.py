from typing import Iterable
import numpy as np
import safitty
import torch
from torch import nn
from torch.optim import Optimizer


def ce_with_logits(logits, target):
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
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def get_optimizable_params(model_or_params):
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


def assert_fp16_available():
    assert torch.backends.cudnn.enabled, \
        "fp16 mode requires cudnn backend to be enabled."

    try:
        __import__("apex")
    except ImportError:
        assert False, \
            "NVidia Apex package must be installed. " \
            "See https://github.com/NVIDIA/apex."


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_activation_fn(activation: str = None):
    if activation is None or activation.lower() == "none":
        activation_fn = lambda x: x  # noqa: E731
    else:
        activation_fn = torch.nn.__dict__[activation]()
    return activation_fn


def any2device(value, device):
    """
    Move tensor, list of tensors, list of list of tensors,
    dict of tensors, tuple of tensors to target device.
    :param value: Object to be moved
    :param device: target device ids
    :return: Save data structure holding tensors on target device
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
