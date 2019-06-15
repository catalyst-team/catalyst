import safitty
from torch.optim import Optimizer


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
