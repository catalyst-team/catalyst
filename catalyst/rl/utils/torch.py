from copy import deepcopy

import torch
from catalyst.rl import utils
from catalyst.rl.registry import \
    CRITERIONS, GRAD_CLIPPERS, OPTIMIZERS, SCHEDULERS


def get_network_weights(network, exclude_norm=False):
    """
    Args:
        network: torch.nn.Module, neural network, e.g. actor or critic
        exclude_norm: True if layers corresponding to norm will be excluded
    Returns:
        state_dict: dictionary which contains neural network parameters
    """
    state_dict = network.state_dict()
    if exclude_norm:
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if all(x not in key for x in ["norm", "lstm"])
        }
    state_dict = {key: value.clone() for key, value in state_dict.items()}
    return state_dict


def set_network_weights(network, weights, strict=True):
    network.load_state_dict(weights, strict=strict)


def _copy_params(params):
    if params is None:
        return {}
    return deepcopy(params)


def get_trainer_components(
    *,
    agent,
    loss_params=None,
    optimizer_params=None,
    scheduler_params=None,
    grad_clip_params=None
):
    # criterion
    loss_params = _copy_params(loss_params)
    criterion = CRITERIONS.get_from_params(**loss_params)
    if criterion is not None \
            and torch.cuda.is_available():
        criterion = criterion.cuda()

    # optimizer
    agent_params = utils.get_optimizable_params(agent.parameters())
    optimizer_params = _copy_params(optimizer_params)
    optimizer = OPTIMIZERS.get_from_params(
        **optimizer_params, params=agent_params
    )

    # scheduler
    scheduler_params = _copy_params(scheduler_params)
    scheduler = SCHEDULERS.get_from_params(
        **scheduler_params, optimizer=optimizer
    )

    # grad clipping
    grad_clip_params = _copy_params(grad_clip_params)
    grad_clip_fn = GRAD_CLIPPERS.get_from_params(**grad_clip_params)

    result = {
        "loss_params": loss_params,
        "criterion": criterion,
        "optimizer_params": optimizer_params,
        "optimizer": optimizer,
        "scheduler_params": scheduler_params,
        "scheduler": scheduler,
        "grad_clip_params": grad_clip_params,
        "grad_clip_fn": grad_clip_fn
    }

    return result
