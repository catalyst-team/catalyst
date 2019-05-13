import torch
from catalyst.dl.utils import UtilsFactory
from catalyst.rl.registry import \
    CRITERIONS, GRAD_CLIPPERS, OPTIMIZERS, SCHEDULERS


def _copy_params(params):
    if params is None:
        return {}
    return params.copy()


def get_agent_stuff_from_params(
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
    agent_params = UtilsFactory.prepare_optimizable_params(
        agent.parameters())
    optimizer_params = _copy_params(optimizer_params)
    optimizer = OPTIMIZERS.get_from_params(
        **optimizer_params,
        params=agent_params
    )

    # scheduler
    scheduler_params = _copy_params(scheduler_params)
    scheduler = SCHEDULERS.get_from_params(
        **scheduler_params,
        optimizer=optimizer
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


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def ce_with_logits(logits, target):
    return torch.sum(-target * torch.log_softmax(logits, -1), -1)


def categorical_loss(
    logits_t, logits_tp1, atoms_target_t, z, delta_z, v_min, v_max
):
    """
    Parameters
    ----------
    logits_t:        logits of categorical VD at (s_t, a_t)
    logits_tp1:      logits of categorical VD at (s_tp1, a_tp1)
    atoms_target_t:  target VD support
    z:               support of categorical VD at (s_t, a_t)
    delta_z:         fineness of categorical VD
    v_min, v_max:    left and right borders of catgorical VD
    """
    probs_tp1 = torch.softmax(logits_tp1, dim=-1)
    tz = torch.clamp(atoms_target_t, v_min, v_max)
    tz_z = torch.abs(tz[:, None, :] - z[None, :, None])
    tz_z = torch.clamp(1.0 - (tz_z / delta_z), 0., 1.)
    probs_target_t = torch.einsum("bij,bj->bi", (tz_z, probs_tp1)).detach()
    loss = ce_with_logits(logits_t, probs_target_t).mean()
    return loss


def quantile_loss(atoms_t, atoms_target_t, tau, num_atoms, criterion):
    """
    Parameters
    ----------
    atoms_t:         support of quantile VD at (s_t, a_t)
    atoms_target_t:  target VD support
    tau:             positions of quantiles where VD is approximated
    num_atoms:       number of atoms in quantile VD
    criterion:       loss function, usually Huber loss
    """
    atoms_diff = atoms_target_t[:, None, :] - atoms_t[:, :, None]
    delta_atoms_diff = atoms_diff.lt(0).to(torch.float32).detach()
    huber_weights = torch.abs(
        tau[None, :, None] - delta_atoms_diff
    ) / num_atoms
    loss = criterion(
        atoms_t[:, :, None], atoms_target_t[:, None, :], huber_weights
    ).mean()
    return loss
