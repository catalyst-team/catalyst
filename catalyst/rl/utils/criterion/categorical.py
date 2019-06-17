import torch
from catalyst.utils import ce_with_logits


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
