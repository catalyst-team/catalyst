import torch


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
