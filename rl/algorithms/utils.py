import torch


def quantile_loss(atoms, target_atoms, tau, n_atoms, criterion):
    atoms_diff = target_atoms[:, None, :] - atoms[:, :, None]
    delta_atoms_diff = atoms_diff.lt(0).to(torch.float32).detach()
    huber_weights = torch.abs(tau[None, :, None] - delta_atoms_diff) / n_atoms
    loss = criterion(
        atoms[:, :, None], target_atoms[:, None, :], huber_weights
    )
    return loss
