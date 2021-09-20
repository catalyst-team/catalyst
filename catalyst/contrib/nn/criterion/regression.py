import torch
from torch import nn


def _ce_with_logits(logits, target):
    """Returns cross entropy for giving logits"""
    return torch.sum(-target * torch.log_softmax(logits, -1), -1)


class HuberLossV0(nn.Module):
    """@TODO: Docs. Contribution is welcome."""

    def __init__(self, clip_delta=1.0, reduction="mean"):
        """@TODO: Docs. Contribution is welcome."""
        super().__init__()
        self.clip_delta = clip_delta
        self.reduction = reduction or "none"

    def forward(self, output: torch.Tensor, target: torch.Tensor, weights=None) -> torch.Tensor:
        """@TODO: Docs. Contribution is welcome."""
        diff = target - output
        diff_abs = torch.abs(diff)
        quadratic_part = torch.clamp(diff_abs, max=self.clip_delta)
        linear_part = diff_abs - quadratic_part
        loss = 0.5 * quadratic_part ** 2 + self.clip_delta * linear_part

        if weights is not None:
            loss = torch.mean(loss * weights, dim=1)
        else:
            loss = torch.mean(loss, dim=1)

        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)

        return loss


class CategoricalRegressionLoss(nn.Module):
    """CategoricalRegressionLoss"""

    def __init__(self, num_atoms: int, v_min: int, v_max: int):
        super().__init__()
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        self.z = torch.linspace(start=self.v_min, end=self.v_max, steps=self.num_atoms)

    def forward(
        self, logits_t: torch.Tensor, logits_tp1: torch.Tensor, atoms_target_t: torch.Tensor
    ) -> torch.Tensor:
        """Compute the loss

        Args:
            logits_t (torch.Tensor): predicted atoms at step T, shape: [bs; num_atoms]
            logits_tp1 (torch.Tensor): predicted atoms at step T+1, shape: [bs; num_atoms]
            atoms_target_t (torch.Tensor): target atoms at step T, shape: [bs; num_atoms]

        Returns:
            torch.Tensor: computed loss
        """
        probs_tp1 = torch.softmax(logits_tp1, dim=-1)
        tz = torch.clamp(atoms_target_t, self.v_min, self.v_max)
        tz_z = torch.abs(tz[:, None, :] - self.z[None, :, None])
        tz_z = torch.clamp(1.0 - (tz_z / self.delta_z), 0.0, 1.0)
        probs_target_t = torch.einsum("bij,bj->bi", (tz_z, probs_tp1)).detach()
        loss = _ce_with_logits(logits_t, probs_target_t).mean()
        return loss


class QuantileRegressionLoss(nn.Module):
    """QuantileRegressionLoss"""

    def __init__(self, num_atoms: int = 51, clip_delta: float = 1.0):
        """Init."""
        super().__init__()
        self.num_atoms = num_atoms
        tau_min = 1 / (2 * self.num_atoms)
        tau_max = 1 - tau_min
        self.tau = torch.linspace(start=tau_min, end=tau_max, steps=self.num_atoms)
        self.criterion = HuberLossV0(clip_delta=clip_delta)

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the loss.

        Args:
            outputs (torch.Tensor): predicted atoms, shape: [bs; num_atoms]
            targets (torch.Tensor): target atoms, shape: [bs; num_atoms]

        Returns:
            torch.Tensor: computed loss
        """
        atoms_diff = targets[:, None, :] - outputs[:, :, None]
        delta_atoms_diff = atoms_diff.lt(0).to(torch.float32).detach()
        huber_weights = torch.abs(self.tau[None, :, None] - delta_atoms_diff) / self.num_atoms
        loss = self.criterion(outputs[:, :, None], targets[:, None, :], huber_weights).mean()
        return loss


__all__ = ["HuberLossV0", "CategoricalRegressionLoss", "QuantileRegressionLoss"]
