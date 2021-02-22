# flake8: noqa
import torch
from torch import nn


class MeanOutputLoss(nn.Module):
    """
    Criterion to compute simple mean of the output, completely ignoring target
    (maybe useful e.g. for WGAN real/fake validity averaging.
    """

    def forward(self, output, target):
        """Compute criterion.
        @TODO: Docs (add typing). Contribution is welcome.
        """
        return output.mean()


class GradientPenaltyLoss(nn.Module):
    """Criterion to compute gradient penalty.

    WARN: SHOULD NOT BE RUN WITH CriterionCallback,
        use special GradientPenaltyCallback instead
    """

    def forward(self, fake_data, real_data, critic, critic_condition_args):
        """Compute gradient penalty.
        @TODO: Docs. Contribution is welcome.
        """
        device = real_data.device
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand((real_data.size(0), 1, 1, 1), device=device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_data + ((1 - alpha) * fake_data)).detach()
        interpolates.requires_grad_(True)
        with torch.set_grad_enabled(True):  # to compute in validation mode
            d_interpolates = critic(interpolates, *critic_condition_args)

        fake = torch.ones((real_data.size(0), 1), device=device, requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty


__all__ = ["MeanOutputLoss", "GradientPenaltyLoss"]
