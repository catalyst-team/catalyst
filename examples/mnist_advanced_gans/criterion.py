"""
All custom criterion modules
"""
import torch
from torch import nn


class MeanOutputLoss(nn.Module):
    """
    Criterion to compute simple mean of the output, completely ignoring target
    (maybe useful e.g. for WGAN real/fake validity averaging
    """

    def forward(self, output, target):
        return output.mean()


class GradientPenaltyLoss(nn.Module):
    """Criterion to compute gradient penalty

    WARN: SHOULD NOT BE RUN WITH CriterionCallback,
        use special MultiKeyCriterionCallback instead
    """

    def forward(self, fake_data, real_data, discriminator):
        device = real_data.device
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand((real_data.size(0), 1, 1, 1), device=device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_data + ((1 - alpha) * fake_data))
        interpolates.requires_grad_(True)
        d_interpolates = discriminator(interpolates)
        if not d_interpolates.requires_grad:
            # TODO:
            #  deal with it (outputs does not require grad in validation mode)
            # raise ValueError("Why the hell??? one of D inputs "
            #                  "has requires_grad=True, so output "
            #                  "should also have requires_grad=True")
            return torch.zeros((real_data.size(0), 1), device=device).mean()
        fake = torch.ones((real_data.size(0), 1), device=device,
                          requires_grad=False)
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
