# flake8: noqa
# @TODO: code formatting issue for 20.07 release
from typing import Dict

from torch import nn


class ABN(nn.Module):
    """Activated Batch Normalization.

    This gathers a `BatchNorm2d` and an activation function in a single module.

    @TODO: Docs (add `Example`). Contribution is welcome.
    """

    def __init__(
        self,
        num_features: int,
        activation: str = "leaky_relu",
        batchnorm_params: Dict = None,
        activation_params: Dict = None,
        use_batchnorm: bool = True,
    ):
        """
        Args:
            num_features: number of feature channels
                in the input and output
            activation: name of the activation functions, one of:
                ``'leaky_relu'``, ``'elu'`` or ``'none'``.
            batchnorm_params: additional ``nn.BatchNorm2d`` params
            activation_params: additional params for activation fucntion
            use_batchnorm: @TODO: Docs. Contribution is welcome
        """
        super().__init__()
        batchnorm_params = batchnorm_params or {}
        activation_params = activation_params or {}

        layers = []
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(num_features=num_features, **batchnorm_params))
        if activation is not None and activation.lower() != "none":
            layers.append(nn.__dict__[activation](inplace=True, **activation_params))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Forward call."""
        x = self.net(x)
        return x


__all__ = ["ABN"]
