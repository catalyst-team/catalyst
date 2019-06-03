from typing import Dict

import torch.nn as nn


class ABN(nn.Module):
    """
    Activated Batch Normalization
    This gathers a `BatchNorm2d` and an activation function in a single module
    """

    def __init__(
        self,
        num_features,
        activation="leaky_relu",
        batchnorm_params: Dict = None,
        activation_params: Dict = None,
        use_batchnorm: bool = True
    ):
        """
        Create an Activated Batch Normalization module
        Parameters
        ----------
        num_features : int
            Number of feature channels in the input and output.
        activation : str
            Name of the activation functions, one of:
                `leaky_relu`, `elu` or `none`.
        """
        super().__init__()
        batchnorm_params = batchnorm_params or {}
        activation_params = activation_params or {}

        layers = []
        if use_batchnorm:
            layers.append(
                nn.BatchNorm2d(num_features=num_features, **batchnorm_params))
        if activation is not None and activation.lower() != "none":
            layers.append(
                nn.__dict__[activation](inplace=True, **activation_params))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return x
