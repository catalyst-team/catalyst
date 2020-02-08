import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    An implementation of [RMS Normalization](
    https://openreview.net/pdf?id=SygkZ3MTJE.
    RMS Normalization stabilises the training of deep neural networks by
    normalising the outputs of neurons from a particular layer. It computes:
    output = (scale * (tensor / (std(tensor) + epsilon)) + bias
    where std denotes standard-deviation
    # Parameters
    dimension : `int`, required.
    The dimension of the layer output to normalize.
    epsilon : `float`, optional, (default = 1e-8)
    An epsilon to prevent dividing by zero in the case
    the layer has zero variance.
    is_bias : `bool`, optional, (default = False)
    A boolean value whether to include bias term while normalization.
    # Returns
    The normalized layer output.
    """  # noqa
    def __init__(self, dimension, epsilon=1e-8, is_bias=False):
        super().__init__()
        self.dimension = dimension
        self.epsilon = epsilon
        self.is_bias = is_bias
        self.scale = nn.Parameter(torch.ones(self.dimension))
        if(self.is_bias):
            self.bias = nn.Parameter(torch.zeros(self.dimension))

    def forward(self, x: torch.Tensor):
        x_std = torch.sqrt(torch.mean(x**2, -1, keepdim=True))
        x_norm = x / (x_std + self.epsilon)
        if(self.is_bias):
            return self.scale * x_norm + self.bias
        return self.scale * x_norm
