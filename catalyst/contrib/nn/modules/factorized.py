from typing import Union

import torch
from torch import nn


class FactorizedLinear(nn.Module):
    """Factorized wrapper for ``nn.Linear``

    Args:
        nn_linear: torch ``nn.Linear`` module
        dim_ratio: dimension ration to use after weights SVD
    """

    def __init__(self, nn_linear: nn.Linear, dim_ratio: Union[int, float] = 1.0):
        super().__init__()
        self.bias = nn.parameter.Parameter(nn_linear.bias.data, requires_grad=True)
        u, vh = self._spectral_init(nn_linear.weight.data, dim_ratio=dim_ratio)
        # print(f"Doing SVD of tensor {or_linear.weight.shape}, U: {u.shape}, Vh: {vh.shape}")
        self.u = nn.parameter.Parameter(u, requires_grad=True)
        self.vh = nn.parameter.Parameter(vh, requires_grad=True)
        self.dim_ratio = dim_ratio
        self.in_features = u.size(0)
        self.out_features = vh.size(1)

    @staticmethod
    def _spectral_init(m, dim_ratio: Union[int, float] = 1):
        u, s, vh = torch.linalg.svd(m, full_matrices=False)
        u = u @ torch.diag(torch.sqrt(s))
        vh = torch.diag(torch.sqrt(s)) @ vh
        if dim_ratio < 1:
            dims = int(u.size(1) * dim_ratio)
            u = u[:, :dims]
            vh = vh[:dims, :]
            # s_share = s[:dims].sum() / s.sum() * 100
            # print(f"SVD eigenvalue share {s_share:.2f}%")
        return u, vh

    def extra_repr(self) -> str:
        """Extra representation log."""
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias=True, dim_ratio={self.dim_ratio}"
        )

    def forward(self, x: torch.Tensor):
        """Forward call."""
        return x @ (self.u @ self.vh).transpose(0, 1) + self.bias


__all__ = ["FactorizedLinear"]
