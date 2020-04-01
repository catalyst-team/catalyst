import torch

from catalyst.contrib import registry


@registry.Model
class Model(torch.nn.Module):
    """
    @TODO: Docs. Contribution is welcome
    """

    def __init__(self, **kwargs):
        """
        @TODO: Docs. Contribution is welcome
        """
        super().__init__()

    def forward(self, x):
        """
        @TODO: Docs. Contribution is welcome
        """
        # CHANGE ME
        return x

    @classmethod
    def get_from_params(cls, **model_params,) -> "Model":
        """
        @TODO: Docs. Contribution is welcome
        """
        # CHANGE ME
        model = cls(**model_params)
        return model
