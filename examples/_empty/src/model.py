import torch

from catalyst.registry import Model


@Model
class Model(torch.nn.Module):
    """
    Your model
    """

    def __init__(self, **kwargs):
        """
        Model init.

        Args:
            **kwargs: model params
        """
        super().__init__()

    def forward(self, x):
        """
        Model forward pass.

        Args:
            x: features

        Returns:
            features
        """
        # CHANGE ME
        return x

    @classmethod
    def get_from_params(cls, **model_params) -> "Model":
        """
        Model init from config.

        Args:
            **model_params: model params

        Returns:
            model
        """
        # CHANGE ME
        model = cls(**model_params)
        return model
