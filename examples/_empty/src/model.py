import torch
from catalyst.contrib import registry


@registry.Model
class Model(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        # CHANGE ME
        return x

    @classmethod
    def get_from_params(
        cls,
        **model_params,
    ) -> "Model":
        # CHANGE ME
        model = cls(**model_params)
        return model
