from torch import nn
from torch.nn import functional as F


class Flatten(nn.Module):
    """
    @TODO: Docs. Contribution is welcome
    """

    def __init__(self):
        """
        @TODO: Docs. Contribution is welcome
        """
        super().__init__()

    def forward(self, x):
        """
        @TODO: Docs. Contribution is welcome
        """
        return x.view(x.shape[0], -1)


class Lambda(nn.Module):
    """
    @TODO: Docs. Contribution is welcome
    """

    def __init__(self, lambda_fn):
        """
        @TODO: Docs. Contribution is welcome
        """
        super().__init__()
        self.lambda_fn = lambda_fn

    def forward(self, x):
        """
        @TODO: Docs. Contribution is welcome
        """
        return self.lambda_fn(x)


class Normalize(nn.Module):
    """
    @TODO: Docs. Contribution is welcome
    """

    def __init__(self, **normalize_kwargs):
        """
        @TODO: Docs. Contribution is welcome
        """
        super().__init__()
        self.normalize_kwargs = normalize_kwargs

    def forward(self, x):
        """
        @TODO: Docs. Contribution is welcome
        """
        return F.normalize(x, **self.normalize_kwargs)
