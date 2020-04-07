from torch import nn
from torch.nn import functional as F


class Flatten(nn.Module):
    """Flattens the input. Does not affect the batch size.

    @TODO: Docs (add `Example`). Contribution is welcome.
    """

    def __init__(self):
        """@TODO: Docs. Contribution is welcome."""
        super().__init__()

    def forward(self, x):
        """Forward call."""
        return x.view(x.shape[0], -1)


class Lambda(nn.Module):
    """@TODO: Docs. Contribution is welcome."""

    def __init__(self, lambda_fn):
        """@TODO: Docs. Contribution is welcome."""
        super().__init__()
        self.lambda_fn = lambda_fn

    def forward(self, x):
        """Forward call."""
        return self.lambda_fn(x)


class Normalize(nn.Module):
    """Performs :math:`L_p` normalization of inputs over specified dimension.

    @TODO: Docs (add `Example`). Contribution is welcome.
    """

    def __init__(self, **normalize_kwargs):
        """
        Args:
            **normalize_kwargs: see ``torch.nn.functional.normalize`` params
        """
        super().__init__()
        self.normalize_kwargs = normalize_kwargs

    def forward(self, x):
        """Forward call."""
        return F.normalize(x, **self.normalize_kwargs)
