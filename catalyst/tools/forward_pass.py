from torch import nn


class ForwardOverrideModel(nn.Module):
    """Model that calls specified method instead of forward.

    (Workaround, single method tracing is not supported)
    """

    def __init__(self, model, method_name):
        super().__init__()
        self.model = model
        self.method_name = method_name

    def forward(self, *args, **kwargs):
        """forward pass

        Returns:
            output: specified method output
        """
        return getattr(self.model, self.method_name)(*args, **kwargs)


__all__ = ["ForwardOverrideModel"]
