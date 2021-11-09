from torch import nn


class ModelForwardWrapper(nn.Module):
    """Model that calls specified method instead of forward.

    Args:
        model: @TODO: docs
        method_name: @TODO: docs

    (Workaround, single method tracing is not supported)
    """

    def __init__(self, model, method_name):
        """Init"""
        super().__init__()
        self.model = model
        self.method_name = method_name

    def forward(self, *args, **kwargs):
        """Forward pass.

        Args:
            *args: some args
            **kwargs: some kwargs

        Returns:
            output: specified method output
        """
        return getattr(self.model, self.method_name)(*args, **kwargs)


__all__ = ["ModelForwardWrapper"]
