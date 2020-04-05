from catalyst.core import State


class GanState(State):
    """
    An object that is used to pass internal state during
    train/valid/infer in GAN Runners.
    """

    def __init__(self, **kwargs):
        """
        Args:
            @TODO: Docs. Contribution is welcome
        """
        self.phase = None
        super().__init__(**kwargs)


__all__ = ["GanState"]
