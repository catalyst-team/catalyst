from catalyst.dl.core import State


class GanState(State):
    """
    An object that is used to pass internal state during
    train/valid/infer in GAN Runners.
    """
    def __init__(self, *, batch_consistant_metrics: bool = False, **kwargs):
        self.phase = None
        super().__init__(
            batch_consistant_metrics=batch_consistant_metrics,
            **kwargs,
        )


__all__ = ["GanState"]
