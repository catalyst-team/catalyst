from typing import Dict  # isort:skip

from catalyst.core import _State
from catalyst.utils.tools.typing import (
    Criterion, Device, Model, Optimizer, Scheduler
)


class State(_State):
    """
    An object that is used to pass internal state during train/valid/infer.
    """

    def __init__(
        self,
        *,
        device: Device = None,
        model: Model = None,
        criterion: Criterion = None,
        optimizer: Optimizer = None,
        scheduler: Scheduler = None,
        logdir: str = None,
        stage: str = "infer",
        num_epochs: int = 1,
        main_metric: str = "loss",
        minimize_metric: bool = True,
        valid_loader: str = "valid",
        verbose: bool = False,
        checkpoint_data: Dict = None,
        batch_consistant_metrics: bool = True,
        **kwargs
    ):

        # @TODO: remove GAN hack
        self.phase = None

        super().__init__(
            device=device,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            logdir=logdir,
            stage=stage,
            num_epochs=num_epochs,
            main_metric=main_metric,
            minimize_metric=minimize_metric,
            valid_loader=valid_loader,
            verbose=verbose,
            checkpoint_data=checkpoint_data,
            batch_consistant_metrics=batch_consistant_metrics,
            **kwargs
        )


__all__ = ["State"]
