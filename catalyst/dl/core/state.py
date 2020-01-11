from typing import Dict  # isort:skip
from collections import defaultdict

from torch.optim.optimizer import Optimizer

from catalyst.core import State


# TODO Deep refactoring
#  - lr/loss/momentum bypass (how to deal when multiple optimizers?)
class DLRunnerState(State):
    """
    An object that is used to pass internal state during train/valid/infer.
    """

    def __init__(
        self,
        *,
        device=None,
        model=None,
        criterion=None,
        optimizer: Optimizer = None,
        scheduler=None,
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
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        # @TODO: remove GAN hack
        self.phase = None

        # data pipeline
        self.input = None
        self.output = None

        # base metrics
        single_optimizer = isinstance(optimizer, Optimizer)
        self.lr = None if single_optimizer else defaultdict(lambda: None)
        self.momentum = None if single_optimizer else defaultdict(lambda: None)
        self.loss = None

        super().__init__(
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

    def _handle_runner_metrics(self):
        values = {}
        for key, value in zip(
            ["_base/lr", "_base/momentum"], [self.lr, self.momentum]
        ):
            if value is not None:
                if isinstance(value, dict):
                    for k, v in value.items():
                        values[f"{key}/{k}"] = v
                else:
                    values[key] = value

        values.update(self.timer.elapsed)

        values["_timers/_fps"] = \
            self.batch_size / self.timer.elapsed["_timers/batch_time"]

        self.metrics.add_batch_value(metrics_dict=values)


__all__ = ["DLRunnerState"]
