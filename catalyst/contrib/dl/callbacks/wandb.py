from typing import Dict, List

import wandb

from catalyst import utils
from catalyst.core import (
    Callback,
    CallbackNode,
    CallbackOrder,
    CallbackScope,
    State,
)


class WandbLogger(Callback):
    """Logger callback, translates ``state.*_metrics`` to Weights & Biases.
    Read about Weights & Biases here https://docs.wandb.com/

    Example:
        .. code-block:: python

            from catalyst import dl
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset

            class Projector(nn.Module):
                def __init__(self, input_size):
                    super().__init__()
                    self.linear = nn.Linear(input_size, 1)

                def forward(self, X):
                    return self.linear(X).squeeze(-1)

            X = torch.rand(16, 10)
            y = torch.rand(X.shape[0])
            model = Projector(X.shape[1])
            dataset = TensorDataset(X, y)
            loader = DataLoader(dataset, batch_size=8)
            runner = dl.SupervisedRunner()

            runner.train(
                model=model,
                loaders={
                    "train": loader,
                    "valid": loader
                },
                criterion=nn.MSELoss(),
                optimizer=optim.Adam(model.parameters()),
                logdir="log_example",
                callbacks=[
                    dl.callbacks.WandbLogger(
                        project="wandb_logger_example"
                    )
                ],
                num_epochs=10
            )
    """

    def __init__(
        self,
        metric_names: List[str] = None,
        log_on_batch_end: bool = False,
        log_on_epoch_end: bool = True,
        **logging_params,
    ):
        """
        Args:
            metric_names (List[str]): list of metric names to log,
                if None - logs everything
            log_on_batch_end (bool): logs per-batch metrics if set True
            log_on_epoch_end (bool): logs per-epoch metrics if set True
            **logging_params: any parameters of function `wandb.init`
                except `reinit` which is automatically set to `True`
                and `dir` which is set to `<logdir>`
        """
        super().__init__(
            order=CallbackOrder.Logging,
            node=CallbackNode.Master,
            scope=CallbackScope.Experiment,
        )
        self.metrics_to_log = metric_names
        self.log_on_batch_end = log_on_batch_end
        self.log_on_epoch_end = log_on_epoch_end

        if not (self.log_on_batch_end or self.log_on_epoch_end):
            raise ValueError("You have to log something!")

        if (self.log_on_batch_end and not self.log_on_epoch_end) or (
            not self.log_on_batch_end and self.log_on_epoch_end
        ):
            self.batch_log_suffix = ""
            self.epoch_log_suffix = ""
        else:
            self.batch_log_suffix = "_batch"
            self.epoch_log_suffix = "_epoch"

        self.logging_params = logging_params

    def _log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        mode: str,
        suffix="",
        commit=True,
    ):
        if self.metrics_to_log is None:
            metrics_to_log = sorted(metrics.keys())
        else:
            metrics_to_log = self.metrics_to_log

        def key_locate(key: str):
            """
            Wandb uses first symbol _ for it service purposes
            because of that fact, we can not send original metric names

            Args:
                key: metric name

            Returns:
                formatted metric name
            """
            if key.startswith("_"):
                return key[1:]
            return key

        metrics = {
            f"{key_locate(key)}/{mode}{suffix}": value
            for key, value in metrics.items()
            if key in metrics_to_log
        }
        wandb.log(metrics, step=step, commit=commit)

    def on_stage_start(self, state: State):
        """Initialize Weights & Biases."""
        wandb.init(**self.logging_params, reinit=True, dir=str(state.logdir))

    def on_stage_end(self, state: State):
        """Finish logging to Weights & Biases."""
        wandb.join()

    def on_batch_end(self, state: State):
        """Translate batch metrics to Weights & Biases."""
        if self.log_on_batch_end:
            mode = state.loader_name
            metrics_ = state.batch_metrics
            self._log_metrics(
                metrics=metrics_,
                step=state.global_step,
                mode=mode,
                suffix=self.batch_log_suffix,
                commit=True,
            )

    def on_loader_end(self, state: State):
        """Translate loader metrics to Weights & Biases."""
        if self.log_on_epoch_end:
            mode = state.loader_name
            metrics_ = state.loader_metrics
            self._log_metrics(
                metrics=metrics_,
                step=state.global_epoch,
                mode=mode,
                suffix=self.epoch_log_suffix,
                commit=False,
            )

    def on_epoch_end(self, state: State):
        """Translate epoch metrics to Weights & Biases."""
        extra_mode = "_base"
        splitted_epoch_metrics = utils.split_dict_to_subdicts(
            dct=state.epoch_metrics,
            prefixes=list(state.loaders.keys()),
            extra_key=extra_mode,
        )

        if self.log_on_epoch_end:
            self._log_metrics(
                metrics=splitted_epoch_metrics[extra_mode],
                step=state.global_epoch,
                mode=extra_mode,
                suffix=self.epoch_log_suffix,
                commit=True,
            )
