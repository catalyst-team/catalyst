from typing import Any, Callable, Dict, List, Mapping, Tuple, Union
from collections import OrderedDict
import logging
import os

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from catalyst.callbacks.batch_overfit import BatchOverfitCallback
from catalyst.callbacks.checkpoint import CheckpointCallback, ICheckpointCallback
from catalyst.callbacks.criterion import CriterionCallback, ICriterionCallback
from catalyst.callbacks.misc import CheckRunCallback, TimerCallback, VerboseCallback
from catalyst.callbacks.optimizer import IOptimizerCallback, OptimizerCallback
from catalyst.callbacks.scheduler import ISchedulerCallback, SchedulerCallback
from catalyst.core.callback import Callback
from catalyst.core.engine import IEngine
from catalyst.core.functional import check_callback_isinstance, sort_callbacks_by_order
from catalyst.core.runner import IRunner, IStageBasedRunner
from catalyst.core.trial import ITrial
from catalyst.experiments.experiment import Experiment
from catalyst.loggers.console import ConsoleLogger
from catalyst.loggers.csv import CSVLogger
from catalyst.loggers.tensorboard import TensorboardLogger
from catalyst.runners.runner import Runner
from catalyst.typing import Criterion, Device, Model, Optimizer, RunnerModel, Scheduler

logger = logging.getLogger(__name__)


class SupervisedRunner(Runner):
    """Runner for experiments with supervised model."""

    def __init__(
        self,
        model: RunnerModel = None,
        engine: IEngine = None,
        input_key: Any = "features",
        output_key: Any = "logits",
        target_key: str = "targets",
        experiment_fn: Callable = Experiment,
    ):
        """
        Args:
            model: Torch model object
            device: Torch device
            input_key: Key in batch dict mapping for model input
            output_key: Key in output dict model output
                will be stored under
            target_key: Key in batch dict mapping for target
            experiment_fn: callable function,
                which defines default experiment type to use
                during ``.train`` and ``.infer`` methods.
        """
        super().__init__(model=model, engine=engine, experiment_fn=experiment_fn)

        self.input_key = input_key
        self.output_key = output_key
        self.target_key = target_key

        if isinstance(self.input_key, str):
            # when model expects value
            self._process_input = self._process_input_str
        elif isinstance(self.input_key, (list, tuple)):
            # when model expects tuple
            self._process_input = self._process_input_list
        elif self.input_key is None:
            # when model expects dict
            self._process_input = self._process_input_none
        else:
            raise NotImplementedError()

        if isinstance(output_key, str):
            # when model returns value
            self._process_output = self._process_output_str
        elif isinstance(output_key, (list, tuple)):
            # when model returns tuple
            self._process_output = self._process_output_list
        elif self.output_key is None:
            # when model returns dict
            self._process_output = self._process_output_none
        else:
            raise NotImplementedError()

    def _process_batch(self, batch):
        if isinstance(batch, (tuple, list)):
            assert len(batch) == 2
            batch = {self.input_key: batch[0], self.target_key: batch[1]}
        return batch

    def _process_input_str(self, batch: Mapping[str, Any], **kwargs):
        output = self.model(batch[self.input_key], **kwargs)
        return output

    def _process_input_list(self, batch: Mapping[str, Any], **kwargs):
        input = {key: batch[key] for key in self.input_key}  # noqa: WPS125
        output = self.model(**input, **kwargs)
        return output

    def _process_input_none(self, batch: Mapping[str, Any], **kwargs):
        output = self.model(**batch, **kwargs)
        return output

    def _process_output_str(self, output: torch.Tensor):
        output = {self.output_key: output}
        return output

    def _process_output_list(self, output: Union[Tuple, List]):
        output = {key: value for key, value in zip(self.output_key, output)}
        return output

    def _process_output_none(self, output: Mapping[str, Any]):
        return output

    def forward(self, batch: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
        """
        Forward method for your Runner.
        Should not be called directly outside of runner.
        If your model has specific interface, override this method to use it

        Args:
            batch (Mapping[str, Any]): dictionary with data batches
                from DataLoaders.
            **kwargs: additional parameters to pass to the model

        Returns:
            dict with model output batch
        """
        output = self._process_input(batch, **kwargs)
        output = self._process_output(output)
        return output

    def on_batch_start(self, runner: "IRunner"):
        self.batch = self._process_batch(self.batch)
        super().on_batch_start(runner)

    def handle_batch(self, batch: Mapping[str, Any]) -> None:
        """
        Inner method to handle specified data batch.
        Used to make a train/valid/infer stage during Experiment run.

        Args:
            batch (Mapping[str, Any]): dictionary with data batches
                from DataLoader.
        """
        self.batch = {**batch, **self.forward(batch)}

    @torch.no_grad()
    def predict_batch(self, batch: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
        """
        Run model inference on specified data batch.

        .. warning::
            You should not override this method. If you need specific model
            call, override forward() method

        Args:
            batch (Mapping[str, Any]): dictionary with data batches
                from DataLoader.
            **kwargs: additional kwargs to pass to the model

        Returns:
            Mapping[str, Any]: model output dictionary
        """
        self._process_batch(batch)
        batch = self.engine.sync_device(batch)
        output = self.forward(batch, **kwargs)
        return output

    def _process_train_callbacks(
        self,
        *,
        # the data
        loaders: "OrderedDict[str, DataLoader]",
        # the core
        model: Model,
        engine: Union["IEngine", str] = None,
        trial: ITrial = None,
        # the components
        criterion: Criterion = None,
        optimizer: Optimizer = None,
        scheduler: Scheduler = None,
        # the callbacks
        callbacks: "Union[List[Callback], OrderedDict[str, Callback]]" = None,
        # extra info (callbacks info)
        logdir: str = None,
        resume: str = None,
        valid_loader: str = "valid",
        main_metric: str = "loss",
        minimize_metric: bool = True,
        verbose: bool = False,
        timeit: bool = False,
        check: bool = False,
        overfit: bool = False,
        load_best_on_end: bool = False,
    ):
        # callbacks handling
        callbacks = super()._process_train_callbacks(
            loaders=loaders,
            model=model,
            engine=engine,
            trial=trial,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            callbacks=callbacks,
            logdir=logdir,
            resume=resume,
            valid_loader=valid_loader,
            main_metric=main_metric,
            minimize_metric=minimize_metric,
            verbose=verbose,
            timeit=timeit,
            check=check,
            overfit=overfit,
            load_best_on_end=load_best_on_end,
        )
        is_callback_exists = lambda callback_fn: any(
            check_callback_isinstance(x, callback_fn) for x in callbacks.values()
        )
        if isinstance(criterion, Criterion) and not is_callback_exists(ICriterionCallback):
            callbacks["_criterion"] = CriterionCallback(
                input_key=self.output_key, target_key=self.target_key, metric_key="loss",
            )
        if isinstance(optimizer, Optimizer) and not is_callback_exists(IOptimizerCallback):
            callbacks["_optimizer"] = OptimizerCallback(metric_key="loss",)
        if isinstance(scheduler, (Scheduler, ReduceLROnPlateau)) and not is_callback_exists(
            ISchedulerCallback
        ):
            callbacks["_scheduler"] = SchedulerCallback(
                loader_key=valid_loader, metric_key=main_metric
            )
        return callbacks


__all__ = ["SupervisedRunner"]
