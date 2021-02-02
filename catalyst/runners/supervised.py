from typing import Any, Callable, Dict, List, Mapping, Tuple, Union
from collections import OrderedDict
import logging
import os

import torch
from torch.jit import ScriptModule
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
from catalyst.runners.runner import Runner
from catalyst.typing import Criterion, Device, Model, Optimizer, RunnerModel, Scheduler
from catalyst.utils import check_amp_available
from catalyst.utils.checkpoint import load_checkpoint, unpack_checkpoint
from catalyst.utils.components import process_components
from catalyst.utils.misc import maybe_recursive_call, set_global_seed
from catalyst.utils.scripts import distributed_cmd_run
from catalyst.utils.torch import get_device, get_requires_grad, set_requires_grad
from catalyst.utils.tracing import save_traced_model, trace_model

logger = logging.getLogger(__name__)


class SupervisedRunner(Runner):
    """Runner for experiments with supervised model."""

    def __init__(
        self,
        model: RunnerModel = None,
        engine: IEngine = None,
        model_input_key: Any = "features",
        model_output_key: Any = "logits",
        target_key: str = "targets",
        experiment_fn: Callable = Experiment,
    ):
        """
        Args:
            model: Torch model object
            device: Torch device
            model_input_key: Key in batch dict mapping for model input
            model_output_key: Key in output dict model output
                will be stored under
            target_key: Key in batch dict mapping for target
            experiment_fn: callable function,
                which defines default experiment type to use
                during ``.train`` and ``.infer`` methods.
        """
        super().__init__(model=model, engine=engine, experiment_fn=experiment_fn)

        self.model_input_key = model_input_key
        self.model_output_key = model_output_key
        self.target_key = target_key

        if isinstance(self.model_input_key, str):
            # when model expects value
            self._process_input = self._process_input_str
        elif isinstance(self.model_input_key, (list, tuple)):
            # when model expects tuple
            self._process_input = self._process_input_list
        elif self.model_input_key is None:
            # when model expects dict
            self._process_input = self._process_input_none
        else:
            raise NotImplementedError()

        if isinstance(model_output_key, str):
            # when model returns value
            self._process_output = self._process_output_str
        elif isinstance(model_output_key, (list, tuple)):
            # when model returns tuple
            self._process_output = self._process_output_list
        elif self.model_output_key is None:
            # when model returns dict
            self._process_output = self._process_output_none
        else:
            raise NotImplementedError()

    def _process_batch(self, batch):
        if isinstance(batch, (tuple, list)):
            assert len(batch) == 2
            batch = {self.model_input_key: batch[0], self.target_key: batch[1]}
        return batch

    def _process_input_str(self, batch: Mapping[str, Any], **kwargs):
        output = self.model(batch[self.model_input_key], **kwargs)
        return output

    def _process_input_list(self, batch: Mapping[str, Any], **kwargs):
        input = {key: batch[key] for key in self.model_input_key}  # noqa: WPS125
        output = self.model(**input, **kwargs)
        return output

    def _process_input_none(self, batch: Mapping[str, Any], **kwargs):
        output = self.model(**batch, **kwargs)
        return output

    def _process_output_str(self, output: torch.Tensor):
        output = {self.model_output_key: output}
        return output

    def _process_output_list(self, output: Union[Tuple, List]):
        output = {key: value for key, value in zip(self.model_output_key, output)}
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

    def train(
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
        # the loggers
        loggers: "Union[Dict[str, ILogger]]" = None,
        # experiment info
        seed: int = 42,
        hparams: Dict[str, Any] = None,
        # stage info
        num_epochs: int = 1,
        # extra info (callbacks info)
        logdir: str = None,
        resume: str = None,
        valid_loader: str = "valid",
        main_metric: str = "loss",
        minimize_metric: bool = True,
        verbose: bool = False,
        timeit: bool = False,
        check_run: bool = False,
        overfit: bool = False,
        # engine extra params, @TODO: what to do with them?
        fp16: Union[Dict, bool] = None,
        distributed: bool = False,
        # user-friendly API, @TODO: what to do with it?
        load_best_on_end: bool = False,
    ) -> None:
        """
        Starts the train stage of the model.

        Args:
            model: model to train
            criterion: criterion function for training
            optimizer: optimizer for training
            scheduler: scheduler for training
            datasets (OrderedDict[str, Union[Dataset, Dict, Any]]): dictionary
                with one or several  ``torch.utils.data.Dataset``
                for training, validation or inference
                used for Loaders automatic creation
                preferred way for distributed training setup
            loaders (OrderedDict[str, DataLoader]): dictionary
                with one or several ``torch.utils.data.DataLoader``
                for training, validation or inference
            callbacks (Union[List[Callback], OrderedDict[str, Callback]]):
                list or dictionary with Catalyst callbacks
            logdir: path to output directory
            resume: path to checkpoint for model
            num_epochs: number of training epochs
            valid_loader: loader name used to calculate
                the metrics and save the checkpoints. For example,
                you can pass `train` and then
                the metrics will be taken from `train` loader.
            main_metric: the key to the name of the metric
                by which the checkpoints will be selected.
            minimize_metric: flag to indicate whether
                the ``main_metric`` should be minimized.
            verbose: if `True`, it displays the status of the training
                to the console.
            fp16: parameters for fp16/distributed training.
                to use pytorch native amp - ``{"amp": True}``.
                to use apex - ``{"apex": True, "opt_level": "O1", ...}``.
                If fp16=True, params by default will be:
                ``{"amp": True}`` if torch>=1.6.0,
                ``{"apex": True, "opt_level": "O1", ...}`` if torch<1.6.0.
                See https://nvidia.github.io/apex/amp.html#properties for
                more params.
            distributed: if `True` will start training
                in distributed mode.
                Note: Works only with python scripts. No jupyter support.
            check_run: if True, then only checks that pipeline is working
                (3 epochs only with 3 batches per loader)
            overfit: if True, then takes only one batch per loader
                for model overfitting, for advance usage please check
                ``BatchOverfitCallback``
            timeit: if True, computes the execution time
                of training process and displays it to the console.
            load_best_on_end: if True, Runner will load
                best checkpoint state (model, optimizer, etc)
                according to validation metrics. Requires specified ``logdir``.
            seed: experiment's initial seed value
            state_kwargs: deprecated, use `stage_kwargs` instead

        Raises:
            NotImplementedError: if both `resume` and `CheckpointCallback`
                already exist
        """
        # fp16 = _resolve_bool_fp16(fp16)

        # callbacks handling
        callbacks = sort_callbacks_by_order(callbacks)
        is_already_present = lambda callback_fn: any(
            check_callback_isinstance(x, callback_fn) for x in callbacks.values()
        )
        if verbose and not is_already_present(VerboseCallback):
            callbacks["_verbose"] = VerboseCallback()
        if timeit and not is_already_present(TimerCallback):
            callbacks["_timer"] = TimerCallback()
        if check_run and not is_already_present(CheckRunCallback):
            callbacks["_check"] = CheckRunCallback()
        if overfit and not is_already_present(BatchOverfitCallback):
            callbacks["_overfit"] = BatchOverfitCallback()

        if resume is not None or load_best_on_end:
            load_on_stage_end = None
            if load_best_on_end:
                load_on_stage_end = "best_full"
                assert logdir is not None, (
                    "For ``load_best_on_end`` feature " "you need to specify ``logdir``"
                )

            if not is_already_present(ICheckpointCallback):
                callbacks["_checkpoint"] = CheckpointCallback(
                    logdir=os.path.join(logdir, "checkpoints"),
                    loader_key=valid_loader,
                    metric_key=main_metric,
                    minimize=minimize_metric,
                    resume=resume,
                    load_on_stage_end=load_on_stage_end,
                )
            else:
                raise NotImplementedError("CheckpointCallback already exist")

        if isinstance(criterion, Criterion) and not is_already_present(ICriterionCallback):
            callbacks["_criterion"] = CriterionCallback(
                input_key=self.model_output_key, target_key=self.target_key, metric_key="loss",
            )
        if isinstance(optimizer, Optimizer) and not is_already_present(IOptimizerCallback):
            callbacks["_optimizer"] = OptimizerCallback(metric_key="loss",)
        if isinstance(scheduler, (Scheduler, ReduceLROnPlateau)) and not is_already_present(
            ISchedulerCallback
        ):
            callbacks["_scheduler"] = SchedulerCallback(
                loader_key=valid_loader, metric_key=main_metric
            )

        experiment = self._experiment_fn(
            # the data
            loaders=loaders,
            # the core
            model=model,
            engine=engine,
            trial=trial,
            # the components
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            # the callbacks
            callbacks=callbacks,
            # the loggers
            loggers=loggers,
            # experiment info
            seed=seed,
            hparams=hparams,
            # stage info
            stage="train",
            num_epochs=num_epochs,
        )
        self.experiment = experiment
        self.run()


__all__ = ["SupervisedRunner"]
