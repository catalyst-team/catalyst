from typing import Any, Dict, Generator, List, Mapping, Optional, Union
from collections import OrderedDict
import os

import torch
from torch.utils.data import DataLoader

from catalyst.callbacks.batch_overfit import BatchOverfitCallback
from catalyst.callbacks.checkpoint import CheckpointCallback, ICheckpointCallback
from catalyst.callbacks.misc import CheckRunCallback, TimerCallback, TqdmCallback
from catalyst.callbacks.profiler import ProfilerCallback
from catalyst.core.callback import Callback
from catalyst.core.engine import Engine
from catalyst.core.logger import ILogger
from catalyst.core.misc import callback_isinstance, sort_callbacks_by_order
from catalyst.core.runner import IRunner, IRunnerError
from catalyst.loggers.console import ConsoleLogger
from catalyst.loggers.csv import CSVLogger
from catalyst.loggers.tensorboard import TensorboardLogger
from catalyst.typing import (
    RunnerCriterion,
    RunnerModel,
    RunnerOptimizer,
    RunnerScheduler,
    TorchCriterion,
    TorchModel,
    TorchOptimizer,
    TorchScheduler,
)
from catalyst.utils.misc import maybe_recursive_call, set_global_seed
from catalyst.utils.torch import get_available_engine, load_checkpoint


class Runner(IRunner):
    """Single-stage deep learning Runner with user-friendly API.

    Runner supports the logic for deep learning pipeline configuration
    with pure python code.
    Please check the examples for intuition.

    Args:
        *args: `IRunner` args (model, engine)
        **kwargs: `IRunner` kwargs (model, engine)

    .. note::
        IRunner supports only base user-friendly callbacks, like
        TqdmCallback, TimerCallback, CheckRunCallback, BatchOverfitCallback,
        and CheckpointCallback.

        It does not automatically add Criterion, Optimizer or Scheduler callbacks.

        That means, that you have do optimization step by yourself during
        ``handle_batch`` method
        or specify the required callbacks in ``.train`` or ``get_callbacks`` methods.

        For more easy-to-go supervised use case please follow
        :py:mod:`catalyst.runners.runner.SupervisedRunner`.

    .. note::
        Please follow the `minimal examples`_ sections for use cases.

        .. _`minimal examples`: https://github.com/catalyst-team/catalyst#minimal-examples  # noqa: E501, W505

    Examples:

    .. code-block:: python

        import os
        from torch import nn, optim
        from torch.nn import functional as F
        from torch.utils.data import DataLoader
        from catalyst import dl, metrics
        from catalyst.contrib.datasets import MNIST

        model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
        optimizer = optim.Adam(model.parameters(), lr=0.02)
        loaders = {
            "train": DataLoader(MNIST(os.getcwd(), train=True), batch_size=32),
            "valid": DataLoader(MNIST(os.getcwd(), train=False), batch_size=32),
        }

        class CustomRunner(dl.Runner):
            def predict_batch(self, batch):
                # model inference step
                return self.model(batch[0].to(self.device))

            def on_loader_start(self, runner):
                super().on_loader_start(runner)
                self.meters = {
                    key: metrics.AdditiveMetric(compute_on_call=False)
                    for key in ["loss", "accuracy01", "accuracy03"]
                }

            def handle_batch(self, batch):
                # model train/valid step
                # unpack the batch
                x, y = batch
                # run model forward pass
                logits = self.model(x)
                # compute the loss
                loss = F.cross_entropy(logits, y)
                # compute other metrics of interest
                accuracy01, accuracy03 = metrics.accuracy(logits, y, topk=(1, 3))
                # log metrics
                self.batch_metrics.update(
                    {"loss": loss, "accuracy01": accuracy01, "accuracy03": accuracy03}
                )
                for key in ["loss", "accuracy01", "accuracy03"]:
                    self.meters[key].update(
                        self.batch_metrics[key].item(), self.batch_size
                    )
                # run model backward pass
                if self.is_train_loader:
                    self.engine.backward(loss)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            def on_loader_end(self, runner):
                for key in ["loss", "accuracy01", "accuracy03"]:
                    self.loader_metrics[key] = self.meters[key].compute()[0]
                super().on_loader_end(runner)

        runner = CustomRunner()
        # model training
        runner.train(
            model=model,
            optimizer=optimizer,
            loaders=loaders,
            logdir="./logs",
            num_epochs=5,
            verbose=True,
            valid_loader="valid",
            valid_metric="loss",
            minimize_valid_metric=True,
        )
        # model inference
        for logits in runner.predict_loader(loader=loaders["valid"]):
            assert logits.detach().cpu().numpy().shape[-1] == 10
    """

    def __init__(self, *args, **kwargs):
        """Init."""
        super().__init__(*args, **kwargs)
        # extra
        self._seed = 42
        self._hparams: Dict = None
        self._num_epochs: int = 1
        # model selection
        self._logdir = None
        self._valid_loader = None
        self._valid_metric = None
        self._minimize_valid_metric = None
        # extras
        self._resume: str = None
        self._verbose = False
        self._timeit = False
        self._check = False
        self._overfit = False
        self._profile = False
        self._load_best_on_end = False

    @property
    def seed(self) -> int:
        """Experiment's initial seed value."""
        return self._seed

    @property
    def hparams(self) -> Dict:
        """Returns hyperparameters."""
        return self._hparams or {}

    @property
    def num_epochs(self) -> int:
        """Returns the number of epochs in the experiment."""
        return self._num_epochs

    def get_engine(self) -> Engine:
        """Returns the engine for the experiment."""
        return self._engine

    def get_loggers(self) -> Dict[str, ILogger]:
        """Returns the loggers for the experiment."""
        loggers = self._loggers or {}
        logger_exists = lambda logger_fn: any(
            isinstance(x, logger_fn) for x in loggers.values()
        )
        if not logger_exists(ConsoleLogger):
            loggers["_console"] = ConsoleLogger()
        if self._logdir is not None and not logger_exists(CSVLogger):
            # @TODO: remove postfix
            loggers["_csv"] = CSVLogger(logdir=self._logdir, use_logdir_postfix=True)
        if self._logdir is not None and not logger_exists(TensorboardLogger):
            # @TODO: remove postfix
            loggers["_tensorboard"] = TensorboardLogger(
                logdir=self._logdir, use_logdir_postfix=True
            )
        return loggers

    def get_loaders(self) -> "OrderedDict[str, DataLoader]":
        """Returns the loaders for the experiment."""
        return self._loaders

    def get_model(self) -> RunnerModel:
        """Returns the model for the experiment."""
        return self._model

    def get_criterion(self) -> Optional[RunnerCriterion]:
        """Returns the criterion for the experiment."""
        return self._criterion

    def get_optimizer(self, model: RunnerModel) -> Optional[RunnerOptimizer]:
        """Returns the optimizer for the experiment."""
        return self._optimizer

    def get_scheduler(self, optimizer: RunnerOptimizer) -> Optional[RunnerScheduler]:
        """Returns the scheduler for the experiment."""
        return self._scheduler

    def get_callbacks(self) -> "OrderedDict[str, Callback]":
        """Returns the callbacks for the experiment."""
        callbacks = sort_callbacks_by_order(self._callbacks)
        callback_exists = lambda callback_fn: any(
            callback_isinstance(x, callback_fn) for x in callbacks.values()
        )
        if self._verbose and not callback_exists(TqdmCallback):
            callbacks["_verbose"] = TqdmCallback()
        if self._timeit and not callback_exists(TimerCallback):
            callbacks["_timer"] = TimerCallback()
        if self._check and not callback_exists(CheckRunCallback):
            callbacks["_check"] = CheckRunCallback()
        if self._overfit and not callback_exists(BatchOverfitCallback):
            callbacks["_overfit"] = BatchOverfitCallback()
        if self._profile and not callback_exists(ProfilerCallback):
            callbacks["_profile"] = ProfilerCallback(
                tensorboard_path=os.path.join(self._logdir, "tb_profile"),
                profiler_kwargs={
                    "activities": [
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    "on_trace_ready": torch.profiler.tensorboard_trace_handler(
                        os.path.join(self._logdir, "tb_profile")
                    ),
                    "with_stack": True,
                    "with_flops": True,
                },
            )

        if self._logdir is not None and not callback_exists(ICheckpointCallback):
            callbacks["_checkpoint"] = CheckpointCallback(
                logdir=os.path.join(self._logdir, "checkpoints"),
                loader_key=self._valid_loader,
                metric_key=self._valid_metric,
                minimize=self._minimize_valid_metric,
                load_best_on_end=self._load_best_on_end,
            )
        return callbacks

    def train(
        self,
        *,
        # the data
        loaders: "OrderedDict[str, DataLoader]",
        # the core
        model: TorchModel = None,
        engine: Union["Engine", str] = None,
        # the components
        criterion: TorchCriterion = None,
        optimizer: TorchOptimizer = None,
        scheduler: TorchScheduler = None,
        # the callbacks
        callbacks: "Union[List[Callback], OrderedDict[str, Callback]]" = None,
        # the loggers
        loggers: "Dict[str, ILogger]" = None,
        # experiment info
        seed: int = 42,
        hparams: Dict[str, Any] = None,
        num_epochs: int = 1,
        # extra info (callbacks info)
        logdir: str = None,
        resume: str = None,
        valid_loader: str = None,
        valid_metric: str = None,
        minimize_valid_metric: bool = None,
        verbose: bool = False,
        timeit: bool = False,
        check: bool = False,
        overfit: bool = False,
        profile: bool = False,
        load_best_on_end: bool = False,
        # engine extra params,
        cpu: bool = False,
        fp16: bool = False,
        ddp: bool = False,
    ) -> None:
        """
        Starts the training of the model.

        Args:
            loaders: dictionary with one or several ``torch.utils.data.DataLoader``
                for training, validation or inference
            model: model to train
            engine: engine to use for model training
            criterion: criterion function for training
            optimizer: optimizer for training
            scheduler: scheduler for training
            callbacks: list or dictionary with Catalyst callbacks
            loggers: dictionary with Catalyst loggers
            seed: experiment's initial seed value
            hparams: hyperparameters for the run
            num_epochs: number of training epochs
            logdir: path to output directory
            resume: path to checkpoint for model
            valid_loader: loader name used to calculate
                the metrics and save the checkpoints. For example,
                you can pass `train` and then
                the metrics will be taken from `train` loader.
            valid_metric: the key to the name of the metric
                by which the checkpoints will be selected.
            minimize_valid_metric: flag to indicate whether
                the ``valid_metric`` should be minimized or not (default: True).
            verbose: if `True`, it displays the status of the training to the console.
            timeit: if True, computes the execution time
                of training process and displays it to the console.
            check: if True, then only checks that pipeline is working
                (3 epochs only with 3 batches per loader)
            overfit: if True, then takes only one batch per loader
                for model overfitting, for advance usage please check
                ``BatchOverfitCallback``
            profile: if True, then uses ProfilerCallback, for advance usage please check
                ``ProfilerCallback``
            load_best_on_end: if True, Runner will load
                best checkpoint state (model, optimizer, etc)
                according to validation metrics. Requires specified ``logdir``.
            cpu: boolean flag to force CPU usage
            fp16: boolean flag to use half-precision
            ddp: if `True` will start training in distributed mode.
                Note: Works only with python scripts. No jupyter support.

        .. note::
            Please follow the `minimal examples`_ sections for use cases.

            .. _`minimal examples`: https://github.com/catalyst-team/catalyst#minimal-examples  # noqa: E501, W505

        """
        # experiment setup
        self._engine = (
            engine or self.engine or get_available_engine(cpu=cpu, fp16=fp16, ddp=ddp)
        )
        # self._trial = trial
        self._loggers = loggers
        # the data
        self._loaders = loaders
        # the components
        self._model = model or self.model
        self._criterion = criterion
        self._optimizer = optimizer
        self._scheduler = scheduler
        # the callbacks
        self._callbacks = callbacks
        # extra
        self._seed = seed
        self._hparams = hparams
        self._num_epochs = num_epochs
        self._logdir = logdir
        self._resume = resume
        self._valid_loader = valid_loader
        self._valid_metric = valid_metric
        self._minimize_valid_metric = minimize_valid_metric
        self._verbose = verbose
        self._timeit = timeit
        self._check = check
        self._overfit = overfit
        self._profile = profile
        self._load_best_on_end = load_best_on_end
        # run
        self.run()

    @torch.no_grad()
    def predict_batch(self, batch: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
        """
        Run model inference on specified data batch.

        Args:
            batch: dictionary with data batches from DataLoader.
            **kwargs: additional kwargs to pass to the model

        Returns:  # noqa: DAR202
            Mapping: model output dictionary

        Raises:
            NotImplementedError: if not implemented yet
        """
        raise NotImplementedError("Please implement `runner.predict_batch` method")

    @torch.no_grad()
    def predict_loader(
        self,
        *,
        loader: DataLoader,
        model: TorchModel = None,
        engine: Union["Engine", str] = None,
        seed: int = 42,
        # extra info
        resume: str = None,
        # engine extra params,
        cpu: bool = False,
        fp16: bool = False,
    ) -> Generator:
        """
        Runs model inference on PyTorch DataLoader and returns
        python generator with model predictions from `runner.predict_batch`.

        Args:
            loader: loader to predict
            model: model to use for prediction
            engine: engine to use for prediction
            seed: random seed to use before prediction
            resume: path to checkpoint for model
            cpu: boolean flag to force CPU usage
            fp16: boolean flag to use half-precision

        Yields:
            bathes with model predictions

        .. note::
            Please follow the `minimal examples`_ sections for use cases.

            .. _`minimal examples`: https://github.com/catalyst-team/catalyst#minimal-examples  # noqa: E501, W505
        """
        self.engine = engine or get_available_engine(cpu=cpu, fp16=fp16)

        if model is not None:
            self.model = model
        assert self.model is not None

        if resume is not None:
            self.engine.wait_for_everyone()
            unwrapped_model = self.engine.unwrap_model(self.model)
            unwrapped_model.load_state_dict(load_checkpoint(resume))

        self.model = self.engine.prepare(self.model)
        maybe_recursive_call(self.model, "train", mode=False)
        loader = self.engine.prepare(loader)

        set_global_seed(seed)
        for batch in loader:
            yield self.predict_batch(batch)

    def evaluate_loader(
        self,
        loader: DataLoader,
        callbacks: "Union[List[Callback], OrderedDict[str, Callback]]" = None,
        model: Optional[TorchModel] = None,
        engine: Union["Engine", str] = None,
        seed: int = 42,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluates dataloader with given model and returns obtained metrics.

        Args:
            loader: loader to predict
            callbacks: list or dictionary with catalyst callbacks
            model: model, compatible with current runner.
                If `None` simply takes current model from runner.
            engine: engine to use for model evaluation
            seed: random seed to use before prediction
            verbose: if `True`, it displays the status of the evaluation to the console.

        Returns:
            Dict with metrics counted on the loader.

        Raises:
            IRunnerError: if ``CheckpointCallback`` found in the callbacks
        """
        callbacks = sort_callbacks_by_order(callbacks)
        for callback in callbacks.values():
            if callback_isinstance(callback, ICheckpointCallback):
                raise IRunnerError(
                    "CheckpointCallback isn`t allowed for evaluation loader method"
                )

        if engine is None:
            engine = self.engine
        if model is None:
            model = self.model
        assert model is not None

        self.train(
            model=model,
            engine=engine,
            loaders=OrderedDict([("valid", loader)]),
            num_epochs=1,
            verbose=verbose,
            callbacks=callbacks,
            valid_loader="valid",
            seed=seed,
        )

        return self.loader_metrics


__all__ = ["Runner"]
