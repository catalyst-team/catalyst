from typing import Any, Callable, Dict, Generator, List, Mapping, Union
from collections import OrderedDict
import os

import torch
from torch.utils.data import DataLoader, Dataset

from catalyst.callbacks.batch_overfit import BatchOverfitCallback
from catalyst.callbacks.checkpoint import CheckpointCallback, ICheckpointCallback
from catalyst.callbacks.misc import CheckRunCallback, TimerCallback, VerboseCallback
from catalyst.core.callback import Callback
from catalyst.core.engine import IEngine
from catalyst.core.functional import check_callback_isinstance, sort_callbacks_by_order
from catalyst.core.logger import ILogger
from catalyst.core.runner import IStageBasedRunner
from catalyst.core.trial import ITrial
from catalyst.experiments.experiment import Experiment
from catalyst.loggers.console import ConsoleLogger
from catalyst.loggers.csv import CSVLogger
from catalyst.loggers.tensorboard import TensorboardLogger
from catalyst.typing import Criterion, Model, Optimizer, RunnerModel, Scheduler
from catalyst.utils import check_amp_available
from catalyst.utils.checkpoint import load_checkpoint, unpack_checkpoint
from catalyst.utils.misc import maybe_recursive_call, set_global_seed


def _resolve_bool_fp16(fp16: Union[Dict, bool]) -> Dict:
    """Resolves fp16/distributed params usage.

    Args:
        fp16: fp16 params

    Returns:
        resolved version of fp16
    """
    if isinstance(fp16, bool):
        if fp16:
            return {"amp": True} if check_amp_available() else {"apex": True, "opt_level": "O1"}
        else:
            return {}
    else:
        return fp16


class Runner(IStageBasedRunner):
    """Deep Learning Runner for supervised, unsupervised, gan, etc runs."""

    def __init__(
        self,
        model: RunnerModel = None,
        engine: IEngine = None,
        experiment_fn: Callable = Experiment,
    ):
        """

        Args:
            model: Torch model object
            device: Torch device
            experiment_fn: callable function,
                which defines default experiment type to use
                during ``.train`` and ``.infer`` methods.
        """
        super().__init__(model=model, engine=engine)
        self._experiment_fn = experiment_fn

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
        callbacks = sort_callbacks_by_order(callbacks)
        is_callback_exists = lambda callback_fn: any(
            check_callback_isinstance(x, callback_fn) for x in callbacks.values()
        )
        if verbose and not is_callback_exists(VerboseCallback):
            callbacks["_verbose"] = VerboseCallback()
        if timeit and not is_callback_exists(TimerCallback):
            callbacks["_timer"] = TimerCallback()
        if check and not is_callback_exists(CheckRunCallback):
            callbacks["_check"] = CheckRunCallback()
        if overfit and not is_callback_exists(BatchOverfitCallback):
            callbacks["_overfit"] = BatchOverfitCallback()

        if logdir is not None or resume is not None or load_best_on_end:
            load_on_stage_end = None
            if load_best_on_end:
                load_on_stage_end = "best_full"
                assert logdir is not None, (
                    "For ``load_best_on_end`` feature " "you need to specify ``logdir``"
                )

            if not is_callback_exists(ICheckpointCallback):
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
        return callbacks

    def _process_train_loggers(self, *, loggers: "Dict[str, ILogger]" = None, logdir: str = None):
        loggers = loggers or {}
        is_logger_exists = lambda logger_fn: any(
            isinstance(x, logger_fn) for x in loggers.values()
        )
        if not is_logger_exists(ConsoleLogger):
            loggers["_console"] = ConsoleLogger()
        if logdir is not None and not is_logger_exists(CSVLogger):
            loggers["_csv"] = CSVLogger(logdir=logdir)
        if logdir is not None and not is_logger_exists(TensorboardLogger):
            loggers["_tensorboard"] = TensorboardLogger(logdir=os.path.join(logdir, "tensorboard"))
        return loggers

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
        loggers: "Dict[str, ILogger]" = None,
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
        check: bool = False,
        overfit: bool = False,
        load_best_on_end: bool = False,
        # engine extra params, @TODO: what to do with them?
        fp16: Union[Dict, bool] = None,
        distributed: bool = False,
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
            check: if True, then only checks that pipeline is working
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

        callbacks = self._process_train_callbacks(
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

        loggers = self._process_train_loggers(loggers=loggers, logdir=logdir)

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

    @torch.no_grad()
    def predict_batch(self, batch: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
        """
        Run model inference on specified data batch.

        Args:
            batch (Mapping[str, Any]): dictionary with data batches
                from DataLoader.
            **kwargs: additional kwargs to pass to the model

        # noqa: DAR202
        Returns:
            Mapping[str, Any]: model output dictionary

        Raises:
            NotImplementedError: if not implemented yet
        """
        raise NotImplementedError("Please implement `runner.predict_batch` method")

    @torch.no_grad()
    def predict_loader(
        self,
        *,
        loader: DataLoader,
        model: Model = None,
        resume: str = None,
        fp16: Union[Dict, bool] = None,
        initial_seed: int = 42,
    ) -> Generator:
        """
        Runs model inference on PyTorch DataLoader and returns
        python generator with model predictions from `runner.predict_batch`.
        Cleans up the experiment info to avoid possible collisions.
        Sets `is_train_loader` and `is_valid_loader` to `False` while
        keeping `is_infer_loader` as True. Moves model to evaluation mode.

        Args:
            loader: loader to predict
            model: model to use for prediction
            resume: path to checkpoint to resume
            fp16 (Union[Dict, bool]): fp16 settings (same as in `train`)
            initial_seed: seed to use before prediction

        Yields:
            bathes with model predictions
        """
        # fp16 = _resolve_bool_fp16(fp16)

        if model is not None:
            self.model = model
        assert self.model is not None

        if resume is not None:
            checkpoint = load_checkpoint(resume)
            unpack_checkpoint(checkpoint, model=self.model)

        self.experiment = None
        # @TODO: we need engine here
        self.model = self.engine.sync_device(self.model)
        maybe_recursive_call(self.model, "train", mode=False)

        set_global_seed(initial_seed)
        for batch in loader:
            yield self.predict_batch(batch)


__all__ = ["Runner"]
