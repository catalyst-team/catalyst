from typing import Any, Callable, Dict, Generator, List, Mapping, Union
from collections import OrderedDict
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
from catalyst.core.runner import IStageBasedRunner
from catalyst.core.trial import ITrial
from catalyst.experiments.experiment import Experiment
from catalyst.typing import Criterion, Device, Model, Optimizer, RunnerModel, Scheduler
from catalyst.utils import check_amp_available
from catalyst.utils.checkpoint import load_checkpoint, unpack_checkpoint
from catalyst.utils.components import process_components
from catalyst.utils.misc import maybe_recursive_call, set_global_seed
from catalyst.utils.scripts import distributed_cmd_run
from catalyst.utils.torch import get_device, get_requires_grad, set_requires_grad
from catalyst.utils.tracing import save_traced_model, trace_model


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

        # if isinstance(criterion, Criterion) and not is_already_present(ICriterionCallback):
        #     callbacks["_criterion"] = CriterionCallback()
        # if isinstance(optimizer, Optimizer) and not is_already_present(IOptimizerCallback):
        #     callbacks["_optimizer"] = OptimizerCallback()
        # if isinstance(scheduler, (Scheduler, ReduceLROnPlateau)) and not is_already_present(
        #         ISchedulerCallback
        # ):
        #     callbacks["_scheduler"] = SchedulerCallback()

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

    # def infer(
    #     self,
    #     *,
    #     model: Model,
    #     datasets: "OrderedDict[str, Union[Dataset, Dict, Any]]" = None,
    #     loaders: "OrderedDict[str, DataLoader]" = None,
    #     callbacks: "Union[List[Callback], OrderedDict[str, Callback]]" = None,
    #     logdir: str = None,
    #     resume: str = None,
    #     verbose: bool = False,
    #     stage_kwargs: Dict = None,
    #     fp16: Union[Dict, bool] = None,
    #     check: bool = False,
    #     timeit: bool = False,
    #     initial_seed: int = 42,
    #     state_kwargs: Dict = None,
    # ) -> None:
    #     """
    #     Starts the inference stage of the model.
    #
    #     Args:
    #         model: model for inference
    #         datasets (OrderedDict[str, Union[Dataset, Dict, Any]]): dictionary
    #             with one or several  ``torch.utils.data.Dataset``
    #             for training, validation or inference
    #             used for Loaders automatic creation
    #             preferred way for distributed training setup
    #         loaders (OrderedDict[str, DataLoader]): dictionary
    #             with one or several ``torch.utils.data.DataLoader``
    #             for training, validation or inference
    #         callbacks (Union[List[Callback], OrderedDict[str, Callback]]):
    #             list or dictionary with Catalyst callbacks
    #         logdir: path to output directory
    #         resume: path to checkpoint to use for resume
    #         verbose: if `True`, it displays the status of the training
    #             to the console.
    #         stage_kwargs: additional stage params
    #         fp16 (Union[Dict, bool]): fp16 settings (same as in `train`)
    #         check: if True, then only checks that pipeline is working
    #             (3 epochs only)
    #         timeit: if True, computes the execution time
    #             of training process and displays it to the console.
    #         initial_seed: experiment's initial seed value
    #         state_kwargs: deprecated, use `stage_kwargs` instead
    #
    #     Raises:
    #         NotImplementedError: if both `resume` and `CheckpointCallback`
    #             already exist
    #     """
    #     assert state_kwargs is None or stage_kwargs is None
    #
    #     fp16 = _resolve_bool_fp16(fp16)
    #
    #     if resume is not None:
    #         callbacks = sort_callbacks_by_order(callbacks)
    #         checkpoint_callback_flag = any(
    #             isinstance(x, CheckpointCallback) for x in callbacks.values()
    #         )
    #         if not checkpoint_callback_flag:
    #             callbacks["loader"] = CheckpointCallback(resume=resume)
    #         else:
    #             raise NotImplementedError("CheckpointCallback already exist")
    #
    #     experiment = self._experiment_fn(
    #         stage="infer",
    #         model=model,
    #         datasets=datasets,
    #         loaders=loaders,
    #         callbacks=callbacks,
    #         logdir=logdir,
    #         verbose=verbose,
    #         check_time=timeit,
    #         check_run=check,
    #         stage_kwargs=stage_kwargs or state_kwargs,
    #         engine_params=fp16,
    #         initial_seed=initial_seed,
    #     )
    #     self.run(experiment)

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
        Runs model inference on PyTorch Dataloader and returns
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

    # def trace(
    #     self,
    #     *,
    #     model: Model = None,
    #     batch: Any = None,
    #     logdir: str = None,
    #     loader: DataLoader = None,
    #     method_name: str = "forward",
    #     mode: str = "eval",
    #     requires_grad: bool = False,
    #     fp16: Union[Dict, bool] = None,
    #     device: Device = "cpu",
    #     predict_params: dict = None,
    # ) -> ScriptModule:
    #     """
    #     Traces model using Torch Jit.
    #
    #     Args:
    #         model: model to trace
    #         batch: batch to forward through the model to trace
    #         logdir (str, optional): If specified,
    #             the result will be written to the directory
    #         loader (DataLoader, optional): if batch is not specified, the batch
    #             will be ``next(iter(loader))``
    #         method_name: model's method name that will be traced
    #         mode: ``train`` or ``eval``
    #         requires_grad: flag to trace with gradients
    #         fp16 (Union[Dict, bool]): fp16 settings (same as in `train`)
    #         device: Torch device or a string
    #         predict_params: additional parameters for model forward
    #
    #     Returns:
    #         ScriptModule: traced model
    #
    #     Raises:
    #         ValueError: if `batch` and `loader` are Nones
    #     """
    #     # @TODO: refactor for easy use
    #     # @TODO: also add quantize, prune, onnx-convert
    #     if batch is None:
    #         if loader is None:
    #             raise ValueError("If batch is not provided the loader must be specified")
    #         batch = next(iter(loader))
    #
    #     if model is not None:
    #         self.model = model
    #     assert self.model is not None
    #
    #     fp16 = _resolve_bool_fp16(fp16)
    #     opt_level = None
    #     if fp16:
    #         opt_level = fp16.get("opt_level", None)
    #
    #     if opt_level is not None:
    #         device = "cuda"
    #     elif device is None:
    #         if self.device is None:
    #             self.device = get_device()
    #         device = self.device
    #
    #     # Dumping previous state of the model, we will need it to restore
    #     device_dump, is_training_dump, requires_grad_dump = (
    #         self.device,
    #         self.model.training,
    #         get_requires_grad(self.model),
    #     )
    #
    #     self.model.to(device)
    #
    #     # function to run prediction on batch
    #     def predict_fn(model, inputs, **kwargs):  # noqa: WPS442
    #         model_dump = self.model
    #         self.model = model
    #         result = self.predict_batch(inputs, **kwargs)
    #         self.model = model_dump
    #         return result
    #
    #     traced_model = trace_model(
    #         model=self.model,
    #         predict_fn=predict_fn,
    #         batch=batch,
    #         method_name=method_name,
    #         mode=mode,
    #         requires_grad=requires_grad,
    #         opt_level=opt_level,
    #         device=device,
    #         predict_params=predict_params,
    #     )
    #
    #     if logdir is not None:
    #         save_traced_model(
    #             model=traced_model,
    #             logdir=logdir,
    #             method_name=method_name,
    #             mode=mode,
    #             requires_grad=requires_grad,
    #             opt_level=opt_level,
    #         )
    #
    #     # Restore previous state of the model
    #     getattr(self.model, "train" if is_training_dump else "eval")()
    #     set_requires_grad(self.model, requires_grad_dump)
    #     self.model.to(device_dump)
    #
    #     return traced_model


__all__ = ["Runner"]
