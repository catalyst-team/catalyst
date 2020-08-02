from typing import Any, Callable, Dict, Generator, List, Mapping, Union
from collections import OrderedDict

import torch
from torch.jit import ScriptModule
from torch.utils.data import DataLoader, Dataset

from catalyst.core import Callback, CheckpointCallback, IStageBasedRunner
from catalyst.dl import utils
from catalyst.dl.experiment.experiment import Experiment
from catalyst.tools.typing import (
    Criterion,
    Device,
    Model,
    Optimizer,
    Scheduler,
)


class Runner(IStageBasedRunner):
    """
    Deep Learning Runner for supervised, unsupervised, gan, etc runs.
    """

    _experiment_fn: Callable = Experiment

    def _init(self, **kwargs):
        self.experiment: Experiment = None

    def train(
        self,
        *,
        model: Model,
        criterion: Criterion = None,
        optimizer: Optimizer = None,
        scheduler: Scheduler = None,
        datasets: "OrderedDict[str, Union[Dataset, Dict, Any]]" = None,
        loaders: "OrderedDict[str, DataLoader]" = None,
        callbacks: "Union[List[Callback], OrderedDict[str, Callback]]" = None,
        logdir: str = None,
        resume: str = None,
        num_epochs: int = 1,
        valid_loader: str = "valid",
        main_metric: str = "loss",
        minimize_metric: bool = True,
        verbose: bool = False,
        stage_kwargs: Dict = None,
        checkpoint_data: Dict = None,
        fp16: Union[Dict, bool] = None,
        distributed: bool = False,
        check: bool = False,
        overfit: bool = False,
        timeit: bool = False,
        load_best_on_end: bool = False,
        initial_seed: int = 42,
        state_kwargs: Dict = None,
    ) -> None:
        """
        Starts the train stage of the model.

        Args:
            model (Model): model to train
            criterion (Criterion): criterion function for training
            optimizer (Optimizer): optimizer for training
            scheduler (Scheduler): scheduler for training
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
            logdir (str): path to output directory
            resume (str): path to checkpoint for model
            num_epochs (int): number of training epochs
            valid_loader (str): loader name used to calculate
                the metrics and save the checkpoints. For example,
                you can pass `train` and then
                the metrics will be taken from `train` loader.
            main_metric (str): the key to the name of the metric
                by which the checkpoints will be selected.
            minimize_metric (bool): flag to indicate whether
                the ``main_metric`` should be minimized.
            verbose (bool): if `True`, it displays the status of the training
                to the console.
            stage_kwargs (dict): additional params for stage
            checkpoint_data (dict): additional data to save in checkpoint,
                for example: ``class_names``, ``date_of_training``, etc
            fp16 (Union[Dict, bool]): If not None, then sets training to FP16.
                See https://nvidia.github.io/apex/amp.html#properties
                if fp16=True, params by default will be ``{"opt_level": "O1"}``
            distributed (bool): if `True` will start training
                in distributed mode.
                Note: Works only with python scripts. No jupyter support.
            check (bool): if True, then only checks that pipeline is working
                (3 epochs only with 3 batches per loader)
            overfit (bool): if True, then takes only one batch per loader
                for model overfitting, for advance usage please check
                ``BatchOverfitCallback``
            timeit (bool): if True, computes the execution time
                of training process and displays it to the console.
            load_best_on_end (bool): if True, Runner will load
                best checkpoint state (model, optimizer, etc)
                according to validation metrics. Requires specified ``logdir``.
            initial_seed (int): experiment's initial seed value
            state_kwargs (dict): deprecated, use `stage_kwargs` instead

        Raises:
            NotImplementedError: if both `resume` and `CheckpointCallback`
                already exist
        """
        assert state_kwargs is None or stage_kwargs is None

        if isinstance(fp16, bool) and fp16:
            fp16 = {"opt_level": "O1"}

        if resume is not None or load_best_on_end:
            load_on_stage_end = None
            if load_best_on_end:
                load_on_stage_end = "best_full"
                assert logdir is not None, (
                    "For ``load_best_on_end`` feature "
                    "you need to specify ``logdir``"
                )
            callbacks = utils.sort_callbacks_by_order(callbacks)
            checkpoint_callback_flag = any(
                isinstance(x, CheckpointCallback) for x in callbacks.values()
            )
            if not checkpoint_callback_flag:
                callbacks["_loader"] = CheckpointCallback(
                    resume=resume, load_on_stage_end=load_on_stage_end,
                )
            else:
                raise NotImplementedError("CheckpointCallback already exist")

        experiment = self._experiment_fn(
            stage="train",
            model=model,
            datasets=datasets,
            loaders=loaders,
            callbacks=callbacks,
            logdir=logdir,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=num_epochs,
            valid_loader=valid_loader,
            main_metric=main_metric,
            minimize_metric=minimize_metric,
            verbose=verbose,
            check_time=timeit,
            check_run=check,
            overfit=overfit,
            stage_kwargs=stage_kwargs or state_kwargs,
            checkpoint_data=checkpoint_data,
            distributed_params=fp16,
            initial_seed=initial_seed,
        )
        self.experiment = experiment
        utils.distributed_cmd_run(self.run_experiment, distributed)

    def infer(
        self,
        *,
        model: Model,
        datasets: "OrderedDict[str, Union[Dataset, Dict, Any]]" = None,
        loaders: "OrderedDict[str, DataLoader]" = None,
        callbacks: "Union[List[Callback], OrderedDict[str, Callback]]" = None,
        logdir: str = None,
        resume: str = None,
        verbose: bool = False,
        stage_kwargs: Dict = None,
        fp16: Union[Dict, bool] = None,
        check: bool = False,
        timeit: bool = False,
        initial_seed: int = 42,
        state_kwargs: Dict = None,
    ) -> None:
        """
        Starts the inference stage of the model.

        Args:
            model (Model): model for inference
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
            logdir (str): path to output directory
            resume (str): path to checkpoint to use for resume
            verbose (bool): if `True`, it displays the status of the training
                to the console.
            stage_kwargs (dict): additional stage params
            fp16 (Union[Dict, bool]): If not None, then sets training to FP16.
                See https://nvidia.github.io/apex/amp.html#properties
                if fp16=True, params by default will be ``{"opt_level": "O1"}``
            check (bool): if True, then only checks that pipeline is working
                (3 epochs only)
            timeit (bool): if True, computes the execution time
                of training process and displays it to the console.
            initial_seed (int): experiment's initial seed value
            state_kwargs (dict): deprecated, use `stage_kwargs` instead

        Raises:
            NotImplementedError: if both `resume` and `CheckpointCallback`
                already exist
        """
        assert state_kwargs is None or stage_kwargs is None

        if isinstance(fp16, bool) and fp16:
            fp16 = {"opt_level": "O1"}

        if resume is not None:
            callbacks = utils.sort_callbacks_by_order(callbacks)
            checkpoint_callback_flag = any(
                isinstance(x, CheckpointCallback) for x in callbacks.values()
            )
            if not checkpoint_callback_flag:
                callbacks["loader"] = CheckpointCallback(resume=resume)
            else:
                raise NotImplementedError("CheckpointCallback already exist")

        experiment = self._experiment_fn(
            stage="infer",
            model=model,
            datasets=datasets,
            loaders=loaders,
            callbacks=callbacks,
            logdir=logdir,
            verbose=verbose,
            check_time=timeit,
            check_run=check,
            stage_kwargs=stage_kwargs or state_kwargs,
            distributed_params=fp16,
            initial_seed=initial_seed,
        )
        self.run_experiment(experiment)

    @torch.no_grad()
    def predict_batch(
        self, batch: Mapping[str, Any], **kwargs
    ) -> Mapping[str, Any]:
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
        raise NotImplementedError(
            "Please implement `runner.predict_batch` method"
        )

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
            loader (DataLoader): loader to predict
            model (Model): model to use for prediction
            resume (str): path to checkpoint to resume
            fp16 (Union[Dict, bool]): fp16 usage flag
            initial_seed (int): seed to use before prediction

        Yields:
            bathes with model predictions
        """
        if isinstance(fp16, bool) and fp16:
            fp16 = {"opt_level": "O1"}

        if model is not None:
            self.model = model
        assert self.model is not None

        if resume is not None:
            checkpoint = utils.load_checkpoint(resume)
            utils.unpack_checkpoint(checkpoint, model=self.model)

        self.experiment = None
        utils.set_global_seed(initial_seed)
        (model, _, _, _, device) = utils.process_components(  # noqa: WPS122
            model=self.model, distributed_params=fp16, device=self.device,
        )
        self._prepare_inner_state(
            stage="infer",
            model=model,
            device=device,
            is_train_loader=False,
            is_valid_loader=False,
            is_infer_loader=True,
        )
        utils.maybe_recursive_call(self.model, "train", mode=False)

        utils.set_global_seed(initial_seed)
        for batch in loader:
            yield self.predict_batch(batch)

    def trace(
        self,
        *,
        model: Model = None,
        batch: Any = None,
        logdir: str = None,
        loader: DataLoader = None,
        method_name: str = "forward",
        mode: str = "eval",
        requires_grad: bool = False,
        fp16: Union[Dict, bool] = None,
        device: Device = "cpu",
        predict_params: dict = None,
    ) -> ScriptModule:
        """
        Traces model using Torch Jit.

        Args:
            model (Model): model to trace
            batch: batch to forward through the model to trace
            logdir (str, optional): If specified,
                the result will be written to the directory
            loader (DataLoader, optional): if batch is not specified, the batch
                will be ``next(iter(loader))``
            method_name (str): model's method name that will be traced
            mode (str): ``train`` or ``eval``
            requires_grad (bool): flag to trace with gradients
            fp16 (Union[Dict, bool]): If not None, then sets
                tracing params to FP16
            device (Device): Torch device or a string
            predict_params (dict): additional parameters for model forward

        Returns:
            ScriptModule: traced model

        Raises:
            ValueError: if `batch` and `loader` are Nones
        """
        if batch is None:
            if loader is None:
                raise ValueError(
                    "If batch is not provided the loader must be specified"
                )
            batch = next(iter(loader))

        if model is not None:
            self.model = model
        assert self.model is not None

        if isinstance(fp16, bool) and fp16:
            opt_level = "O1"
        elif isinstance(fp16, bool) and not fp16:
            opt_level = None
        elif isinstance(fp16, dict):
            opt_level = fp16["opt_level"]
        else:
            opt_level = fp16

        if opt_level is not None:
            device = "cuda"
        elif device is None:
            if self.device is None:
                self.device = utils.get_device()
            device = self.device

        # Dumping previous state of the model, we will need it to restore
        device_dump, is_training_dump, requires_grad_dump = (
            self.device,
            self.model.training,
            utils.get_requires_grad(self.model),
        )

        self.model.to(device)

        # function to run prediction on batch
        def predict_fn(model, inputs, **kwargs):  # noqa: WPS442
            model_dump = self.model
            self.model = model
            result = self.predict_batch(inputs, **kwargs)
            self.model = model_dump
            return result

        traced_model = utils.trace_model(
            model=self.model,
            predict_fn=predict_fn,
            batch=batch,
            method_name=method_name,
            mode=mode,
            requires_grad=requires_grad,
            opt_level=opt_level,
            device=device,
            predict_params=predict_params,
        )

        if logdir is not None:
            utils.save_traced_model(
                model=traced_model,
                logdir=logdir,
                method_name=method_name,
                mode=mode,
                requires_grad=requires_grad,
                opt_level=opt_level,
            )

        # Restore previous state of the model
        getattr(self.model, "train" if is_training_dump else "eval")()
        utils.set_requires_grad(self.model, requires_grad_dump)
        self.model.to(device_dump)

        return traced_model


__all__ = ["Runner"]
