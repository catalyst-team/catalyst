from typing import Any, Callable, Dict, List, Union
from collections import OrderedDict

from torch.utils.data import DataLoader, Dataset

from catalyst.core import Callback, CheckpointCallback, StageBasedRunner, State
from catalyst.dl import Experiment, utils
from catalyst.utils.tools.typing import Criterion, Model, Optimizer, Scheduler


class Runner(StageBasedRunner):
    """
    Deep Learning Runner for different supervised, unsupervised, gan, etc runs.
    """

    _experiment_fn: Callable = Experiment
    _state_fn: Callable = State

    def _init(self):
        self.experiment: Experiment = None
        self.state: State = None

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
        state_kwargs: Dict = None,
        checkpoint_data: Dict = None,
        fp16: Union[Dict, bool] = None,
        distributed: bool = False,
        monitoring_params: Dict = None,
        check: bool = False,
        timeit: bool = False,
    ) -> None:
        """
        Starts the training process of the model.

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
            state_kwargs (dict): additional state params to ``State``
            checkpoint_data (dict): additional data to save in checkpoint,
                for example: ``class_names``, ``date_of_training``, etc
            fp16 (Union[Dict, bool]): If not None, then sets training to FP16.
                See https://nvidia.github.io/apex/amp.html#properties
                if fp16=True, params by default will be ``{"opt_level": "O1"}``
            distributed (bool): if `True` will start training
                in distributed mode.
                Note: Works only with python scripts. No jupyter support.
            monitoring_params (dict): If not None, then create monitoring
                through Alchemy or other tools.
                For example,
                ``{"token": "api_token", "experiment": "experiment_name"}``
            check (bool): if True, then only checks that pipeline is working
                (3 epochs only)
            timeit (bool): if True, computes the execution time
                of training process and displays it to the console.
        """
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
            stage="train",
            model=model,
            loaders=loaders,
            datasets=datasets,
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
            check_run=check,
            check_time=timeit,
            state_kwargs=state_kwargs,
            checkpoint_data=checkpoint_data,
            distributed_params=fp16,
            monitoring_params=monitoring_params,
        )
        self.experiment = experiment
        utils.distributed_cmd_run(self.run_experiment, distributed)


__all__ = ["Runner"]
