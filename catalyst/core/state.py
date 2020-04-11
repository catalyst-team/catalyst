from typing import Any, Dict, Optional, TYPE_CHECKING, Union
from collections import defaultdict, OrderedDict
from pathlib import Path
import warnings

import numpy as np

from torch.utils.data import DataLoader

from catalyst.core import utils
from catalyst.utils.tools.frozen_class import FrozenClass
from catalyst.utils.tools.settings import (
    LOADER_VALID_PREFIX,
    STAGE_INFER_PREFIX,
    STATE_MAIN_METRIC,
)
from catalyst.utils.tools.typing import (
    Criterion,
    Device,
    Model,
    Optimizer,
    Scheduler,
)

if TYPE_CHECKING:
    from .callback import Callback  # noqa: F401

StateModel = Union[Model, Dict[str, Model]]
StateCriterion = Union[Criterion, Dict[str, Criterion]]
StateOptimizer = Union[Optimizer, Dict[str, Optimizer]]
StateScheduler = Union[Scheduler, Dict[str, Scheduler]]


class State(FrozenClass):
    """
    Some intermediate storage between Experiment and Runner
    that saves the current state of the Experiments –
    model, criterion, optimizer, schedulers, metrics, loggers, loaders, etc

    .. note::
        To learn more about Catalyst Core concepts, please check out

            - :py:mod:`catalyst.core.experiment._Experiment`
            - :py:mod:`catalyst.core.runner._Runner`
            - :py:mod:`catalyst.core.state.State`
            - :py:mod:`catalyst.core.callback.Callback`

    **state.loaders** - ordered dictionary with torch.DataLoaders; \
    for example,
    ::

        state.loaders = {
            "train": MnistTrainLoader(),
            "valid": MnistValidLoader()
        }

    .. note::
        - "*train*" prefix is used for training loaders - \
          metrics computations, backward pass, optimization
        - "*valid*" prefix is used for validation loaders - \
          metrics computations only
        - "*infer*" prefix is used for inference loaders - \
          dataset prediction


    **state.model** - an instance of torch.nn.Module class, \
    (should implement ``forward`` method); \
    for example,
    ::

        state.model = torch.nn.Linear(10, 10)

    **state.criterion** - an instance of torch.nn.Module class\
    or torch.nn.modules.loss._Loss (should implement ``forward`` method); \
    for example,
    ::

        state.criterion = torch.nn.CrossEntropyLoss()

    **state.optimizer** - an instance of torch.optim.optimizer.Optimizer\
    (should implement ``step`` method); \
    for example,
    ::

        state.optimizer = torch.optim.Adam()

    **state.scheduler** - an instance of torch.optim.lr_scheduler._LRScheduler\
    (should implement ``step`` method); \
    for example,
    ::

        state.scheduler = htorch.optim.lr_scheduler.ReduceLROnPlateau()

    **state.device** - an instance of torch.device (CPU, GPU, TPU); \
    for example,
    ::

        state.device = torch.device("cpu")

    **state.callbacks** - ordered dictionary with Catalyst.Callback instances;\
    for example,
    ::

        state.callbacks = {
            "accuracy": AccuracyCallback(),
            "criterion": CriterionCallback(),
            "optim": OptimizerCallback(),
            "saver": CheckpointCallback()
        }


    **state.batch_in** - dictionary, \
    containing batch of data from currents DataLoader; \
    for example,
    ::

        state.batch_in = {
            "images": np.ndarray(batch_size, c, h, w),
            "targets": np.ndarray(batch_size, 1),
        }

    **state.batch_out** - dictionary, \
    containing model output for current batch; \
    for example,
    ::

        state.batch_out = {"logits": torch.Tensor(batch_size, num_classes)}

    **state.batch_metrics** - dictionary, flatten storage for batch metrics; \
    for example,
    ::

        state.batch_metrics = {"loss": ..., "accuracy": ..., "iou": ...}

    **state.loader_metrics** - dictionary with aggregated batch statistics \
    for loader (mean over all batches) and global loader metrics, like AUC; \
    for example,
    ::

        state.loader_metrics = {"loss": ..., "accuracy": ..., "auc": ...}

    **state.epoch_metrics** - dictionary with summarized metrics \
    for different loaders and global epoch metrics, like lr, momentum; \
    for example,
    ::

        state.epoch_metrics = {
            "train_loss": ..., "train_auc": ..., "valid_loss": ...,
            "lr": ..., "momentum": ...,
        }


    **state.is_best_valid** - bool, indicator flag

        - ``True`` if this training epoch is best over all epochs
        - ``False`` if not

    **state.valid_metrics** - dictionary with validation metrics\
    for currect epoch; \
    for example,
    ::

        state.valid_metrics = {"loss": ..., "accuracy": ..., "auc": ...}

    .. note::
        subdictionary of epoch_metrics

    **state.best_valid_metrics** - dictionary with best validation metrics \
    during whole training process


    **state.distributed_rank** - distributed rank of current worker

    **state.is_distributed_worker** - bool, indicator flag

        - ``True`` if is worker node (state.distributed_rank > 0)
        - ``False`` if is master node (state.distributed_rank == 0)


    **state.stage_name** - string, current stage name,\
    for example,
    ::

        state.stage_name = "pretraining" / "training" / "finetuning" / etc

    **state.epoch** - int, numerical indicator for current stage epoch

    **state.num_epochs** - int, maximum number of epochs, \
    required for this stage


    **state.loader_name** - string, current loader name\
    for example,
    ::

        state.loader_name = "train_dataset1" / "valid_data2" / "infer_golden"

    **state.loader_step** - int, numerical indicator \
    for batch index in current loader

    **state.loader_len** - int, maximum number of batches in current loaders


    **state.batch_size** - int, typical Deep Learning batch size parameter


    **state.global_step** - int, numerical indicator, counter for all batches,\
    that passes through our model during training, validation and\
    inference stages

    **state.global_epoch** - int, numerical indicator, counter for all epochs,\
    that have passed during model training, validation and\
    inference stages


    **state.main_metric** - string, containing name of metric of interest \
    for optimization, validation and checkpointing during training

    **state.minimize_metric** - bool, indicator flag

        - ``True`` if we need to minimize metric during training,\
          like `Cross Entropy loss`
        - ``False`` if we need to maximize metric during training, \
          like `Accuracy` or `Intersection over Union`

    **state.valid_loader** - string, name of validation loader \
    for metric selection, validation and model checkpoining


    **state.logdir** - string, path to logging directory to save\
    all logs, metrics, checkpoints and artifacts

    **state.checkpoint_data** - dictionary\
    with all extra data for experiment tracking


    **state.is_check_run** - bool, indicator flag

        - ``True`` if you want to check you pipeline and \
          run only 2 batches per loader and 2 epochs per stage
        - ``False`` (default) if you want to just the pipeline

    **state.is_train_loader** - bool, indicator flag

        - ``True`` for training loaders
        - ``False`` otherwise

    **state.is_valid_loader** - bool, indicator flag

        - ``True`` for validation loaders
        - ``False`` otherwise

    **state.is_infer_loader** - bool, indicator flag

        - ``True`` for inference loaders
        - ``False`` otherwise

    **state.is_infer_stage** - bool, indicator flag

        - ``True`` for inference stages
        - ``False`` otherwise

    **state.need_early_stop** - bool, indicator flag \
    used for EarlyStopping and CheckRun Callbacks

        - ``True`` if we need to stop the training
        - ``False`` (default) otherwise

    **state.need_exception_reraise** - bool, indicator flag

        - ``True`` (default) if you want to show exception \
          during pipeline and stop the training process
        - ``False`` otherwise

    **state.exception** - python Exception instance to raise (or not ;) )
    """

    def __init__(
        self,
        *,
        device: Device = None,
        model: StateModel = None,
        criterion: StateCriterion = None,
        optimizer: StateOptimizer = None,
        scheduler: StateScheduler = None,
        callbacks: Dict[str, "Callback"] = None,
        logdir: str = None,
        stage: str = STAGE_INFER_PREFIX,
        num_epochs: int = None,
        main_metric: str = STATE_MAIN_METRIC,
        minimize_metric: bool = True,
        valid_loader: str = LOADER_VALID_PREFIX,
        checkpoint_data: Dict = None,
        is_check_run: bool = False,
        **kwargs,
    ):
        """
        Args:
            @TODO: Docs. Contribution is welcome
        """
        # main part
        # data
        self.loaders: OrderedDict[str, DataLoader] = None
        # components
        self.model: StateModel = model
        self.criterion: StateCriterion = criterion
        self.optimizer: StateOptimizer = optimizer
        self.scheduler: StateScheduler = scheduler
        # extra components - PyTorch device
        self.device: Device = device
        # extra components - Catalyst callbacks
        self.callbacks: Dict[str, "Callback"] = callbacks

        # dataflow - model input, model output, metrics
        self.batch_in = None
        self.batch_out = None
        # let's use flatten storage for batch metrics
        # batch_metrics = {'loss': ..., 'accuracy': ..., 'iou': ...}
        self.batch_metrics = defaultdict(None)
        # just aggregated (aka mean over all batches)
        # batch statistics for loader
        # and global loader metrics, like AUC
        # loader_metrics = {'loss': ..., 'accuracy': ..., `auc`: ...}
        self.loader_metrics = defaultdict(None)
        # summarized metrics for different loaders
        # and global epoch metrics, like lr, momentum
        # epoch_metrics = {
        # 'train_loss': ..., 'train_auc': ..., 'valid_loss': ...,
        # 'lr': ..., 'momentum': ...,
        # }
        self.epoch_metrics = defaultdict(None)

        # validation
        self.is_best_valid = False
        self.valid_metrics = defaultdict(None)
        self.best_valid_metrics = defaultdict(None)

        # pipeline info
        self.distributed_rank = utils.get_rank()
        self.is_distributed_worker = self.distributed_rank > 0

        self.stage_name: str = stage
        self.epoch: int = 1
        self.num_epochs: int = num_epochs or np.iinfo(np.int32).max

        self.loader_name: str = None
        self.loader_step: int = 0
        self.loader_len: int = 0

        self.batch_size: int = 0

        self.global_step: int = 0
        self.global_epoch: int = 1

        # metrics & validation
        self.main_metric: str = main_metric
        self.minimize_metric: bool = minimize_metric
        self.valid_loader: str = valid_loader

        # logging
        self.logdir: Path = Path(logdir) if logdir is not None else None
        # extra checkpoint data for saving in checkpoint files
        self.checkpoint_data: Dict = checkpoint_data or {}

        # other
        self.is_check_run: bool = is_check_run
        self.is_train_loader: bool = False
        self.is_valid_loader: bool = False
        self.is_infer_loader: bool = False
        self.is_infer_stage: bool = self.stage_name.startswith(
            STAGE_INFER_PREFIX
        )
        self.need_early_stop: bool = False
        self.need_exception_reraise: bool = True
        self.exception: Optional[Exception] = None

        # kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

        self._freeze()

    @property
    def input(self):
        """Alias for `state.batch_in`.

        .. warning::
            Deprecated, saved for backward compatibility.
            Please use `state.batch_in` instead.
        """
        warnings.warn(
            "`input` was deprecated, " "please use `batch_in` instead",
            DeprecationWarning,
        )
        return self.batch_in

    @property
    def output(self):
        """Alias for `state.batch_out`.

        .. warning::
            Deprecated, saved for backward compatibility.
            Please use `state.batch_out` instead.
        """
        warnings.warn(
            "`output` was deprecated, " "please use `batch_out` instead",
            DeprecationWarning,
        )
        return self.batch_out

    @property
    def need_backward_pass(self):
        """Alias for `state.is_train_loader`.

        .. warning::
            Deprecated, saved for backward compatibility.
            Please use `state.is_train_loader` instead.
        """
        warnings.warn(
            "`need_backward_pass` was deprecated, "
            "please use `is_train_loader` instead",
            DeprecationWarning,
        )
        return self.is_train_loader

    def get_attr(self, key: str, inner_key: str = None) -> Any:
        """
        Alias for python `getattr` method. Useful for Callbacks preparation
        and cases with multi-criterion, multi-optimizer setup.
        For example, when you would like to train multi-task classification.

        Used to get a named attribute from a `State` by `key` keyword;
        for example\
        ::

            # example 1
            state.get_attr("criterion")
            # is equivalent to
            state.criterion

            # example 2
            state.get_attr("optimizer")
            # is equivalent to
            state.optimizer

            # example 3
            state.get_attr("scheduler")
            # is equivalent to
            state.scheduler

        With `inner_key` usage, it suppose to find a dictionary under `key`\
        and would get `inner_key` from this dict; for example,
        ::

            # example 1
            state.get_attr("criterion", "bce")
            # is equivalent to
            state.criterion["bce"]

            # example 2
            state.get_attr("optimizer", "adam")
            # is equivalent to
            state.optimizer["adam"]

            # example 3
            state.get_attr("scheduler", "adam")
            # is equivalent to
            state.scheduler["adam"]

        Args:
            key (str): name for attribute of interest,
                like `criterion`, `optimizer`, `scheduler`
            inner_key (str): name of inner dictionary key
        """
        if inner_key is None:
            return getattr(self, key)
        else:
            return getattr(self, key)[inner_key]
