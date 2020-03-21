# flake8: noqa
from typing import Dict, Optional, Union, TYPE_CHECKING  # isort:skip
from collections import defaultdict, OrderedDict
from pathlib import Path
import warnings

import numpy as np

from torch.utils.data import DataLoader

from catalyst import utils
from catalyst.utils.tools.frozen_class import FrozenClass
from catalyst.utils.tools.typing import (
    Criterion, Device, Model, Optimizer, Scheduler
)

if TYPE_CHECKING:
    from .callback import Callback  # noqa: F401

StateModel = Union[Model, Dict[str, Model]]
StateCriterion = Union[Criterion, Dict[str, Criterion]]
StateOptimizer = Union[Optimizer, Dict[str, Optimizer]]
StateScheduler = Union[Scheduler, Dict[str, Scheduler]]


class State(FrozenClass):
    r"""
    Object containing all information about current state of the experiment.

    state.loaders - ordered dictionary with torch.DataLoaders
        - "train" prefix is used for training loaders \
          (metrics computations, backward pass, optimization)
        - "valid" prefix is used for validation loaders - metrics only
        - "infer" prefix is used for inference loaders - dataset prediction

        ::

            state.loaders = {
                "train": MnistTrainLoader(),
                "valid": MnistValidLoader()
            }

    state.model - an instance of torch.nn.Module class
        should implement ``forward`` method
        ::

            state.model = torch.nn.Linear(10, 10)

    state.criterion - an instance of torch.nn.Module class or torch.nn.modules.loss._Loss
        should implement ``forward`` method
        ::

            state.criterion = torch.nn.CrossEntropyLoss()

    state.optimizer - an instance of torch.optim.optimizer.Optimizer
        should implement ``step`` method
        ::

            state.optimizer = torch.optim.Adam()

    state.scheduler - an instance of torch.optim.lr_scheduler._LRScheduler
        should implement ``step`` method
        ::

            state.scheduler = htorch.optim.lr_scheduler.ReduceLROnPlateau()

    state.device - an instance of torch.device (CPU, GPU, TPU)
        ::

            state.device = torch.device("cpu")

    state.callbacks - ordered dictionary with Catalyst.Callback instances
        ::

            state.callbacks = {
                "accuracy": AccuracyCallback(),
                "criterion": CriterionCallback(),
                "optim": OptimizerCallback(),
                "saver": CheckpointCallback()
            }

    state.batch_in - dictionary, containing current batch of data from DataLoader
        ::

            state.batch_in = {
                "images": np.ndarray(batch_size, c, h, w),
                "targets": np.ndarray(batch_size, 1),
            }

    state.batch_out - dictionary, containing model output based on current batch
        ::

            state.batch_out = {"logits": torch.Tensor(batch_size, num_classes)}

    state.batch_metrics - dictionary, flatten storage for batch metrics
        ::

            state.batch_metrics = {"loss": ..., "accuracy": ..., "iou": ...}

    state.loader_metrics - dictionary with aggregated batch statistics for loader (mean over all batches) and global loader metrics, like AUC
        ::

            state.loader_metrics = {"loss": ..., "accuracy": ..., "auc": ...}

    state.epoch_metrics - dictionary with summarized metrics for different loaders and global epoch metrics, like lr, momentum
        ::

            state.epoch_metrics = {
                "train_loss": ..., "train_auc": ..., "valid_loss": ...,
                "lr": ..., "momentum": ...,
            }

    state.is_best_valid - bool, indicator flag
        - ``True`` if this training epoch is best over all epochs
        - ``False`` if not

    state.valid_metrics - dictionary with validation metrics for currect epoch
        just a subdictionary of epoch_metrics
        ::

            state.valid_metrics = {"loss": ..., "accuracy": ..., "auc": ...}

    state.best_valid_metrics - dictionary with best validation metrics during whole training process

    state.distributed_rank

    state.is_distributed_worker

    state.stage_name

    state.epoch

    state.num_epochs

    state.loader_name

    state.loader_step

    state.loader_len

    state.batch_size

    state.global_step

    state.global_epoch

    state.main_metric

    state.minimize_metric

    state.valid_loader

    state.logdir - path to logging directory to save
        all logs, metrics, checkpoints and artifacts

    state.checkpoint_data - dictionary
        with all extra data for experiment tracking

    state.is_check_run - bool, indicator flag
        - ``True`` if you want to check you pipeline and run only 2 batches per loader and 2 epochs per stage
        - ``False`` (default) if you want to just the pipeline

    state.need_backward_pass - bool, indicator flag
        - ``True`` for training loaders
        - ``False`` otherwise

    state.need_early_stop - bool, indicator flag
        used for EarlyStopping and CheckRun Callbacks

        - ``True`` if we need to stop the training
        - ``False`` (default) otherwise

    state.need_exception_reraise - bool, indicator flag
        - ``True`` (default) if you want to show exception during pipeline and stop the training process
        - ``False`` otherwise

    state.exception - python Exception instance to raise
        (or not ;) )
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
        stage: str = "infer",
        num_epochs: int = None,
        main_metric: str = "loss",
        minimize_metric: bool = True,
        valid_loader: str = "valid",
        checkpoint_data: Dict = None,
        is_check_run: bool = False,
        **kwargs,
    ):
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
        self.is_infer_stage: bool = self.stage_name.startswith("infer")
        self.need_early_stop: bool = False
        self.need_exception_reraise: bool = True
        self.exception: Optional[Exception] = None

        # kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

        self._freeze()

    @property
    def input(self):
        # backward compatibility
        return self.batch_in

    @property
    def output(self):
        # backward compatibility
        return self.batch_out

    @property
    def need_backward_pass(self):
        warnings.warn(
            "`need_backward_pass` was deprecated, "
            "please use `is_train_loader` instead", DeprecationWarning
        )
        return self.is_train_loader

    def get_attr(self, key, inner_key=None):
        if inner_key is None:
            return getattr(self, key)
        else:
            return getattr(self, key)[inner_key]

    def set_attr(self, value, key, inner_key=None):
        if inner_key is None:
            setattr(self, key, value)
        else:
            getattr(self, key)[inner_key] = value
