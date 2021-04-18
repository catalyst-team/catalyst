from typing import Any, Dict, List
from collections import OrderedDict
from copy import deepcopy
from functools import partial
import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from catalyst.callbacks import CheckpointCallback, ICheckpointCallback
from catalyst.callbacks.batch_overfit import BatchOverfitCallback
from catalyst.callbacks.misc import CheckRunCallback, TimerCallback, TqdmCallback
from catalyst.core import Callback
from catalyst.core.logger import ILogger
from catalyst.core.misc import callback_isinstance
from catalyst.core.runner import IRunner
from catalyst.core.trial import ITrial
from catalyst.engines import IEngine
from catalyst.loggers.console import ConsoleLogger
from catalyst.loggers.csv import CSVLogger
from catalyst.loggers.tensorboard import TensorboardLogger
from catalyst.runners.misc import do_lr_linear_scaling, get_model_parameters
from catalyst.runners.supervised import ISupervisedRunner
from catalyst.typing import (
    Criterion,
    Model,
    Optimizer,
    RunnerCriterion,
    RunnerModel,
    RunnerOptimizer,
    RunnerScheduler,
    Scheduler,
)
from catalyst.utils.data import get_loaders_from_params
from catalyst.utils.misc import get_short_hash, get_utcnow_time
from catalyst.utils.torch import get_available_engine

logger = logging.getLogger(__name__)


class HydraRunner(IRunner):
    """Runner created from a hydra configuration file.

    Args:
        cfg: Hydra dictionary with parameters
    """

    def __init__(self, cfg: DictConfig):
        """Init."""
        super().__init__()
        self._config: DictConfig = deepcopy(cfg)

        self._apex: bool = self._config.args.apex or False
        self._amp: bool = self._config.args.amp or False
        self._ddp: bool = self._config.args.ddp or False
        self._fp16: bool = self._config.args.fp16 or False

        self._seed: int = self._config.args.seed or 42
        self._verbose: bool = self._config.args.verbose or False
        self._timeit: bool = self._config.args.timeit or False
        self._check: bool = self._config.args.check or False
        self._overfit: bool = self._config.args.overfit or False

        self._name: str = self._get_run_name()
        self._logdir: str = self._get_run_logdir()

        # @TODO: hack for catalyst-dl tune, could be done better
        self._trial = None

    def _get_run_name(self) -> str:
        timestamp = get_utcnow_time()
        config_hash = get_short_hash(self._config)
        default_name = f"{timestamp}-{config_hash}"
        name = self._config.args.name or default_name
        return name

    def _get_logdir(self, config: Dict) -> str:
        timestamp = get_utcnow_time()
        config_hash = get_short_hash(config)
        logdir = f"{timestamp}.{config_hash}"
        return logdir

    def _get_run_logdir(self) -> str:  # noqa: WPS112
        output = None
        exclude_tag = "none"

        logdir: str = self._config.args.logdir
        baselogdir: str = self._config.args.baselogdir

        if logdir is not None and logdir.lower() != exclude_tag:
            output = logdir
        elif baselogdir is not None and baselogdir.lower() != exclude_tag:
            logdir = self._get_logdir(self._config)
            output = f"{baselogdir}/{logdir}"
        return output

    @property
    def logdir(self) -> str:
        """@TODO: docs."""
        return self._logdir

    @property
    def seed(self) -> int:
        """Experiment's seed for reproducibility."""
        return self._seed

    @property
    def name(self) -> str:
        """Returns run name for monitoring tools."""
        return self._name

    @property
    def hparams(self) -> OrderedDict:
        """Hyperparameters"""
        return OrderedDict(OmegaConf.to_container(self._config, resolve=True))

    @property
    def stages(self) -> List[str]:
        """Experiment's stage names."""
        stages_keys = list(self._config.stages.keys())
        return stages_keys

    def get_stage_len(self, stage: str) -> int:
        """Returns number of epochs for the selected stage.

        Args:
            stage: current stage

        Returns:
            number of epochs in stage

        Example::

            >>> runner.get_stage_len("pretraining")
            3
        """
        return self._config.stages[stage].num_epochs or 1

    def get_trial(self) -> ITrial:
        """Returns the trial for the run."""
        return self._trial

    def get_engine(self) -> IEngine:
        """Returns the engine for the run."""
        engine_params = self._config.engine
        if engine_params is not None:
            engine = hydra.utils.instantiate(engine_params)
        else:
            engine = get_available_engine(
                fp16=self._fp16, ddp=self._ddp, amp=self._amp, apex=self._apex
            )
        return engine

    def get_loggers(self) -> Dict[str, ILogger]:
        """Returns the loggers for the run."""
        loggers_params = self._config.loggers or {}
        loggers = {key: hydra.utils.instantiate(params) for key, params in loggers_params.items()}

        is_logger_exists = lambda logger_fn: any(
            isinstance(x, logger_fn) for x in loggers.values()
        )
        if not is_logger_exists(ConsoleLogger):
            loggers["_console"] = ConsoleLogger()
        if self._logdir is not None and not is_logger_exists(CSVLogger):
            loggers["_csv"] = CSVLogger(logdir=self._logdir, use_logdir_postfix=True)
        if self._logdir is not None and not is_logger_exists(TensorboardLogger):
            loggers["_tensorboard"] = TensorboardLogger(
                logdir=self._logdir, use_logdir_postfix=True,
            )

        return loggers

    def get_loaders(self, stage: str) -> Dict[str, DataLoader]:
        """
        Returns loaders for a given stage.

        Args:
            stage: stage name

        Returns:
            Dict: loaders objects

        """
        loaders_params = self._config.stages[stage].loaders
        loaders_params = OmegaConf.to_container(loaders_params, resolve=True)
        loaders = get_loaders_from_params(
            datasets_fn=partial(self.get_datasets, stage=stage),
            initial_seed=self.seed,
            stage=stage,
            **loaders_params,
        )
        return loaders

    @staticmethod
    def _get_model_from_params(params: DictConfig) -> RunnerModel:
        params = deepcopy(params)
        is_key_value = params._key_value or False
        if is_key_value:
            model = {
                key: HydraRunner._get_model_from_params(value)
                for key, value in params.items()  # noqa: WPS437
            }
            # model = nn.ModuleDict(model)
        else:
            model: Model = hydra.utils.instantiate(params)
        return model

    def get_model(self, stage: str) -> RunnerModel:
        """Returns the model for a given stage."""
        assert "model" in self._config, "config must contain 'model' key"
        model_params: DictConfig = self._config.model
        model: RunnerModel = self._get_model_from_params(model_params)
        return model

    @staticmethod
    def _get_criterion_from_params(params: DictConfig) -> RunnerCriterion:
        params = deepcopy(params)
        is_key_value = params._key_value or False
        if is_key_value:
            criterion = {
                key: HydraRunner._get_criterion_from_params(value)  # noqa: WPS437
                for key, value in params.items()
            }
        else:
            criterion: Criterion = hydra.utils.instantiate(params)
        return criterion

    def get_criterion(self, stage: str) -> RunnerCriterion:
        """Returns the criterion for a given stage."""
        if "criterion" not in self._config.stages[stage]:
            return None
        criterion_params: DictConfig = self._config.stages[stage].criterion
        criterion = self._get_criterion_from_params(criterion_params)
        return criterion

    def _get_optimizer_from_params(
        self, model: RunnerModel, stage: str, params: DictConfig
    ) -> Optimizer:
        # @TODO 1: refactor; this method is too long
        params = deepcopy(params)
        # learning rate linear scaling
        lr_scaling_params = params.pop("lr_linear_scaling", None)
        if lr_scaling_params:
            loaders_params = self._config.stages[stage].loaders
            lr, lr_scaling = do_lr_linear_scaling(
                lr_scaling_params=lr_scaling_params,
                batch_size=loaders_params.get("batch_size", 1),
                per_gpu_scaling=loaders_params.get("per_gpu_scaling", False),
            )
            params["lr"] = lr
        else:
            lr_scaling = 1.0
        # getting layer-wise parameters
        layerwise_params = params.pop("layerwise_params", OrderedDict())
        no_bias_weight_decay = params.pop("no_bias_weight_decay", True)
        # getting model parameters
        model_key = params.pop("_model", None)
        model_params = get_model_parameters(
            models=model,
            models_keys=model_key,
            layerwise_params=layerwise_params,
            no_bias_weight_decay=no_bias_weight_decay,
            lr_scaling=lr_scaling,
        )

        # instantiate optimizer
        optimizer: Optimizer = hydra.utils.instantiate(params, params=model_params)
        return optimizer

    def get_optimizer(self, model: RunnerModel, stage: str) -> RunnerOptimizer:
        """
        Returns the optimizer for a given stage and epoch.

        Args:
            model: model or a dict of models
            stage: current stage name

        Returns:
            optimizer for selected stage and epoch
        """
        if "optimizer" not in self._config.stages[stage]:
            return None

        optimizer_params: DictConfig = self._config.stages[stage].optimizer
        optimizer_params = deepcopy(optimizer_params)
        is_key_value = optimizer_params._key_value or False

        if is_key_value:
            optimizer = {}
            for key, params in optimizer_params.items():
                optimizer[key] = self._get_optimizer_from_params(
                    model=model, stage=stage, params=params
                )
        else:
            optimizer = self._get_optimizer_from_params(
                model=model, stage=stage, params=optimizer_params
            )

        return optimizer

    @staticmethod
    def _get_scheduler_from_params(
        *, optimizer: RunnerOptimizer, params: DictConfig
    ) -> RunnerScheduler:
        params = deepcopy(params)
        is_key_value = params._key_value or False
        if is_key_value:
            scheduler: Dict[str, Scheduler] = {}
            for key, scheduler_params in params.items():
                scheduler_params = deepcopy(scheduler_params)
                optimizer_key = scheduler_params._optimizer or None
                optim = optimizer[optimizer_key] if optimizer_key else optimizer
                scheduler[key] = HydraRunner._get_scheduler_from_params(  # noqa: WPS437
                    optimizer=optim, params=scheduler_params
                )
        else:
            optimizer_key = params._optimizer or None
            optimizer = optimizer[optimizer_key] if optimizer_key else optimizer
            scheduler = hydra.utils.instantiate(params, optimizer=optimizer)

        return scheduler

    def get_scheduler(self, optimizer: RunnerOptimizer, stage: str) -> RunnerScheduler:
        """Returns the schedulers for a given stage."""
        if "scheduler" not in self._config.stages[stage]:
            return None
        scheduler_params: DictConfig = self._config.stages[stage].scheduler
        scheduler = self._get_scheduler_from_params(optimizer=optimizer, params=scheduler_params)
        return scheduler

    @staticmethod
    def _get_callback_from_params(params: DictConfig):
        params = deepcopy(params)
        wrapper_params = params.pop("_wrapper", None)
        target = params.pop("_target_")
        callback_class = hydra.utils.get_class(target)
        params = OmegaConf.to_container(params, resolve=True)
        callback = callback_class(**params)
        if wrapper_params is not None:
            wrapper_params["base_callback"] = callback
            callback = HydraRunner._get_callback_from_params(**wrapper_params)  # noqa: WPS437
        return callback

    def get_callbacks(self, stage: str) -> "OrderedDict[str, Callback]":
        """Returns the callbacks for a given stage."""
        callbacks_params = self._config.stages[stage].callbacks or {}

        callbacks: Dict[str, Callback] = {
            name: self._get_callback_from_params(callback_params)
            for name, callback_params in callbacks_params.items()
        }

        is_callback_exists = lambda callback_fn: any(
            callback_isinstance(x, callback_fn) for x in callbacks.values()
        )
        if self._verbose and not is_callback_exists(TqdmCallback):
            callbacks["_verbose"] = TqdmCallback()
        if self._timeit and not is_callback_exists(TimerCallback):
            callbacks["_timer"] = TimerCallback()
        if self._check and not is_callback_exists(CheckRunCallback):
            callbacks["_check"] = CheckRunCallback()
        if self._overfit and not is_callback_exists(BatchOverfitCallback):
            callbacks["_overfit"] = BatchOverfitCallback()

        if self._logdir is not None and not is_callback_exists(ICheckpointCallback):
            callbacks["_checkpoint"] = CheckpointCallback(
                logdir=os.path.join(self._logdir, "checkpoints"),
            )

        return callbacks


class SupervisedHydraRunner(ISupervisedRunner, HydraRunner):
    """HydraRunner for supervised tasks

    Args:
        cfg: Hydra dictionary with parameters
        input_key: key in ``runner.batch`` dict mapping for model input
        output_key: key for ``runner.batch`` to store model output
        target_key: key in ``runner.batch`` dict mapping for target
        loss_key: key for ``runner.batch_metrics`` to store criterion loss output
    """

    def __init__(
        self,
        cfg: DictConfig = None,
        input_key: Any = "features",
        output_key: Any = "logits",
        target_key: str = "targets",
        loss_key: str = "loss",
    ):
        """Init."""
        ISupervisedRunner.__init__(
            self,
            input_key=input_key,
            output_key=output_key,
            target_key=target_key,
            loss_key=loss_key,
        )
        HydraRunner.__init__(self, cfg=cfg)


__all__ = ["HydraRunner", "SupervisedHydraRunner"]
