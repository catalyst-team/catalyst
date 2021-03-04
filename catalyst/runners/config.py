from typing import Any, Dict, List
from collections import OrderedDict
from copy import deepcopy
from functools import partial
import logging
import os

from torch import nn
from torch.utils.data import DataLoader

from catalyst.callbacks import CheckpointCallback, ICheckpointCallback
from catalyst.callbacks.batch_overfit import BatchOverfitCallback
from catalyst.callbacks.misc import CheckRunCallback, TimerCallback, TqdmCallback
from catalyst.core.callback import Callback
from catalyst.core.functional import check_callback_isinstance
from catalyst.core.logger import ILogger
from catalyst.core.runner import IRunner
from catalyst.core.trial import ITrial
from catalyst.engines import IEngine
from catalyst.loggers.console import ConsoleLogger
from catalyst.loggers.csv import CSVLogger
from catalyst.loggers.tensorboard import TensorboardLogger
from catalyst.registry import REGISTRY
from catalyst.runners.misc import do_lr_linear_scaling, get_model_parameters
from catalyst.runners.supervised import ISupervisedRunner
from catalyst.typing import (
    RunnerCriterion,
    RunnerModel,
    RunnerOptimizer,
    RunnerScheduler,
    Scheduler,
)
from catalyst.utils.data import get_loaders_from_params
from catalyst.utils.misc import get_by_keys, get_short_hash, get_utcnow_time

logger = logging.getLogger(__name__)


class ConfigRunner(IRunner):
    """@TODO: docs."""

    def __init__(self, config: Dict):
        """@TODO: docs.

        Args:
            config: dictionary with parameters
        """
        super().__init__()
        self._config: Dict = deepcopy(config)
        self._stage_config: Dict = self._config["stages"]

        self._seed: int = get_by_keys(self._config, "args", "seed", default=42)
        self._verbose: bool = get_by_keys(self._config, "args", "verbose", default=False)
        self._timeit: bool = get_by_keys(self._config, "args", "timeit", default=False)
        self._check: bool = get_by_keys(self._config, "args", "check", default=False)
        self._overfit: bool = get_by_keys(self._config, "args", "overfit", default=False)
        self._name: str = self._get_init_name()
        self._logdir: str = self._get_init_logdir()
        self._trial = None  # @TODO: hack for catalyst-dl tune

    def _get_init_name(self) -> str:
        timestamp = get_utcnow_time()
        config_hash = get_short_hash(self._config)
        default_name = f"{timestamp}-{config_hash}"
        name = get_by_keys(self._config, "args", "name", default=default_name)
        return name

    def _get_logdir(self, config: Dict) -> str:
        timestamp = get_utcnow_time()
        config_hash = get_short_hash(config)
        logdir = f"{timestamp}.{config_hash}"
        return logdir

    def _get_init_logdir(self) -> str:  # noqa: WPS112
        output = None
        exclude_tag = "none"

        logdir: str = get_by_keys(self._config, "args", "logdir", default=None)
        baselogdir: str = get_by_keys(self._config, "args", "baselogdir", default=None)

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
        """@TODO: docs."""
        return self._seed

    @property
    def name(self) -> str:
        """@TODO: docs."""
        return self._name

    @property
    def hparams(self) -> Dict:
        """Returns hyper parameters"""
        return OrderedDict(self._config)

    @property
    def stages(self) -> List[str]:
        """Experiment's stage names."""
        stages_keys = list(self._stage_config.keys())
        return stages_keys

    def get_stage_len(self, stage: str) -> int:
        """@TODO: docs."""
        return get_by_keys(self._stage_config, stage, "num_epochs", default=1)

    def get_trial(self) -> ITrial:
        """@TODO: docs."""
        return self._trial

    def get_engine(self) -> IEngine:
        """@TODO: docs."""
        engine_params = self._config.get("engine")
        engine = REGISTRY.get_from_params(**engine_params)
        return engine

    def get_loggers(self) -> Dict[str, ILogger]:
        """@TODO: docs."""
        loggers_params = self._config.get("loggers", {})
        loggers = {
            key: REGISTRY.get_from_params(**params) for key, params in loggers_params.items()
        }

        is_logger_exists = lambda logger_fn: any(
            isinstance(x, logger_fn) for x in loggers.values()
        )
        if not is_logger_exists(ConsoleLogger):
            loggers["_console"] = ConsoleLogger()
        if self._logdir is not None and not is_logger_exists(CSVLogger):
            loggers["_csv"] = CSVLogger(logdir=self._logdir)
        if self._logdir is not None and not is_logger_exists(TensorboardLogger):
            loggers["_tensorboard"] = TensorboardLogger(
                logdir=os.path.join(self._logdir, "tensorboard")
            )

        return loggers

    def get_loaders(self, stage: str) -> "OrderedDict[str, DataLoader]":
        """@TODO: docs."""
        loaders_params = dict(self._stage_config[stage]["loaders"])
        loaders = get_loaders_from_params(
            datasets_fn=partial(self.get_datasets, stage=stage),
            initial_seed=self.seed,
            stage=stage,
            **loaders_params,
        )
        return loaders

    @staticmethod
    def _get_model_from_params(**params) -> RunnerModel:
        params = deepcopy(params)
        is_key_value = params.pop("_key_value", False)

        if is_key_value:
            model = {
                model_key: ConfigRunner._get_model_from_params(**model_params)  # noqa: WPS437
                for model_key, model_params in params.items()
            }
            model = nn.ModuleDict(model)
        else:
            model = REGISTRY.get_from_params(**params)
        return model

    def get_model(self, stage: str) -> RunnerModel:
        """Returns the model for a given stage."""
        assert "model" in self._config, "config must contain 'model' key"
        model_params: Dict = self._config["model"]
        model: RunnerModel = self._get_model_from_params(**model_params)
        return model

    @staticmethod
    def _get_criterion_from_params(**params) -> RunnerCriterion:
        params = deepcopy(params)
        key_value_flag = params.pop("_key_value", False)

        if key_value_flag:
            criterion = {
                key: ConfigRunner.get_criterion_(**key_params)
                for key, key_params in params.items()  # noqa: WPS437
            }
        else:
            criterion = REGISTRY.get_from_params(**params)
        return criterion

    def get_criterion(self, stage: str) -> RunnerCriterion:
        """Returns the criterion for a given stage."""
        if "criterion" not in self._stage_config[stage]:
            return None
        criterion_params = get_by_keys(self._stage_config, stage, "criterion", default={})
        criterion = self._get_criterion_from_params(**criterion_params)
        return criterion

    def _get_optimizer_from_params(
        self, model: RunnerModel, stage: str, **params
    ) -> RunnerOptimizer:
        # @TODO 1: refactor; this method is too long

        # learning rate linear scaling
        lr_scaling_params = params.pop("lr_linear_scaling", None)
        if lr_scaling_params:
            loaders_params = dict(self._stage_config[stage]["loaders"])
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
        optimizer = REGISTRY.get_from_params(**params, params=model_params)
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
        if "optimizer" not in self._stage_config[stage]:
            return None

        optimizer_params = get_by_keys(self._stage_config, stage, "optimizer", default={})
        optimizer_params = deepcopy(optimizer_params)
        is_key_value = optimizer_params.pop("_key_value", False)

        if is_key_value:
            optimizer = {}
            for key, params in optimizer_params.items():
                # load specified optimizer from checkpoint
                optimizer_key = "_optimizer"
                assert optimizer_key not in params, "keyword reserved"
                params[optimizer_key] = key

                optimizer[key] = self._get_optimizer_from_params(
                    model=model, stage=stage, **params
                )
        else:
            optimizer = self._get_optimizer_from_params(
                model=model, stage=stage, **optimizer_params
            )

        return optimizer

    @staticmethod
    def _get_scheduler_from_params(*, optimizer: RunnerOptimizer, **params) -> RunnerScheduler:
        params = deepcopy(params)

        is_key_value = params.pop("_key_value", False)
        optimizer_key = params.pop("_optimizer", None)
        optimizer = optimizer[optimizer_key] if optimizer_key else optimizer

        if is_key_value:
            scheduler: Dict[str, Scheduler] = {}
            for key, scheduler_params in params.items():
                scheduler[key] = ConfigRunner.get_scheduler_(
                    **scheduler_params, optimizer=optimizer
                )  # noqa: WPS437
        else:
            scheduler = REGISTRY.get_from_params(**params, optimizer=optimizer)
        return scheduler

    def get_scheduler(self, optimizer: RunnerOptimizer, stage: str) -> RunnerScheduler:
        """Returns the scheduler for a given stage."""
        if "scheduler" not in self._stage_config[stage]:
            return None
        scheduler_params = get_by_keys(self._stage_config, stage, "scheduler", default={})
        scheduler = self._get_scheduler_from_params(optimizer=optimizer, **scheduler_params)
        return scheduler

    @staticmethod
    def _get_callback_from_params(**params):
        params = deepcopy(params)
        wrapper_params = params.pop("_wrapper", None)
        callback = REGISTRY.get_from_params(**params)
        if wrapper_params is not None:
            wrapper_params["base_callback"] = callback
            callback = ConfigRunner._get_callback_from_params(**wrapper_params)  # noqa: WPS437
        return callback

    def get_callbacks(self, stage: str) -> "OrderedDict[str, Callback]":
        """Returns the callbacks for a given stage."""
        callbacks_params = get_by_keys(self._stage_config, stage, "callbacks", default={})

        callbacks = OrderedDict(
            [
                (key, self._get_callback_from_params(**callback_params))
                for key, callback_params in callbacks_params.items()
            ]
        )

        is_callback_exists = lambda callback_fn: any(
            check_callback_isinstance(x, callback_fn) for x in callbacks.values()
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


class SupervisedConfigRunner(ISupervisedRunner, ConfigRunner):
    """@TODO: docs."""

    def __init__(
        self,
        config: Dict = None,
        input_key: Any = "features",
        output_key: Any = "logits",
        target_key: str = "targets",
        loss_key: str = "loss",
    ):
        """@TODO: docs."""
        ISupervisedRunner.__init__(
            self,
            input_key=input_key,
            output_key=output_key,
            target_key=target_key,
            loss_key=loss_key,
        )
        ConfigRunner.__init__(self, config=config)


__all__ = ["ConfigRunner", "SupervisedConfigRunner"]
