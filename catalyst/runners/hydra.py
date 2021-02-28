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
from catalyst.core.functional import check_callback_isinstance
from catalyst.core.logger import ILogger
from catalyst.core.runner import IRunner
from catalyst.core.trial import ITrial
from catalyst.engines import IEngine, process_engine
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

logger = logging.getLogger(__name__)


class HydraRunner(IRunner):
    """Runner created from a hydra configuration file."""

    def __init__(self, cfg: DictConfig):
        """
        Args:
            cfg: dictionary with parameters
        """
        super().__init__()
        self._config: DictConfig = deepcopy(cfg)

        self._seed: int = self._config.args.seed
        self._verbose: bool = self._config.args.verbose
        self._timeit: bool = self._config.args.timeit
        self._check: bool = self._config.args.check
        self._overfit: bool = self._config.args.overfit
        self._name: str = self._get_name_()
        self._logdir: str = self._get_logdir_()
        self._trial = None  # @TODO: hack for catalyst-dl tune

    def _get_name_(self) -> str:
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

    def _get_logdir_(self) -> str:  # noqa: WPS112
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
        return self._logdir

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def name(self) -> str:
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
        return self._config.stages[stage].num_epochs or 1

    def get_trial(self) -> ITrial:
        return self._trial

    def get_engine(self) -> IEngine:
        engine_params = self._config.engine
        # @TODO: remove the trick
        try:
            engine = hydra.utils.instantiate(engine_params)
        except:
            engine = process_engine(**engine_params)
        return engine

    def get_loggers(self) -> Dict[str, ILogger]:
        loggers_params = self._config.loggers or {}
        loggers = {key: hydra.utils.instantiate(params) for key, params in loggers_params.items()}

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

    # @staticmethod
    # def _get_transform(params: DictConfig) -> Any:
    #     transform: Any = hydra.utils.instantiate(params)
    #     return transform
    #
    # def get_transforms(self, stage: str = None, dataset: str = None) -> Dict[str, Any]:
    #     """
    #     Returns transforms for a given stage.
    #
    #     Args:
    #         stage: (str) stage name
    #         dataset: (str) dataset name
    #
    #     Returns:
    #         Dict[str, Any]: transform objects
    #
    #     """
    #     assert (
    #         "transforms" in self._config.stages[stage]
    #     ), "stages config must contain 'transforms' key"
    #     transforms_params = self._config.stages[stage].transforms
    #     transforms: Dict[str, Any] = {
    #         name: self._get_transform(transform_params)
    #         for name, transform_params in transforms_params.items()
    #     }
    #     return transforms
    #
    # @staticmethod
    # def _get_dataset(transform: Any, params: DictConfig) -> Dataset:
    #     dataset: Dataset = hydra.utils.instantiate(params, transform=transform)
    #     return dataset
    #
    # def get_datasets(self, stage: str, **kwargs) -> Dict[str, Dataset]:
    #     """
    #     Returns datasets for a given stage.
    #
    #     Args:
    #         stage (str): stage name
    #         epoch (int): epoch number
    #         **kwargs: kwargs
    #
    #     Returns:
    #         Dict[str, Dataset]: datasets objects
    #
    #     """
    #     transforms = self.get_transforms(stage)
    #     assert (
    #             "datasets" in self._config.stages[stage]
    #     ), "stages config must contain 'datasets' key"
    #     datasets_params = self._config.stages[stage].datasets
    #     datasets: Dict[str, Dataset] = {
    #         name: self._get_dataset(transforms.get(name, None), dataset_params)
    #         for name, dataset_params in datasets_params.items()
    #     }
    #     return datasets
    #
    # @staticmethod
    # def _get_sampler(params: DictConfig) -> Sampler:
    #     sampler: Sampler = hydra.utils.instantiate(params)
    #     return sampler
    #
    # def get_samplers(self, stage: str) -> Dict[str, Sampler]:
    #     """
    #     Returns samplers for a given stage.
    #
    #     Args:
    #         stage: (str) stage name
    #
    #     Returns:
    #         Dict[str, Sampler]: samplers objects
    #
    #     """
    #     samplers_params = self._config.stages[stage].get("samplers", None)
    #     if samplers_params is None:
    #         return {}
    #     samplers: Dict[str, Sampler] = {
    #         name: self._get_sampler(sampler_params)
    #         for name, sampler_params in samplers_params.items()
    #     }
    #     return samplers
    #
    # @staticmethod
    # def _get_loader(
    #         dataset: Dataset, sampler: Sampler, initial_seed: int, params: DictConfig,
    # ) -> DataLoader:
    #     params = OmegaConf.to_container(params, resolve=True)
    #     per_gpu_scaling = params.pop("per_gpu_scaling", False)
    #     params["dataset"] = dataset
    #     distributed_rank = get_rank()
    #     distributed = distributed_rank > -1
    #     if per_gpu_scaling and not distributed:
    #         num_gpus = max(1, torch.cuda.device_count())
    #         assert "batch_size" in params, "loader config must contain 'batch_size' key"
    #         assert "num_workers" in params, "loader config must contain 'num_workers' key"
    #         params["batch_size"] *= num_gpus
    #         params["num_workers"] *= num_gpus
    #     if distributed:
    #         if sampler is not None:
    #             if not isinstance(sampler, DistributedSampler):
    #                 sampler = DistributedSamplerWrapper(sampler=sampler)
    #         else:
    #             sampler = DistributedSampler(dataset=params["dataset"])
    #     params["shuffle"] = params.get("shuffle", False) and sampler is None
    #     params["sampler"] = sampler
    #     worker_init_fn = params.pop("worker_init_fn", None)
    #     if worker_init_fn is None:
    #         params["worker_init_fn"] = lambda x: set_global_seed(initial_seed + x)
    #     else:
    #         params["worker_init_fn"] = hydra.utils.get_method(worker_init_fn)
    #     collate_fn = params.pop("collate_fn", None)
    #     if collate_fn is None:
    #         params["collate_fn"] = None
    #     else:
    #         params["collate_fn"] = hydra.utils.get_method(collate_fn)
    #     loader: DataLoader = DataLoader(**params)
    #     return loader

    def get_loaders(self, stage: str) -> Dict[str, DataLoader]:
        """
        Returns loaders for a given stage.

        Args:
            stage: (str) stage name
            epoch: (int) epoch number

        Returns:
            Dict[str, DataLoader]: loaders objects

        """
        # datasets = self.get_datasets(stage)
        # samplers = self.get_samplers(stage)
        # assert "loaders" in self._config.stages[stage], "stages config must contain 'loaders' key"
        # loaders_params = self._config.stages[stage].loaders
        # loaders: Dict[str, DataLoader] = {
        #     name: self._get_loader(
        #         datasets.get(name, None), samplers.get(name, None), self._seed, loader_params,
        #     )
        #     for name, loader_params in loaders_params.items()
        # }
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
    def get_model_(params: DictConfig) -> RunnerModel:
        params = deepcopy(params)
        is_key_value = params._key_value or False
        if is_key_value:
            model = {
                key: HydraRunner.get_model_(value) for key, value in params.items()  # noqa: WPS437
            }
            # model = nn.ModuleDict(model)
        else:
            model: Model = hydra.utils.instantiate(params)
        return model

    def get_model(self, stage: str) -> RunnerModel:
        """Returns the model for a given stage."""
        assert "model" in self._config, "config must contain 'model' key"
        model_params: DictConfig = self._config.model
        model: RunnerModel = self.get_model_(model_params)
        return model

    @staticmethod
    def get_criterion_(params: DictConfig) -> RunnerCriterion:
        params = deepcopy(params)
        is_key_value = params._key_value or False
        if is_key_value:
            criterion = {
                key: HydraRunner.get_criterion_(value)  # noqa: WPS437
                for key, value in params.items()
            }
        else:
            criterion: Criterion = hydra.utils.instantiate(params)
        return criterion

    def get_criterion(self, stage: str) -> RunnerCriterion:
        """Returns the criterions for a given stage."""
        if "criterion" not in self._config.stages[stage]:
            return None
        criterion_params: DictConfig = self._config.stages[stage].criterion
        criterion = self.get_criterion_(criterion_params)
        return criterion

    def get_optimizer_(self, model: RunnerModel, stage: str, params: DictConfig,) -> Optimizer:
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
            epoch: current epoch index

        Returns:
            optimizer for selected stage and epoch
        """
        if "optimizer" not in self._config.stages[stage]:
            return None

        optimizer_params: DictConfig = self._config.stages[stage].optimizer
        is_key_value = optimizer_params._key_value or False

        if is_key_value:
            optimizer = {}
            for key, params in optimizer_params.items():
                # load specified optimizer from checkpoint
                optimizer_key = "_optimizer"
                assert optimizer_key not in params, "keyword reserved"
                params[optimizer_key] = key

                optimizer[key] = self.get_optimizer_(model=model, stage=stage, params=params)
        else:
            optimizer = self.get_optimizer_(model=model, stage=stage, params=optimizer_params)

        return optimizer

    @staticmethod
    def get_scheduler_(*, optimizer: RunnerOptimizer, params: DictConfig) -> RunnerScheduler:
        params = deepcopy(params)
        is_key_value = params._key_value or False
        optimizer_key = params._optimizer or None
        optimizer = optimizer[optimizer_key] if optimizer_key else optimizer

        if is_key_value:
            scheduler: Dict[str, Scheduler] = {}
            for key, scheduler_params in params.items():
                scheduler[key] = HydraRunner.get_scheduler_(  # noqa: WPS437
                    optimizer=optimizer, params=scheduler_params
                )
        else:
            scheduler = hydra.utils.instantiate(params, optimizer=optimizer)

        return scheduler

    def get_scheduler(self, optimizer: RunnerOptimizer, stage: str) -> RunnerScheduler:
        """Returns the schedulers for a given stage."""
        if "scheduler" not in self._config.stages[stage]:
            return None
        scheduler_params: DictConfig = self._config.stages[stage].scheduler
        scheduler = self.get_scheduler_(optimizer=optimizer, params=scheduler_params)
        return scheduler

    @staticmethod
    def _get_callback(params: DictConfig):
        params = deepcopy(params)
        wrapper_params = params.pop("_wrapper", None)
        target = params.pop("_target_")
        callback_class = hydra.utils.get_class(target)
        params = OmegaConf.to_container(params, resolve=True)
        callback = callback_class(**params)
        if wrapper_params is not None:
            wrapper_params["base_callback"] = callback
            callback = HydraRunner._get_callback(**wrapper_params)  # noqa: WPS437
        return callback

    def get_callbacks(self, stage: str) -> "OrderedDict[str, Callback]":
        """Returns the callbacks for a given stage."""
        callbacks_params = self._config.stages[stage].callbacks or {}

        callbacks: Dict[str, Callback] = {
            name: self._get_callback(callback_params)
            for name, callback_params in callbacks_params.items()
        }

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


class SupervisedHydraRunner(ISupervisedRunner, HydraRunner):
    def __init__(
        self,
        cfg: DictConfig = None,
        input_key: Any = "features",
        output_key: Any = "logits",
        target_key: str = "targets",
        loss_key: str = "loss",
    ):
        ISupervisedRunner.__init__(
            self,
            input_key=input_key,
            output_key=output_key,
            target_key=target_key,
            loss_key=loss_key,
        )
        HydraRunner.__init__(self, cfg=cfg)


__all__ = ["HydraRunner", "SupervisedHydraRunner"]
