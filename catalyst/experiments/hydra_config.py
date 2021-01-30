from typing import Any, Dict, List, Mapping
from collections import OrderedDict
from copy import deepcopy
import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import Sampler

from catalyst.core.callback import Callback
from catalyst.core.experiment import IExperiment
from catalyst.data.sampler import DistributedSamplerWrapper
from catalyst.experiments.functional import (
    add_default_callbacks,
    do_lr_linear_scaling,
    get_model_parameters,
    load_optimizer_from_checkpoint,
    process_callbacks,
)
from catalyst.typing import Criterion, Model, Optimizer, Scheduler
from catalyst.utils.distributed import get_rank
from catalyst.utils.misc import set_global_seed

logger = logging.getLogger(__name__)


class HydraConfigExperiment(IExperiment):
    """
    Experiment created from a hydra configuration file.
    """

    def __init__(self, cfg: DictConfig):
        """
        Args:
            cfg (dict): dictionary with parameters
        """
        self._config: DictConfig = deepcopy(cfg)
        self._trial = None
        self._initial_seed: int = self._config.args.seed
        self._verbose: bool = self._config.args.verbose
        self._check_time: bool = self._config.args.timeit
        self._check_run: bool = self._config.args.check
        self._overfit: bool = self._config.args.overfit
        self._logdir: str = os.getcwd()

    @property
    def seed(self) -> int:
        """Experiment's initial seed value."""
        return self._initial_seed

    @property
    def logdir(self) -> str:
        """Path to the directory where the experiment logs."""
        return self._logdir

    @property
    def hparams(self) -> OrderedDict:
        """Returns hyperparameters"""
        return OrderedDict(OmegaConf.to_container(self._config, resolve=True))

    @property
    def trial(self) -> Any:
        """Returns hyperparameter trial for current experiment"""
        return self._trial

    @property
    def engine_params(self) -> Dict:
        """Dict with the parameters for distributed and FP16 methond."""
        return self._config.engine

    @property
    def stages(self) -> List[str]:
        """Experiment's stage names."""
        stages_keys = list(self._config.stages.keys())
        return stages_keys

    def get_stage_params(self, stage: str) -> Mapping[str, Any]:
        """Returns the state parameters for a given stage."""
        return self._config.stages[stage].params

    @staticmethod
    def _get_model(params: DictConfig) -> Model:
        model: Model = hydra.utils.instantiate(params)
        return model

    def get_model(self, stage: str) -> Dict[str, Model]:
        """Returns the models for a given stage."""
        assert "models" in self._config, "config must contain 'models' key"
        models_params: DictConfig = self._config.models
        model: Dict[str, Model] = {
            name: self._get_model(model_params) for name, model_params in models_params.items()
        }
        return model

    @staticmethod
    def _get_criterion(params: DictConfig) -> Criterion:
        criterion: Criterion = hydra.utils.instantiate(params)
        return criterion

    def get_criterion(self, stage: str) -> Dict[str, Criterion]:
        """Returns the criterions for a given stage."""
        if "criterions" not in self._config.stages[stage]:
            return {}
        criterions_params: DictConfig = self._config.stages[stage].criterions
        criterion: Dict[str, Criterion] = {
            name: self._get_criterion(criterion_params)
            for name, criterion_params in criterions_params.items()
        }
        return criterion

    def _get_optimizer(
        self, stage: str, models: Dict[str, Model], name: str, params: DictConfig,
    ) -> Optimizer:
        # lr linear scaling
        lr_scaling_params = params.pop("lr_linear_scaling", None)
        if lr_scaling_params:
            assert (
                "datasets" in self._config.stages[stage]
            ), "stages config must contain 'datasets' key"
            assert "train" in self._config.stages[stage]["datasets"], (
                "datasets config must contain 'train' " "dataset for lr linear scaling"
            )
            data_params = dict(self._config.stages[stage]["datasets"]["train"])
            batch_size = data_params.get("batch_size")
            per_gpu_scaling = data_params.get("per_gpu_scaling", False)
            lr, lr_scaling = do_lr_linear_scaling(
                lr_scaling_params=lr_scaling_params,
                batch_size=batch_size,
                per_gpu_scaling=per_gpu_scaling,
            )
            params["lr"] = lr
        else:
            lr_scaling = 1.0
        # getting layer-wise parameters
        layerwise_params = params.pop("layerwise_params", OrderedDict())
        no_bias_weight_decay = params.pop("no_bias_weight_decay", True)
        # getting model parameters
        models_keys = params.pop("models", None)
        model_params = get_model_parameters(
            models=models,
            models_keys=models_keys,
            layerwise_params=layerwise_params,
            no_bias_weight_decay=no_bias_weight_decay,
            lr_scaling=lr_scaling,
        )
        # getting load-from-previous-stage flag
        load_from_previous_stage = params.pop("load_from_previous_stage", False)
        # instantiate optimizer
        optimizer: Optimizer = hydra.utils.instantiate(params, params=model_params)
        # load from previous stage
        if load_from_previous_stage and self.stages.index(stage) != 0:
            checkpoint_path = f"{self.logdir}/checkpoints/best_full.pth"
            optimizer = load_optimizer_from_checkpoint(
                optimizer,
                checkpoint_path=checkpoint_path,
                checkpoint_optimizer_key=name,
                model_parameters=model_params,
                optimizer_params=params,
            )

        return optimizer

    def get_optimizer(self, stage: str, model: Dict[str, Model]) -> Dict[str, Optimizer]:
        """
        Returns the optimizers for a given stage.

        Args:
            stage (str):  stage name
            model (Dict[str, Model]): dict of models

        Returns:
            Dict[str, Optimizer]: optimizer for selected stage

        """
        if "optimizers" not in self._config.stages[stage]:
            return {}
        optimizers_params: DictConfig = self._config.stages[stage].optimizers
        optimizer = {
            name: self._get_optimizer(stage, model, name, optimizer_params)
            for name, optimizer_params in optimizers_params.items()
        }
        return optimizer

    @staticmethod
    def _get_scheduler(optimizers: Dict[str, Optimizer], params: DictConfig) -> Scheduler:
        assert "optimizer" in params, "scheduler config must contain 'optimizer' key"
        optimizer_key: str = params.pop("optimizer")
        scheduler: Scheduler = hydra.utils.instantiate(params, optimizer=optimizers[optimizer_key])
        return scheduler

    def get_scheduler(self, stage: str, optimizer: Dict[str, Optimizer]) -> Dict[str, Scheduler]:
        """Returns the schedulers for a given stage."""
        if "schedulers" not in self._config.stages[stage]:
            return {}
        schedulers_params: DictConfig = self._config.stages[stage].schedulers
        scheduler: Dict[str, Scheduler] = {
            key: self._get_scheduler(optimizers=optimizer, params=scheduler_params)
            for key, scheduler_params in schedulers_params.items()
        }
        return scheduler

    @staticmethod
    def _get_transform(params: DictConfig) -> Any:
        transform: Any = hydra.utils.instantiate(params)
        return transform

    def get_transforms(self, stage: str = None, dataset: str = None) -> Dict[str, Any]:
        """
        Returns transforms for a given stage.

        Args:
            stage: (str) stage name
            dataset: (str) dataset name

        Returns:
            Dict[str, Any]: transform objects

        """
        assert (
            "transforms" in self._config.stages[stage]
        ), "stages config must contain 'transforms' key"
        transforms_params = self._config.stages[stage].transforms
        transforms: Dict[str, Any] = {
            name: self._get_transform(transform_params)
            for name, transform_params in transforms_params.items()
        }
        return transforms

    @staticmethod
    def _get_dataset(transform: Any, params: DictConfig) -> Dataset:
        dataset: Dataset = hydra.utils.instantiate(params, transform=transform)
        return dataset

    def get_datasets(self, stage: str, epoch: int = None, **kwargs) -> Dict[str, Dataset]:
        """
        Returns datasets for a given stage.

        Args:
            stage (str): stage name
            epoch (int): epoch number
            **kwargs: kwargs

        Returns:
            Dict[str, Dataset]: datasets objects

        """
        transforms = self.get_transforms(stage)
        assert (
            "datasets" in self._config.stages[stage]
        ), "stages config must contain 'datasets' key"
        datasets_params = self._config.stages[stage].datasets
        datasets: Dict[str, Dataset] = {
            name: self._get_dataset(transforms.get(name, None), dataset_params)
            for name, dataset_params in datasets_params.items()
        }
        return datasets

    @staticmethod
    def _get_sampler(params: DictConfig) -> Sampler:
        sampler: Sampler = hydra.utils.instantiate(params)
        return sampler

    def get_samplers(self, stage: str) -> Dict[str, Sampler]:
        """
        Returns samplers for a given stage.

        Args:
            stage: (str) stage name

        Returns:
            Dict[str, Sampler]: samplers objects

        """
        samplers_params = self._config.stages[stage].get("samplers", None)
        if samplers_params is None:
            return {}
        samplers: Dict[str, Sampler] = {
            name: self._get_sampler(sampler_params)
            for name, sampler_params in samplers_params.items()
        }
        return samplers

    @staticmethod
    def _get_loader(
        dataset: Dataset, sampler: Sampler, initial_seed: int, params: DictConfig,
    ) -> DataLoader:
        params = OmegaConf.to_container(params, resolve=True)
        per_gpu_scaling = params.pop("per_gpu_scaling", False)
        params["dataset"] = dataset
        distributed_rank = get_rank()
        distributed = distributed_rank > -1
        if per_gpu_scaling and not distributed:
            num_gpus = max(1, torch.cuda.device_count())
            assert "batch_size" in params, "loader config must contain 'batch_size' key"
            assert "num_workers" in params, "loader config must contain 'num_workers' key"
            params["batch_size"] *= num_gpus
            params["num_workers"] *= num_gpus
        if distributed:
            if sampler is not None:
                if not isinstance(sampler, DistributedSampler):
                    sampler = DistributedSamplerWrapper(sampler=sampler)
            else:
                sampler = DistributedSampler(dataset=params["dataset"])
        params["shuffle"] = params.get("shuffle", False) and sampler is None
        params["sampler"] = sampler
        worker_init_fn = params.pop("worker_init_fn", None)
        if worker_init_fn is None:
            params["worker_init_fn"] = lambda x: set_global_seed(initial_seed + x)
        else:
            params["worker_init_fn"] = hydra.utils.get_method(worker_init_fn)
        collate_fn = params.pop("collate_fn", None)
        if collate_fn is None:
            params["collate_fn"] = None
        else:
            params["collate_fn"] = hydra.utils.get_method(collate_fn)
        loader: DataLoader = DataLoader(**params)
        return loader

    def get_loaders(self, stage: str, epoch: int = None) -> Dict[str, DataLoader]:
        """
        Returns loaders for a given stage.

        Args:
            stage: (str) stage name
            epoch: (int) epoch number

        Returns:
            Dict[str, DataLoader]: loaders objects

        """
        datasets = self.get_datasets(stage)
        samplers = self.get_samplers(stage)
        assert "loaders" in self._config.stages[stage], "stages config must contain 'loaders' key"
        loaders_params = self._config.stages[stage].loaders
        loaders: Dict[str, DataLoader] = {
            name: self._get_loader(
                datasets.get(name, None),
                samplers.get(name, None),
                self._initial_seed,
                loader_params,
            )
            for name, loader_params in loaders_params.items()
        }
        return loaders

    @staticmethod
    def _get_callback(params):
        target = params.pop("_target_")
        callback_class = hydra.utils.get_class(target)
        params = OmegaConf.to_container(params, resolve=True)
        callback = callback_class(**params)
        return callback

    def get_callbacks(self, stage: str) -> Dict[str, Callback]:
        """
        Returns callbacks for a given stage.

        Args:
            stage (str): stage name

        Returns:
            Dict[str, Callback]: callbacks objects

        """
        assert (
            "callbacks" in self._config.stages[stage]
        ), "stages config must contain 'callbacks' key"
        callbacks_params = self._config.stages[stage].callbacks
        callbacks: Dict[str, Callback] = {
            name: self._get_callback(callback_params)
            for name, callback_params in callbacks_params.items()
        }

        callbacks = add_default_callbacks(
            callbacks,
            verbose=self._verbose,
            check_time=self._check_time,
            check_run=self._check_run,
            overfit=self._overfit,
            is_infer=stage.startswith("infer"),
            is_logger=self.logdir is not None,
            is_criterion=self._config.stages[stage].get("criterions", {}),
            is_optimizer=self._config.stages[stage].get("optimizers", {}),
            is_scheduler=self._config.stages[stage].get("schedulers", {}),
        )

        # NOTE: stage should be in self._config.stages
        #       othervise will be raised ValueError
        stage_index = list(self._config.stages.keys()).index(stage)
        process_callbacks(callbacks, stage_index)

        return callbacks


__all__ = ["HydraConfigExperiment"]
