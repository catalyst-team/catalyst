from typing import Any, Callable, Dict, List, Mapping, Union
from collections import OrderedDict
from copy import deepcopy

import torch
from torch import nn
from torch.utils.data import DataLoader

from catalyst.contrib.data.augmentor import Augmentor, AugmentorCompose
from catalyst.core.callback import Callback
from catalyst.core.experiment import IExperiment
from catalyst.core.functional import check_callback_isinstance
from catalyst.core.logger import ILogger
from catalyst.core.trial import get_trial_by_params, ITrial
from catalyst.engines import IEngine, process_engine
from catalyst.experiments.functional import (
    add_default_callbacks,
    do_lr_linear_scaling,
    get_model_parameters,
    load_optimizer_from_checkpoint,
    process_callbacks,
)
from catalyst.registry import REGISTRY
from catalyst.typing import Criterion, Model, Optimizer, Scheduler
from catalyst.utils.loaders import get_loaders_from_params
from catalyst.utils.misc import get_by_keys, get_short_hash, get_utcnow_time, merge_dicts


class ConfigExperiment(IExperiment):
    """
    Experiment created from a configuration file.
    """

    STAGE_KEYWORDS = [  # noqa: WPS115
        "criterion_params",
        "optimizer_params",
        "scheduler_params",
        "data_params",
        "transform_params",
        "stage_params",
        "callbacks_params",
    ]

    def __init__(self, config: Dict):
        """
        Args:
            config: dictionary with parameters
        """
        self._config: Dict = deepcopy(config)
        self._config["stages"]["stage_params"] = merge_dicts(
            deepcopy(self._config["stages"].get("stage_params", {})),
            deepcopy(self._config.get("args", {})),
        )
        self.stages_config: Dict = self._get_stages_config(self._config["stages"])

    def _get_stages_config(self, stages_config: Dict):
        stages_defaults = {
            key: deepcopy(stages_config.get(key, {})) for key in self.STAGE_KEYWORDS
        }

        stages_config_out = OrderedDict()
        for stage in stages_config:
            if stage in self.STAGE_KEYWORDS or stages_config.get(stage) is None:
                continue

            stages_config_out[stage] = {}
            for key2 in self.STAGE_KEYWORDS:
                stages_config_out[stage][key2] = merge_dicts(
                    deepcopy(stages_defaults.get(key2, {})),
                    deepcopy(stages_config[stage].get(key2, {})),
                )

        return stages_config_out

    @property
    def seed(self) -> int:
        """Experiment's initial seed value."""
        seed = get_by_keys(self._config, "args", "seed", default=42)
        return seed

    @property
    def name(self) -> str:
        timestamp = get_utcnow_time()
        config_hash = get_short_hash(self._config)
        default_name = f"{timestamp}-{config_hash}"

        name = get_by_keys(self._config, "args", "name", default=default_name)
        return name

    @property
    def hparams(self) -> OrderedDict:
        """Returns hyperparameters"""
        return OrderedDict(self._config)

    @property
    def stages(self) -> List[str]:
        """Experiment's stage names."""
        stages_keys = list(self.stages_config.keys())
        return stages_keys

    def get_stage_params(self, stage: str) -> Mapping[str, Any]:
        """Returns the state parameters for a given stage."""
        return get_by_keys(self.stages_config, stage, "stage_params", default={})

    @staticmethod
    def _get_model(**params):
        key_value_flag = params.pop("_key_value", False)

        if key_value_flag:
            model = {
                model_key: ConfigExperiment._get_model(**model_params)  # noqa: WPS437
                for model_key, model_params in params.items()
            }
            model = nn.ModuleDict(model)
        else:
            model = REGISTRY.get_from_params(**params)
        return model

    def get_model(self, stage: str):
        """Returns the model for a given stage."""
        model_params = self._config["model_params"]
        model = self._get_model(**model_params)
        return model

    @staticmethod
    def _get_criterion(**params):
        key_value_flag = params.pop("_key_value", False)

        if key_value_flag:
            criterion = {
                key: ConfigExperiment._get_criterion(**key_params)  # noqa: WPS437
                for key, key_params in params.items()
            }
        else:
            criterion = REGISTRY.get_from_params(**params)
        return criterion

    def get_criterion(self, stage: str) -> Criterion:
        """Returns the criterion for a given stage."""
        criterion_params = get_by_keys(self.stages_config, stage, "criterion_params", default={})
        criterion = self._get_criterion(**criterion_params)
        return criterion

    def _get_optimizer(
        self, stage: str, model: Union[Model, Dict[str, Model]], **params
    ) -> Optimizer:
        # @TODO 1: refactoring; this method is too long
        # @TODO 2: load state dicts for schedulers & criterion

        # learning rate linear scaling
        lr_scaling_params = params.pop("lr_linear_scaling", None)
        if lr_scaling_params:
            data_params = dict(self.stages_config[stage]["data_params"])
            lr, lr_scaling = do_lr_linear_scaling(
                lr_scaling_params=lr_scaling_params,
                batch_size=data_params.get("batch_size", 1),
                per_gpu_scaling=data_params.get("per_gpu_scaling", False),
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

        # # getting load-from-previous-stage flags
        # load_from_previous_stage = params.pop("load_from_previous_stage", False)
        # optimizer_key = params.pop("optimizer_key", None)

        optimizer = REGISTRY.get_from_params(**params, params=model_params)
        # if load_from_previous_stage and self.stages.index(stage) != 0:
        #     checkpoint_path = f"{self.logdir}/checkpoints/best_full.pth"
        #     optimizer = load_optimizer_from_checkpoint(
        #         optimizer,
        #         checkpoint_path=checkpoint_path,
        #         checkpoint_optimizer_key=optimizer_key,
        #         model_parameters=model_params,
        #         optimizer_params=params,
        #     )

        return optimizer

    def get_optimizer(
        self, stage: str, model: Union[Model, Dict[str, Model]]
    ) -> Union[Optimizer, Dict[str, Optimizer]]:
        """
        Returns the optimizer for a given stage.

        Args:
            stage: stage name
            model (Union[Model, Dict[str, Model]]): model or a dict of models

        Returns:
            optimizer for selected stage
        """
        optimizer_params = get_by_keys(self.stages_config, stage, "optimizer_params", default={})

        optimizer_params = deepcopy(optimizer_params)
        key_value_flag = optimizer_params.pop("_key_value", False)
        if key_value_flag:
            optimizer = {}
            for key, params in optimizer_params.items():
                # load specified optimizer from checkpoint
                optimizer_key = "optimizer_key"
                assert optimizer_key not in params, "keyword reserved"
                params[optimizer_key] = key

                optimizer[key] = self._get_optimizer(stage, model, **params)
        else:
            optimizer = self._get_optimizer(stage, model, **optimizer_params)

        return optimizer

    @staticmethod
    def _get_scheduler(
        *, optimizer: Union[Optimizer, Dict[str, Optimizer]], **params: Any
    ) -> Union[Scheduler, Dict[str, Scheduler]]:
        key_value_flag = params.pop("_key_value", False)

        optimizer_key = params.pop("_optimizer", None)
        optimizer = optimizer[optimizer_key] if optimizer_key else optimizer

        if key_value_flag:
            scheduler: Dict[str, Scheduler] = {}
            for key, scheduler_params in params.items():
                scheduler[key] = ConfigExperiment._get_scheduler(  # noqa: WPS437
                    **scheduler_params, optimizer=optimizer
                )
        else:
            scheduler = REGISTRY.get_from_params(**params, optimizer=optimizer)
        return scheduler

    def get_scheduler(
        self, stage: str, optimizer: Union[Optimizer, Dict[str, Optimizer]]
    ) -> Union[Scheduler, Dict[str, Scheduler]]:
        """Returns the scheduler for a given stage."""
        scheduler_params = get_by_keys(self.stages_config, stage, "scheduler_params", default={})
        scheduler = self._get_scheduler(optimizer=optimizer, **scheduler_params)
        return scheduler

    @staticmethod
    def _get_transform(**params) -> Callable:
        key_value_flag = params.pop("_key_value", False)

        if key_value_flag:
            transforms_composition = {
                transform_key: ConfigExperiment._get_transform(**transform_params)  # noqa: WPS437
                for transform_key, transform_params in params.items()
            }

            augmentors = {
                key: Augmentor(dict_key=key, augment_fn=transform, input_key=key, output_key=key)
                for key, transform in transforms_composition.items()
            }
            transform = AugmentorCompose(augmentors)
        else:
            if "transforms" in params:
                transforms_composition = [
                    ConfigExperiment._get_transform(**transform_params)  # noqa: WPS437
                    for transform_params in params["transforms"]
                ]
                params.update(transforms=transforms_composition)
            transform = REGISTRY.get_from_params(**params)

        return transform

    def get_transforms(self, stage: str = None, dataset: str = None) -> Callable:
        """
        Returns transform for a given stage and dataset.

        Args:
            stage: stage name
            dataset: dataset name (e.g. "train", "valid"),
                will be used only if the value of `_key_value`` is ``True``

        Returns:
            Callable: transform function
        """
        transform_params = get_by_keys(self.stages_config, stage, "transform_params", default={})

        transform_params = deepcopy(transform_params)
        key_value_flag = transform_params.pop("_key_value", False)
        if key_value_flag:
            transform_params = transform_params.get(dataset, {})

        transform_fn = self._get_transform(**transform_params)
        if transform_fn is None:

            def transform_fn(dict_):  # noqa: WPS440
                return dict_

        elif not isinstance(transform_fn, AugmentorCompose):
            transform_fn_origin = transform_fn

            def transform_fn(dict_):  # noqa: WPS440
                return transform_fn_origin(**dict_)

        return transform_fn

    def get_loaders(self, stage: str, epoch: int = None,) -> "OrderedDict[str, DataLoader]":
        """Returns the loaders for a given stage."""
        data_params = dict(self.stages_config[stage]["data_params"])
        loaders = get_loaders_from_params(
            datasets_fn=self.get_datasets, initial_seed=self.seed, stage=stage, **data_params,
        )
        return loaders

    @staticmethod
    def _get_callback(**params):
        wrapper_params = params.pop("_wrapper", None)
        callback = REGISTRY.get_from_params(**params)
        if wrapper_params is not None:
            wrapper_params["base_callback"] = callback
            callback = ConfigExperiment._get_callback(**wrapper_params)  # noqa: WPS437
        return callback

    def get_callbacks(self, stage: str) -> "OrderedDict[str, Callback]":
        """Returns the callbacks for a given stage."""
        callbacks_params = get_by_keys(self.stages_config, stage, "callbacks_params", default={})

        callbacks_list = [
            (key, self._get_callback(**callback_params))
            for key, callback_params in callbacks_params.items()
        ]

        callbacks = add_default_callbacks(
            OrderedDict(callbacks_list),
            verbose=get_by_keys(self._config, "args", "verbose", default=False),
            timeit=get_by_keys(self._config, "args", "timeit", default=False),
            check=get_by_keys(self._config, "args", "check", default=False),
            overfit=get_by_keys(self._config, "args", "overfit", default=False),
            is_infer=stage.startswith("infer"),
            is_logger=(not self.get_loggers()),
            is_criterion=get_by_keys(self.stages_config, stage, "criterion_params", default=False),
            is_optimizer=get_by_keys(self.stages_config, stage, "optimizer_params", default=False),
            is_scheduler=get_by_keys(self.stages_config, stage, "scheduler_params", default=False),
        )

        # NOTE: stage should be in self._config.stages
        #       otherwise will be raised ValueError
        stage_index = list(self.stages_config.keys()).index(stage)
        process_callbacks(callbacks, stage_index)

        return callbacks

    def get_engine(self) -> IEngine:
        engine_params = self._config.get("engine")
        return process_engine(engine_params)

    def get_trial(self) -> ITrial:
        trial_params = self._config.get("trial", {})
        return get_trial_by_params(trial_params)

    def get_loggers(self) -> Dict[str, ILogger]:
        loggers_params = self._config.get("loggers", {})
        loggers = {
            key: REGISTRY.get_from_params(**params) for key, params in loggers_params.items()
        }
        return loggers


__all__ = ["ConfigExperiment"]
