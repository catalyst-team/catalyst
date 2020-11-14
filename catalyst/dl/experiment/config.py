# flake8: noqa
# @TODO: code formatting issue for 20.07 release
from typing import Any, Callable, Dict, List, Mapping, Union
from collections import OrderedDict
from copy import deepcopy

import torch
from torch import nn
from torch.utils.data import DataLoader

from catalyst.core import IExperiment
from catalyst.data import Augmentor, AugmentorCompose
from catalyst.dl import (
    AMPOptimizerCallback,
    BatchOverfitCallback,
    Callback,
    CheckpointCallback,
    CheckRunCallback,
    ConsoleLogger,
    CriterionCallback,
    ExceptionCallback,
    IOptimizerCallback,
    ISchedulerCallback,
    MetricManagerCallback,
    OptimizerCallback,
    SchedulerCallback,
    TensorboardLogger,
    TimerCallback,
    utils,
    ValidationManagerCallback,
    VerboseLogger,
)
from catalyst.dl.utils import check_amp_available, check_callback_isinstance
from catalyst.registry import (
    CALLBACKS,
    CRITERIONS,
    MODELS,
    OPTIMIZERS,
    SCHEDULERS,
    TRANSFORMS,
)
from catalyst.tools.typing import Criterion, Model, Optimizer, Scheduler


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
            config (dict): dictionary with parameters
        """
        self._config: Dict = deepcopy(config)
        self._initial_seed: int = self._config.get("args", {}).get("seed", 42)
        self._verbose: bool = self._config.get("args", {}).get(
            "verbose", False
        )
        self._check_time: bool = self._config.get("args", {}).get(
            "timeit", False
        )
        self._check_run: bool = self._config.get("args", {}).get(
            "check", False
        )
        self._overfit: bool = self._config.get("args", {}).get(
            "overfit", False
        )

        self.__prepare_logdir()

        self._config["stages"]["stage_params"] = utils.merge_dicts(
            deepcopy(
                self._config["stages"].get("state_params", {})
            ),  # saved for backward compatibility
            deepcopy(self._config["stages"].get("stage_params", {})),
            deepcopy(self._config.get("args", {})),
            {"logdir": self._logdir},
        )
        self.stages_config: Dict = self._get_stages_config(
            self._config["stages"]
        )

    def __prepare_logdir(self):  # noqa: WPS112
        exclude_tag = "none"

        logdir = self._config.get("args", {}).get("logdir", None)
        baselogdir = self._config.get("args", {}).get("baselogdir", None)

        if logdir is not None and logdir.lower() != exclude_tag:
            self._logdir = logdir
        elif baselogdir is not None and baselogdir.lower() != exclude_tag:
            logdir_postfix = self._get_logdir(self._config)
            self._logdir = f"{baselogdir}/{logdir_postfix}"
        else:
            self._logdir = None

    @property
    def hparams(self) -> OrderedDict:
        """Returns hyperparameters"""
        return OrderedDict(self._config)

    def _get_stages_config(self, stages_config: Dict):
        stages_defaults = {}
        stages_config_out = OrderedDict()
        for key in self.STAGE_KEYWORDS:
            if key == "stage_params":
                # backward compatibility
                stages_defaults[key] = utils.merge_dicts(
                    deepcopy(stages_config.get("state_params", {})),
                    deepcopy(stages_config.get(key, {})),
                )
            else:
                stages_defaults[key] = deepcopy(stages_config.get(key, {}))
        for stage in stages_config:
            if (
                stage in self.STAGE_KEYWORDS
                or stage == "state_params"
                or stages_config.get(stage) is None
            ):
                continue
            stages_config_out[stage] = {}
            for key2 in self.STAGE_KEYWORDS:
                if key2 == "stage_params":
                    # backward compatibility
                    stages_config_out[stage][key2] = utils.merge_dicts(
                        deepcopy(stages_defaults.get("state_params", {})),
                        deepcopy(stages_defaults.get(key2, {})),
                        deepcopy(stages_config[stage].get("state_params", {})),
                        deepcopy(stages_config[stage].get(key2, {})),
                    )
                else:
                    stages_config_out[stage][key2] = utils.merge_dicts(
                        deepcopy(stages_defaults.get(key2, {})),
                        deepcopy(stages_config[stage].get(key2, {})),
                    )

        return stages_config_out

    def _get_logdir(self, config: Dict) -> str:
        timestamp = utils.get_utcnow_time()
        config_hash = utils.get_short_hash(config)
        logdir = f"{timestamp}.{config_hash}"
        return logdir

    @property
    def initial_seed(self) -> int:
        """Experiment's initial seed value."""
        return self._initial_seed

    @property
    def logdir(self):
        """Path to the directory where the experiment logs."""
        return self._logdir

    @property
    def stages(self) -> List[str]:
        """Experiment's stage names."""
        stages_keys = list(self.stages_config.keys())
        return stages_keys

    @property
    def distributed_params(self) -> Dict:
        """Dict with the parameters for distributed and FP16 methond."""
        return self._config.get("distributed_params", {})

    def get_stage_params(self, stage: str) -> Mapping[str, Any]:
        """Returns the state parameters for a given stage."""
        return self.stages_config[stage].get("stage_params", {})

    @staticmethod
    def _get_model(**params):
        key_value_flag = params.pop("_key_value", False)

        if key_value_flag:
            model = {}
            for model_key, model_params in params.items():
                model[model_key] = ConfigExperiment._get_model(  # noqa: WPS437
                    **model_params
                )
            model = nn.ModuleDict(model)
        else:
            model = MODELS.get_from_params(**params)
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
            criterion = {}
            for key, key_params in params.items():
                criterion[
                    key
                ] = ConfigExperiment._get_criterion(  # noqa: WPS437
                    **key_params
                )
        else:
            criterion = CRITERIONS.get_from_params(**params)
            if criterion is not None and torch.cuda.is_available():
                criterion = criterion.cuda()
        return criterion

    def get_criterion(self, stage: str) -> Criterion:
        """Returns the criterion for a given stage."""
        criterion_params = self.stages_config[stage].get(
            "criterion_params", {}
        )
        criterion = self._get_criterion(**criterion_params)
        return criterion

    def _get_optimizer(
        self, stage: str, model: Union[Model, Dict[str, Model]], **params
    ) -> Optimizer:
        # @TODO 1: refactoring; this method is too long
        # @TODO 2: load state dicts for schedulers & criterion
        layerwise_params = params.pop("layerwise_params", OrderedDict())
        no_bias_weight_decay = params.pop("no_bias_weight_decay", True)

        # linear scaling rule from https://arxiv.org/pdf/1706.02677.pdf
        lr_scaling_params = params.pop("lr_linear_scaling", None)
        if lr_scaling_params:
            data_params = dict(self.stages_config[stage]["data_params"])
            batch_size = data_params.get("batch_size")
            per_gpu_scaling = data_params.get("per_gpu_scaling", False)
            distributed_rank = utils.get_rank()
            distributed = distributed_rank > -1
            if per_gpu_scaling and not distributed:
                num_gpus = max(1, torch.cuda.device_count())
                batch_size *= num_gpus

            base_lr = lr_scaling_params.get("lr")
            base_batch_size = lr_scaling_params.get("base_batch_size", 256)
            lr_scaling = batch_size / base_batch_size
            params["lr"] = base_lr * lr_scaling  # scale default lr
        else:
            lr_scaling = 1.0

        # getting model parameters
        model_key = params.pop("_model", None)
        if model_key is None:
            assert isinstance(
                model, nn.Module
            ), "model is key-value, but optimizer has no specified model"
            model_params = utils.process_model_params(
                model, layerwise_params, no_bias_weight_decay, lr_scaling
            )
        elif isinstance(model_key, str):
            model_params = utils.process_model_params(
                model[model_key],
                layerwise_params,
                no_bias_weight_decay,
                lr_scaling,
            )
        elif isinstance(model_key, (list, tuple)):
            model_params = []
            for model_key_el in model_key:
                model_params_el = utils.process_model_params(
                    model[model_key_el],
                    layerwise_params,
                    no_bias_weight_decay,
                    lr_scaling,
                )
                model_params.extend(model_params_el)
        else:
            raise ValueError("unknown type of model_params")

        load_from_previous_stage = params.pop(
            "load_from_previous_stage", False
        )
        optimizer_key = params.pop("optimizer_key", None)
        optimizer = OPTIMIZERS.get_from_params(**params, params=model_params)

        if load_from_previous_stage and self.stages.index(stage) != 0:
            checkpoint_path = f"{self.logdir}/checkpoints/best_full.pth"
            checkpoint = utils.load_checkpoint(checkpoint_path)

            dict2load = optimizer
            if optimizer_key is not None:
                dict2load = {optimizer_key: optimizer}
            utils.unpack_checkpoint(checkpoint, optimizer=dict2load)

            # move optimizer to device
            device = utils.get_device()
            for param in model_params:
                param = param["params"][0]
                optimizer_state = optimizer.state[param]
                for state_key, state_value in optimizer_state.items():
                    optimizer_state[state_key] = utils.any2device(
                        state_value, device
                    )

            # update optimizer params
            for key, value in params.items():
                for optimizer_param_group in optimizer.param_groups:
                    optimizer_param_group[key] = value

        return optimizer

    def get_optimizer(
        self, stage: str, model: Union[Model, Dict[str, Model]]
    ) -> Union[Optimizer, Dict[str, Optimizer]]:
        """
        Returns the optimizer for a given stage.

        Args:
            stage (str): stage name
            model (Union[Model, Dict[str, Model]]): model or a dict of models

        Returns:
            optimizer for selected stage
        """
        optimizer_params = self.stages_config[stage].get(
            "optimizer_params", {}
        )
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
        optimizer_key = params.pop("_optimizer", None)
        optimizer_ = optimizer[optimizer_key] if optimizer_key else optimizer
        scheduler = SCHEDULERS.get_from_params(**params, optimizer=optimizer_)

        return scheduler

    def get_scheduler(
        self, stage: str, optimizer: Union[Optimizer, Dict[str, Optimizer]]
    ) -> Union[Scheduler, Dict[str, Scheduler]]:
        """Returns the scheduler for a given stage."""
        params = self.stages_config[stage].get("scheduler_params", {})
        key_value_flag = params.pop("_key_value", False)

        if key_value_flag:
            scheduler: Dict[str, Scheduler] = {}
            for key, scheduler_params in params.items():
                scheduler[key] = self._get_scheduler(
                    optimizer=optimizer, **scheduler_params
                )
        else:
            scheduler = self._get_scheduler(optimizer=optimizer, **params)

        return scheduler

    @staticmethod
    def _get_transform(**params) -> Callable:
        key_value_flag = params.pop("_key_value", False)

        if key_value_flag:
            transforms_composition = {
                transform_key: ConfigExperiment._get_transform(  # noqa: WPS437
                    **transform_params
                )
                for transform_key, transform_params in params.items()
            }

            transform = AugmentorCompose(
                {
                    key: Augmentor(
                        dict_key=key,
                        augment_fn=transform,
                        input_key=key,
                        output_key=key,
                    )
                    for key, transform in transforms_composition.items()
                }
            )
        else:
            if "transforms" in params:
                transforms_composition = [
                    ConfigExperiment._get_transform(  # noqa: WPS437
                        **transform_params
                    )
                    for transform_params in params["transforms"]
                ]
                params.update(transforms=transforms_composition)

            transform = TRANSFORMS.get_from_params(**params)

        return transform

    def get_transforms(
        self, stage: str = None, dataset: str = None
    ) -> Callable:
        """
        Returns transform for a given stage and dataset.

        Args:
            stage (str): stage name
            dataset (str): dataset name (e.g. "train", "valid"),
                will be used only if the value of `_key_value`` is ``True``

        Returns:
            Callable: transform function
        """
        transform_params = deepcopy(
            self.stages_config[stage].get("transform_params", {})
        )

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

    def get_loaders(
        self, stage: str, epoch: int = None,
    ) -> "OrderedDict[str, DataLoader]":
        """Returns the loaders for a given stage."""
        data_params = dict(self.stages_config[stage]["data_params"])
        loaders = utils.get_loaders_from_params(
            get_datasets_fn=self.get_datasets,
            initial_seed=self.initial_seed,
            stage=stage,
            **data_params,
        )
        return loaders

    @staticmethod
    def _get_callback(**params):
        wrapper_params = params.pop("_wrapper", None)
        callback = CALLBACKS.get_from_params(**params)
        if wrapper_params is not None:
            wrapper_params["base_callback"] = callback
            callback = ConfigExperiment._get_callback(  # noqa: WPS437
                **wrapper_params
            )
        return callback

    @staticmethod
    def _process_callbacks(
        callbacks: OrderedDict, stage_index: int = None
    ) -> None:
        """
        Iterate over each of the callbacks and update
        appropriate parameters required for success
        run of config experiment.

        Arguments:
            callbacks (OrderedDict): finalized order of callbacks.
            stage_index (int): number of a current stage
        """
        if stage_index is None:
            stage_index = -float("inf")
        for callback in callbacks.values():
            # NOTE: in experiments with multiple stages need to omit
            #       loading of a best model state for the first stage
            #       but for the other stages by default should
            #       load best state of a model
            # @TODO: move this logic to ``CheckpointCallback``
            if isinstance(callback, CheckpointCallback) and stage_index > 0:
                if callback.load_on_stage_start is None:
                    callback.load_on_stage_start = "best"
                if (
                    isinstance(callback.load_on_stage_start, dict)
                    and "model" not in callback.load_on_stage_start
                ):
                    callback.load_on_stage_start["model"] = "best"

    def get_callbacks(self, stage: str) -> "OrderedDict[Callback]":
        """Returns the callbacks for a given stage."""
        callbacks_params = self.stages_config[stage].get(
            "callbacks_params", {}
        )

        callbacks = OrderedDict()
        for key, callback_params in callbacks_params.items():
            callback = self._get_callback(**callback_params)
            callbacks[key] = callback

        # default_callbacks = [(Name, InterfaceClass, InstanceFactory)]
        default_callbacks = []

        is_amp_enabled = (
            self.distributed_params.get("amp", False) and check_amp_available()
        )
        optimizer_cls = (
            AMPOptimizerCallback if is_amp_enabled else OptimizerCallback
        )

        if self._verbose:
            default_callbacks.append(("_verbose", None, VerboseLogger))
        if self._check_time:
            default_callbacks.append(("_timer", None, TimerCallback))
        if self._check_run:
            default_callbacks.append(("_check", None, CheckRunCallback))
        if self._overfit:
            default_callbacks.append(("_overfit", None, BatchOverfitCallback))

        if not stage.startswith("infer"):
            default_callbacks.append(("_metrics", None, MetricManagerCallback))
            default_callbacks.append(
                ("_validation", None, ValidationManagerCallback)
            )
            default_callbacks.append(("_console", None, ConsoleLogger))

            if self.logdir is not None:
                default_callbacks.append(("_saver", None, CheckpointCallback))
                default_callbacks.append(
                    ("_tensorboard", None, TensorboardLogger)
                )

            if self.stages_config[stage].get("criterion_params", {}):
                default_callbacks.append(
                    ("_criterion", None, CriterionCallback)
                )
            if self.stages_config[stage].get("optimizer_params", {}):
                default_callbacks.append(
                    ("_optimizer", IOptimizerCallback, optimizer_cls)
                )
            if self.stages_config[stage].get("scheduler_params", {}):
                default_callbacks.append(
                    ("_scheduler", ISchedulerCallback, SchedulerCallback)
                )

        default_callbacks.append(("_exception", None, ExceptionCallback))

        for (
            callback_name,
            callback_interface,
            callback_fn,
        ) in default_callbacks:
            callback_interface = callback_interface or callback_fn
            is_already_present = any(
                check_callback_isinstance(x, callback_interface)
                for x in callbacks.values()
            )
            if not is_already_present:
                callbacks[callback_name] = callback_fn()

        # NOTE: stage should be in self.stages_config
        #       othervise will be raised ValueError
        stage_index = list(self.stages_config.keys()).index(stage)
        self._process_callbacks(callbacks, stage_index)

        return callbacks


__all__ = ["ConfigExperiment"]
