from typing import Any, Callable, Dict, List, Mapping, Union
from collections import OrderedDict
from copy import deepcopy

import torch
from torch import nn
from torch.utils.data import DataLoader  # noqa F401

from catalyst.core import StageBasedExperiment
from catalyst.data import Augmentor, AugmentorCompose
from catalyst.dl import (
    Callback,
    CheckpointCallback,
    CheckRunCallback,
    ConsoleLogger,
    CriterionCallback,
    ExceptionCallback,
    MetricManagerCallback,
    OptimizerCallback,
    PhaseWrapperCallback,
    SchedulerCallback,
    TensorboardLogger,
    TimerCallback,
    utils,
    ValidationManagerCallback,
    VerboseLogger,
)
from catalyst.dl.registry import (
    CALLBACKS,
    CRITERIONS,
    MODELS,
    OPTIMIZERS,
    SCHEDULERS,
    TRANSFORMS,
)
from catalyst.utils.tools.typing import Criterion, Model, Optimizer, Scheduler


class ConfigExperiment(StageBasedExperiment):
    """
    Experiment created from a configuration file.
    """

    STAGE_KEYWORDS = [
        "criterion_params",
        "optimizer_params",
        "scheduler_params",
        "data_params",
        "transform_params",
        "state_params",
        "callbacks_params",
    ]

    def __init__(self, config: Dict):
        """
        Args:
            config (dict): dictionary of parameters
        """
        self._config: Dict = deepcopy(config)
        self._initial_seed: int = self._config.get("args", {}).get("seed", 42)
        self._verbose: bool = self._config.get("args", {}).get(
            "verbose", False
        )
        self._check_run: bool = self._config.get("args", {}).get(
            "check", False
        )
        self._check_time: bool = self._config.get("args", {}).get(
            "timeit", False
        )
        self.__prepare_logdir()

        self._config["stages"]["state_params"] = utils.merge_dicts(
            deepcopy(self._config["stages"].get("state_params", {})),
            deepcopy(self._config.get("args", {})),
            {"logdir": self._logdir},
        )
        self.stages_config: Dict = self._get_stages_config(
            self._config["stages"]
        )

    def __prepare_logdir(self):
        EXCLUDE_TAG = "none"

        logdir = self._config.get("args", {}).get("logdir", None)
        baselogdir = self._config.get("args", {}).get("baselogdir", None)

        if logdir is not None and logdir.lower() != EXCLUDE_TAG:
            self._logdir = logdir
        elif baselogdir is not None and baselogdir.lower() != EXCLUDE_TAG:
            logdir_postfix = self._get_logdir(self._config)
            self._logdir = f"{baselogdir}/{logdir_postfix}"
        else:
            self._logdir = None

    def _get_stages_config(self, stages_config: Dict):
        stages_defaults = {}
        stages_config_out = OrderedDict()
        for key in self.STAGE_KEYWORDS:
            stages_defaults[key] = deepcopy(stages_config.get(key, {}))
        for stage in stages_config:
            if (
                stage in self.STAGE_KEYWORDS
                or stages_config.get(stage) is None
            ):
                continue
            stages_config_out[stage] = {}
            for key in self.STAGE_KEYWORDS:
                stages_config_out[stage][key] = utils.merge_dicts(
                    deepcopy(stages_defaults.get(key, {})),
                    deepcopy(stages_config[stage].get(key, {})),
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

        # @TODO: return the feature
        # # Change start `stages_keys` if resume data were founded
        # state_params = self.get_state_params(stages_keys[0])
        # resume, resume_dir = [
        #     state_params.get(key, None) for key in ["resume", "resume_dir"]
        # ]
        #
        # if resume_dir is not None:
        #     resume = resume_dir / str(resume)
        #
        # if resume is not None and Path(resume).is_file():
        #     checkpoint = utils.load_checkpoint(resume)
        #     start_stage = checkpoint["stage"]
        #     start_idx = stages_keys.index(start_stage)
        #     stages_keys = stages_keys[start_idx:]

        return stages_keys

    @property
    def distributed_params(self) -> Dict:
        """Dict with the parameters for distributed and FP16 methond."""
        return self._config.get("distributed_params", {})

    @property
    def monitoring_params(self) -> Dict:
        """Dict with the parameters for monitoring services."""
        return self._config.get("monitoring_params", {})

    def get_state_params(self, stage: str) -> Mapping[str, Any]:
        """Returns the state parameters for a given stage."""
        return self.stages_config[stage].get("state_params", {})

    def _preprocess_model_for_stage(self, stage: str, model: Model):
        stage_index = self.stages.index(stage)
        if stage_index > 0:
            checkpoint_path = f"{self.logdir}/checkpoints/best.pth"
            checkpoint = utils.load_checkpoint(checkpoint_path)
            utils.unpack_checkpoint(checkpoint, model=model)
        return model

    def _postprocess_model_for_stage(self, stage: str, model: Model):
        return model

    @staticmethod
    def _get_model(**params):
        key_value_flag = params.pop("_key_value", False)

        if key_value_flag:
            model = {}
            for key, params_ in params.items():
                model[key] = ConfigExperiment._get_model(**params_)
            model = nn.ModuleDict(model)
        else:
            model = MODELS.get_from_params(**params)
        return model

    def get_model(self, stage: str):
        """Returns the model for a given stage."""
        model_params = self._config["model_params"]
        model = self._get_model(**model_params)

        model = self._preprocess_model_for_stage(stage, model)
        model = self._postprocess_model_for_stage(stage, model)
        return model

    @staticmethod
    def _get_criterion(**params):
        key_value_flag = params.pop("_key_value", False)

        if key_value_flag:
            criterion = {}
            for key, params_ in params.items():
                criterion[key] = ConfigExperiment._get_criterion(**params_)
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
            for model_key_ in model_key:
                model_params_ = utils.process_model_params(
                    model[model_key_],
                    layerwise_params,
                    no_bias_weight_decay,
                    lr_scaling,
                )
                model_params.extend(model_params_)
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
                state = optimizer.state[param]
                for key, value in state.items():
                    state[key] = utils.any2device(value, device)

            # update optimizer params
            for key, value in params.items():
                for pg in optimizer.param_groups:
                    pg[key] = value

        return optimizer

    def get_optimizer(
        self, stage: str, model: Union[Model, Dict[str, Model]]
    ) -> Union[Optimizer, Dict[str, Optimizer]]:
        """Returns the optimizer for a given stage.

        Args:
            stage (str): stage name
            model (Union[Model, Dict[str, Model]]): model or a dict of models
        """
        optimizer_params = self.stages_config[stage].get(
            "optimizer_params", {}
        )
        key_value_flag = optimizer_params.pop("_key_value", False)

        if key_value_flag:
            optimizer = {}
            for key, params_ in optimizer_params.items():
                # load specified optimizer from checkpoint
                optimizer_key = "optimizer_key"
                assert optimizer_key not in params_, "keyword reserved"
                params_[optimizer_key] = key

                optimizer[key] = self._get_optimizer(stage, model, **params_)
        else:
            optimizer = self._get_optimizer(stage, model, **optimizer_params)

        return optimizer

    @staticmethod
    def _get_scheduler(*, optimizer, **params):
        key_value_flag = params.pop("_key_value", False)

        if key_value_flag:
            scheduler = {}
            for key, params_ in params.items():
                scheduler[key] = ConfigExperiment._get_scheduler(
                    optimizer=optimizer, **params_
                )
        else:
            scheduler = SCHEDULERS.get_from_params(
                **params, optimizer=optimizer
            )
        return scheduler

    def get_scheduler(self, stage: str, optimizer: Optimizer) -> Scheduler:
        """Returns the scheduler for a given stage."""
        scheduler_params = self.stages_config[stage].get(
            "scheduler_params", {}
        )
        scheduler = self._get_scheduler(
            optimizer=optimizer, **scheduler_params
        )
        return scheduler

    @staticmethod
    def _get_transform(**params) -> Callable:
        key_value_flag = params.pop("_key_value", False)

        if key_value_flag:
            transforms_composition = {
                key: ConfigExperiment._get_transform(**params_)
                for key, params_ in params.items()
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
                    ConfigExperiment._get_transform(**transform_params)
                    for transform_params in params["transforms"]
                ]
                params.update(transforms=transforms_composition)

            transform = TRANSFORMS.get_from_params(**params)

        return transform

    def get_transforms(
        self, stage: str = None, dataset: str = None
    ) -> Callable:
        """Returns transform for a given stage and mode.

        Args:
            stage (str): stage name
            dataset (str): dataset name (e.g. "train", "valid"),
                will be used only if the value of `_key_value`` is ``True``
        """
        transform_params = deepcopy(
            self.stages_config[stage].get("transform_params", {})
        )

        key_value_flag = transform_params.pop("_key_value", False)
        if key_value_flag:
            transform_params = transform_params.get(dataset, {})

        transform = self._get_transform(**transform_params)
        if transform is None:

            def transform(dict_):
                return dict_

        elif not isinstance(transform, AugmentorCompose):
            transform_ = transform

            def transform(dict_):
                return transform_(**dict_)

        return transform

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
            return ConfigExperiment._get_callback(**wrapper_params)
        return callback

    def get_callbacks(self, stage: str) -> "OrderedDict[Callback]":
        """Returns the callbacks for a given stage."""
        callbacks_params = self.stages_config[stage].get(
            "callbacks_params", {}
        )

        callbacks = OrderedDict()
        for key, callback_params in callbacks_params.items():
            callback = self._get_callback(**callback_params)
            callbacks[key] = callback

        default_callbacks = []
        if self._verbose:
            default_callbacks.append(("_verbose", VerboseLogger))
        if self._check_time:
            default_callbacks.append(("_timer", TimerCallback))
        if self._check_run:
            default_callbacks.append(("_check", CheckRunCallback))

        if not stage.startswith("infer"):
            default_callbacks.append(("_metrics", MetricManagerCallback))
            default_callbacks.append(
                ("_validation", ValidationManagerCallback)
            )
            default_callbacks.append(("_console", ConsoleLogger))

            if self.logdir is not None:
                default_callbacks.append(("_saver", CheckpointCallback))
                default_callbacks.append(("_tensorboard", TensorboardLogger))

            if self.stages_config[stage].get("criterion_params", {}):
                default_callbacks.append(("_criterion", CriterionCallback))
            if self.stages_config[stage].get("optimier_params", {}):
                default_callbacks.append(("_optimizer", OptimizerCallback))
            if self.stages_config[stage].get("scheduler_params", {}):
                default_callbacks.append(("_scheduler", SchedulerCallback))

        default_callbacks.append(("_exception", ExceptionCallback))

        for callback_name, callback_fn in default_callbacks:
            is_already_present = False
            for x in callbacks.values():
                if isinstance(x, PhaseWrapperCallback):
                    x = x.callback
                if isinstance(x, callback_fn):
                    is_already_present = True
                    break
            if not is_already_present:
                callbacks[callback_name] = callback_fn()

        return callbacks


__all__ = ["ConfigExperiment"]
