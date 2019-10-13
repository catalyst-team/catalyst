from typing import Any, Mapping, Dict, List, Union
from copy import deepcopy
from collections import OrderedDict

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset  # noqa F401
from torch.utils.data import DistributedSampler

from catalyst.dl.registry import \
    MODELS, CRITERIONS, OPTIMIZERS, SCHEDULERS, CALLBACKS
from catalyst.dl.core import Experiment, Callback
from catalyst.dl import utils
from catalyst.dl.utils.torch import _Model, _Criterion, _Optimizer, \
    _Scheduler


class ConfigExperiment(Experiment):
    STAGE_KEYWORDS = [
        "criterion_params",
        "optimizer_params",
        "scheduler_params",
        "data_params",
        "state_params",
        "callbacks_params",
    ]

    def __init__(self, config: Dict):
        self._config = deepcopy(config)
        self._initial_seed = self._config.get("args", {}).get("seed", 42)
        self.__prepare_logdir()

        self._config["stages"]["state_params"] = utils.merge_dicts(
            deepcopy(self._config["stages"].get("state_params", {})),
            deepcopy(self._config.get("args", {})), {"logdir": self._logdir}
        )
        self.stages_config = self._get_stages_config(self._config["stages"])

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

    def _get_stages_config(self, stages_config):
        stages_defaults = {}
        stages_config_out = OrderedDict()
        for key in self.STAGE_KEYWORDS:
            stages_defaults[key] = deepcopy(stages_config.get(key, {}))
        for stage in stages_config:
            if stage in self.STAGE_KEYWORDS \
                    or stages_config.get(stage) is None:
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
        distributed_rank = self.distributed_params.get("rank", -1)
        if distributed_rank > -1:
            logdir = f"{logdir}.rank{distributed_rank:02d}"
        return logdir

    @property
    def initial_seed(self) -> int:
        return self._initial_seed

    @property
    def logdir(self):
        return self._logdir

    @property
    def stages(self) -> List[str]:
        stages_keys = list(self.stages_config.keys())
        return stages_keys

    @property
    def distributed_params(self) -> Dict:
        return self._config.get("distributed_params", {})

    @property
    def monitoring_params(self) -> Dict:
        return self._config.get("monitoring_params", {})

    def get_state_params(self, stage: str) -> Mapping[str, Any]:
        return self.stages_config[stage].get("state_params", {})

    def _preprocess_model_for_stage(self, stage: str, model: _Model):
        stage_index = self.stages.index(stage)
        if stage_index > 0:
            checkpoint_path = \
                f"{self.logdir}/checkpoints/best.pth"
            checkpoint = utils.load_checkpoint(checkpoint_path)
            utils.unpack_checkpoint(checkpoint, model=model)
        return model

    def _postprocess_model_for_stage(self, stage: str, model: _Model):
        return model

    @staticmethod
    def _get_model(**params):
        key_value_flag = params.pop("_key_value", False)

        if key_value_flag:
            model = {}
            for key, params_ in params.items():
                model[key] = ConfigExperiment._get_model(**params_)
        else:
            model = MODELS.get_from_params(**params)
        return model

    def get_model(self, stage: str):
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

    def get_criterion(self, stage: str) -> _Criterion:
        criterion_params = \
            self.stages_config[stage].get("criterion_params", {})
        criterion = self._get_criterion(**criterion_params)
        return criterion

    def _get_optimizer(
        self,
        stage: str,
        model: Union[_Model, Dict[str, _Model]],
        **params
    ) -> _Optimizer:
        # @TODO 1: refactoring; this method is too long
        # @TODO 2: load state dicts for schedulers & criteria
        layerwise_params = \
            params.pop("layerwise_params", OrderedDict())
        no_bias_weight_decay = \
            params.pop("no_bias_weight_decay", True)

        # linear scaling rule from https://arxiv.org/pdf/1706.02677.pdf
        lr_scaling_params = params.pop("lr_linear_scaling", None)
        if lr_scaling_params:
            data_params = dict(self.stages_config[stage]["data_params"])
            batch_size = data_params.get("batch_size")
            per_gpu_scaling = data_params.get("per_gpu_scaling", False)
            distributed_rank = self.distributed_params.get("rank", -1)
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
            assert isinstance(model, nn.Module), \
                "model is keyvalue, but optimizer has no specified model"
            model_params = utils.process_model_params(
                model, layerwise_params, no_bias_weight_decay, lr_scaling
            )
        elif isinstance(model_key, str):
            model_params = utils.process_model_params(
                model[model_key], layerwise_params, no_bias_weight_decay,
                lr_scaling
            )
        elif isinstance(model_key, (list, tuple)):
            model_params = []
            for model_key_ in model_key:
                model_params_ = utils.process_model_params(
                    model[model_key_], layerwise_params, no_bias_weight_decay,
                    lr_scaling
                )
                model_params.extend(model_params_)
        else:
            raise ValueError("unknown type of model_params")

        load_from_previous_stage = \
            params.pop("load_from_previous_stage", False)
        optimizer_key = params.pop("optimizer_key", None)
        optimizer = OPTIMIZERS.get_from_params(**params, params=model_params)

        if load_from_previous_stage:
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
        self,
        stage: str,
        model: Union[_Model, Dict[str, _Model]]
    ) -> Union[_Optimizer, Dict[str, _Optimizer]]:
        optimizer_params = \
            self.stages_config[stage].get("optimizer_params", {})
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

    def get_scheduler(self, stage: str, optimizer) -> _Scheduler:
        scheduler_params = \
            self.stages_config[stage].get("scheduler_params", {})
        scheduler = self._get_scheduler(
            optimizer=optimizer, **scheduler_params
        )
        return scheduler

    def get_loaders(self, stage: str) -> "OrderedDict[str, DataLoader]":
        data_params = dict(self.stages_config[stage]["data_params"])

        batch_size = data_params.pop("batch_size", 1)
        num_workers = data_params.pop("num_workers")
        drop_last = data_params.pop("drop_last", False)
        per_gpu_scaling = data_params.pop("per_gpu_scaling", False)
        distributed_rank = self.distributed_params.get("rank", -1)
        distributed = distributed_rank > -1

        datasets = self.get_datasets(stage=stage, **data_params)

        overridden_loaders_params = data_params.pop("loaders_params", {})
        assert isinstance(overridden_loaders_params, dict), \
            f"{overridden_loaders_params} should be Dict"

        loaders = OrderedDict()
        for name, ds_ in datasets.items():
            assert isinstance(ds_, (Dataset, dict)), \
                f"{ds_} should be Dataset or Dict"

            overridden_loader_params = overridden_loaders_params.pop(name, {})
            assert isinstance(overridden_loader_params, dict), \
                f"{overridden_loader_params} should be Dict"

            batch_size = overridden_loader_params.pop("batch_size", batch_size)
            num_workers = overridden_loader_params.\
                pop("num_workers", num_workers)

            if per_gpu_scaling and not distributed:
                num_gpus = max(1, torch.cuda.device_count())
                batch_size *= num_gpus
                num_workers *= num_gpus

            loader_params = {
                "batch_size": batch_size,
                "num_workers": num_workers,
                "pin_memory": torch.cuda.is_available(),
                "drop_last": drop_last,
                **overridden_loader_params
            }

            if isinstance(ds_, Dataset):
                loader_params["dataset"] = ds_
            elif isinstance(ds_, dict):
                assert "dataset" in ds_, \
                    "You need to specify dataset for dataloader"
                loader_params = utils.merge_dicts(ds_, loader_params)
            else:
                raise NotImplementedError

            if distributed:
                sampler = loader_params.get("sampler")
                if sampler is not None:
                    assert isinstance(sampler, DistributedSampler)
                else:
                    loader_params["sampler"] = DistributedSampler(
                        dataset=loader_params["dataset"]
                    )

            loader_params["shuffle"] = (
                name.startswith("train")
                and loader_params.get("sampler") is None
            )

            if "batch_sampler" in loader_params:
                if distributed:
                    raise ValueError(
                        "batch_sampler option is mutually "
                        "exclusive with distributed"
                    )

                for k in ("batch_size", "shuffle", "sampler", "drop_last"):
                    loader_params.pop(k, None)

            if "worker_init_fn" not in loader_params:
                loader_params["worker_init_fn"] = \
                    lambda x: utils.set_global_seed(self.initial_seed + x)

            loaders[name] = DataLoader(**loader_params)

        return loaders

    @staticmethod
    def _get_callback(**params):
        wrapper_params = params.pop("_wrapper", None)
        callback = CALLBACKS.get_from_params(**params)
        if wrapper_params:
            wrapper_params["base_callback"] = callback
            return ConfigExperiment._get_callback(**wrapper_params)
        return callback

    def get_callbacks(self, stage: str) -> "OrderedDict[Callback]":
        callbacks_params = (
            self.stages_config[stage].get("callbacks_params", {})
        )

        callbacks = OrderedDict()
        for key, callback_params in callbacks_params.items():
            callback = self._get_callback(**callback_params)
            callbacks[key] = callback

        return callbacks


__all__ = ["ConfigExperiment"]
