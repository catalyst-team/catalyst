from typing import Any, Mapping, Dict, List
from copy import deepcopy
from collections import OrderedDict
import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset  # noqa F401
from torch.utils.data import DistributedSampler

from catalyst.dl.registry import \
    MODELS, CRITERIONS, OPTIMIZERS, SCHEDULERS, CALLBACKS
from catalyst.dl import utils
from catalyst.utils.misc import merge_dicts
from catalyst.utils.hash import get_short_hash
from catalyst.dl.core import Experiment, Callback
from catalyst.dl.utils.torch import _Model, _Criterion, _Optimizer, _Scheduler


class ConfigExperiment(Experiment):
    STAGE_KEYWORDS = [
        "criterion_params", "optimizer_params", "scheduler_params",
        "data_params", "state_params", "callbacks_params",
    ]

    def __init__(self, config: Dict):
        self._config = deepcopy(config)
        self.__prepare_logdir()

        self._config["stages"]["state_params"] = merge_dicts(
            deepcopy(self._config["stages"].get("state_params", {})),
            deepcopy(self._config.get("args", {})),
            {"logdir": self._logdir}
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
                stages_config_out[stage][key] = merge_dicts(
                    deepcopy(stages_defaults.get(key, {})),
                    deepcopy(stages_config[stage].get(key, {})),
                )

        return stages_config_out

    @property
    def logdir(self):
        return self._logdir

    def _get_logdir(self, config: Dict) -> str:
        timestamp = datetime.datetime.utcnow().strftime("%y%m%d.%H%M%S")
        config_hash = get_short_hash(config)
        logdir = f"{timestamp}.{config_hash}"
        distributed_rank = self.distributed_params.get("rank", -1)
        if distributed_rank > -1:
            logdir = f"{logdir}.rank{distributed_rank:02d}"
        return logdir

    @property
    def stages(self) -> List[str]:
        stages_keys = list(self.stages_config.keys())
        return stages_keys

    @property
    def distributed_params(self) -> Dict:
        return self._config.get("distributed_params", {})

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

    def get_model(self, stage: str) -> _Model:
        model_params = self._config["model_params"]
        model = MODELS.get_from_params(**model_params)

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

    def _get_optimizer(self, *, model_params, **params):
        key_value_flag = params.pop("_key_value", False)

        if key_value_flag:
            optimizer = {}
            for key, params_ in params.items():
                optimizer[key] = self._get_optimizer(
                    model_params=model_params, **params_)
        else:
            load_from_previous_stage = \
                params.pop("load_from_previous_stage", False)
            optimizer = OPTIMIZERS.get_from_params(
                **params,
                params=model_params
            )

            if load_from_previous_stage:
                checkpoint_path = \
                    f"{self.logdir}/checkpoints/best.pth"
                checkpoint = utils.load_checkpoint(checkpoint_path)
                utils.unpack_checkpoint(checkpoint, optimizer=optimizer)
                for key, value in params.items():
                    for pg in optimizer.param_groups:
                        pg[key] = value

        return optimizer

    def get_optimizer(
        self,
        stage: str,
        model: nn.Module
    ) -> _Optimizer:

        model_params = utils.get_optimizable_params(model.parameters())
        optimizer_params = \
            self.stages_config[stage].get("optimizer_params", {})

        optimizer = self._get_optimizer(
            model_params=model_params, **optimizer_params)

        return optimizer

    @staticmethod
    def _get_scheduler(*, optimizer, **params):
        key_value_flag = params.pop("_key_value", False)

        if key_value_flag:
            scheduler = {}
            for key, params_ in params.items():
                scheduler[key] = ConfigExperiment._get_scheduler(
                    optimizer=optimizer, **params_)
        else:
            scheduler = SCHEDULERS.get_from_params(
                **params,
                optimizer=optimizer
            )
        return scheduler

    def get_scheduler(self, stage: str, optimizer) -> _Scheduler:
        scheduler_params = \
            self.stages_config[stage].get("scheduler_params", {})
        scheduler = self._get_scheduler(
            optimizer=optimizer, **scheduler_params)
        return scheduler

    def get_loaders(self, stage: str) -> "OrderedDict[str, DataLoader]":
        data_params = dict(self.stages_config[stage]["data_params"])

        batch_size = data_params.pop("batch_size")
        num_workers = data_params.pop("num_workers")
        drop_last = data_params.pop("drop_last", False)
        per_gpu_scaling = data_params.pop("per_gpu_scaling", False)
        distributed_rank = self.distributed_params.get("rank", -1)
        distributed = distributed_rank > -1

        if per_gpu_scaling and not distributed:
            num_gpus = max(1, torch.cuda.device_count())
            batch_size *= num_gpus
            num_workers *= num_gpus

        datasets = self.get_datasets(stage=stage, **data_params)

        loaders = OrderedDict()
        for name, ds_ in datasets.items():
            assert isinstance(ds_, (Dataset, dict)), \
                f"{ds_} should be Dataset of Dict"

            loader_params = {
                "batch_size": batch_size,
                "num_workers": num_workers,
                "pin_memory": torch.cuda.is_available(),
                "drop_last": drop_last,
            }

            if isinstance(ds_, Dataset):
                loader_params["dataset"] = ds_
            elif isinstance(ds_, dict):
                assert "dataset" in ds_, \
                    "You need to specify dataset for dataloader"
                loader_params = merge_dicts(ds_, loader_params)
            else:
                raise NotImplementedError

            if distributed:
                sampler = loader_params.get("sampler")
                if sampler is not None:
                    assert isinstance(sampler, DistributedSampler)
                else:
                    loader_params["sampler"] = DistributedSampler(
                        dataset=loader_params["dataset"])

            loader_params["shuffle"] = (
                name.startswith("train")
                and loader_params.get("sampler") is None)

            loaders[name] = DataLoader(**loader_params)

        return loaders

    def get_callbacks(self, stage: str) -> "List[Callback]":
        callbacks_params = (
            self.stages_config[stage].get("callbacks_params", {}))

        callbacks = []
        for key, callback_params in callbacks_params.items():
            callback = CALLBACKS.get_from_params(**callback_params)
            callbacks.append(callback)

        return callbacks


__all__ = ["ConfigExperiment"]
