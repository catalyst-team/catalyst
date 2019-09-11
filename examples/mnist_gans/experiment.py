from collections import OrderedDict
from typing import List
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

from catalyst.dl import ConfigExperiment
from catalyst import utils
from catalyst.dl.registry import \
    MODELS, CRITERIONS, OPTIMIZERS, SCHEDULERS, CALLBACKS


class Phase:
    def __init__(self, callbacks=None, steps=None, name=None):
        self.steps = steps
        self.curr_step = 0
        self.name = name
        self.callbacks = callbacks


class PhaseManager:
    def __init__(self, train_phases: "List[Phase]", valid_phases: "List[Phase]"):
        self.train_phases = train_phases
        self.valid_phases = valid_phases

        self.train_index = 0
        self.valid_index = 0

    def step(self, state, step_size=1):
        if state.need_backward:
            if len(self.train_phases) > 1:
                phase = self.train_phases[self.train_index]
                phase.curr_step += step_size
                if phase.curr_step >= phase.steps:
                    phase.curr_step = 0
                    self.train_index = (self.train_index + 1) % len(self.train_phases)
        else:
            if len(self.valid_phases) > 1:
                phase = self.valid_phases[self.valid_index]
                phase.curr_step += step_size
                if phase.curr_step >= phase.steps:
                    phase.curr_step = 0
                    self.valid_index = (self.valid_index + 1) % len(self.valid_phases)

    def get_phase_name(self, state):  # for runner predict_batch (may behave differently in different phases)
        if state.need_backward:
            return self.train_phases[self.train_index].name
        return self.valid_phases[self.valid_index].name

    def get_callbacks(self, state):
        if state.need_backward:
            return self.train_phases[self.train_index].callbacks
        return self.valid_phases[self.valid_index].callbacks


class GANExperiment(ConfigExperiment):
    def get_phase_manager(self, stage):
        state_params = self.get_state_params(stage)
        callbacks = self.get_callbacks(stage)

        runner_phases = state_params.get("runner_phases", None)

        train_phases = []
        valid_phases = []
        if runner_phases is None:
            train_phases = [Phase(callbacks=callbacks, steps=None, name=None)]
            valid_phases = train_phases
        else:
            VM_ALL = "all"
            VM_SAME = "same"
            allowed_valid_modes = [VM_SAME, VM_ALL]

            valid_mode = runner_phases.pop("_valid_mode", VM_ALL)
            if valid_mode not in allowed_valid_modes:
                raise ValueError(f"_valid_mode must be one of {allowed_valid_modes}, got '{valid_mode}'")
            # train phases
            for phase_name, phase_params in runner_phases.items():
                steps = phase_params.get("steps", 1)
                inactive_callbacks = phase_params.get("inactive_callbacks", None)
                active_callbacks = phase_params.get("active_callbacks", None)
                if active_callbacks is not None and inactive_callbacks is not None:
                    raise ValueError("Only one of '[active_callbacks/inactive_callbacks]' may be specified")
                phase_callbacks = callbacks
                if active_callbacks:
                    phase_callbacks = OrderedDict(x for x in callbacks.items() if x[0] in active_callbacks)
                if inactive_callbacks:
                    phase_callbacks = OrderedDict(x for x in callbacks.items() if x[0] not in inactive_callbacks)
                phase = Phase(callbacks=phase_callbacks, steps=steps, name=phase_name)
                train_phases.append(phase)
                # valid
                if valid_mode == VM_SAME:
                    valid_phases.append(
                        Phase(callbacks=phase_callbacks, steps=steps, name=phase_name)
                    )
            # valid
            if valid_mode == VM_ALL:
                valid_phases.append(Phase(callbacks=callbacks))

        return PhaseManager(
            train_phases=train_phases,
            valid_phases=valid_phases
        )

    def get_model(self, stage: str):
        model_params = self._config["model_params"]
        model = GANExperiment._get_model(**model_params)

        model = self._preprocess_model_for_stage(stage, model)
        model = self._postprocess_model_for_stage(stage, model)
        return model

    @staticmethod
    def _get_model(**params):
        key_value_flag = params.pop("_key_value", False)

        if key_value_flag:
            model = {}
            for key, params_ in params.items():
                model[key] = GANExperiment._get_model(**params_)
        else:
            model = MODELS.get_from_params(**params)
            if model is not None and torch.cuda.is_available():
                model = model.cuda()
        return model

    def __get_optimizer(self, stage, model, **params):
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

        model_key = params.pop("_model", None)
        if model_key is None:
            assert isinstance(model, nn.Module), "model is keyvalue, but optimizer has no specified model"
            model_params = utils.process_model_params(
                model, layerwise_params, no_bias_weight_decay, lr_scaling
            )
        elif isinstance(model_key, str):
            model_params = utils.process_model_params(
                model[model_key], layerwise_params, no_bias_weight_decay, lr_scaling
            )
        elif isinstance(model_key, (list, tuple)):
            model_params = []
            for model_key_ in model_key:
                model_params_ = utils.process_model_params(
                    model[model_key_], layerwise_params, no_bias_weight_decay, lr_scaling
                )
                model_params.extend(model_params_)
        else:
            raise ValueError("unknown type of model_params")

        load_from_previous_stage = \
            params.pop("load_from_previous_stage", False)
        optimizer = OPTIMIZERS.get_from_params(**params, params=model_params)

        if load_from_previous_stage:
            checkpoint_path = f"{self.logdir}/checkpoints/best.pth"
            checkpoint = utils.load_checkpoint(checkpoint_path)
            utils.unpack_checkpoint(checkpoint, optimizer=optimizer)
            for key, value in params.items():
                for pg in optimizer.param_groups:
                    pg[key] = value

        return optimizer

    def get_optimizer(self, stage: str, model: nn.Module):
        optimizer_params = \
            self.stages_config[stage].get("optimizer_params", {})
        key_value_flag = optimizer_params.pop("_key_value", False)

        if key_value_flag:
            optimizer = {}
            for key, params_ in optimizer_params.items():
                optimizer[key] = self.__get_optimizer(stage, model, **params_)
        else:
            optimizer = self.__get_optimizer(stage, model, **optimizer_params)

        return optimizer

    def get_callbacks(self, stage: str):
        callbacks_params = (
            self.stages_config[stage].get("callbacks_params", {})
        )

        callbacks = OrderedDict()
        for key, callback_params in callbacks_params.items():
            callback = CALLBACKS.get_from_params(**callback_params)
            callbacks[key] = callback

        return callbacks

    @staticmethod
    def get_transforms(stage: str = None, mode: str = None):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, ), (0.5, ))
            ]
        )

    def get_datasets(self, stage: str, **kwargs):
        datasets = OrderedDict()

        trainset = torchvision.datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=GANExperiment.get_transforms(stage=stage, mode="train")
        )
        testset = torchvision.datasets.MNIST(
            root="./data",
            train=False,
            download=True,
            transform=GANExperiment.get_transforms(stage=stage, mode="valid")
        )

        datasets["train"] = trainset
        datasets["valid"] = testset

        return datasets


