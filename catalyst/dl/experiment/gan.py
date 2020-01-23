from collections import OrderedDict
from typing import Dict, Union, List

from catalyst.dl import Callback
from torch.utils.data import DataLoader
from catalyst.dl.callbacks import (
    CheckpointCallback,
    ConsoleLogger,
    RaiseExceptionCallback,
    VerboseLogger,
    PhaseBatchWrapperCallback,
    PhaseManagerCallback,
)
from catalyst.utils.typing import Model, Criterion, Optimizer, Scheduler
from .base import BaseExperiment


class GanExperiment(BaseExperiment):
    """
    One-staged GAN experiment
    Mainly used in notebooks
    """

    def __init__(
        self,
        model: Union[Model, Dict[str, Model]],
        loaders: "OrderedDict[str, DataLoader]",
        callbacks: "OrderedDict[str, Callback]" = None,
        logdir: str = None,
        stage: str = "train",
        criterion: Criterion = None,
        optimizer: Optimizer = None,
        scheduler: Scheduler = None,
        num_epochs: int = 1,
        valid_loader: str = "valid",
        main_metric: str = "loss",
        minimize_metric: bool = True,
        verbose: bool = False,
        state_kwargs: Dict = None,
        checkpoint_data: Dict = None,
        distributed_params: Dict = None,
        monitoring_params: Dict = None,
        initial_seed: int = 42,
        phase2callbacks: Dict[str, List[str]] = None,
    ):
        """
        Args:
            model (Model or Dict[str, Model]): models,
                usually generator and discriminator
            loaders (dict): dictionary containing one or several
                ``torch.utils.data.DataLoader`` for training and validation
            callbacks (List[catalyst.dl.Callback]): list of callbacks
            logdir (str): path to output directory
            stage (str): current stage
            criterion (Criterion): criterion function
            optimizer (Optimizer): optimizer
            scheduler (Scheduler): scheduler
            num_epochs (int): number of experiment's epochs
            valid_loader (str): loader name used to calculate
                the metrics and save the checkpoints. For example,
                you can pass `train` and then
                the metrics will be taken from `train` loader.
            main_metric (str): the key to the name of the metric
                by which the checkpoints will be selected.
            minimize_metric (bool): flag to indicate whether
                the ``main_metric`` should be minimized.
            verbose (bool): ff true, it displays the status of the training
                to the console.
            state_kwargs (dict): additional state params to ``RunnerState``
            checkpoint_data (dict): additional data to save in checkpoint,
                for example: ``class_names``, ``date_of_training``, etc
            distributed_params (dict): dictionary with the parameters
                for distributed and FP16 method
            monitoring_params (dict): dict with the parameters
                for monitoring services
            initial_seed (int): experiment's initial seed value
            phase2callbacks (dict): dictionary with lists of callback names
                which should be wrapped for appropriate phase
                for example: {"generator_train": "loss_g", "optim_g"}
                "loss_g" and "optim_g" callbacks from callbacks dict
                will be wrapped for "generator_train" phase
                in wrap_callbacks method
        """
        super().__init__(
            model,
            loaders,
            callbacks,
            logdir,
            stage,
            criterion,
            optimizer,
            scheduler,
            num_epochs,
            valid_loader,
            main_metric,
            minimize_metric,
            verbose,
            state_kwargs,
            checkpoint_data,
            distributed_params,
            monitoring_params,
            initial_seed,
        )
        self.wrap_callbacks(phase2callbacks)

    def wrap_callbacks(self, phase2callbacks) -> None:
        """Phase wrapping procedure for callbacks"""
        discriminator_phase_name = self._additional_state_kwargs[
            "discriminator_train_phase"
        ]
        discriminator_phase_num = self._additional_state_kwargs[
            "discriminator_train_num"
        ]
        generator_phase_name = self._additional_state_kwargs[
            "generator_train_phase"
        ]
        generator_phase_num = self._additional_state_kwargs[
            "generator_train_num"
        ]
        self._callbacks["phase_manager"] = PhaseManagerCallback(
            train_phases=OrderedDict(
                [
                    (discriminator_phase_name, discriminator_phase_num),
                    (generator_phase_name, generator_phase_num),
                ]
            ),
            valid_mode="all",
        )
        for phase_name, callback_name_list in phase2callbacks.items():
            # TODO: Check for phase in state_params?
            for callback_name in callback_name_list:
                callback = self._callbacks.pop(callback_name)
                self._callbacks[callback_name] = PhaseBatchWrapperCallback(
                    base_callback=callback, active_phases=[phase_name],
                )

    def get_callbacks(self, stage: str) -> "OrderedDict[str, Callback]":
        callbacks = super().get_callbacks(stage="")
        default_callbacks = []
        if self._verbose:
            default_callbacks.append(("verbose", VerboseLogger))
        if not stage.startswith("infer"):
            default_callbacks.append(("saver", CheckpointCallback))
            default_callbacks.append(("console", ConsoleLogger))
        default_callbacks.append(("exception", RaiseExceptionCallback))
        # Check for absent callbacks and add them
        for callback_name, callback_fn in default_callbacks:
            is_already_present = any(
                isinstance(x, callback_fn) for x in callbacks.values()
            )
            if not is_already_present:
                callbacks[callback_name] = callback_fn()
        return callbacks

    @staticmethod
    def get_transforms(stage: str = None, mode: str = None):
        raise NotImplementedError(
            "No static transforms are used in GAN experiment"
        )


__all__ = ["GanExperiment"]
