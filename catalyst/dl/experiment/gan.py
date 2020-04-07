from typing import Dict, List
from collections import OrderedDict

from catalyst.dl import PhaseBatchWrapperCallback, PhaseManagerCallback

from .core import Experiment


class GanExperiment(Experiment):
    """One-staged GAN experiment."""

    def __init__(
        self, *, phase2callbacks: Dict[str, List[str]] = None, **kwargs,
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
        super().__init__(**kwargs)
        self.wrap_callbacks(phase2callbacks or {})

    def wrap_callbacks(self, phase2callbacks) -> None:
        """Phase wrapping procedure for callbacks."""
        discriminator_phase_name = self._state_kwargs[
            "discriminator_train_phase"
        ]
        discriminator_phase_num = self._state_kwargs["discriminator_train_num"]
        generator_phase_name = self._state_kwargs["generator_train_phase"]
        generator_phase_num = self._state_kwargs["generator_train_num"]
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


__all__ = ["GanExperiment"]
