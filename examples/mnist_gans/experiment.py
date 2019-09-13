from collections import OrderedDict

import torchvision
from catalyst.dl import ConfigExperiment
from torchvision import transforms

from .phase_managers import Phase, PhaseManager


# abstract; supports multiple phases
class MultiPhaseConfigExperiment(ConfigExperiment):
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
                raise ValueError(
                    f"_valid_mode must be one of {allowed_valid_modes}, "
                    f"got '{valid_mode}'")
            # train phases
            for phase_name, phase_params in runner_phases.items():
                steps = phase_params.get("steps", 1)
                inactive_callbacks = phase_params.get("inactive_callbacks",
                                                      None)
                active_callbacks = phase_params.get("active_callbacks", None)
                if (active_callbacks is not None
                        and inactive_callbacks is not None):
                    raise ValueError(
                        "Only one of '[active_callbacks/inactive_callbacks]'"
                        " may be specified")
                phase_callbacks = callbacks
                if active_callbacks:
                    phase_callbacks = OrderedDict(
                        x for x in callbacks.items() if
                        x[0] in active_callbacks)
                if inactive_callbacks:
                    phase_callbacks = OrderedDict(
                        x for x in callbacks.items() if
                        x[0] not in inactive_callbacks)
                phase = Phase(callbacks=phase_callbacks, steps=steps,
                              name=phase_name)
                train_phases.append(phase)
                # valid
                if valid_mode == VM_SAME:
                    valid_phases.append(
                        Phase(callbacks=phase_callbacks, steps=steps,
                              name=phase_name)
                    )
            # valid
            if valid_mode == VM_ALL:
                valid_phases.append(Phase(callbacks=callbacks))

        return PhaseManager(
            train_phases=train_phases,
            valid_phases=valid_phases
        )


# data loaders & transforms
class MNISTGANExperiment(MultiPhaseConfigExperiment):
    @staticmethod
    def get_transforms(stage: str = None, mode: str = None):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ]
        )

    def get_datasets(self, stage: str, **kwargs):
        datasets = OrderedDict()

        trainset = torchvision.datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=MNISTGANExperiment.get_transforms(stage=stage,
                                                        mode="train")
        )
        testset = torchvision.datasets.MNIST(
            root="./data",
            train=False,
            download=True,
            transform=MNISTGANExperiment.get_transforms(stage=stage,
                                                        mode="valid")
        )

        datasets["train"] = trainset
        datasets["valid"] = testset

        return datasets
