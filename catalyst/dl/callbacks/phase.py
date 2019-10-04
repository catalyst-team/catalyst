from typing import List
from collections import OrderedDict

from catalyst.dl.core import Callback, CallbackOrder


class Phase:
    """
    Class for storing information about certain phase, including
    - phase name
    - number of steps (batches) in phase before next phase is chosen
    - how many steps (batches) are done already
    """
    def __init__(self, name: str = None, steps: int = None):
        self.steps = int(steps) if steps is not None else None
        self.curr_step = 0
        self.name = name


class PhaseManager:
    """
    Class for storing & managing all phases in experiment configuration

    Stores separately current phases in train & validation modes

    By calling `.step(...)` method current phase is updated by step-size
    and if current phase is finished, the next phase becomes current
    """
    def __init__(
        self,
        train_phases: List[Phase],
        valid_phases: List[Phase]
    ):
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
                    self.train_index = \
                        (self.train_index + 1) % len(self.train_phases)
        else:
            if len(self.valid_phases) > 1:
                phase = self.valid_phases[self.valid_index]
                phase.curr_step += step_size
                if phase.curr_step >= phase.steps:
                    phase.curr_step = 0
                    self.valid_index = \
                        (self.valid_index + 1) % len(self.valid_phases)

    def get_phase_name(self, state):
        if state.need_backward:
            return self.train_phases[self.train_index].name
        return self.valid_phases[self.valid_index].name


class PhaseManagerCallback(Callback):
    """
    PhaseManagerCallback updates state.phase
    """

    VALIDATION_MODE_ALL = "all"  # (in validation) use all callbacks
    VALIDATION_MODE_SAME = "same"  # (in validation) same phases as in training
    allowed_valid_modes = [VALIDATION_MODE_SAME, VALIDATION_MODE_ALL]

    def __init__(
        self,
        train_phases: "OrderedDict[str, int]" = None,
        valid_phases: "OrderedDict[str, int]" = None,
        valid_mode: str = None
    ):
        super().__init__(CallbackOrder.Internal)
        self.phase_manager = self._get_phase_manager(
            train_phases=train_phases,
            valid_phases=valid_phases,
            valid_mode=valid_mode
        )

    def _get_phase_manager(
        self,
        train_phases: "OrderedDict[str, int]" = None,
        valid_phases: "OrderedDict[str, int]" = None,
        valid_mode: str = None
    ):
        assert (valid_phases is None) ^ (valid_mode is None), \
            "Exactly one of them must be specified"

        if train_phases is None:
            train_phases = [Phase(name=None, steps=None)]
        else:
            train_phases = [
                Phase(name=name, steps=steps)
                for name, steps in train_phases.items()
            ]

        if valid_phases is None:
            if valid_mode == self.VALIDATION_MODE_ALL:
                valid_phases = [Phase(name=None, steps=None)]
            elif valid_mode == self.VALIDATION_MODE_SAME:
                valid_phases = [
                    Phase(name=p.name, steps=p.steps)
                    for p in train_phases
                ]
            else:
                raise ValueError(
                    f"Unsupported validation_mode, should be one of "
                    f"{self.allowed_valid_modes}"
                )

        return PhaseManager(
            train_phases=train_phases,
            valid_phases=valid_phases
        )

    def on_batch_start(self, state):
        state.phase = self.phase_manager.get_phase_name(state)

    def on_batch_end(self, state):
        self.phase_manager.step(state)


__all__ = ["PhaseManagerCallback"]
