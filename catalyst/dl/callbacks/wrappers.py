from typing import List

from catalyst.dl.core import Callback, RunnerState


class PhaseWrapperCallback(Callback):
    """
    CallbackWrapper which disables/enables handlers
    dependant on current phase and event type

    May be useful i.e. to disable/enable optimizers & losses
    """
    LEVEL_STAGE = "stage"
    LEVEL_EPOCH = "epoch"
    LEVEL_LOADER = "loader"
    LEVEL_BATCH = "batch"

    TIME_START = "start"
    TIME_END = "end"

    def __init__(
        self,
        base_callback: Callback,
        active_phases: List[str] = None,
        inactive_phases: List[str] = None
    ):
        super().__init__(base_callback.order)
        assert (active_phases is None) ^ (inactive_phases is None), \
            "Exactly one of active/inactive phases must be specified"
        self.callback = base_callback
        self.active_phases = active_phases or []
        self.inactive_phases = inactive_phases or []
        assert len(self.active_phases) + len(self.inactive_phases) > 0, \
            "Wrapper has no sense if callback is always active/inactive"

    def is_active_on_phase(self, phase, level, time):
        return self._is_active_on_phase(phase=phase)

    def _is_active_on_phase(self, phase):
        if phase is None:
            # if phase is None every callback is active
            return True

        if phase in self.active_phases:
            return True
        if self.inactive_phases and phase not in self.inactive_phases:
            return True
        return False

    def on_stage_start(self, state: RunnerState):
        if self.is_active_on_phase(
            phase=state.phase,
            level=self.LEVEL_STAGE,
            time=self.TIME_START
        ):
            self.callback.on_stage_start(state)

    def on_stage_end(self, state: RunnerState):
        if self.is_active_on_phase(
            phase=state.phase,
            level=self.LEVEL_STAGE,
            time=self.TIME_END
        ):
            self.callback.on_stage_end(state)

    def on_epoch_start(self, state: RunnerState):
        if self.is_active_on_phase(
            phase=state.phase,
            level=self.LEVEL_EPOCH,
            time=self.TIME_START
        ):
            self.callback.on_epoch_start(state)

    def on_epoch_end(self, state: RunnerState):
        if self.is_active_on_phase(
            phase=state.phase,
            level=self.LEVEL_EPOCH,
            time=self.TIME_END
        ):
            self.callback.on_epoch_end(state)

    def on_loader_start(self, state: RunnerState):
        if self.is_active_on_phase(
            phase=state.phase,
            level=self.LEVEL_LOADER,
            time=self.TIME_START
        ):
            self.callback.on_loader_start(state)

    def on_loader_end(self, state: RunnerState):
        if self.is_active_on_phase(
            phase=state.phase,
            level=self.LEVEL_LOADER,
            time=self.TIME_END
        ):
            self.callback.on_loader_end(state)

    def on_batch_start(self, state: RunnerState):
        if self.is_active_on_phase(
            phase=state.phase,
            level=self.LEVEL_BATCH,
            time=self.TIME_START
        ):
            self.callback.on_batch_start(state)

    def on_batch_end(self, state: RunnerState):
        if self.is_active_on_phase(
            phase=state.phase,
            level=self.LEVEL_BATCH,
            time=self.TIME_END
        ):
            self.callback.on_batch_end(state)

    def on_exception(self, state: RunnerState):
        self.callback.on_exception(state)


class PhaseBatchWrapperCallback(PhaseWrapperCallback):
    def is_active_on_phase(self, phase, level, time):
        if level != self.LEVEL_BATCH:
            return True
        return self._is_active_on_phase(phase)


__all__ = ["PhaseWrapperCallback", "PhaseBatchWrapperCallback"]
