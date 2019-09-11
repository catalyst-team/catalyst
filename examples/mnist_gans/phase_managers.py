from typing import List


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

    def get_phase_name(self, state):
        if state.need_backward:
            return self.train_phases[self.train_index].name
        return self.valid_phases[self.valid_index].name

    def get_callbacks(self, state):
        if state.need_backward:
            return self.train_phases[self.train_index].callbacks
        return self.valid_phases[self.valid_index].callbacks
