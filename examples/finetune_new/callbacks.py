from tqdm import tqdm

from catalyst.dl.callbacks import Callback, RunnerState


class VerboseCallback(Callback):
    def __init__(self):
        self.tqdm: tqdm = None

    def on_loader_start(self, state: RunnerState):
        self.tqdm = tqdm(total=state.loader_len)

    def on_batch_end(self, state: RunnerState):
        self.tqdm.update(state.step + 1)

    def on_loader_end(self, state: RunnerState):
        self.tqdm.close()
        self.tqdm = None
