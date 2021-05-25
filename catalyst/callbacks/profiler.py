from catalyst.core.callback import Callback
from catalyst.core.runner import IRunner


class ProfilerCallback(Callback):
    def __init__(self, profiler=None):
        # Order = 0 means that this has the highest execution priority
        super().__init__(
            order=0, node=1
        )  # Node = 1 means that the callback is only executed on the Main thread.
        self._profiler = profiler

    def on_batch_end(self, runner: IRunner):
        if self._profiler is not None:
            self._profiler.step()
