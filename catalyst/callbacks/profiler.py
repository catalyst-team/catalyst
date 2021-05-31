import torch

from catalyst.core.callback import Callback, CallbackNode, CallbackOrder
from catalyst.core.runner import IRunner


# TODO: how to show logs/trace ?
class ProfilerCallback(Callback):
    """Performs the profiler step for the PyTorch:1.8 profiler"""

    def __init__(self, mode: str = None, loader_key: str = None, **profiler_kwargs):
        super().__init__(order=CallbackOrder.Internal, node=CallbackNode.master)

        mode = "batch" if mode is None else mode
        if mode not in ("epoch", "batch"):
            raise ValueError(f"Expected mode 'epoch'/'batch' but got '{mode}'!")

        self.mode = mode
        self.loader_key = loader_key
        self.profiler_kwargs = profiler_kwargs
        self.profiler = None

    def on_experiment_start(self, runner: IRunner) -> None:
        if self.loader_key is None:
            self.loader_key = runner.loader_key  # use first loader for profile

    def _enter_profiler(self, loader_key: str, mode: str) -> None:
        if self.loader_key != loader_key or self.mode != mode:
            return
        self.profiler = torch.profiler.profile(**self.profiler_kwargs)
        self.profiler.__enter__()

    def _exit_profiler(self, loader_key: str, mode: str) -> None:
        if self.loader_key != loader_key or self.mode != mode:
            return
        self.profiler.__exit__()

    def on_epoch_start(self, runner: IRunner) -> None:
        self._enter_profiler(self.loader_key, "epoch")

    def on_epoch_end(self, runner: IRunner) -> None:
        self._exit_profiler(self.loader_key, "epoch")

    def on_batch_start(self, runner: IRunner) -> None:
        self._enter_profiler(self.loader_key, "batch")

    def on_batch_end(self, runner: IRunner) -> None:
        # do a profiling step after each batch
        self.profiler.step()
        self._exit_profiler(self.loader_key, "batch")
