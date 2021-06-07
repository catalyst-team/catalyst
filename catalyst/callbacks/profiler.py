import torch

from catalyst.core.callback import Callback, CallbackNode, CallbackOrder
from catalyst.core.runner import IRunner


class ProfilerCallback(Callback):
    """Performs the profiler step for the PyTorch:1.8 profiler"""

    def __init__(
        self, loader_key: str = None, epoch: int = 1, num_batches: int = None, **profiler_kwargs,
    ):
        super().__init__(order=CallbackOrder.Internal, node=CallbackNode.Master)

        self.loader_key = loader_key
        self.epoch = epoch
        self.num_batches = num_batches
        self.batch_cnt = 0

        self.profiler_kwargs = profiler_kwargs
        self.profiler = None
        self.stats = None

    def on_experiment_start(self, runner: IRunner) -> None:
        """
        On batch end action

        Args:
            runner: current runner
        """
        if self.loader_key is None:
            self.loader_key = runner.loader_key  # use first loader for profile

    def _should_use_profiler(self, loader_key: str, epoch: int):
        if self.loader_key == loader_key and self.epoch == epoch:
            if self.num_batches is not None:
                return self.batch_cnt < self.num_batches
            return True
        return False

    def _enter_profiler(self, loader_key: str, epoch: int) -> None:
        if not self._should_use_profiler(loader_key, epoch):
            return

        if self.profiler is None:
            self.profiler = torch.profiler.profile(**self.profiler_kwargs)
            self.profiler = self.profiler.__enter__()

    def _exit_profiler(self, loader_key: str, epoch: int) -> None:
        if not self._should_use_profiler(loader_key, epoch) or self.profiler is None:
            return

        self.profiler.__exit__(None, None, None)
        self.stats = self.profiler.key_averages()

        # TODO: how to show table in other logers ?
        print(self.stats.table(sort_by="cpu_time_total", row_limit=10))

    def on_loader_start(self, runner: IRunner) -> None:
        """
        On loader start action

        Args:
            runner: current runner
        """
        self._enter_profiler(runner.loader_key, runner.stage_epoch_step)

    def on_loader_end(self, runner: IRunner) -> None:
        """
        On loader end action

        Args:
            runner: current runner
        """
        self._exit_profiler(runner.loader_key, runner.stage_epoch_step)

    def on_batch_start(self, runner: IRunner) -> None:
        """
        On batch start action

        Args:
            runner: current runner
        """
        self._enter_profiler(runner.loader_key, runner.stage_epoch_step)

    def on_batch_end(self, runner: IRunner) -> None:
        """
        On batch end action

        Args:
            runner: current runner
        """
        if self.profiler is None:
            return

        if self.num_batches is not None and self.batch_cnt < self.num_batches:
            # do a profiling step after each batch
            self.profiler.step()
            self.batch_cnt += 1

        self._exit_profiler(runner.loader_key, runner.stage_epoch_step)
