import torch

from catalyst.core.callback import Callback, CallbackNode, CallbackOrder
from catalyst.core.runner import IRunner


class ProfilerCallback(Callback):
    """Profile specified epoch or some fixed number of batches.

    Args:
        loader_key: name of the loader to use for profiling.
            If `None` then will be used first loader from experiment.
            Default is `None`.
        epoch: epoch number to use for profiling.
            Default is `1`.
        num_batches: number of batches to use in epoch to do a profiling.
            If `None` then will be used all batches in loader.
            Default is `None`.
        **profiler_kwargs: arguments to pass to a profiler.
            To get more info about possible arguments please use PyTorch
            `profiler docs`_.

    Example:
        .. code-block:: python

            import os

            import torch
            from torch import nn
            from torch.utils.data import DataLoader

            from catalyst import dl
            from catalyst.data import ToTensor
            from catalyst.contrib.datasets import MNIST
            from catalyst.contrib.nn.modules import Flatten

            loaders = {
                "train": DataLoader(
                    MNIST(os.getcwd(), train=False, download=True, transform=ToTensor()),
                    batch_size=32,
                ),
                "valid": DataLoader(
                    MNIST(os.getcwd(), train=False, download=True, transform=ToTensor()),
                    batch_size=32,
                ),
            }

            model = nn.Sequential(Flatten(), nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 10))
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
            runner = dl.SupervisedRunner()
            runner.train(
                model=model,
                callbacks=[dl.ProfilerCallback(
                    loader_key="train", epoch=3,
                    # profiler arguments
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    with_stack=True,
                    with_flops=True,
                )],
                loaders=loaders,
                criterion=criterion,
                optimizer=optimizer,
                num_epochs=5,
                logdir="./logs",
            )

    .. _profiler docs: https://pytorch.org/docs/stable/profiler.html

    """

    def __init__(
        self,
        loader_key: str = None,
        epoch: int = 1,
        num_batches: int = None,
        **profiler_kwargs,
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
            self.profiler.__enter__()

    def _exit_profiler(self, loader_key: str, epoch: int) -> None:
        if not self._should_use_profiler(loader_key, epoch) or self.profiler is None:
            return

        if self.stats is None:
            self.profiler.__exit__(None, None, None)
            self.stats = self.profiler.key_averages()
            # TODO: how to show table in other logers ?
            print(self.stats.table(sort_by="cpu_time_total", row_limit=100))

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
