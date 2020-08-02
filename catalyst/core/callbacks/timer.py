from catalyst.core.callback import Callback, CallbackNode, CallbackOrder
from catalyst.core.runner import IRunner
from catalyst.tools.time_manager import TimeManager

EPS = 1e-8


class TimerCallback(Callback):
    """Logs pipeline execution time."""

    def __init__(self):
        """@TODO: Docs. Contribution is welcome."""
        super().__init__(order=CallbackOrder.metric + 1, node=CallbackNode.all)
        self.timer = TimeManager()

    def on_loader_start(self, runner: IRunner) -> None:
        """Loader start hook.

        Args:
            runner (IRunner): current runner
        """
        self.timer.reset()
        self.timer.start("_timer/batch_time")
        self.timer.start("_timer/data_time")

    def on_loader_end(self, runner: IRunner) -> None:
        """Loader end hook.

        Args:
            runner (IRunner): current runner
        """
        self.timer.reset()

    def on_batch_start(self, runner: IRunner) -> None:
        """Batch start hook.

        Args:
            runner (IRunner): current runner
        """
        self.timer.stop("_timer/data_time")
        self.timer.start("_timer/model_time")

    def on_batch_end(self, runner: IRunner) -> None:
        """Batch end hook.

        Args:
            runner (IRunner): current runner
        """
        self.timer.stop("_timer/model_time")
        self.timer.stop("_timer/batch_time")

        # @TODO: just a trick
        self.timer.elapsed["_timer/_fps"] = runner.batch_size / (
            self.timer.elapsed["_timer/batch_time"] + EPS
        )
        for key, value in self.timer.elapsed.items():
            runner.batch_metrics[key] = value

        self.timer.reset()
        self.timer.start("_timer/batch_time")
        self.timer.start("_timer/data_time")


__all__ = ["TimerCallback"]
