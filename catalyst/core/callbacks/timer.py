from catalyst.core import Callback, CallbackNode, CallbackOrder, State
from catalyst.utils.tools.time_manager import TimeManager


class TimerCallback(Callback):
    """Logs pipeline execution time."""

    def __init__(self):
        """@TODO: Docs. Contribution is welcome."""
        super().__init__(order=CallbackOrder.Metric + 1, node=CallbackNode.All)
        self.timer = TimeManager()

    def on_loader_start(self, state: State) -> None:
        """Loader start hook.

        Args:
            state (State): current state
        """
        self.timer.reset()
        self.timer.start("_timer/batch_time")
        self.timer.start("_timer/data_time")

    def on_loader_end(self, state: State) -> None:
        """Loader end hook.

        Args:
            state (State): current state
        """
        self.timer.reset()

    def on_batch_start(self, state: State) -> None:
        """Batch start hook.

        Args:
            state (State): current state
        """
        self.timer.stop("_timer/data_time")
        self.timer.start("_timer/model_time")

    def on_batch_end(self, state: State) -> None:
        """Batch end hook.

        Args:
            state (State): current state
        """
        self.timer.stop("_timer/model_time")
        self.timer.stop("_timer/batch_time")

        # @TODO: just a trick
        self.timer.elapsed["_timer/_fps"] = (
            state.batch_size / self.timer.elapsed["_timer/batch_time"]
        )
        for key, value in self.timer.elapsed.items():
            state.batch_metrics[key] = value

        self.timer.reset()
        self.timer.start("_timer/batch_time")
        self.timer.start("_timer/data_time")


__all__ = ["TimerCallback"]
