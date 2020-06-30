from collections import defaultdict

from catalyst.core.callback import Callback, CallbackNode, CallbackOrder
from catalyst.core.runner import IRunner


class ValidationManagerCallback(Callback):
    """
    A callback to aggregate runner.valid_metrics from runner.epoch_metrics.
    """

    def __init__(self):
        """@TODO: Docs. Contribution is welcome."""
        super().__init__(
            order=CallbackOrder.validation, node=CallbackNode.all,
        )

    def on_epoch_start(self, runner: IRunner) -> None:
        """Epoch start hook.

        Args:
            runner (IRunner): current runner
        """
        runner.valid_metrics = defaultdict(None)
        runner.is_best_valid = False

    def on_epoch_end(self, runner: IRunner) -> None:
        """Epoch end hook.

        Args:
            runner (IRunner): current runner
        """
        if runner.stage_name.startswith("infer"):
            return

        runner.valid_metrics = {
            k.replace(f"{runner.valid_loader}_", ""): v
            for k, v in runner.epoch_metrics.items()
            if k.startswith(runner.valid_loader)
        }
        assert (
            runner.main_metric in runner.valid_metrics
        ), f"{runner.main_metric} value is not available by the epoch end"

        current_valid_metric = runner.valid_metrics[runner.main_metric]
        if runner.minimize_metric:
            best_valid_metric = runner.best_valid_metrics.get(
                runner.main_metric, float("+inf")
            )
            is_best = current_valid_metric < best_valid_metric
        else:
            best_valid_metric = runner.best_valid_metrics.get(
                runner.main_metric, float("-inf")
            )
            is_best = current_valid_metric > best_valid_metric

        if is_best:
            runner.is_best_valid = True
            runner.best_valid_metrics = runner.valid_metrics.copy()


__all__ = ["ValidationManagerCallback"]
