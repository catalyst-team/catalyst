from typing import Union, Optional, List
from pathlib import Path

from catalyst.utils.plotly import plot_tensorboard_log


def plot_metrics(
    logdir: Union[str, Path],
    step: Optional[str] = "epoch",
    metrics: Optional[List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None
) -> None:
    """Plots your learning results.

    Args:
        logdir: the logdir that was specified during training.
        step: 'batch' or 'epoch' - what logs to show: for batches or
            for epochs
        metrics: list of metrics to plot. The loss should be specified as
            'loss', learning rate = '_base/lr' and other metrics should be
            specified as names in metrics dict
            that was specified during training
        height: the height of the whole resulting plot
        width: the width of the whole resulting plot

    """
    assert step in ["batch", "epoch"], \
        f"Step should be either 'batch' or 'epoch', got '{step}'"
    metrics = metrics or ["loss"]
    plot_tensorboard_log(logdir, step, metrics, height, width)


__all__ = ["plot_metrics"]
