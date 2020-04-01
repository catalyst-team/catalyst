from typing import List, Optional, Union
import itertools
from pathlib import Path

import numpy as np

from .image import tensor_from_rgb_image
from .plotly import plot_tensorboard_log


def plot_confusion_matrix(
    cm,
    class_names=None,
    normalize=False,
    title="confusion matrix",
    fname=None,
    show=True,
    figsize=12,
    fontsize=32,
    colormap="Blues",
):
    """
    Render the confusion matrix and return matplotlib"s figure with it.
    Normalization can be applied by setting `normalize=True`.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.ioff()

    cmap = plt.cm.__dict__[colormap]

    if class_names is None:
        class_names = [str(i) for i in range(len(np.diag(cm)))]

    if normalize:
        cm = cm.astype(np.float32) / cm.sum(axis=1)[:, np.newaxis]

    plt.rcParams.update(
        {"font.size": int(fontsize / np.log2(len(class_names)))}
    )

    f = plt.figure(figsize=(figsize, figsize))
    plt.title(title)
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")

    plt.yticks(tick_marks, class_names)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    if fname is not None:
        plt.savefig(fname=fname)

    if show:
        plt.show()

    return f


def render_figure_to_tensor(figure):
    """@TODO: Docs. Contribution is welcome."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.ioff()

    figure.canvas.draw()

    image = np.array(figure.canvas.renderer._renderer)
    plt.close(figure)
    del figure

    image = tensor_from_rgb_image(image)
    return image


def plot_metrics(
    logdir: Union[str, Path],
    step: Optional[str] = "epoch",
    metrics: Optional[List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
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
    assert step in [
        "batch",
        "epoch",
    ], f"Step should be either 'batch' or 'epoch', got '{step}'"
    metrics = metrics or ["loss"]
    plot_tensorboard_log(logdir, step, metrics, height, width)


__all__ = ["plot_confusion_matrix", "render_figure_to_tensor", "plot_metrics"]
