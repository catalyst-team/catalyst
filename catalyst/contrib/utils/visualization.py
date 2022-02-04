import itertools

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names=None,
    normalize=False,
    title="confusion matrix",
    fname=None,
    show=True,
    figsize=12,
    fontsize=32,
    colormap="Blues",
):
    """Render the confusion matrix and return matplotlib"s figure with it.
    Normalization can be applied by setting `normalize=True`.

    Args:
        cm: numpy confusion matrix
        class_names: class names
        normalize: boolean flag to normalize confusion matrix
        title: title
        fname: filename to save confusion matrix
        show: boolean flag for preview
        figsize: matplotlib figure size
        fontsize: matplotlib font size
        colormap: matplotlib color map

    Returns:
        matplotlib figure
    """
    plt.ioff()

    cmap = plt.cm.__dict__[colormap]

    if class_names is None:
        class_names = [str(i) for i in range(len(np.diag(cm)))]

    if normalize:
        cm = cm.astype(np.float32) / cm.sum(axis=1)[:, np.newaxis]

    plt.rcParams.update({"font.size": int(fontsize / np.log2(len(class_names)))})

    figure = plt.figure(figsize=(figsize, figsize))
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

    plt.ion()
    return figure


def render_figure_to_array(figure):
    """Renders matplotlib"s figure to tensor."""
    plt.ioff()

    figure.canvas.draw()
    image = np.array(figure.canvas.renderer._renderer)
    plt.close(figure)
    del figure

    plt.ion()
    return image


__all__ = [
    "plot_confusion_matrix",
    "render_figure_to_array",
]
