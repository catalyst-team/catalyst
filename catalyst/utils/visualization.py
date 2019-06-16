import itertools
import numpy as np

from .image import tensor_from_rgb_image


def plot_confusion_matrix(
    cm,
    class_names=None,
    normalize=False,
    title="confusion matrix",
    fname=None,
    show=True,
    figsize=12,
    fontsize=32,
    colormap="Blues"
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
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black"
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
