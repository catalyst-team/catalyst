import itertools

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# def tensor_to_ndimage(
#     images: torch.Tensor,
#     denormalize: bool = True,
#     mean: Tuple[float, float, float] = _IMAGENET_MEAN,
#     std: Tuple[float, float, float] = _IMAGENET_STD,
#     move_channels_dim: bool = True,
#     dtype=np.float32,
# ) -> np.ndarray:
#     """
#     Convert float image(s) with standard normalization to
#     np.ndarray with [0..1] when dtype is np.float32 and [0..255]
#     when dtype is `np.uint8`.
#
#     Args:
#         images: [B]xCxHxW float tensor
#         denormalize: if True, multiply image(s) by std and add mean
#         mean (Tuple[float, float, float]): per channel mean to add
#         std (Tuple[float, float, float]): per channel std to multiply
#         move_channels_dim: if True, convert tensor to [B]xHxWxC format
#         dtype: result ndarray dtype. Only float32 and uint8 are supported
#
#     Returns:
#         [B]xHxWxC np.ndarray of dtype
#     """
#     if denormalize:
#         has_batch_dim = len(images.shape) == 4
#
#         mean = images.new_tensor(mean).view(*((1) if has_batch_dim else ()), len(mean), 1, 1)
#         std = images.new_tensor(std).view(*((1) if has_batch_dim else ()), len(std), 1, 1)
#
#         images = images * std + mean
#
#     images = images.clamp(0, 1).numpy()
#
#     if move_channels_dim:
#         images = np.moveaxis(images, -3, -1)
#
#     if dtype == np.uint8:
#         images = (images * 255).round().astype(dtype)
#     else:
#         assert dtype == np.float32, "Only float32 and uint8 are supported"
#
#     return images


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


# def render_figure_to_numpy(figure):
#     """@TODO: Docs. Contribution is welcome."""
#     import matplotlib
#
#     matplotlib.use("Agg")
#     import matplotlib.pyplot as plt
#
#     plt.ioff()
#
#     figure.canvas.draw()
#
#     image = np.array(figure.canvas.renderer._renderer)  # noqa: WPS437
#     plt.close(figure)
#     del figure
#
#     return image


def render_figure_to_array(figure):
    """Renders matplotlib"s figure to tensor."""
    plt.ioff()

    figure.canvas.draw()

    image = np.array(figure.canvas.renderer._renderer)  # noqa: WPS437
    plt.close(figure)
    del figure

    plt.ion()
    return image


__all__ = [
    "plot_confusion_matrix",
    "render_figure_to_array",
]
