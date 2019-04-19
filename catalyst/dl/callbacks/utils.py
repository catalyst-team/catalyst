from typing import List
import itertools
import numpy as np
import torch
from skimage.color import label2rgb


def get_val_from_metric(metric_value):
    if isinstance(metric_value, (int, float)):
        pass
    elif torch.is_tensor(metric_value):
        metric_value = metric_value.item()
    else:
        metric_value = metric_value.value()
        if isinstance(metric_value, (tuple, list)):
            metric_value = metric_value[0]
        if torch.is_tensor(metric_value):
            metric_value = metric_value.item()
    return metric_value


def process_epoch_metrics(
    epoch_metrics,
    best_metrics,
    valid_loader="valid",
    main_metric="loss",
    minimize=True
):
    valid_metrics = epoch_metrics[valid_loader]
    is_best = True \
        if best_metrics is None \
        else (minimize != (
            valid_metrics[main_metric] > best_metrics[main_metric]))
    best_metrics = valid_metrics if is_best else best_metrics
    return best_metrics, valid_metrics, is_best


def to_batch_metrics(*, state, metric_key, state_key=None):
    metric = state.get_key(state_key or metric_key)
    if isinstance(metric, dict):
        for key, value in metric.items():
            state.batch_metrics[f"{metric_key}_{key}"] = \
                get_val_from_metric(value)
    else:
        state.batch_metrics[f"{metric_key}"] = \
            get_val_from_metric(metric)


def get_optimizer_momentum(optimizer):
    if isinstance(optimizer, torch.optim.Adam):
        return list(optimizer.param_groups)[0]["betas"][0]
    elif isinstance(optimizer, torch.optim.SGD):
        return list(optimizer.param_groups)[0]["momentum"]
    else:
        return None


def scheduler_step(scheduler, valid_metric=None):
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(valid_metric)
        lr = list(scheduler.optimizer.param_groups)[0]["lr"]
    else:
        scheduler.step()
        lr = scheduler.get_lr()[0]

    momentum = get_optimizer_momentum(scheduler.optimizer)

    return lr, momentum


def binary_mask_to_overlay_image(image: np.ndarray, masks: List[np.ndarray]):
    """Draws every mask for with some color over image"""
    h, w = image.shape[:2]
    labels = np.zeros((h, w), np.uint8)

    for idx, mask in enumerate(masks):
        labels[mask > 0] = idx + 1

    image_with_overlay = label2rgb(labels, image)

    image_with_overlay = (image_with_overlay * 255).round().astype(np.uint8)
    return image_with_overlay


def tensor_from_rgb_image(image: np.ndarray) -> torch.Tensor:
    image = np.moveaxis(image, -1, 0)
    image = np.ascontiguousarray(image)
    image = torch.from_numpy(image)
    return image


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

    plt.rcParams.update({"font.size": int(fontsize/np.log2(len(class_names)))})

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
            j, i, format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

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
