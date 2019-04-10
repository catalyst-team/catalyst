from typing import List

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
