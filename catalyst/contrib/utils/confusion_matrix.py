# flake8: noqa
# @TODO: code formatting issue for 20.07 release
import numpy as np

import torch


def calculate_tp_fp_fn(confusion_matrix: np.ndarray) -> np.ndarray:
    """@TODO: Docs. Contribution is welcome."""
    true_positives = np.diag(confusion_matrix)
    false_positives = confusion_matrix.sum(axis=0) - true_positives
    false_negatives = confusion_matrix.sum(axis=1) - true_positives
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def calculate_confusion_matrix_from_arrays(
    predictions: np.ndarray, labels: np.ndarray, num_classes: int
) -> np.ndarray:
    """Calculate confusion matrix for a given set of classes.
    If labels value is outside of the [0, num_classes) it is excluded.

    Args:
        predictions (np.ndarray): model predictions
        labels (np.ndarray): ground truth labels
        num_classes (int): number of classes

    Returns:
        np.ndarray: confusion matrix
    """
    # @TODO: add `num_class`=None handling
    # a long 2xn array with each column being a pixel pair
    replace_indices = np.vstack((labels.flatten(), predictions.flatten()))

    valid_index = replace_indices[0, :] < num_classes
    replace_indices = replace_indices[:, valid_index].T

    # add up confusion matrix
    confusion_matrix, _ = np.histogramdd(
        replace_indices,
        bins=(num_classes, num_classes),
        range=[(0, num_classes), (0, num_classes)],
    )
    return confusion_matrix.astype(np.uint64)


def calculate_confusion_matrix_from_tensors(
    y_pred_logits: torch.Tensor, y_true: torch.Tensor
) -> np.ndarray:
    """
    Calculate confusion matrix from tensors.

    Args:
        y_pred_logits: model logits
        y_true: true labels

    Returns:
        np.ndarray: confusion matrix
    """
    num_classes = y_pred_logits.shape[1]
    y_pred = torch.argmax(y_pred_logits, dim=1)
    predictions = y_pred.cpu().numpy()
    labels = y_true.cpu().numpy()

    return calculate_confusion_matrix_from_arrays(
        predictions, labels, num_classes
    )


__all__ = [
    "calculate_tp_fp_fn",
    "calculate_confusion_matrix_from_arrays",
    "calculate_confusion_matrix_from_tensors",
]
