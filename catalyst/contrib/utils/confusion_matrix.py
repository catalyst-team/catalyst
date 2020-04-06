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
    ground_truth: np.ndarray, prediction: np.ndarray, num_classes: int
) -> np.ndarray:
    """Calculate confusion matrix for a given set of classes.
    If GT value is outside of the [0, num_classes) it is excluded.

    Args:
        ground_truth (np.ndarray):
        prediction (np.ndarray):
        num_classes (int):

    @TODO: Docs . Contribution is welcome
    """
    # a long 2xn array with each column being a pixel pair
    replace_indices = np.vstack((ground_truth.flatten(), prediction.flatten()))

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
    """@TODO: Docs. Contribution is welcome."""
    num_classes = y_pred_logits.shape[1]
    y_pred = torch.argmax(y_pred_logits, dim=1)
    ground_truth = y_true.cpu().numpy()
    prediction = y_pred.cpu().numpy()

    return calculate_confusion_matrix_from_arrays(
        ground_truth, prediction, num_classes
    )


__all__ = [
    "calculate_tp_fp_fn",
    "calculate_confusion_matrix_from_arrays",
    "calculate_confusion_matrix_from_tensors",
]
