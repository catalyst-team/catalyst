from typing import Callable, List, Tuple
from collections import defaultdict
import enum
from functools import partial

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

METRIC_FN = Callable[[np.ndarray, np.ndarray], float]


class ThresholdMode(str, enum.Enum):
    """Available threshold search strategies types."""

    NOOP = noop = "noop"  # noqa: WPS115
    MULTILABEL = multilabel = "multilabel"  # noqa: WPS115
    MULTICLASS = multiclass = "multiclass"  # noqa: WPS115


def get_baseline_thresholds(
    scores: np.ndarray, labels: np.ndarray, objective: METRIC_FN,
) -> Tuple[float, List[float]]:
    """Returns baseline thresholds for multiclass/multilabel classification.

    Args:
        scores: estimated per-class scores/probabilities predicted by the model,
            numpy array with shape [num_examples, num_classes]
        labels: ground truth labels,
            numpy array with shape [num_examples, num_classes]
        objective: callable function, metric which we want to maximize

    Returns:
        tuple with best found objective score and per-class thresholds
    """
    num_classes = scores.shape[1]
    thresholds = [0.5] * num_classes
    predictions = np.greater(scores, thresholds).astype(np.int32)
    best_metric = objective(labels, predictions)
    return best_metric, thresholds


def get_binary_threshold(
    scores: np.ndarray, labels: np.ndarray, objective: METRIC_FN, num_thresholds: int = 100,
) -> Tuple[float, float]:
    """Finds best threshold for binary classification task based on cross-validation estimates.

    Args:
        scores: estimated per-class scores/probabilities predicted by the model,
            numpy array with shape [num_examples, ]
        labels: ground truth labels, numpy array with shape [num_examples, ]
        objective: callable function, metric which we want to maximize
        num_thresholds: number of thresholds ot try for each class

    Returns:
        tuple with best found objective score and threshold
    """
    thresholds = np.linspace(scores.min(), scores.max(), num=num_thresholds)
    metric_values = []

    for threshold in thresholds:
        predictions = (scores >= threshold).astype(np.int32)
        if np.sum(predictions) > 0:
            metric_value = objective(labels, predictions)
            metric_values.append(metric_value)
        else:
            metric_values.append(0.0)

    if np.max(metric_values) == 0.0:
        best_metric_value = 0.0
        best_threshold = 1.0
    else:
        best_metric_value = metric_values[np.argmax(metric_values)]
        best_threshold = thresholds[np.argmax(metric_values)]

    return best_metric_value, best_threshold


def get_multiclass_thresholds(
    scores: np.ndarray, labels: np.ndarray, objective: METRIC_FN,
) -> Tuple[List[float], List[float]]:
    """Finds best thresholds for multiclass classification task.

    Args:
        scores: estimated per-class scores/probabilities predicted by the model,
            numpy array with shape [num_examples, num_classes]
        labels: ground truth labels, numpy array with shape [num_examples, num_classes]
        objective: callable function, metric which we want to maximize

    Returns:
        tuple with best found objective score and per-class thresholds
    """
    num_classes = scores.shape[1]
    metrics = [0.0] * num_classes
    thresholds = [0.0] * num_classes

    # score threshold -> classes with such score
    classes_by_threshold = defaultdict(list)
    for class_index in range(num_classes):
        for score in np.unique(scores[:, class_index]):
            classes_by_threshold[score].append(class_index)

    for threshold in sorted(classes_by_threshold):
        for class_index in classes_by_threshold[threshold]:
            metric_value = objective(labels[:, class_index], scores[:, class_index] >= threshold)
            if metric_value > metrics[class_index]:
                metrics[class_index] = metric_value
                thresholds[class_index] = threshold

    return metrics, thresholds


def get_multilabel_thresholds(
    scores: np.ndarray, labels: np.ndarray, objective: METRIC_FN,
):
    """Finds best thresholds for multilabel classification task.

    Args:
        scores: estimated per-class scores/probabilities predicted by the model,
            numpy array with shape [num_examples, num_classes]
        labels: ground truth labels, numpy array with shape [num_examples, num_classes]
        objective: callable function, metric which we want to maximize

    Returns:
        tuple with best found objective score and per-class thresholds
    """
    num_classes = labels.shape[1]
    metrics = [0.0] * num_classes
    thresholds = [0.0] * num_classes

    for class_index in range(num_classes):
        best_metric, best_threshold = get_binary_threshold(
            labels=labels[:, class_index], scores=scores[:, class_index], objective=objective,
        )
        metrics[class_index] = best_metric
        thresholds[class_index] = best_threshold

    return metrics, thresholds


def get_binary_threshold_cv(
    scores: np.ndarray,
    labels: np.ndarray,
    objective: METRIC_FN,
    num_splits: int = 5,
    num_repeats: int = 1,
    random_state: int = 42,
):
    """Finds best threshold for binary classification task
    based on cross-validation estimates.

    Args:
        scores: estimated per-class scores/probabilities predicted by the model,
            numpy array with shape [num_examples, ]
        labels: ground truth labels, numpy array with shape [num_examples, ]
        objective: callable function, metric which we want to maximize
        num_splits: number of splits to use for cross-validation
        num_repeats: number of repeats to use for cross-validation
        random_state: random state to use for cross-validation

    Returns:
        tuple with best found objective score and threshold
    """
    rkf = RepeatedStratifiedKFold(
        n_splits=num_splits, n_repeats=num_repeats, random_state=random_state
    )
    fold_metrics, fold_thresholds = [], []

    for train_index, valid_index in rkf.split(labels, labels):
        labels_train, labels_valid = labels[train_index], labels[valid_index]
        scores_train, scores_valid = scores[train_index], scores[valid_index]

        _, best_threshold = get_binary_threshold(
            labels=labels_train, scores=scores_train, objective=objective,
        )

        valid_predictions = (scores_valid >= best_threshold).astype(np.int32)
        best_metric_value = objective(labels_valid, valid_predictions)

        fold_metrics.append(best_metric_value)
        fold_thresholds.append(best_threshold)

    return np.mean(fold_metrics), np.mean(fold_thresholds)


def get_multilabel_thresholds_cv(
    scores: np.ndarray,
    labels: np.ndarray,
    objective: METRIC_FN,
    num_splits: int = 5,
    num_repeats: int = 1,
    random_state: int = 42,
):
    """Finds best thresholds for multilabel classification task
    based on cross-validation estimates.

    Args:
        scores: estimated per-class scores/probabilities predicted by the model,
            numpy array with shape [num_examples, num_classes]
        labels: ground truth labels, numpy array with shape [num_examples, num_classes]
        objective: callable function, metric which we want to maximize
        num_splits: number of splits to use for cross-validation
        num_repeats: number of repeats to use for cross-validation
        random_state: random state to use for cross-validation

    Returns:
        tuple with best found objective score and per-class thresholds
    """
    num_classes = labels.shape[1]
    metrics = [0.0] * num_classes
    thresholds = [0.0] * num_classes

    for class_index in range(num_classes):
        best_metric, best_threshold = get_binary_threshold_cv(
            labels=labels[:, class_index],
            scores=scores[:, class_index],
            objective=objective,
            num_splits=num_splits,
            num_repeats=num_repeats,
            random_state=random_state,
        )
        metrics[class_index] = best_metric
        thresholds[class_index] = best_threshold

    return metrics, thresholds


def get_thresholds_greedy(
    scores: np.ndarray,
    labels: np.ndarray,
    score_fn: Callable,
    num_iterations: int = 100,
    num_thresholds: int = 100,
    thresholds: np.ndarray = None,
    patience: int = 3,
    atol: float = 0.01,
) -> Tuple[float, List[float]]:
    """Finds best thresholds for classification task with brute-force algorithm.

    Args:
        scores: estimated per-class scores/probabilities predicted by the model
        labels: ground truth labels
        score_fn: callable function, based on (scores, labels, thresholds)
        num_iterations: number of iteration for brute-force algorithm
        num_thresholds: number of thresholds ot try for each class
        thresholds: baseline thresholds, which we want to optimize
        patience: maximum number of iteration before early stop exit
        atol: minimum required improvement per iteration for early stop exit

    Returns:
        tuple with best found objective score and per-class thresholds
    """
    num_classes = scores.shape[1]
    if thresholds is None:
        thresholds = [0.5] * num_classes
    best_metric = score_fn(scores, labels, thresholds)
    iteration_metrics = []

    for i in range(num_iterations):
        if len(iteration_metrics) >= patience:
            if best_metric < iteration_metrics[i - patience] + atol:
                break

        for class_index in range(num_classes):
            current_thresholds = thresholds.copy()
            class_scores = []
            class_thresholds = np.linspace(
                scores[:, class_index].min(), scores[:, class_index].max(), num=num_thresholds,
            )

            for threshold in class_thresholds:
                current_thresholds[class_index] = threshold
                class_score = score_fn(scores, labels, current_thresholds)
                class_scores.append(class_score)

            best_class_score = np.max(class_scores)
            best_score_index = np.argmax(class_scores)
            if best_class_score > best_metric:
                best_metric = best_class_score
                thresholds[class_index] = class_thresholds[best_score_index]
        iteration_metrics.append(best_metric)

    return best_metric, thresholds


def _multilabel_score_fn(scores, labels, thresholds, objective):
    predictions = np.greater(scores, thresholds).astype(np.int32)
    return objective(labels, predictions)


def get_multilabel_thresholds_greedy(
    scores: np.ndarray,
    labels: np.ndarray,
    objective: METRIC_FN,
    num_iterations: int = 100,
    num_thresholds: int = 100,
    thresholds: np.ndarray = None,
    patience: int = 3,
    atol: float = 0.01,
) -> Tuple[float, List[float]]:
    """Finds best thresholds for multilabel classification task with brute-force algorithm.

    Args:
        scores: estimated per-class scores/probabilities predicted by the model
        labels: ground truth labels
        objective: callable function, metric which we want to maximize
        num_iterations: number of iteration for brute-force algorithm
        num_thresholds: number of thresholds ot try for each class
        thresholds: baseline thresholds, which we want to optimize
        patience: maximum number of iteration before early stop exit
        atol: minimum required improvement per iteration for early stop exit

    Returns:
        tuple with best found objective score and per-class thresholds
    """
    best_metric, thresholds = get_thresholds_greedy(
        scores=scores,
        labels=labels,
        score_fn=partial(_multilabel_score_fn, objective=objective),
        num_iterations=num_iterations,
        num_thresholds=num_thresholds,
        thresholds=thresholds,
        patience=patience,
        atol=atol,
    )

    return best_metric, thresholds


def _multiclass_score_fn(scores, labels, thresholds, objective):
    scores_copy = scores.copy()
    scores_copy[np.less(scores, thresholds)] = 0
    predictions = scores_copy.argmax(axis=1)
    return objective(labels, predictions)


def get_multiclass_thresholds_greedy(
    scores: np.ndarray,
    labels: np.ndarray,
    objective: METRIC_FN,
    num_iterations: int = 100,
    num_thresholds: int = 100,
    thresholds: np.ndarray = None,
    patience: int = 3,
    atol: float = 0.01,
) -> Tuple[float, List[float]]:
    """Finds best thresholds for multiclass classification task with brute-force algorithm.

    Args:
        scores: estimated per-class scores/probabilities predicted by the model
        labels: ground truth labels
        objective: callable function, metric which we want to maximize
        num_iterations: number of iteration for brute-force algorithm
        num_thresholds: number of thresholds ot try for each class
        thresholds: baseline thresholds, which we want to optimize
        patience: maximum number of iteration before early stop exit
        atol: minimum required improvement per iteration for early stop exit

    Returns:
        tuple with best found objective score and per-class thresholds
    """
    best_metric, thresholds = get_thresholds_greedy(
        scores=scores,
        labels=labels,
        score_fn=partial(_multiclass_score_fn, objective=objective),
        num_iterations=num_iterations,
        num_thresholds=num_thresholds,
        thresholds=thresholds,
        patience=patience,
        atol=atol,
    )

    return best_metric, thresholds


def get_best_multilabel_thresholds(
    scores: np.ndarray, labels: np.ndarray, objective: METRIC_FN,
) -> Tuple[float, List[float]]:
    """Finds best thresholds for multilabel classification task.

    Args:
        scores: estimated per-class scores/probabilities predicted by the model
        labels: ground truth labels
        objective: callable function, metric which we want to maximize

    Returns:
        tuple with best found objective score and per-class thresholds
    """
    num_classes = scores.shape[1]
    best_metric, best_thresholds = 0.0, []

    for baseline_thresholds_fn in [
        get_baseline_thresholds,
        get_multiclass_thresholds,
        get_binary_threshold,
        get_multilabel_thresholds,
    ]:
        _, baseline_thresholds = baseline_thresholds_fn(
            labels=labels, scores=scores, objective=objective,
        )
        if isinstance(baseline_thresholds, (int, float)):
            baseline_thresholds = [baseline_thresholds] * num_classes
        metric_value, thresholds_value = get_multilabel_thresholds_greedy(
            labels=labels, scores=scores, objective=objective, thresholds=baseline_thresholds,
        )
        if metric_value > best_metric:
            best_metric = metric_value
            best_thresholds = thresholds_value

    return best_metric, best_thresholds


def get_best_multiclass_thresholds(
    scores: np.ndarray, labels: np.ndarray, objective: METRIC_FN,
) -> Tuple[float, List[float]]:
    """Finds best thresholds for multiclass classification task.

    Args:
        scores: estimated per-class scores/probabilities predicted by the model
        labels: ground truth labels
        objective: callable function, metric which we want to maximize

    Returns:
        tuple with best found objective score and per-class thresholds
    """
    num_classes = scores.shape[1]
    best_metric, best_thresholds = 0.0, []
    labels_onehot = np.zeros((labels.size, labels.max() + 1))
    labels_onehot[np.arange(labels.size), labels] = 1

    for baseline_thresholds_fn in [
        get_baseline_thresholds,
        get_multiclass_thresholds,
        get_binary_threshold,
        get_multilabel_thresholds,
    ]:
        _, baseline_thresholds = baseline_thresholds_fn(
            labels=labels_onehot, scores=scores, objective=objective,
        )
        if isinstance(baseline_thresholds, (int, float)):
            baseline_thresholds = [baseline_thresholds] * num_classes
        metric_value, thresholds_value = get_multiclass_thresholds_greedy(
            labels=labels, scores=scores, objective=objective, thresholds=baseline_thresholds,
        )
        if metric_value > best_metric:
            best_metric = metric_value
            best_thresholds = thresholds_value

    return best_metric, best_thresholds


__all__ = [
    "get_baseline_thresholds",
    "get_binary_threshold",
    "get_multiclass_thresholds",
    "get_multilabel_thresholds",
    "get_binary_threshold_cv",
    "get_multilabel_thresholds_cv",
    "get_thresholds_greedy",
    "get_multilabel_thresholds_greedy",
    "get_multiclass_thresholds_greedy",
    "get_best_multilabel_thresholds",
    "get_best_multiclass_thresholds",
]
