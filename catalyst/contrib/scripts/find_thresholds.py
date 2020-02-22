from typing import Any, Callable, Dict, List, Tuple  # isort:skip
import argparse
from itertools import repeat
import json
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn import metrics
from sklearn.model_selection import RepeatedStratifiedKFold

from catalyst import utils

BINARY_PER_CLASS_METRICS = [
    "accuracy_score",
    "precision_score",
    "recall_score",
    "f1_score",
    "roc_auc_score",
]

RANK_METRICS = [
    "ndcg_score",
    "coverage_error",
    "label_ranking_loss",
    "label_ranking_average_precision_score",
]


def build_args(parser):
    parser.add_argument(
        "--in-csv",
        type=Path,
        help="Path to .csv with labels column",
        required=True
    )
    parser.add_argument(
        "--in-label-column",
        type=str,
        help="Column to get labels",
        required=False,
        default="labels",
    )
    parser.add_argument(
        "--in-npy",
        type=Path,
        help="Path to .npy with class logits",
        required=True
    )
    parser.add_argument(
        "--out-thresholds",
        type=Path,
        help="Path to save .json with thresholds",
        required=True
    )

    parser.add_argument(
        "--metric",
        type=str,
        help="Metric to use",
        required=False,
        choices=BINARY_PER_CLASS_METRICS,
        default="roc_auc_score"
    )
    # parser.add_argument(
    #     "--ignore-label", type=int,
    #     required=False,
    #     default=None
    # )
    parser.add_argument(
        "--num-splits", type=int, help="NUM_SPLITS", required=False, default=5
    )
    parser.add_argument(
        "--num-repeats",
        type=int,
        help="NUM_REPEATS",
        required=False,
        default=1
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="CPU pool size",
        required=False,
        default=1
    )

    utils.boolean_flag(parser, "verbose", default=False)
    utils.boolean_flag(parser, "sigmoid", default=False)

    return parser


def parse_args():
    parser = argparse.ArgumentParser()
    build_args(parser)
    args = parser.parse_args()
    return args


def get_binary_labels(labels: np.array, label: int, ignore_label: int = None):
    binary_labels = labels == label
    if ignore_label is not None:
        binary_labels[labels == ignore_label] = 0
    return (binary_labels).astype(int)


def find_best_split_threshold(
    y_pred: np.array,
    y_true: np.array,
    metric: Callable,
):
    thresholds = np.linspace(0.0, 1.0, num=100)
    metric_values = []
    for t in thresholds:
        predictions = (y_pred >= t).astype(int)
        if sum(predictions) > 0:
            metric_values.append(metric(y_true, predictions))
        else:
            metric_values.append(0.)

    best_threshold = thresholds[np.argmax(metric_values)]
    return best_threshold


def find_best_threshold(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    metric_fn: Callable = metrics.roc_auc_score,
    num_splits: int = 5,
    num_repeats: int = 1,
    random_state: int = 42,
):
    rkf = RepeatedStratifiedKFold(
        n_splits=num_splits, n_repeats=num_repeats, random_state=random_state
    )
    fold_thresholds = []
    fold_metrics = {k: [] for k in BINARY_PER_CLASS_METRICS}

    for train_index, test_index in rkf.split(y_true, y_true):
        y_pred_train, y_pred_test = y_pred[train_index], y_pred[test_index]
        y_true_train, y_true_test = y_true[train_index], y_true[test_index]

        best_threshold = find_best_split_threshold(
            y_pred_train, y_true_train, metric=metric_fn
        )
        best_predictions = (y_pred_test >= best_threshold).astype(int)

        for metric_name in BINARY_PER_CLASS_METRICS:
            try:
                metric_value = metrics.__dict__[metric_name](
                    y_true_test, best_predictions
                )
            except ValueError:
                metric_value = 0.

            fold_metrics[metric_name].append(metric_value)
        fold_thresholds.append(best_threshold)

    fold_best_threshold = np.mean(fold_thresholds)
    for metric_name in fold_metrics:
        fold_metrics[metric_name] = np.mean(fold_metrics[metric_name])

    return fold_best_threshold, fold_metrics


def wrap_find_best_threshold(args: Tuple[Any]):
    class_id, function_args = args[0], args[1:]
    threshold, metrics = find_best_threshold(*function_args)
    return class_id, threshold, metrics


def optimize_thresholds(
    predictions: np.ndarray,
    labels: np.ndarray,
    classes: List[int],
    metric_fn: Callable = metrics.roc_auc_score,
    num_splits: int = 5,
    num_repeats: int = 1,
    num_workers: int = 0,
    ignore_label: int = None
) -> Tuple[Dict, Dict]:
    pool = utils.get_pool(num_workers)

    predictions_ = predictions.copy()

    predictions_list, labels_list = [], []
    for cls in classes:
        predictions_list.append(predictions_[:, cls])
        labels_list.append(
            get_binary_labels(labels, cls, ignore_label=ignore_label)
        )

    results = utils.tqdm_parallel_imap(
        wrap_find_best_threshold,
        zip(
            classes,
            predictions_list,
            labels_list,
            repeat(metric_fn),
            repeat(num_splits),
            repeat(num_repeats),
        ), pool
    )
    results = [(r[1], r[2]) for r in sorted(results, key=lambda x: x[0])]

    result_thresholds = [r[0] for r in results]
    result_metrics = [r[1] for r in results]
    class_thresholds = {c: t for (c, t) in zip(classes, result_thresholds)}
    class_metrics = {c: m for (c, m) in zip(classes, result_metrics)}
    return class_thresholds, class_metrics


def get_model_confidences(
    confidences: np.ndarray,
    thresholds: Dict[int, float] = None,
    classes: List[int] = None,
):
    """
    Args:
        confidences (np.ndarray): model predictions of shape
            [dataset_len; class_confidences]
        thresholds (Dict[int, float]): thresholds for each class
        classes (List[int]): classes of interest for evaluation
    """
    if classes is not None:
        classes = np.array(classes)
        confidences = confidences[:, classes]

    confidences_th = confidences.copy()
    if thresholds is not None:
        assert confidences.shape[1] == len(thresholds)
        thresholds = np.array(list(thresholds.values()))
        confidences_th = confidences - thresholds

    return confidences_th


def score_model_coverage(
    confidences: np.ndarray,
    labels: np.ndarray,
):
    candidates = np.argsort(-confidences, axis=1)
    confidences = -np.sort(-confidences, axis=1)
    candidates[confidences < 0] = -1
    labels = labels[:, None]

    coverage_metrics = {}

    for top_k in [1, 3, 5]:
        metric = (candidates[:, :top_k] == labels).sum(axis=1).mean()
        coverage_metrics[f"Recall@{top_k:02d}"] = metric

    return coverage_metrics


def _sort_dict_by_keys(disordered: Dict):
    key = lambda item: item[0]
    sorted_dict = {k: v for k, v in sorted(disordered.items(), key=key)}
    return sorted_dict


def _save_json(dct: Dict, outpath: Path, suffix: str = None):
    outpath = str(outpath)
    if suffix is not None:
        outpath = outpath.replace(".json", f"{suffix}.json")
    dct = _sort_dict_by_keys({str(k): v for k, v in dct.copy().items()})
    with open(outpath, "w") as fout:
        json.dump(dct, fout, ensure_ascii=False, indent=4)


def main(args, _=None):
    predictions = expit(np.load(args.in_npy))
    if args.sigmoid:
        predictions = expit(predictions)
    labels = pd.read_csv(args.in_csv)[args.in_label_column].values
    classes = list(set(labels))  # - set([args.ignore_label]))

    assert args.metric in metrics.__dict__.keys()
    metric_fn = metrics.__dict__[args.metric]

    class_thresholds, class_metrics = optimize_thresholds(
        predictions=predictions,
        labels=labels,
        classes=classes,
        metric_fn=metric_fn,
        num_splits=args.num_splits,
        num_repeats=args.num_repeats,
        ignore_label=None,  # args.ignore_label,
        num_workers=args.num_workers
    )
    _save_json(class_thresholds, outpath=args.out_thresholds)

    class_metrics["_mean"] = {
        key_metric: np.mean(
            [
                class_metrics[key_class][key_metric]
                for key_class in class_metrics.keys()
            ]
        )
        for key_metric in BINARY_PER_CLASS_METRICS
    }

    _save_json(class_metrics, args.out_thresholds, suffix=".class.metrics")

    if args.verbose:
        print("CLASS METRICS")
        pprint(class_metrics)
        print("CLASS THRESHOLDS")
        pprint(class_thresholds)

    labels_scores = np.zeros(predictions.shape)
    labels_scores[:, labels] = 1.0
    for class_thresholds_ in [None, class_thresholds]:
        thresholds_used = class_thresholds_ is not None

        confidences = get_model_confidences(
            confidences=predictions,
            thresholds=class_thresholds_,
            classes=classes,
        )

        rank_metrics = {
            key: metrics.__dict__[key](labels_scores, confidences)
            for key in RANK_METRICS
        }
        postfix = (
            ".rank.metrics"
            if not thresholds_used else ".rank.metrics.thresholds"
        )
        _save_json(rank_metrics, args.out_thresholds, suffix=postfix)

        coverage_metrics = score_model_coverage(confidences, labels)
        postfix = (
            ".coverage.metrics.json"
            if not thresholds_used else ".coverage.metrics.thresholds.json"
        )
        _save_json(coverage_metrics, args.out_thresholds, suffix=postfix)

        if args.verbose:
            print(
                "RANK METRICS"
                if not thresholds_used else "RANK METRICS WITH THRESHOLD"
            )
            pprint(rank_metrics)
            print(
                "COVERAGE METRICS"
                if not thresholds_used else "COVERAGE METRICS WITH THRESHOLD"
            )
            pprint(coverage_metrics)


if __name__ == "__main__":
    args = parse_args()
    main(args)
