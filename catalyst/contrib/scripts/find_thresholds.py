from typing import Callable, Dict, List, Tuple
from path import Path
import json
import argparse
import numpy as np
import pandas as pd
from pprint import pprint

from scipy.special import expit
from sklearn import metrics
from sklearn.model_selection import RepeatedStratifiedKFold

from catalyst.utils import boolean_flag

BINARY_PER_CLASS_METRICS = [
    "accuracy_score", "precision_score", "recall_score",
    "f1_score", "roc_auc_score",
]

RANK_METRICS = [
    "ndcg_score", "coverage_error",
    "label_ranking_loss", "label_ranking_average_precision_score",
]


def build_args(parser):
    parser.add_argument(
        "--in-csv", type=Path,
        help="Path to .csv with labels column", required=True
    )
    parser.add_argument(
        "--in-label-column", type=str,
        help="Column to get labels", required=False, default="labels",
    )
    parser.add_argument(
        "--in-npy", type=Path,
        help="Path to .npy with class logits", required=True
    )
    parser.add_argument(
        "--out-thresholds", type=Path,
        help="Path to save .json with thresholds", required=True
    )

    parser.add_argument(
        "--metric", type=str,
        help="Metric to use ['f1_score', 'roc_auc_score']", required=False,
        default="roc_auc_score"
    )
    parser.add_argument(
        "--ignore-label", type=int,
        required=False,
        default=None
    )
    parser.add_argument(
        "--num-splits", type=int,
        help="NUM_SPLITS", required=False,
        default=2
    )
    parser.add_argument(
        "--num-repeats", type=int,
        help="NUM_REPEATS", required=False,
        default=1
    )
    parser.add_argument(
        "--num-workers", type=int,
        help="CPU pool size", required=False,
        default=1
    )

    boolean_flag(parser, "verbose", default=False)
    boolean_flag(parser, "use-sigmoid", default=False)

    return parser


def parse_args():
    parser = argparse.ArgumentParser()
    build_args(parser)
    args = parser.parse_args()
    return args


def get_label_for_class(
    labels: np.array,
    cls: int,
    ignore_label: int = None
):
    is_correct = labels == cls
    if ignore_label is not None:
        is_correct[labels == ignore_label] = 0
    return (is_correct).astype(int)


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
        n_splits=num_splits,
        n_repeats=num_repeats,
        random_state=random_state
    )
    fold_thresholds = []
    fold_metrics = {k: [] for k in BINARY_PER_CLASS_METRICS}

    for train_index, test_index in rkf.split(y_true, y_true):
        y_pred_train, y_pred_test = y_pred[train_index], y_pred[test_index]
        y_true_train, y_true_test = y_true[train_index], y_true[test_index]

        best_threshold = find_best_split_threshold(
            y_pred_train,
            y_true_train,
            metric=metric_fn
        )
        best_predictions = (y_pred_test >= best_threshold).astype(int)

        for metric_name in BINARY_PER_CLASS_METRICS:
            try:
                metric_value = metrics.__dict__[metric_name](
                    y_true_test,
                    best_predictions
                )
            except ValueError:
                metric_value = 0.

            fold_metrics[metric_name].append(metric_value)
        fold_thresholds.append(best_threshold)

    fold_best_threshold = np.mean(fold_thresholds)
    for metric_name in fold_metrics:
        fold_metrics[metric_name] = np.mean(fold_metrics[metric_name])

    return fold_best_threshold, fold_metrics


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
    predictions_ = predictions.copy()

    predictions_list, labels_list = [], []
    for cls in classes:
        predictions_list.append(predictions_[:, cls])
        labels_list.append(
            get_label_for_class(
                labels, cls, ignore_label=ignore_label
            ))

    results = [
        find_best_threshold(
            predictions_list_,
            labels_list_,
            metric_fn,
            num_splits,
            num_repeats
        ) for predictions_list_, labels_list_ in \
            zip(predictions_list, labels_list)
    ]

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

    # candidates = np.argsort(-confidences_th, axis=1)
    # confidences_th = -np.sort(-confidences_th, axis=1)
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


def main(args, _=None):
    predictions = expit(np.load(args.in_npy))
    if args.use_sigmoid:
        predictions = expit(predictions)
    labels = pd.read_csv(args.in_csv)[args.in_label_column].values
    classes = list(set(labels) - set([args.ignore_label]))

    assert args.metric in metrics.__dict__.keys()
    metric_fn = metrics.__dict__[args.metric]

    class_thresholds, class_metrics = optimize_thresholds(
        predictions=predictions,
        labels=labels,
        classes=classes,
        metric_fn=metric_fn,
        num_splits=args.num_splits,
        num_repeats=args.num_repeats,
        ignore_label=args.ignore_label,
        num_workers=args.num_workers
    )
    class_thresholds = _sort_dict_by_keys(class_thresholds)

    with open(args.out_thresholds, "w") as fout:
        class_thresholds_ = {str(k): v for k, v in class_thresholds.items()}
        json.dump(class_thresholds_, fout, ensure_ascii=False, indent=4)

    class_metrics["_mean"] = {
        key_metric: np.mean([
            class_metrics[key_class][key_metric]
            for key_class in class_metrics.keys()
        ])
        for key_metric in BINARY_PER_CLASS_METRICS
    }

    out_metrics = args.out_thresholds.replace(".json", ".class.metrics.json")
    class_metrics = _sort_dict_by_keys(
        {str(k): v for k, v in class_metrics.items()}
    )
    with open(out_metrics, "w") as fout:
        json.dump(class_metrics, fout, ensure_ascii=False, indent=4)

    if args.verbose:
        print("CLASS METRICS")
        pprint(class_metrics)
        print("CLASS THRESHOLDS")
        pprint(class_thresholds)

    labels_scores = np.zeros(predictions.shape)
    labels_scores[:,labels] = 1.0
    for class_thresholds_ in [None, class_thresholds]:
        thresolds_used = class_thresholds_ is not None

        confidences = get_model_confidences(
            confidences=predictions,
            thresholds=class_thresholds_,
            classes=classes,
        )

        rank_metrics = {
            key: metrics.__dict__[key](labels_scores, confidences)
            for key in RANK_METRICS
        }
        out_metrics = args.out_thresholds.replace(
            ".json",
            ".rank.metrics.json" \
                if not thresolds_used \
                else ".rank.metrics.thresholds.json"
        )
        rank_metrics = _sort_dict_by_keys(
            {str(k): v for k, v in rank_metrics.items()}
        )
        with open(out_metrics, "w") as fout:
            json.dump(rank_metrics, fout, ensure_ascii=False, indent=4)

        coverage_metrics = score_model_coverage(confidences, labels)
        out_metrics = args.out_thresholds.replace(
            ".json",
            ".coverage.metrics.json" \
                if not thresolds_used \
                else ".rank.coverage.thresholds.json"
        )
        coverage_metrics = _sort_dict_by_keys(
            {str(k): v for k, v in coverage_metrics.items()}
        )
        with open(out_metrics, "w") as fout:
            json.dump(coverage_metrics, fout, ensure_ascii=False, indent=4)

        if args.verbose:
            print(
                "RANK METRICS"
                if not thresolds_used
                else "RANK METRICS WITH THRESHOLD"
            )
            pprint(rank_metrics)
            print(
                "COVERAGE METRICS"
                if not thresolds_used
                else "COVERAGE METRICS WITH THRESHOLD"
            )
            pprint(coverage_metrics)


if __name__ == "__main__":
    args = parse_args()
    main(args)
