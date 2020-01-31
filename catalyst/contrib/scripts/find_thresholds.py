from typing import Callable, Dict, List, Tuple
from path import Path
import json
import argparse
import numpy as np
import pandas as pd
from pathos.multiprocessing import ProcessingPool as Pool
from itertools import repeat
from pprint import pprint

from scipy.special import expit
from sklearn import metrics
from sklearn.model_selection import RepeatedStratifiedKFold

from catalyst.utils import boolean_flag

BINARY_METRICS = [
    "f1_score", "roc_auc_score", "precision_score", "recall_score"
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

    return parser


def parse_args():
    parser = argparse.ArgumentParser()
    build_args(parser)
    args = parser.parse_args()
    return args


def get_label_for_class(
    labels: np.array,
    cls: int,
    ignore_label: bool = None
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
    fold_metrics = {k: [] for k in BINARY_METRICS}

    for train_index, test_index in rkf.split(y_true, y_true):
        y_pred_train, y_pred_test = y_pred[train_index], y_pred[test_index]
        y_true_train, y_true_test = y_true[train_index], y_true[test_index]

        best_threshold = find_best_split_threshold(
            y_pred_train,
            y_true_train,
            metric=metric_fn
        )
        best_predictions = (y_pred_test >= best_threshold).astype(int)

        for metric_name in BINARY_METRICS:
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

    # pool = Pool(min(num_workers, len(classes)))
    predictions_list, labels_list = [], []
    for cls in classes:
        predictions_list.append(predictions[:, cls])
        labels_list.append(
            get_label_for_class(
                labels, cls, ignore_label=ignore_label
            ))

    # results = pool.map(
    #     find_best_threshold,
    #     predictions_list,
    #     labels_list,
    #     repeat(metric_fn),
    #     repeat(num_splits),
    #     repeat(num_repeats),
    # )
    results = [
        find_best_threshold(
            predictions_list_,
            labels_list_,
            metric_fn,
            num_splits,
            num_repeats
        ) for predictions_list_, labels_list_ in zip(predictions_list, labels_list)
    ]

    thresholds = [r[0] for r in results]
    metrics = [r[1] for r in results]
    class_thresholds = {c: t for (c, t) in zip(classes, thresholds)}
    class_metrics = {c: m for (c, m) in zip(classes, metrics)}
    return class_thresholds, class_metrics


def get_model_predictions(
    predictions: np.ndarray,
    thresholds: Dict[int, float],
    classes: List[int],
    top_k: int = 3,
    use_threshold: bool = False
):
    classes = np.array(classes)
    predictions = predictions[:, classes]
    assert predictions.shape[1] == len(thresholds)
    thresholds = np.array(list(thresholds.values()))

    predictions_th = predictions - thresholds
    args_th = np.argsort(-predictions_th, axis=1)
    predictions_th = -np.sort(-predictions_th, axis=1)

    model_predictions = []
    for i in range(predictions.shape[0]):
        candidates = (
            args_th[i, :][np.nonzero(predictions_th[i, :] > 0)]
            if use_threshold
            else args_th[i, :]
        )
        model_predictions.append(list(classes[candidates[:top_k]]))

    return model_predictions


def score_model_predictions(
    predictions: List[int],
    labels: List[int]
):
    overall_metrics = {
        "Accuracy@1": 0.,
        "Accuracy@3": 0.,
        "Coverage": 0.
    }
    labels = np.array(labels)
    top1_predictions = []
    for p in predictions:
        single_prediction = 0 if len(p) == 0 else p[0]
        top1_predictions.append(single_prediction)
    top1_predictions = np.array(top1_predictions)

    positive_predictions_mask = top1_predictions > 0

    overall_metrics["Coverage"] = np.mean(np.array(top1_predictions) > 0)
    overall_metrics["Accuracy@1"] = np.mean(
        top1_predictions[positive_predictions_mask] ==
        labels[positive_predictions_mask])

    top_n_predictions = np.zeros((len(predictions), 3), dtype=np.int)
    for i, p in enumerate(predictions):
        top_n_predictions[i, :len(p)] = np.array(p)

    overall_metrics["Accuracy@3"] = np.mean(
        np.any(
            np.expand_dims(labels[positive_predictions_mask], axis=-1) ==
            top_n_predictions[positive_predictions_mask, :],
            axis=1
        )
    )
    return overall_metrics


def _sort_dict_by_keys(disordered: Dict):
    key = lambda item: item[0]
    sorted_dict = {k: v for k, v in sorted(disordered.items(), key=key)}
    return sorted_dict


def main(args, _=None):
    predictions = expit(np.load(args.in_npy))
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

    if not args.verbose:
        return

    print("CLASS METRICS")
    pprint(_sort_dict_by_keys(class_metrics))
    print("CLASS THRESHOLDS")
    pprint(_sort_dict_by_keys(class_thresholds))

    for use_threshold in [False, True]:
        model_predictions = get_model_predictions(
            predictions=predictions,
            thresholds=class_thresholds,
            classes=classes,
            use_threshold=use_threshold
        )
        model_metrics = score_model_predictions(model_predictions, labels)

        print(
            "MODEL METRICS WITH THRESHOLDS"
            if use_threshold
            else "MODEL METRICS WITHOUT THRESHOLDS"
        )
        pprint(_sort_dict_by_keys(model_metrics))


if __name__ == "__main__":
    args = parse_args()
    main(args)
