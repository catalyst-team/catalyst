from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


def get_classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray = None, beta: float = None
) -> pd.DataFrame:
    """Generates pandas-based per-class and aggregated classification metrics.

    Args:
        y_true (np.ndarray): ground truth labels
        y_pred (np.ndarray): predicted model labels
        y_scores (np.ndarray): predicted model scores. Defaults to None.
        beta (float, optional): Beta parameter for custom Fbeta score computation.
            Defaults to None.

    Returns:
        pd.DataFrame: pandas dataframe with main classification metrics.

    Examples:

    .. code-block:: python

        from sklearn import datasets, linear_model, metrics
        from sklearn.model_selection import train_test_split
        from catalyst import utils

        digits = datasets.load_digits()

        # flatten the images
        n_samples = len(digits.images)
        data = digits.images.reshape((n_samples, -1))

        # Create a classifier
        clf = linear_model.LogisticRegression(multi_class="ovr")

        # Split data into 50% train and 50% test subsets
        X_train, X_test, y_train, y_test = train_test_split(
            data, digits.target, test_size=0.5, shuffle=False)

        # Learn the digits on the train subset
        clf.fit(X_train, y_train)

        # Predict the value of the digit on the test subset
        y_scores = clf.predict_proba(X_test)
        y_pred = clf.predict(X_test)

        utils.get_classification_report(
            y_true=y_test,
            y_pred=y_pred,
            y_scores=y_scores,
            beta=0.5
        )
    """
    metrics = defaultdict(lambda: {})
    metrics_names = [
        "precision",
        "recall",
        "f1-score",
        "auc",
        "support",
        "support (%)",
    ]
    avg_names = ["macro", "micro", "weighted"]
    labels = sorted(set(y_true).union(y_pred))
    auc = np.zeros(len(labels))
    if y_scores is not None:
        for i, label in enumerate(labels):
            auc[i] = roc_auc_score((y_true == label).astype(int), y_scores[:, i])

    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true=y_true, y_pred=y_pred, average=None, labels=labels
    )

    r_support = support / support.sum()
    for average in avg_names:
        avg_precision, avg_recall, avg_f1, _ = precision_recall_fscore_support(
            y_true=y_true, y_pred=y_pred, average=average, labels=labels
        )

        avg_metrics = avg_precision, avg_recall, avg_f1
        for k, v in zip(metrics_names[:4], avg_metrics):
            metrics[k][average] = v

    report = pd.DataFrame(
        [precision, recall, f1, auc, support, r_support], columns=labels, index=metrics_names
    ).T

    if beta is not None:
        _, _, fbeta, _ = precision_recall_fscore_support(
            y_true=y_true, y_pred=y_pred, average=None, beta=beta, labels=labels
        )
        avg_fbeta = np.zeros(len(avg_names))
        for i, average in enumerate(avg_names):
            _, _, avg_beta, _ = precision_recall_fscore_support(
                y_true=y_true, y_pred=y_pred, average=average, beta=beta, labels=labels
            )
            avg_fbeta[i] = avg_beta
        report.insert(3, "f-beta", fbeta, True)

    metrics["support"]["macro"] = support.sum()
    metrics["precision"]["accuracy"] = accuracy
    if y_scores is not None:
        metrics["auc"]["macro"] = roc_auc_score(
            y_true, y_scores, multi_class="ovr", average="macro"
        )
        metrics["auc"]["weighted"] = roc_auc_score(
            y_true, y_scores, multi_class="ovr", average="weighted"
        )
    metrics = pd.DataFrame(metrics, index=avg_names + ["accuracy"])

    result = pd.concat((report, metrics)).fillna("")

    if beta:
        result["f-beta"]["macro"] = avg_fbeta[0]
        result["f-beta"]["micro"] = avg_fbeta[1]
        result["f-beta"]["weighted"] = avg_fbeta[2]
    return result


__all__ = ["get_classification_report"]
