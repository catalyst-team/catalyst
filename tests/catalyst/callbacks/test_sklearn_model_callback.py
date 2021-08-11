# flake8: noqa

import pytest

from catalyst.callbacks import SklearnModelCallback
from catalyst.settings import SETTINGS


@pytest.mark.skipif(
    not (SETTINGS.ml_required), reason="catalyst[ml] required",
)
def test_init_from_str():

    pathes = [
        "ensemble.RandomForestClassifier",
        "linear_model.LogisticRegression",
        "cluster.KMeans",
    ]

    for fn in pathes:
        SklearnModelCallback(
            feature_key="feature_key",
            target_key="target_key",
            train_loader="train",
            valid_loader="valid_loader",
            sklearn_classifier_fn=fn,
        )

    pathes_with_transform = ["cluster.KMeans", "decomposition.PCA"]

    for fn in pathes_with_transform:
        SklearnModelCallback(
            feature_key="feature_key",
            target_key="target_key",
            train_loader="train",
            valid_loader="valid_loader",
            sklearn_classifier_fn=fn,
            predict_method="transform",
        )
