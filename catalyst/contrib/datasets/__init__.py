# flake8: noqa

from catalyst.settings import SETTINGS

if SETTINGS.cv_required:  # we need imread function here
    from catalyst.contrib.datasets.market1501 import (
        Market1501MLDataset,
        Market1501QGDataset,
    )

from catalyst.contrib.datasets.mnist import (
    MnistMLDataset,
    MnistQGDataset,
    MNIST,
)

if SETTINGS.ml_required:
    from catalyst.contrib.datasets.movielens import MovieLens

if SETTINGS.cv_required:
    from catalyst.contrib.datasets.cv import *
