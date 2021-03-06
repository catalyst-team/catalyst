# flake8: noqa

from catalyst.settings import SETTINGS

from catalyst.contrib.datasets.mnist import (
    MnistMLDataset,
    MnistQGDataset,
    MNIST,
)

if SETTINGS.ml_required:
    from catalyst.contrib.datasets.movielens import MovieLens

if SETTINGS.cv_required:
    from catalyst.contrib.datasets.cv import *
