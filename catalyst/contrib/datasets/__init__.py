# flake8: noqa

from catalyst.settings import SETTINGS

from catalyst.contrib.datasets.mnist import (
    MnistMLDataset,
    MnistQGDataset,
    MNIST,
)

if SETTINGS.use_ml:
    from catalyst.contrib.datasets.movielens import MovieLens

if SETTINGS.use_cv:
    from catalyst.contrib.datasets.cv import *
