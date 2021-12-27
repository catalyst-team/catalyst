# flake8: noqa

from catalyst.settings import SETTINGS

from catalyst.contrib.datasets.cifar import CIFAR10, CIFAR100

if SETTINGS.cv_required:
    from catalyst.contrib.datasets.imagecar import CarvanaOneCarDataset

    from catalyst.contrib.datasets.imagenette import (
        Imagenette,
        Imagenette160,
        Imagenette320,
    )
    from catalyst.contrib.datasets.imagewang import (
        Imagewang,
        Imagewang160,
        Imagewang320,
    )
    from catalyst.contrib.datasets.imagewoof import (
        Imagewoof,
        Imagewoof160,
        Imagewoof320,
    )

    from catalyst.contrib.datasets.market1501 import (
        Market1501MLDataset,
        Market1501QGDataset,
    )

from catalyst.contrib.datasets.mnist import (
    MnistMLDataset,
    MnistQGDataset,
    MNIST,
    PartialMNIST,
)

if SETTINGS.ml_required:
    from catalyst.contrib.datasets.movielens import MovieLens

    if SETTINGS.is_torch_1_7_0:
        from catalyst.contrib.datasets.movielens import MovieLens20M
