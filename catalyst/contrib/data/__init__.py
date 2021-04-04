# flake8: noqa

from catalyst.settings import SETTINGS

from catalyst.contrib.data.augmentor import (
    Augmentor,
    AugmentorCompose,
    AugmentorKeys,
)
from catalyst.contrib.data.reader import (
    IReader,
    ScalarReader,
    LambdaReader,
    ReaderCompose,
)

if SETTINGS.cv_required:
    from catalyst.contrib.data.cv import ImageReader, MaskReader, ImageFolderDataset

if SETTINGS.nifti_required:
    from catalyst.contrib.data.nifti import NiftiReader
