# flake8: noqa

from catalyst.settings import SETTINGS

from catalyst.contrib.data.reader import (
    IReader,
    ScalarReader,
    LambdaReader,
    ReaderCompose,
)

if SETTINGS.cv_required:
    from catalyst.contrib.data.reader_cv import ImageReader, MaskReader
    from catalyst.contrib.data.dataset_cv import ImageFolderDataset

if SETTINGS.nifti_required:
    from catalyst.contrib.data.reader_nifti import NiftiReader
