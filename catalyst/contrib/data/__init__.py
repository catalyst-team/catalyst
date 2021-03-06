# flake8: noqa
import logging

from catalyst.settings import SETTINGS

logger = logging.getLogger(__name__)

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

try:
    from catalyst.contrib.data.cv import ImageReader, MaskReader, ImageFolderDataset
except ImportError as ex:
    if SETTINGS.cv_required:
        logger.warning(
            "some of catalyst-cv dependencies are not available,"
            " to install dependencies, run `pip install catalyst[cv]`."
        )
        raise ex
