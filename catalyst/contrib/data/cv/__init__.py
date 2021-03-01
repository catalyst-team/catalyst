# flake8: noqa
import logging

from catalyst.settings import SETTINGS

logger = logging.getLogger(__name__)


try:
    from catalyst.contrib.data.cv.reader import ImageReader, MaskReader
    from catalyst.contrib.data.cv.dataset import ImageFolderDataset
except ImportError as ex:
    if SETTINGS.cv_required:
        logger.warning(
            "some of catalyst-cv dependencies are not available,"
            " to install dependencies, run `pip install catalyst[cv]`."
        )
        raise ex
