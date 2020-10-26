# flake8: noqa
import logging

from catalyst.settings import SETTINGS

from catalyst.data.cv.transforms.torch import (
    Compose,
    Normalize,
    ToTensor,
)


logger = logging.getLogger(__name__)


try:
    from catalyst.data.cv.reader import ImageReader, MaskReader
    from catalyst.data.cv.dataset import ImageFolderDataset

    from catalyst.data.cv.mixins import BlurMixin, FlareMixin, RotateMixin

    from catalyst.data.cv.transforms.albumentations import (
        TensorToImage,
        ImageToTensor,
    )
    from catalyst.data.cv.transforms.kornia import (
        OneOfPerBatch,
        OneOfPerSample,
    )
except ImportError as ex:
    if SETTINGS.cv_required:
        logger.warning(
            "some of catalyst-cv dependencies are not available,"
            " to install dependencies, run `pip install catalyst[cv]`."
        )
        raise ex
